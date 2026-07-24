"""The `-ping` health-dashboard feature: rendering + the live-edit orchestration.

Layering (docs/PING_METADATA_PLAN.md §4):

  src/diagnostics.py  probes + versions, Discord-agnostic (reusable by healthz)
  src/ping.py         THIS FILE — turns probe results into an embed and drives the
                      optimistic-send + live-edit message loop
  src/musicbot.py     command registration only; the cog delegates straight here

Keeping the loop and the renderer out of musicbot.py is deliberate: that module is
the Discord command surface, not a home for per-feature logic.
"""

import asyncio
import math
from typing import Optional

import discord
import redis.asyncio as aioredis
from discord.ext import commands
from opentelemetry import trace
from opentelemetry.trace import Span

from src import diagnostics
from src.config import ENVIRONMENT
from src.diagnostics import ProbeResult, ProbeState
from src.spotify import Spotify
from src.util import get_logger, latency_color, send_embed, trace_footer

log = get_logger(__name__)

# ── Rendering (docs/PING_METADATA_PLAN.md §6.2) ────────────────────────────────
# One band table drives BOTH the status dot and the embed accent so a probe can't
# show, say, a green dot under a yellow accent. These bands (≤100/≤200) differ on
# purpose from util.latency_color's (≤50/≤100/≤200), which backs the lighter
# send_latency_line reply — don't swap one for the other.
_LATENCY_BANDS: tuple[tuple[float, str, int], ...] = (
    (100, "🟢", 0x44FF44),
    (200, "🟡", 0xFFD000),
    (float("inf"), "🟠", 0xFF6600),
)
_STATE_DOT = {
    ProbeState.PENDING: "⏳",
    ProbeState.NA: "⚪",
    ProbeState.OFF: "⚪",
    ProbeState.DOWN: "🔴",
    ProbeState.FAILED: "🔴",
}
_PING_RED = 0x990000
_PING_PROBING = 0x5865F2  # blurple: at least one row still pending


def _latency_band(ms: float) -> tuple[str, int]:
    # Total over every float, INCLUDING nan (nan <= cap is always False): fall back
    # to the worst band so an unknown latency can never StopIteration. discord.py's
    # Client.latency is nan whenever the gateway ws is down (reconnect window), which
    # is exactly when -ping is most likely to be run.
    return next(
        ((dot, hue) for cap, dot, hue in _LATENCY_BANDS if ms <= cap),
        _LATENCY_BANDS[-1][1:],
    )


def _ping_dot(r: ProbeResult) -> str:
    if r.state is not ProbeState.OK:
        return _STATE_DOT[r.state]
    return _latency_band(r.latency_ms or 0)[0]


def _ping_value(r: ProbeResult) -> str:
    if r.state is ProbeState.OK:
        return f"{round(r.latency_ms or 0)} ms"
    return {
        ProbeState.PENDING: "pending…",
        ProbeState.NA: "n/a",
        ProbeState.OFF: "off",
        ProbeState.DOWN: "down",
        ProbeState.FAILED: "failed",
    }[r.state]


def _ping_line(r: ProbeResult) -> str:
    return f"{_ping_dot(r)} {r.label:<16}{_ping_value(r)}"


def render_ping_embed(
    results: dict[str, ProbeResult],
    versions: dict[str, str],
    discord_ms: float,
    span: Span,
) -> discord.Embed:
    """Build the live health embed from the current probe results + versions.

    Accent: any down/failed → red; else any still-pending → blurple; else the
    worst OK latency's band colour (same bands as the dots)."""
    # discord.py reports nan latency while the gateway ws is reconnecting — show it
    # as down (red) rather than a bogus number, matching the old latency_color(nan).
    disc = (
        ProbeResult("Discord gateway", ProbeState.DOWN, detail="reconnecting")
        if math.isnan(discord_ms)
        else ProbeResult("Discord gateway", ProbeState.OK, latency_ms=discord_ms)
    )
    rows = [disc, *results.values()]
    lat_lines = [_ping_line(r) for r in rows]

    if any(r.state in (ProbeState.DOWN, ProbeState.FAILED) for r in rows):
        color = _PING_RED
    elif any(r.state is ProbeState.PENDING for r in rows):
        color = _PING_PROBING
    else:
        worst = max((r.latency_ms or 0) for r in rows)
        color = _latency_band(worst)[1]

    ver_lines = [
        f"Bot        {versions['bot']}",
        f"yt-dlp     {versions['yt-dlp']}",
        f"ffmpeg     {versions['ffmpeg']}",
        f"Python     {versions['python']}  ·  discord.py  {versions['discord.py']}",
    ]
    embed = discord.Embed(title="🏓 Pong — service health", color=discord.Color(color))
    embed.add_field(
        name="Latency", value="```\n" + "\n".join(lat_lines) + "\n```", inline=False
    )
    embed.add_field(
        name="Versions", value="```\n" + "\n".join(ver_lines) + "\n```", inline=False
    )
    footer = f"environment: {ENVIRONMENT}"
    if any(r.state is ProbeState.PENDING for r in rows):
        footer += " · probing…"
    if (tf := trace_footer(span)) is not None:
        footer += f" · {tf}"
    embed.set_footer(text=footer)
    return embed


def _ping_embed_changed(new: discord.Embed, old: discord.Embed) -> bool:
    """True when a re-render actually differs — the only things that move are the
    two field values, the colour, and the footer, all captured by to_dict()."""
    return new.to_dict() != old.to_dict()


async def _safe_edit(
    message: discord.Message, embed: discord.Embed
) -> Optional[discord.Embed]:
    """Edit a message, tolerating a host the user deleted mid-loop (mirrors
    musicplayer._progress_updater). Returns the embed on success, None if gone."""
    try:
        await message.edit(embed=embed)
        return embed
    except discord.NotFound:
        return None


# ── Commands' bodies ──────────────────────────────────────────────────────────


async def send_latency_line(ctx: commands.Context, bot_latency: float) -> None:
    """The lightweight one-line WS-latency reply. Used by -join (and cold -play,
    via join) so the common connect path never pays for the full health dashboard
    — see docs/PING_METADATA_PLAN.md §2.3. Sent through ctx.send so it still
    honours the Now Playing host machinery."""
    ms = bot_latency * 1000
    await send_embed(
        ctx,
        "Ping - latency in ms",
        f"Ping: **{round(ms)}** milliseconds!",
        latency_color(ms),
    )


async def run_health_dashboard(
    ctx: commands.Context,
    *,
    bot_latency: float,
    redis: Optional[aioredis.Redis],
    spotify: Spotify,
    pg_pool: Optional[object] = None,
) -> None:
    """Optimistic-send + live-edit health dashboard (docs/PING_METADATA_PLAN.md §5).

    Fires a skeleton embed immediately, then edits it in place each tick as probes
    return; a hard deadline fails any straggler. Runs inside the caller's span.
    Exceptions propagate — the command owns the user-facing error reply.
    """
    span = trace.get_current_span()
    loop = asyncio.get_running_loop()
    tasks: dict[str, asyncio.Task[ProbeResult]] = {}

    def _drain() -> bool:
        """Fold every finished probe task into `results`; True if a row moved.
        Only genuinely-completed tasks are done() here (cancellation happens at
        the deadline/finally), so .result() never re-raises."""
        changed = False
        for label in [lbl for lbl in pending if tasks[lbl].done()]:
            results[label] = tasks[label].result()
            pending.discard(label)
            changed = True
        return changed

    try:
        # 1. launch probes INSIDE try so `finally` cancels them no matter where
        #    a later await raises. create_task copies the otel context, so child
        #    probe spans (auto-instrumented Redis/aiohttp) nest under bot.ping.
        tasks = {
            "Redis": asyncio.create_task(diagnostics.probe_redis(redis)),
            "Spotify API": asyncio.create_task(diagnostics.probe_spotify(spotify)),
            "Postgres": asyncio.create_task(diagnostics.probe_postgres(pg_pool)),
            "OTEL collector": asyncio.create_task(diagnostics.probe_otel()),
        }
        results = {label: ProbeResult(label, ProbeState.PENDING) for label in tasks}
        pending = set(tasks)

        # 2. instant data + skeleton. collect_versions() awaits (executor hop),
        #    which lets the immediate NA/OFF probes complete; pre-drain them so
        #    those rows never flash "pending…" for a tick.
        versions = await diagnostics.collect_versions()
        _drain()
        discord_ms = bot_latency * 1000
        last = render_ping_embed(results, versions, discord_ms, span)
        message = await ctx.channel.send(embed=last)  # bypass NP host (§5.3)

        # 3. live-edit loop: tick, drain, edit-on-change; exit early when done.
        deadline = loop.time() + diagnostics.PING_DEADLINE_SECS
        while pending and (remaining := deadline - loop.time()) > 0:
            await asyncio.sleep(min(diagnostics.PING_TICK_SECS, remaining))
            if _drain():
                embed = render_ping_embed(results, versions, discord_ms, span)
                if _ping_embed_changed(embed, last):
                    last = await _safe_edit(message, embed) or last

        # 4. deadline: fail only what is STILL pending. Re-check done() first —
        #    a probe can finish during step 3's final edit await, and flipping
        #    it to FAILED unconditionally would report a healthy dep as red.
        if pending:
            for label in pending:
                if tasks[label].done():
                    results[label] = tasks[label].result()
                else:
                    tasks[label].cancel()
                    results[label] = ProbeResult(label, ProbeState.FAILED)
            await _safe_edit(
                message, render_ping_embed(results, versions, discord_ms, span)
            )

        for r in results.values():  # self-documenting trace
            span.set_attribute(f"ping.{r.label}.state", r.state.name.lower())
            if r.latency_ms is not None:
                span.set_attribute(f"ping.{r.label}.latency_ms", round(r.latency_ms, 2))
    finally:
        for t in tasks.values():  # never leak a probe task
            if not t.done():
                t.cancel()
