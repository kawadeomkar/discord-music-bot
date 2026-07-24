"""
One-shot migration CLI: Redis guild:{id}:history lists → Postgres play_history
(docs/POSTGRES_HISTORY_PLAN.md §5.6, runbook §6).

    poetry run backfill-history [--verify] [--demote]

Always backfills first (idempotent by construction — the play_history_dedup
unique index + ON CONFLICT DO NOTHING make re-runs and overlap with the live
bot's dual-writes harmless; a second pass inserts only what was played since
the first). Then:

  --verify   per guild: parseable Redis entries ≤ Postgres rows (PG may
             exceed via dual-writes; smaller = fail) and the newest Redis
             entry exists in PG (drain lag makes this transiently fail —
             re-run). Exit 1 on any failure.
  --demote   refuses to run unless --verify passed in this same invocation;
             LTRIM 0,49 + EXPIRE each history list — catches at-rest lists
             from idle guilds that the (lazy, per-push) cutover writer never
             touches.

Safe to run while either bot build is live. Reads REDIS_URL and POSTGRES_URL
from the environment (same variables as the bot). Unlike the bot — which gets
.env injected by docker compose's env_file — this CLI runs on the host, so
main() loads the project .env itself (existing shell env always wins).
"""

import argparse
import asyncio
import dataclasses
import os
import sys
from collections.abc import Callable
from typing import Optional

import redis.asyncio as aioredis
from dotenv import load_dotenv

from src.db import Database
from src.guild_state import HistoryEntry, parse_history_entry
from src.history_archive import PostgresHistoryArchive
from src.redis_client import GUILD_TTL, HISTORY_CACHE_LIMIT
from src.util import get_logger

log = get_logger(__name__)

_HISTORY_KEY_PATTERN = "guild:*:history"
_BATCH = 500


def _guild_id_from_key(key: bytes) -> Optional[int]:
    # b"guild:{id}:history" — the key, not the entry, is authoritative for
    # guild ownership (pre-Phase-A entries carry no guild_id at all, and a
    # per-guild list can only hold that guild's plays).
    parts = key.split(b":")
    if len(parts) != 3 or parts[0] != b"guild" or parts[2] != b"history":
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


async def _read_guild_entries(
    redis: aioredis.Redis, key: bytes, guild_id: int
) -> tuple[list[HistoryEntry], int]:
    """All parseable entries of one list, oldest-first, guild_id stamped from
    the key. Returns (entries, corrupt_count)."""
    raw: list[bytes] = await redis.lrange(key, 0, -1)  # type: ignore[misc]
    entries: list[HistoryEntry] = []
    corrupt = 0
    for item in raw:
        parsed = parse_history_entry(item)
        if parsed is None:
            corrupt += 1
            continue
        entries.append(dataclasses.replace(parsed, guild_id=guild_id))
    entries.reverse()  # LPUSH stores newest-first; insert oldest-first
    return entries, corrupt


async def run(
    redis: aioredis.Redis,
    archive: PostgresHistoryArchive,
    *,
    verify: bool = False,
    demote: bool = False,
    out: Callable[[str], object] = print,
) -> int:
    """The testable core; returns the process exit code."""
    if demote and not verify:
        out("--demote requires --verify in the same invocation")
        return 2

    # ── backfill (always) ────────────────────────────────────────────────
    keys: list[bytes] = []
    async for key in redis.scan_iter(match=_HISTORY_KEY_PATTERN):
        if _guild_id_from_key(key) is not None:
            keys.append(key)
    keys.sort()

    per_guild: dict[int, tuple[bytes, list[HistoryEntry], int]] = {}
    for key in keys:
        gid = _guild_id_from_key(key)
        assert gid is not None
        entries, corrupt = await _read_guild_entries(redis, key, gid)
        per_guild[gid] = (key, entries, corrupt)
        before = await archive.count(gid)
        for i in range(0, len(entries), _BATCH):
            await archive.insert_batch(entries[i : i + _BATCH])
        inserted = await archive.count(gid) - before
        out(
            f"guild {gid}: redis={len(entries)} inserted={inserted} "
            f"dup={len(entries) - inserted} corrupt={corrupt}"
        )
    if not keys:
        out("no guild history lists found")

    # ── --verify ─────────────────────────────────────────────────────────
    if verify:
        failures = 0
        for gid, (key, entries, _corrupt) in per_guild.items():
            pg_count = await archive.count(gid)
            if pg_count < len(entries):
                out(f"VERIFY FAIL guild {gid}: pg={pg_count} < redis={len(entries)}")
                failures += 1
                continue
            if entries:
                newest = entries[-1]  # oldest-first list → newest is last
                pg_newest = await archive.recent(gid, 1)
                identity = (newest.played_at, newest.webpage_url)
                if (
                    not pg_newest
                    or (
                        pg_newest[0].played_at,
                        pg_newest[0].webpage_url,
                    )
                    != identity
                ):
                    out(
                        f"VERIFY FAIL guild {gid}: newest Redis entry not "
                        f"newest in PG (drain lag? re-run)"
                    )
                    failures += 1
                    continue
            out(f"verify ok guild {gid}: pg={pg_count} >= redis={len(entries)}")
        if failures:
            out(f"verification FAILED for {failures} guild(s)")
            return 1
        out("verification passed")

    # ── --demote (gated on verify having passed above) ───────────────────
    if demote:
        for gid, (key, _entries, _corrupt) in per_guild.items():
            pipe = redis.pipeline()
            pipe.ltrim(key, 0, HISTORY_CACHE_LIMIT - 1)
            pipe.expire(key, GUILD_TTL)
            await pipe.execute()
            out(f"demoted guild {gid}: trimmed to {HISTORY_CACHE_LIMIT}, TTL armed")
        out(f"demoted {len(per_guild)} history list(s)")

    return 0


async def _amain(args: argparse.Namespace) -> int:
    postgres_url = os.getenv("POSTGRES_URL")
    if not postgres_url:
        print("POSTGRES_URL is not set")
        return 2
    redis = aioredis.from_url(
        os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=False
    )
    db = Database(postgres_url)
    try:
        return await run(
            redis,
            PostgresHistoryArchive(db),
            verify=args.verify,
            demote=args.demote,
        )
    finally:
        await db.close()
        await redis.aclose()


def main() -> None:
    # Host-run CLI: compose isn't in the loop to inject .env, so load it here.
    # load_dotenv never overrides variables already set in the shell.
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="One-shot migration: Redis guild history lists → Postgres "
        "play_history (docs/POSTGRES_HISTORY_PLAN.md §5.6)"
    )
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--demote", action="store_true")
    sys.exit(asyncio.run(_amain(parser.parse_args())))


if __name__ == "__main__":
    main()
