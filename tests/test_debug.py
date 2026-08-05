"""Debug mode's footer: what every reply grows while the mode is on.

Observation-only by rule, so what is asserted here is entirely what is DISPLAYED.
The seam that applies it to real command responses is tested in test_context.py.
"""

import discord
from opentelemetry import trace as trace_api

from src import debug


class TestDebugFooter:
    """Every segment is optional because every one has a genuine absent case: a
    send outside any command has no elapsed time, a DM has no shard, and an
    unsampled span has no trace id."""

    def test_nothing_known_renders_nothing(self) -> None:
        """An empty string, not a lone marker — a footer that says only "🐞" is
        noise on every reply and tells the operator nothing."""
        assert debug.debug_footer() == ""

    def test_elapsed_and_shard_render(self) -> None:
        footer = debug.debug_footer(elapsed_ms=12.4, shard_id=3)
        assert "12 ms" in footer
        assert "shard 3" in footer

    def test_shard_zero_is_not_mistaken_for_absent(self) -> None:
        """Shard 0 is a real shard and falsy; `is not None` is what separates them."""
        assert "shard 0" in debug.debug_footer(shard_id=0)

    def test_an_embeds_own_footer_is_kept(self) -> None:
        embed = discord.Embed(description="x")
        embed.set_footer(text="Requested by someone")
        debug.decorate_embeds([embed], elapsed_ms=5.0)
        text = embed.footer.text or ""
        assert text.startswith("Requested by someone")
        assert "5 ms" in text

    def test_an_icon_url_survives_decoration(self) -> None:
        """set_footer replaces the whole footer object, so the icon has to be
        carried across explicitly or every decorated reply silently loses it."""
        embed = discord.Embed(description="x")
        embed.set_footer(text="by someone", icon_url="https://example.invalid/a.png")
        debug.decorate_embeds([embed], elapsed_ms=5.0)
        assert embed.footer.icon_url == "https://example.invalid/a.png"

    def test_an_embed_with_nothing_to_add_is_left_alone(self) -> None:
        embed = discord.Embed(description="x")
        debug.decorate_embeds([embed])
        assert embed.footer.text is None

    def test_the_footer_is_truncated_to_discords_cap(self) -> None:
        embed = discord.Embed(description="x")
        embed.set_footer(text="A" * (debug.FOOTER_LIMIT + 500))
        debug.decorate_embeds([embed], elapsed_ms=1.0)
        assert len(embed.footer.text or "") <= debug.FOOTER_LIMIT

    def test_every_embed_in_the_list_is_decorated(self) -> None:
        embeds = [discord.Embed(description=str(i)) for i in range(3)]
        debug.decorate_embeds(embeds, elapsed_ms=7.0)
        assert all("7 ms" in (e.footer.text or "") for e in embeds)


class TestTraceIdInTheFooter:
    """The trace id is the whole point of debug mode — the help text says "paste
    that id to the operator". Nothing else in the suite renders one: conftest
    installs no TracerProvider, so `trace_id_of()` answers "" everywhere and every
    trace assertion elsewhere holds vacuously. These build a real span context.
    """

    TRACE_ID = 0x4BF92F3577B34DA6A3CE929D0E0E4736
    HEX = "4bf92f3577b34da6a3ce929d0e0e4736"

    @classmethod
    def _span(cls) -> trace_api.Span:
        return trace_api.NonRecordingSpan(
            trace_api.SpanContext(
                trace_id=cls.TRACE_ID,
                span_id=0x00F067AA0BA902B7,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(trace_api.TraceFlags.SAMPLED),
            )
        )

    def test_the_footer_carries_the_trace_id(self) -> None:
        footer = debug.debug_footer(span=self._span(), elapsed_ms=12.0)
        assert f"trace {self.HEX}" in footer

    def test_an_embed_gets_the_trace_id(self) -> None:
        embed = discord.Embed(description="x")
        debug.decorate_embeds([embed], span=self._span(), shard_id=0)
        assert self.HEX in (embed.footer.text or "")

    def test_an_error_embeds_existing_trace_is_not_repeated(self) -> None:
        """_command_error already puts `trace: <id>` on error embeds. Twice in one
        footer reads as two different traces."""
        embed = discord.Embed(description="x")
        embed.set_footer(text=f"trace: {self.HEX}")
        debug.decorate_embeds([embed], span=self._span(), elapsed_ms=5.0)
        assert (embed.footer.text or "").count(self.HEX) == 1

    def test_skip_trace_is_what_suppresses_it(self) -> None:
        """Pins the guard itself, not just its effect: flipping skip_trace has to
        be the thing that changes the output."""
        span = self._span()
        assert self.HEX in debug.debug_footer(span=span, skip_trace=False)
        assert self.HEX not in debug.debug_footer(span=span, skip_trace=True)

    def test_an_unsampled_span_contributes_nothing(self) -> None:
        """INVALID_SPAN has an all-zero trace id, which trace_id_of reports as "".
        Rendering it would put a meaningless id in front of a user who is being
        told to paste it to an operator."""
        assert debug.debug_footer(span=trace_api.INVALID_SPAN) == ""
