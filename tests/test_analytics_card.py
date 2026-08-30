"""Tests for src/analytics_card.py — the -analytics window allowlist, cache codec
and embed, plus the cog command that drives them.

The command lives on MusicBot because that is where dispatch and the archive handle
are, but every test of it belongs with the module it renders through — the same split
test_leaderboard.py and test_debug.py already make.
"""

import asyncio
import contextlib
import time
from concurrent.futures.process import BrokenProcessPool
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import orjson
import pytest

from src import analytics_card, analytics_render, chart_pool
from src.analytics_card import ALLOWED_DAYS, TOP_N
from src.guild_state import (
    WAIT_UNAVAILABLE,
    AnalyticsMetrics,
    CompletionBucket,
    DailyPoint,
    DurationBucket,
    HeatCell,
    SourceCompletion,
    SourceDay,
    TopArtist,
    TopListener,
    TopSong,
)
from src.history_archive import PostgresHistoryArchive
from src.musicbot import MusicBot
from src.ytdlp_pool import PoolClosedError, RemoteCallError
from tests.helpers import command_callback

# A UTC midnight, so the window arithmetic in the tests reads as dates rather than
# as arbitrary offsets.
_TODAY = 1_787_961_600.0


def _metrics(**over: Any) -> AnalyticsMetrics:
    """A REAL frozen AnalyticsMetrics with plausible defaults, overridable per field.

    Real rather than mocked because the embed builder reads ~15 attributes and a bare
    MagicMock answers all of them truthily — including ones a regression deleted — so
    a dispatch test would pass against a card rendering `<MagicMock id=0x…>`.
    conftest.py records two prior incidents of exactly this shape.
    """
    base: dict[str, Any] = dict(
        days=30,
        window_start_epoch=_TODAY - 30 * 86400,
        window_end_epoch=_TODAY,
        today_start_epoch=_TODAY,
        bucket_unit="day",
        archived_days=30,
        plays=156,
        listen_secs=26_909,
        unique_songs=118,
        unique_listeners=4,
        unique_artists=82,
        wait_p50_secs=38.0,
        livestream_plays=1,
        daily=(DailyPoint(day="2026-08-27", plays=12, listen_secs=2400),),
        daily_by_source=(SourceDay(day="2026-08-27", source="search", plays=12),),
        heat=(HeatCell(dow=4, hour=22, plays=41),),
        completion=(CompletionBucket(source="search", bucket=10, plays=122),),
        durations=(DurationBucket(minutes=3, plays=60),),
        source_completion=(
            SourceCompletion(source="search", played_secs=20_000, duration_secs=22_000),
        ),
        wait_pcts=(1.0, 3.2, 38.0, 192.0, 414.0),
        top_listeners=(
            TopListener(
                requester_id=7, requester_name="Ann", plays=133, played_secs=20_000
            ),
        ),
        top_artists=(TopArtist(uploader="Lofi Girl", plays=11, played_secs=3000),),
        top_songs=(
            TopSong(
                title="Know My Name",
                webpage_url="https://yt.com/v=1",
                query_source="search",
                plays=3,
                played_secs=600,
            ),
        ),
    )
    return AnalyticsMetrics(**(base | over))


def _flags(days: int = 30) -> SimpleNamespace:
    """Stand-in for a parsed AnalyticsFlags — a FlagConverter cannot be constructed
    directly, and the command body only reads .days."""
    return SimpleNamespace(days=days)


def _fake_archive(metrics: AnalyticsMetrics) -> MagicMock:
    """Specced, because ArchiveReader is a Protocol: structural and unchecked at
    runtime, so a bare double that forgot analytics() satisfies both the old and the
    new protocol and fails later, at an await on a non-coroutine, somewhere
    unrelated."""
    archive = MagicMock(spec=PostgresHistoryArchive)
    archive.analytics = AsyncMock(return_value=metrics)
    return archive


class TestWindowAllowlist:
    @pytest.mark.parametrize("days", ALLOWED_DAYS)
    def test_the_four_windows_resolve(self, days: int) -> None:
        assert analytics_card.resolve_days(days) == days

    @pytest.mark.parametrize("days", [0, 1, 6, 8, 29, 31, 364, 366, -7, 10_000])
    def test_everything_else_is_refused(self, days: int) -> None:
        assert analytics_card.resolve_days(days) is None

    def test_the_key_space_is_closed(self) -> None:
        """The whole point: a free-form 1..365 range means the cache can never hit —
        `--days 1, 2, 3, …` is 365 guaranteed misses, restartable indefinitely, each
        holding one of the archive pool's four connections while the drainer competes
        for the same four."""
        assert len(set(ALLOWED_DAYS)) == len(ALLOWED_DAYS)
        assert analytics_card.DEFAULT_DAYS in ALLOWED_DAYS

    def test_the_notice_names_every_allowed_value(self) -> None:
        notice = analytics_card.invalid_days_notice()
        for days in ALLOWED_DAYS:
            assert str(days) in notice


class TestCacheKeys:
    def test_the_key_carries_guild_window_and_codec_version(self) -> None:
        key = analytics_card.cache_key(99, 30)
        assert key.startswith("analytics:agg:v")
        assert key.endswith(":99:30")

    def test_windows_do_not_share_an_entry(self) -> None:
        keys = {analytics_card.cache_key(99, d) for d in ALLOWED_DAYS}
        assert len(keys) == len(ALLOWED_DAYS)

    def test_there_is_no_timezone_component(self) -> None:
        """The query takes no zone parameter, so a second key dimension would be
        meaningless — and an unbounded user-controlled one would reopen the
        key-space hole the allowlist closes."""
        assert analytics_card.cache_key(99, 30).count(":") == 4

    def test_the_png_key_is_bound_to_the_aggregate_it_rendered(self) -> None:
        """A stale PNG must MISS, not render beside numbers recomputed hours later
        with nothing on the card to say the two disagree."""
        a = analytics_card.png_cache_key(99, 30, "abc")
        b = analytics_card.png_cache_key(99, 30, "def")
        assert a != b
        assert "analytics:png:v" in a

    def test_the_png_and_aggregate_key_spaces_do_not_collide(self) -> None:
        assert not analytics_card.png_cache_key(99, 30, "d").startswith(
            analytics_card.cache_key(99, 30)
        )


class TestPngKeyVersioning:
    def test_the_png_key_moves_with_the_renderer(self) -> None:
        """The digest fingerprints the AGGREGATE, which cannot see a palette or
        layout change. Without the render's own version in the key, deploying one at
        09:00 UTC serves every guild that ran earlier that day the old chart beside
        identical numbers, until midnight."""
        key = analytics_card.png_cache_key(1, 30, "deadbeef")
        assert analytics_render.RENDER_VERSION in key
        assert analytics_render.RENDER_VERSION not in analytics_card.cache_key(1, 30)


class TestCacheTtl:
    def test_the_ttl_runs_to_the_next_utc_midnight(self) -> None:
        """Not a fixed interval: that is exactly when the answer changes. The window
        is N COMPLETE days, so the cached artifact is immutable for the whole of its
        life — there is no partial 'today' inside it to go stale."""
        m = _metrics(today_start_epoch=_TODAY)
        assert analytics_card.cache_ttl_secs(m, now=_TODAY) == 86400
        assert analytics_card.cache_ttl_secs(m, now=_TODAY + 86399) == 1

    def test_a_query_that_straddled_midnight_is_not_cached(self) -> None:
        """The race the derivation exists for: an aggregate computed a millisecond
        before midnight, stored a millisecond after, would otherwise be served for
        the whole of a day it does not cover. A non-positive TTL is the signal."""
        m = _metrics(today_start_epoch=_TODAY)
        assert analytics_card.cache_ttl_secs(m, now=_TODAY + 86400 + 0.5) <= 0

    def test_the_ttl_uses_the_querys_clock_not_a_second_read(self) -> None:
        # today_start_epoch comes back from the SQL that resolved the buckets, so
        # the cache and the chart cannot disagree about when the day turned.
        m = _metrics(today_start_epoch=_TODAY - 86400)
        assert analytics_card.cache_ttl_secs(m, now=_TODAY) == 0

    def test_an_unstamped_aggregate_is_not_cached(self) -> None:
        assert analytics_card.cache_ttl_secs(_metrics(today_start_epoch=0.0)) == 0

    def test_the_weekly_window_still_expires_daily(self) -> None:
        """Its content could not change until next week, so a day is conservative —
        which is the right direction. One extra recompute per guild per day beats
        reasoning about a seven-day entry."""
        m = _metrics(days=365, bucket_unit="week", today_start_epoch=_TODAY)
        assert analytics_card.cache_ttl_secs(m, now=_TODAY) == 86400


class TestCacheCodec:
    def test_round_trip_through_orjson_is_lossless(self) -> None:
        m = _metrics()
        decoded = analytics_card.from_cache(
            orjson.loads(orjson.dumps(analytics_card.to_cache(m)))
        )
        assert decoded == m

    def test_an_empty_window_round_trips(self) -> None:
        """None must mean MALFORMED, never 'empty': an empty window is a valid
        cached value, and caching it is what stops an idle guild re-querying
        Postgres on every invocation."""
        m = AnalyticsMetrics(days=30, today_start_epoch=_TODAY)
        decoded = analytics_card.from_cache(
            orjson.loads(orjson.dumps(analytics_card.to_cache(m)))
        )
        assert decoded == m
        assert decoded is not None and decoded.is_empty

    @pytest.mark.parametrize(
        "raw", [None, [], "x", 3, {"days": 30}, {"daily": "notalist"}]
    )
    def test_malformed_entries_decode_to_none(self, raw: object) -> None:
        assert analytics_card.from_cache(raw) is None

    def test_a_renamed_field_is_a_miss_not_a_silent_wrong_card(self) -> None:
        """The wire names are spelled out in _WIRE rather than taken from the
        dataclass, so a Python attribute rename breaks the decode loudly."""
        blob = analytics_card.to_cache(_metrics())
        blob["daily"] = [{"date": "2026-08-27", "plays": 1, "listen_secs": 1}]
        assert analytics_card.from_cache(blob) is None

    def test_every_metrics_field_is_carried(self) -> None:
        """A new field that to_cache forgets survives a round trip as its DEFAULT,
        so the cached card silently differs from the fresh one. Compares against the
        dataclass itself, so adding a field fails this rather than shipping."""
        carried = (
            set(analytics_card._SCALARS) | set(analytics_card._WIRE) | {"wait_pcts"}
        )
        assert carried == set(AnalyticsMetrics.__dataclass_fields__)


class TestEmbed:
    def test_the_title_names_the_requested_window(self) -> None:
        """FlagConverter silently defaults every input it does not recognise —
        `--days=7` (its delimiter is a space), a bare `--days`, a positional `7` —
        so all three render 30 days while the user believes they asked for 7. The
        period is named where the answer is read; a footer under an 1100px image is
        not that place."""
        embed = analytics_card.build_embed(_metrics(days=90, archived_days=90))
        assert "last 90 days" in (embed.title or "")

    def test_a_sparse_guild_sees_both_the_window_and_its_coverage(self) -> None:
        """ "with plays", not "archived": first_play is min() over the window SLICE, so
        a guild archiving for a year that went quiet last month reports the gap, not
        its coverage. Naming it "archived" contradicts -leaderboard --days 365."""
        embed = analytics_card.build_embed(_metrics(days=30, archived_days=2))
        title = embed.title or ""
        assert "last 30 days" in title
        assert "2 days with plays" in title

    def test_full_coverage_does_not_mention_it(self) -> None:
        assert "with plays" not in (analytics_card.build_embed(_metrics()).title or "")

    def test_a_weekly_window_is_named_in_weeks(self) -> None:
        """365 mod 7 is 1, so date_trunc snaps back six days and the window covers 53
        whole weeks. "last 365 days" names six fewer days than the chart draws."""
        embed = analytics_card.build_embed(
            _metrics(days=365, bucket_unit="week", archived_days=365)
        )
        assert "last 53 weeks" in (embed.title or "")

    def test_a_cached_scalar_of_the_wrong_type_is_a_miss(self) -> None:
        """Shape was validated and TYPE was not, so a build with a different field
        type decoded into a plausible AnalyticsMetrics and raised inside build_embed
        -- outside every catch that degrades to a card."""
        blob = analytics_card.to_cache(_metrics())
        blob["days"] = "thirty"
        assert analytics_card.from_cache(blob) is None

    def test_sections_live_in_the_description_not_in_fields(self) -> None:
        """A field value caps at 1024 characters and five masked-link lines do not
        reliably fit; the 4096-char description does."""
        embed = analytics_card.build_embed(_metrics())
        assert embed.fields == []
        for header in ("Top listeners", "Top artists", "Top songs"):
            assert header in (embed.description or "")

    def test_the_topline_carries_every_headline_number(self) -> None:
        """Each number asserted WITH its label. Bare substrings let any two of these
        transpose and still pass, and four of them are plain counts."""
        embed = analytics_card.build_embed(
            _metrics(
                daily=(
                    DailyPoint(day="2026-08-26", plays=6, listen_secs=1200),
                    DailyPoint(day="2026-08-27", plays=12, listen_secs=2400),
                )
            )
        )
        desc = embed.description or ""
        for fragment in (
            "**156** plays",
            "**7:28:29** listened",
            "**118** unique songs",
            "**4** listeners",
            "**82** artists",
            "Median queue wait **38s**",
            # 156 // 2 buckets. A one-bucket fixture makes the division an identity,
            # so dropping it entirely would read as correct.
            "**78** plays on the average active day",
        ):
            assert fragment in desc, f"{fragment!r} missing from {desc!r}"

    def test_a_member_still_present_is_mentioned(self) -> None:
        guild = MagicMock(spec=discord.Guild)
        guild.get_member.return_value = MagicMock()
        embed = analytics_card.build_embed(_metrics(), guild=guild)
        assert "<@7>" in (embed.description or "")

    def test_a_departed_member_falls_back_to_the_archived_name(self) -> None:
        """Discord renders a mention for a non-member as a raw id."""
        guild = MagicMock(spec=discord.Guild)
        guild.get_member.return_value = None
        embed = analytics_card.build_embed(_metrics(), guild=guild)
        desc = embed.description or ""
        assert "<@7>" not in desc
        assert "Ann" in desc

    def test_the_footer_states_utc_and_that_today_is_missing(self) -> None:
        """Both are surprising: a UTC day boundary is 17:00 in US/Pacific, and a
        guild whose only plays are from this morning is empty to this command."""
        footer = (analytics_card.build_embed(_metrics()).footer.text or "").lower()
        assert "utc" in footer
        assert "today is not included" in footer

    def test_the_image_is_attached_only_when_one_was_rendered(self) -> None:
        assert analytics_card.build_embed(_metrics()).image.url is None
        embed = analytics_card.build_embed(
            _metrics(), image_filename=analytics_card.IMAGE_FILENAME
        )
        assert embed.image.url == f"attachment://{analytics_card.IMAGE_FILENAME}"

    def test_an_unavailable_wait_is_not_reported_as_zero(self) -> None:
        """Every queued_at in the window was the epoch-0 backfill sentinel. A
        confident "0s" would read as "songs play instantly here"."""
        embed = analytics_card.build_embed(
            _metrics(wait_p50_secs=WAIT_UNAVAILABLE, wait_pcts=())
        )
        desc = embed.description or ""
        assert "n/a" in desc
        assert "0s" not in desc


class TestEmbedSafety:
    """Every string on this card comes from the archive, where title, uploader and
    requester_name are bare `text` columns with no CHECK and HistoryEntry strips only
    NUL — so they arrive unbounded and attacker-influenceable."""

    def test_a_bracket_bearing_uploader_cannot_forge_a_masked_link(self) -> None:
        hostile = "x](https://evil.example)[y"
        embed = analytics_card.build_embed(
            _metrics(top_artists=(TopArtist(uploader=hostile, plays=1),))
        )
        desc = embed.description or ""
        assert "](https://evil.example)" not in desc
        assert "evil.example" in desc  # flattened, not silently dropped

    def test_a_bracket_bearing_title_cannot_escape_its_own_label(self) -> None:
        embed = analytics_card.build_embed(
            _metrics(
                top_songs=(
                    TopSong(
                        title="a](https://evil.example)[b",
                        webpage_url="https://yt.com/v=1",
                        plays=1,
                    ),
                )
            )
        )
        assert "](https://evil.example)" not in (embed.description or "")

    def test_a_departed_members_archived_name_is_sanitized(self) -> None:
        guild = MagicMock(spec=discord.Guild)
        guild.get_member.return_value = None
        embed = analytics_card.build_embed(
            _metrics(
                top_listeners=(
                    TopListener(requester_id=7, requester_name="**bold**", plays=1),
                )
            ),
            guild=guild,
        )
        assert "**bold**" not in (embed.description or "")

    @pytest.mark.parametrize(
        "url",
        [
            "https://y/a(b)",
            "https://y/a b",
            "https://y/a\nb",
            "https://y/" + "x" * 300,
            "",
        ],
        ids=["parens", "space", "newline", "too-long", "empty"],
    )
    def test_an_unsafe_url_renders_as_plain_text(self, url: str) -> None:
        """Both halves of a masked link come from the archive, and webpage_url has
        no CHECK either. A paren, whitespace or control character ends the markdown
        early and leaks the rest of the line."""
        embed = analytics_card.build_embed(
            _metrics(top_songs=(TopSong(title="T", webpage_url=url, plays=1),))
        )
        assert "[T](" not in (embed.description or "")

    def test_a_blank_title_still_renders_a_visible_label(self) -> None:
        """A blank title is a real archived value, and an empty masked-link label
        renders as an invisible link."""
        embed = analytics_card.build_embed(
            _metrics(top_songs=(TopSong(title="", webpage_url="https://y/1", plays=1),))
        )
        assert "[Unknown](https://y/1)" in (embed.description or "")

    def test_the_worst_case_card_fits_discords_budget(self) -> None:
        """6000 characters across EVERY embed in the message, which this shares with
        the <=3-embed Now Playing block MusicContext.send prepends; a description
        caps at 4096. Both failures are a 400 on the whole send, AFTER the query and
        the render are paid, so neither is recoverable where it fires."""
        long = "Z" * 400
        embed = analytics_card.build_embed(
            _metrics(
                top_listeners=tuple(
                    TopListener(requester_id=0, requester_name=long, plays=999_999)
                    for _ in range(TOP_N)
                ),
                top_artists=tuple(
                    TopArtist(uploader=long, plays=999_999, played_secs=999_999)
                    for _ in range(TOP_N)
                ),
                top_songs=tuple(
                    TopSong(
                        title=long,
                        webpage_url="https://yt.com/" + "u" * 130,
                        plays=999_999,
                        played_secs=999_999,
                    )
                    for _ in range(TOP_N)
                ),
            )
        )
        assert len(embed.description or "") <= 4096
        # ~1500 reserved for the NP block the send prepends.
        assert len(embed) <= 6000 - 1500

    def test_top_n_is_five_because_ten_would_not_fit(self) -> None:
        """The arithmetic picks the constant; this is the guard on raising it
        without redoing it. -leaderboard affords ten because it renders TWO lists —
        this card renders three plus a topline."""
        assert TOP_N == 5


class TestAnalyticsCommand:
    async def test_archive_disabled_sends_notice_and_queries_nothing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.history_archive = None
        await command_callback(MusicBot.analytics)(music_bot, mock_ctx, flags=_flags())
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "long-term play archive" in embed.description

    async def test_dm_invocation_sends_server_only_notice(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.guild = None
        archive = _fake_archive(_metrics())
        music_bot.history_archive = archive
        await command_callback(MusicBot.analytics)(music_bot, mock_ctx, flags=_flags())
        assert "per server" in mock_ctx.send.call_args[1]["embed"].description
        archive.analytics.assert_not_awaited()

    @pytest.mark.parametrize("days", [1, 45, 366, 0, -3])
    async def test_an_unlisted_window_never_reaches_postgres(
        self, music_bot: MusicBot, mock_ctx: MagicMock, days: int
    ) -> None:
        """A rejection must not take a read slot either — that is the whole point of
        rejecting before the archive rather than inside it. It also hands the guild's
        cooldown slot back, since the cooldown's only job is protecting the archive
        and this never reached it."""
        archive = _fake_archive(_metrics())
        music_bot.history_archive = archive
        mock_ctx.command.reset_cooldown = MagicMock()
        await command_callback(MusicBot.analytics)(
            music_bot, mock_ctx, flags=_flags(days)
        )
        archive.analytics.assert_not_awaited()
        mock_ctx.command.reset_cooldown.assert_called_once_with(mock_ctx)
        embed = mock_ctx.send.call_args[1]["embed"]
        assert embed.color == discord.Color.red()
        assert "7, 30, 90, 365" in embed.description

    @pytest.mark.parametrize(
        "setup",
        [
            lambda ctx, bot: setattr(ctx, "guild", None),
            lambda ctx, bot: setattr(bot, "history_archive", None),
        ],
        ids=["no-guild", "no-archive"],
    )
    async def test_a_refusal_that_never_reaches_postgres_hands_the_slot_back(
        self, music_bot: MusicBot, mock_ctx: MagicMock, setup: Any
    ) -> None:
        """discord.py charges the cooldown in prepare(), before the body. These two
        refusals never reach Postgres, so holding the guild's slot for 30s afterwards
        refuses a corrected retry over a query that never ran."""
        setup(mock_ctx, music_bot)
        mock_ctx.command.reset_cooldown = MagicMock()
        await command_callback(MusicBot.analytics)(music_bot, mock_ctx, flags=_flags())
        mock_ctx.command.reset_cooldown.assert_called_once_with(mock_ctx)

    async def test_a_listed_window_renders_the_card(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.history_archive = _fake_archive(_metrics())
        await command_callback(MusicBot.analytics)(music_bot, mock_ctx, flags=_flags())
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "Analytics" in embed.title
        assert "Top songs" in embed.description

    async def test_the_command_asks_for_its_own_row_limit(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        archive = _fake_archive(_metrics())
        music_bot.history_archive = archive
        await command_callback(MusicBot.analytics)(
            music_bot, mock_ctx, flags=_flags(90)
        )
        assert archive.analytics.await_args.kwargs == {"days": 90, "top_n": TOP_N}
        # The guild id rides POSITIONALLY, so kwargs alone leaves it unpinned: any
        # other snowflake in scope reads another guild's archive and then caches the
        # answer under this guild's key for the rest of the day.
        assert archive.analytics.await_args.args == (mock_ctx.guild.id,)

    async def test_an_empty_window_sends_a_notice_rather_than_a_blank_card(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.history_archive = _fake_archive(
            AnalyticsMetrics(days=30, today_start_epoch=_TODAY)
        )
        await command_callback(MusicBot.analytics)(music_bot, mock_ctx, flags=_flags())
        embed = mock_ctx.send.call_args[1]["embed"]
        assert embed.color == discord.Color.orange()
        # A guild whose only plays are from this morning is empty to this command,
        # and would otherwise read the notice as data loss.
        assert "before today" in embed.description
        assert "-history" in embed.description

    async def test_a_cache_hit_does_not_query_postgres(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        archive = _fake_archive(_metrics())
        music_bot.history_archive = archive
        cached = analytics_card.to_cache(_metrics(plays=999))
        with patch("src.musicbot.cache_get", AsyncMock(return_value=cached)):
            await command_callback(MusicBot.analytics)(
                music_bot, mock_ctx, flags=_flags()
            )
        archive.analytics.assert_not_awaited()
        assert "999" in mock_ctx.send.call_args[1]["embed"].description

    async def test_a_miss_writes_the_aggregate_with_a_midnight_ttl(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.history_archive = _fake_archive(
            _metrics(today_start_epoch=time.time())
        )
        setter = AsyncMock()
        with (
            patch("src.musicbot.cache_get", AsyncMock(return_value=None)),
            patch("src.musicbot.cache_set", setter),
        ):
            await command_callback(MusicBot.analytics)(
                music_bot, mock_ctx, flags=_flags()
            )
        setter.assert_awaited_once()
        assert setter.await_args is not None
        ttl = setter.await_args.args[3]
        assert 0 < ttl <= 86400

    async def test_an_aggregate_that_straddled_midnight_is_not_written(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.history_archive = _fake_archive(
            _metrics(today_start_epoch=time.time() - 2 * 86400)
        )
        setter = AsyncMock()
        with (
            patch("src.musicbot.cache_get", AsyncMock(return_value=None)),
            patch("src.musicbot.cache_set", setter),
        ):
            await command_callback(MusicBot.analytics)(
                music_bot, mock_ctx, flags=_flags()
            )
        setter.assert_not_awaited()
        # The card is still sent — a cache it cannot write is not a failure.
        assert mock_ctx.send.call_args[1]["embed"].title is not None

    async def test_an_archive_failure_publishes_no_infrastructure_detail(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """-leaderboard's rule: with detail=None, _command_error falls through to
        `type(e).__name__: e`, which here would publish a DSN host and port or
        SchemaVersionError's operator runbook into the guild."""
        archive = MagicMock(spec=PostgresHistoryArchive)
        archive.analytics = AsyncMock(
            side_effect=RuntimeError("connect to 10.0.0.4:5432 failed: run just x")
        )
        music_bot.history_archive = archive
        await command_callback(MusicBot.analytics)(music_bot, mock_ctx, flags=_flags())
        embed = mock_ctx.send.call_args[1]["embed"]
        text = f"{embed.title} {embed.description}"
        assert "10.0.0.4" not in text
        assert "just x" not in text
        assert "could not be reached" in text

    async def test_it_sends_through_ctx_send_so_the_np_block_rides_along(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Unlike -ping and -debug, this command sends once and never edits, so it
        behaves like -history and -leaderboard: MusicContext.send prepends the Now
        Playing block and the message may adopt the NP host."""
        music_bot.history_archive = _fake_archive(_metrics())
        await command_callback(MusicBot.analytics)(music_bot, mock_ctx, flags=_flags())
        mock_ctx.send.assert_awaited_once()
        mock_ctx.channel.send.assert_not_called()


class TestCommandRegistration:
    def test_the_command_is_bounded_on_both_axes(self) -> None:
        """max_concurrency bounds how many run at once; the cooldown bounds how
        OFTEN, which is the axis a closed key space and a day-long cache cannot
        cover on their own — four cold windows are four real queries."""
        cmd = MusicBot.analytics
        assert cmd._max_concurrency is not None
        assert cmd._max_concurrency.number == 1
        assert cmd._max_concurrency.wait is False
        cooldown = cmd._buckets._cooldown
        assert cooldown is not None
        assert (cooldown.rate, cooldown.per) == (1, 30.0)

    def test_stats_is_not_an_alias(self) -> None:
        """-ping already answers to `status`, and the two are one character apart
        while doing opposite things: a sub-second health check versus the heaviest
        read the bot makes."""
        assert "stats" not in MusicBot.analytics.aliases
        assert "an" in MusicBot.analytics.aliases

    def test_it_needs_no_voice_channel(self) -> None:
        assert MusicBot.analytics._before_invoke is None


class TestChartFallback:
    """Every way the render can fail degrades to the embed-only card. The data is
    already in hand by then — a chart failure must never take the numbers with it."""

    @staticmethod
    def _bot(music_bot: MusicBot) -> MusicBot:
        music_bot.history_archive = _fake_archive(_metrics())
        return music_bot

    @staticmethod
    def _sent_kwargs(mock_ctx: MagicMock) -> dict[str, Any]:
        return mock_ctx.send.call_args[1]

    async def test_the_card_carries_the_chart_when_it_renders(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        with patch(
            "src.chart_pool.chart_pool.run", AsyncMock(return_value=b"\x89PNG-bytes")
        ):
            await command_callback(MusicBot.analytics)(
                self._bot(music_bot), mock_ctx, flags=_flags()
            )
        kwargs = self._sent_kwargs(mock_ctx)
        assert kwargs["file"].filename == analytics_card.IMAGE_FILENAME
        assert kwargs["embed"].image.url == (
            f"attachment://{analytics_card.IMAGE_FILENAME}"
        )

    @pytest.mark.parametrize(
        "error",
        [
            RemoteCallError("worker blew up", "ValueError"),
            PoolClosedError("chart render pool is shut down"),
            TimeoutError(),
            OSError("no space left on device"),
            analytics_render.UnsafeGlyphError("non-ASCII glyph"),
            BrokenProcessPool("worker died twice"),
            ValueError("matplotlib said no"),
        ],
        ids=["remote", "closed", "timeout", "oserror", "glyph", "broken", "value"],
    )
    async def test_a_render_failure_still_sends_the_numbers(
        self, music_bot: MusicBot, mock_ctx: MagicMock, error: Exception
    ) -> None:
        """The promise is EVERY failure, so the list here is deliberately wider than
        the four types a narrow catch named. UnsafeGlyphError and a second
        BrokenProcessPool both reach the handler as themselves."""
        with patch("src.chart_pool.chart_pool.run", AsyncMock(side_effect=error)):
            await command_callback(MusicBot.analytics)(
                self._bot(music_bot), mock_ctx, flags=_flags()
            )
        kwargs = self._sent_kwargs(mock_ctx)
        assert "file" not in kwargs
        assert kwargs["embed"].image.url is None
        assert "Top songs" in kwargs["embed"].description

    async def test_a_glyph_guard_trip_falls_back_rather_than_erroring(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Raised through the POOL rather than injected past it. Substituting the
        wrapped type here is what let a catch list that never sees the real one look
        covered."""
        with patch(
            "src.analytics_render.render_dashboard",
            side_effect=analytics_render.UnsafeGlyphError("non-ASCII"),
        ):
            await command_callback(MusicBot.analytics)(
                self._bot(music_bot), mock_ctx, flags=_flags()
            )
        kwargs = self._sent_kwargs(mock_ctx)
        embed = kwargs["embed"]
        assert "file" not in kwargs
        assert "Top songs" in (embed.description or "")
        assert "UnsafeGlyphError" not in f"{embed.title} {embed.description}"
        assert "Analytics" in embed.title

    async def test_the_glyph_guard_crosses_the_boundary_as_itself(self) -> None:
        """The premise the fallback rests on: _picklable_call wraps only what FAILS
        to pickle, and this pickles — so a catch naming RemoteCallError never sees
        it. Asserted directly, because every caller downstream depends on it."""

        def _boom(_ignored: object) -> bytes:
            raise analytics_render.UnsafeGlyphError("non-ASCII")

        with pytest.raises(analytics_render.UnsafeGlyphError):
            await chart_pool.chart_pool.run(_boom, None)

    async def test_no_worker_is_spawned_without_matplotlib(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Detected in the PARENT with find_spec — a finder lookup, so the ~690ms
        import is still never paid there. Deferring the check to the worker would
        spawn a ~173MB process only to have it fail on the import and stay resident."""
        run = AsyncMock()
        with (
            patch("src.chart_pool.chart_available", return_value=False),
            patch("src.chart_pool.chart_pool.run", run),
        ):
            await command_callback(MusicBot.analytics)(
                self._bot(music_bot), mock_ctx, flags=_flags()
            )
        run.assert_not_awaited()
        assert "file" not in self._sent_kwargs(mock_ctx)

    async def test_missing_attach_files_is_preflighted_not_discovered(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """This is the bot's FIRST file upload, and many existing installs' grants
        predate it. Checked before the render, so ~460ms is not paid for a message
        Discord will refuse."""
        # Every OTHER permission granted, so only the right bit can distinguish this
        # from the positive case below. Permissions(attach_files=False) is
        # Permissions(0), which reading any attribute at all would satisfy.
        perms = discord.Permissions.all()
        perms.attach_files = False
        mock_ctx.channel.permissions_for.return_value = perms
        run = AsyncMock()
        with patch("src.chart_pool.chart_pool.run", run):
            await command_callback(MusicBot.analytics)(
                self._bot(music_bot), mock_ctx, flags=_flags()
            )
        run.assert_not_awaited()
        assert "file" not in self._sent_kwargs(mock_ctx)

    async def test_the_chart_renders_when_only_attach_files_is_granted(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The positive half. Without it the preflight could read any permission bit
        and still pass its negative case."""
        perms = discord.Permissions.none()
        perms.attach_files = True
        mock_ctx.channel.permissions_for.return_value = perms
        run = AsyncMock(return_value=b"\x89PNGx")
        with patch("src.chart_pool.chart_pool.run", run):
            await command_callback(MusicBot.analytics)(
                self._bot(music_bot), mock_ctx, flags=_flags()
            )
        run.assert_awaited()
        assert "file" in self._sent_kwargs(mock_ctx)

    async def test_a_missing_grant_says_so_on_the_card(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The only cause of a missing chart the guild can ACT on. Left to the log
        line alone, an operator sees a permanently chart-less card and nothing that
        names the reason."""
        perms = discord.Permissions.all()
        perms.attach_files = False
        mock_ctx.channel.permissions_for.return_value = perms
        with patch("src.chart_pool.chart_pool.run", AsyncMock()):
            await command_callback(MusicBot.analytics)(
                self._bot(music_bot), mock_ctx, flags=_flags()
            )
        footer = self._sent_kwargs(mock_ctx)["embed"].footer.text or ""
        assert "Attach Files" in footer

    async def test_a_render_failure_stays_silent_on_the_card(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The other causes are transient or host-side, so naming them would only
        hand the guild something it cannot act on."""
        with patch(
            "src.chart_pool.chart_pool.run", AsyncMock(side_effect=TimeoutError())
        ):
            await command_callback(MusicBot.analytics)(
                self._bot(music_bot), mock_ctx, flags=_flags()
            )
        footer = self._sent_kwargs(mock_ctx)["embed"].footer.text or ""
        assert "Attach Files" not in footer

    async def test_a_server_error_does_not_repost_the_card(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Forbidden subclasses HTTPException, so catching the base type caught 5xx
        too -- and a 5xx can be raised after Discord already created the message.
        Retrying then posts the card twice, the second adopting the NP host."""
        mock_ctx.send = AsyncMock(
            side_effect=[
                discord.HTTPException(MagicMock(status=503), "upstream"),
                MagicMock(),
            ]
        )
        with patch(
            "src.chart_pool.chart_pool.run", AsyncMock(return_value=b"\x89PNGx")
        ):
            await command_callback(MusicBot.analytics)(
                self._bot(music_bot), mock_ctx, flags=_flags()
            )
        # The second send is the error card, not the analytics card a second time.
        assert mock_ctx.send.await_count == 2
        second = mock_ctx.send.await_args_list[1].kwargs["embed"]
        assert "unavailable" in (second.title or "").lower()

    async def test_a_refused_upload_retries_without_the_image(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A permission can change between the preflight and the send. Losing the
        whole card to a lost image would be the worse trade."""
        mock_ctx.send = AsyncMock(
            side_effect=[discord.Forbidden(MagicMock(status=403), "nope"), MagicMock()]
        )
        with patch("src.chart_pool.chart_pool.run", AsyncMock(return_value=b"\x89PNG")):
            await command_callback(MusicBot.analytics)(
                self._bot(music_bot), mock_ctx, flags=_flags()
            )
        assert mock_ctx.send.await_count == 2
        assert "file" not in mock_ctx.send.await_args[1]

    async def test_the_render_is_bounded_by_its_own_deadline(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """It bounds the CALLER, not the pool — a ProcessPoolExecutor cannot cancel a
        running call, so the worker finishes into nothing. Without it a wedged worker
        holds the guild's max_concurrency slot indefinitely."""

        async def _hang(*_a: object, **_k: object) -> bytes:
            await asyncio.sleep(3600)
            return b""

        with (
            patch("src.chart_pool.chart_pool.run", AsyncMock(side_effect=_hang)),
            patch("src.analytics_card.ANALYTICS_RENDER_DEADLINE_SECS", 0.01),
        ):
            await command_callback(MusicBot.analytics)(
                self._bot(music_bot), mock_ctx, flags=_flags()
            )
        assert "file" not in self._sent_kwargs(mock_ctx)
        assert "Analytics" in self._sent_kwargs(mock_ctx)["embed"].title

    async def test_the_render_happens_after_the_archive_call_returns(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Rendering inside the archive's deadline would hold one of four pool
        connections for ~460ms of matplotlib — the starvation the bound exists to
        prevent, reproduced exactly."""
        order: list[str] = []
        archive = MagicMock(spec=PostgresHistoryArchive)

        async def _query(*_a: object, **_k: object) -> AnalyticsMetrics:
            order.append("query")
            return _metrics()

        async def _render(*_a: object, **_k: object) -> bytes:
            order.append("render")
            return b"\x89PNG"

        archive.analytics = AsyncMock(side_effect=_query)
        music_bot.history_archive = archive
        with patch("src.chart_pool.chart_pool.run", AsyncMock(side_effect=_render)):
            await command_callback(MusicBot.analytics)(
                music_bot, mock_ctx, flags=_flags()
            )
        assert order == ["query", "render"]


class TestNowPlayingEditSites:
    """The testable half of the attachment contract. discord.py's
    handle_message_parameters writes
    payload['attachments'] only when the argument is not MISSING, and Message.edit
    defaults it to MISSING — so an edit passing neither `attachments=` nor `files=`
    leaves the analytics PNG attached. One added kwarg silently destroys the image
    three seconds after posting, and no other test covers it."""

    def _edit_kwargs(self, name: str) -> set[str]:
        import ast
        import pathlib

        source = pathlib.Path("src/musicplayer.py").read_text()
        tree = ast.parse(source)
        target = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == name
        )
        kwargs: set[str] = set()
        for node in ast.walk(target):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "edit"
            ):
                kwargs |= {kw.arg for kw in node.keywords if kw.arg}
        return kwargs

    @pytest.mark.parametrize("site", ["_push_np_edit", "_retire_np_host"])
    def test_neither_edit_site_passes_attachments_or_files(self, site: str) -> None:
        kwargs = self._edit_kwargs(site)
        assert kwargs, f"no message.edit() call found in {site} — the walk is broken"
        assert kwargs == {"embeds"}, (
            f"{site} now passes {sorted(kwargs)} to Message.edit — anything beyond "
            "embeds= makes Discord rewrite the attachment list and drops the "
            "-analytics chart from the card"
        )


class TestPngCache:
    """The PNG layer. The AGGREGATE is authoritative and this cannot serve a request
    alone: every text field on the card is built from the aggregate, so a PNG hit with
    an evicted aggregate still runs the SQL."""

    async def test_a_hit_skips_the_render(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.history_archive = _fake_archive(_metrics())
        run = AsyncMock()
        with (
            patch(
                "src.analytics_card.analytics_png_get",
                AsyncMock(return_value=b"\x89PNGx"),
            ),
            patch("src.chart_pool.chart_pool.run", run),
        ):
            await command_callback(MusicBot.analytics)(
                music_bot, mock_ctx, flags=_flags()
            )
        run.assert_not_awaited()
        # The bytes, not just "a file": returning b"" or the wrong variable uploads a
        # 0-byte PNG on every cache hit, which is the common path after the first
        # render of a window.
        sent = mock_ctx.send.call_args[1]["file"]
        assert sent.fp.getvalue() == b"\x89PNGx"

    async def test_a_late_render_still_populates_the_cache(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A ProcessPoolExecutor cannot cancel a running call, so the worker finishes
        whether or not the caller waited. Dropping the result means a pool at
        capacity can NEVER fill the cache: every caller times out on work that keeps
        completing unseen, which is a brownout rather than a backlog."""
        gate = asyncio.Event()

        async def _slow(*_a: object, **_k: object) -> bytes:
            await gate.wait()
            return b"\x89PNGlate"

        setter = AsyncMock()
        music_bot.history_archive = _fake_archive(
            _metrics(today_start_epoch=time.time())
        )
        with (
            patch("src.chart_pool.chart_pool.run", _slow),
            patch("src.analytics_card.ANALYTICS_RENDER_DEADLINE_SECS", 0.01),
            patch("src.analytics_card.analytics_png_get", AsyncMock(return_value=None)),
            patch("src.analytics_card.analytics_png_set", setter),
        ):
            await command_callback(MusicBot.analytics)(
                music_bot, mock_ctx, flags=_flags()
            )
            # The caller gave up: the card went out with no image.
            assert "file" not in mock_ctx.send.call_args[1]
            setter.assert_not_awaited()
            gate.set()
            for task in list(music_bot._restore_tasks):
                await task
        assert setter.await_args is not None
        assert setter.await_args.args[2] == b"\x89PNGlate"

    async def test_the_render_runs_inside_the_typing_indicator(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Around the QUERY alone, an aggregate hit with a PNG miss -- every
        invocation for the rest of the day after a render change or an eviction --
        showed the user nothing at all until the card arrived."""
        order: list[str] = []

        @contextlib.asynccontextmanager
        async def _typing(_ctx: object) -> AsyncIterator[None]:
            order.append("typing-open")
            try:
                yield
            finally:
                order.append("typing-close")

        async def _render(*_a: object, **_k: object) -> bytes:
            order.append("render")
            return b"\x89PNGx"

        music_bot.history_archive = _fake_archive(_metrics())
        with (
            patch("src.musicbot.background_typing", _typing),
            patch("src.chart_pool.chart_pool.run", _render),
        ):
            await command_callback(MusicBot.analytics)(
                music_bot, mock_ctx, flags=_flags()
            )
        assert order == ["typing-open", "render", "typing-close"]

    async def test_a_miss_renders_and_stores_under_the_midnight_ttl(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.history_archive = _fake_archive(
            _metrics(today_start_epoch=time.time())
        )
        setter = AsyncMock()
        with (
            patch("src.analytics_card.analytics_png_get", AsyncMock(return_value=None)),
            patch("src.analytics_card.analytics_png_set", setter),
            patch("src.chart_pool.chart_pool.run", AsyncMock(return_value=b"\x89PNGy")),
        ):
            await command_callback(MusicBot.analytics)(
                music_bot, mock_ctx, flags=_flags()
            )
        setter.assert_awaited_once()
        assert setter.await_args is not None
        _, key, png, ttl = setter.await_args.args
        assert png == b"\x89PNGy"
        assert 0 < ttl <= 86400
        assert key.startswith("analytics:png:v")

    def test_the_digest_ignores_key_order(self) -> None:
        """OPT_SORT_KEYS is what the docstring credits, but to_cache builds its dict
        in one fixed order on both paths, so the round-trip below cannot exercise it.
        Reversing the mapping is what does."""
        blob = analytics_card.to_cache(_metrics())
        rebuilt = analytics_card.from_cache(dict(reversed(list(blob.items()))))
        assert rebuilt is not None
        assert analytics_card.aggregate_digest(
            rebuilt
        ) == analytics_card.aggregate_digest(_metrics())

    async def test_the_digest_is_stable_across_the_cache_round_trip(self) -> None:
        """A hit rebuilds the metrics through from_cache, so the two paths must agree
        on the digest or every aggregate hit would re-render."""
        fresh = _metrics()
        rebuilt = analytics_card.from_cache(
            orjson.loads(orjson.dumps(analytics_card.to_cache(fresh)))
        )
        assert rebuilt is not None
        assert analytics_card.aggregate_digest(rebuilt) == (
            analytics_card.aggregate_digest(fresh)
        )

    def test_different_data_gets_a_different_key(self) -> None:
        a = analytics_card.aggregate_digest(_metrics())
        b = analytics_card.aggregate_digest(_metrics(plays=157))
        assert a != b

    async def test_a_render_failure_is_not_cached(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.history_archive = _fake_archive(_metrics())
        setter = AsyncMock()
        with (
            patch("src.analytics_card.analytics_png_get", AsyncMock(return_value=None)),
            patch("src.analytics_card.analytics_png_set", setter),
            patch(
                "src.chart_pool.chart_pool.run",
                AsyncMock(side_effect=RemoteCallError("boom")),
            ),
        ):
            await command_callback(MusicBot.analytics)(
                music_bot, mock_ctx, flags=_flags()
            )
        setter.assert_not_awaited()
