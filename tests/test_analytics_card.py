"""Tests for src/analytics_card.py — the -analytics window allowlist, cache codec
and embed.
"""

from typing import Any
from unittest.mock import MagicMock

import discord
import orjson
import pytest

from src import analytics_card, analytics_render
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
