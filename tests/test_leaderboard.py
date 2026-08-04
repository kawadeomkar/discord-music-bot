"""Tests for src/leaderboard.py — the -leaderboard cache codec and renderer,
plus the cog command that drives them.

The command lives on MusicBot (musicbot.py) because that is where dispatch and
the archive handle are, but every test of it belongs with the module it renders
through: splitting them would leave a reader checking two files to learn what one
board looks like."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from redis.asyncio import Redis

from src import leaderboard
from src.history_archive import (
    Leaderboard,
    RequesterLeader,
    SchemaVersionError,
    SongLeader,
)
from src.leaderboard import LeaderboardFlags
from src.musicbot import MusicBot
from tests.helpers import command_callback, mocked


def _lb_flags(days: int = 0) -> SimpleNamespace:
    """Stand-in for a parsed LeaderboardFlags (FlagConverter can't be constructed
    directly; the command body only reads .days)."""
    return SimpleNamespace(days=days)


def _lb_key(guild_id: int, days: int) -> str:
    """The cache key the command writes, at the command's own row limit."""
    return leaderboard.cache_key(guild_id, days, leaderboard.TOP_N)


def _board(
    requesters: list[RequesterLeader] | None = None,
    songs: list[SongLeader] | None = None,
) -> Leaderboard:
    return Leaderboard(requesters=tuple(requesters or ()), songs=tuple(songs or ()))


def _requester(n: int, *, plays: int = 2, played_secs: int = 3600) -> RequesterLeader:
    return RequesterLeader(
        requester_id=n, requester_name=f"user{n}", plays=plays, played_secs=played_secs
    )


def _song(
    n: int,
    *,
    title: str | None = None,
    url: str | None = None,
    plays: int = 2,
) -> SongLeader:
    return SongLeader(
        title=f"Song {n}" if title is None else title,
        webpage_url=f"https://yt.com/v={n}" if url is None else url,
        duration_secs=210,
        plays=plays,
        played_secs=400,
    )


def _fake_archive(board: Leaderboard) -> MagicMock:
    """An explicit archive double. The music_bot fixture pins history_archive to
    None, and mock_bot's is a truthy MagicMock, so every enabled-path test must
    install one of these rather than rely on a default."""
    archive = MagicMock()
    archive.leaderboard = AsyncMock(return_value=board)
    return archive


class TestLeaderboardCommand:
    async def test_archive_disabled_sends_notice_and_queries_nothing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.history_archive = None
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "long-term play archive" in embed.description

    async def test_dm_invocation_sends_server_only_notice(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.guild = None
        archive = _fake_archive(_board([_requester(1)]))
        music_bot.history_archive = archive
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "per server" in embed.description
        archive.leaderboard.assert_not_awaited()

    async def test_renders_both_sections(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.history_archive = _fake_archive(
            _board(
                [
                    _requester(1, plays=2, played_secs=3661),
                    _requester(2, plays=1, played_secs=90),
                ],
                [_song(1, plays=2), _song(2, plays=1)],
            )
        )
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        embed = mock_ctx.send.call_args[1]["embed"]
        desc = embed.description
        assert "**Top listeners**" in desc and "**Top songs**" in desc
        # Mentions, not names: embed mentions never ping, and the query filters
        # requester_id > 0 so one always resolves.
        assert "<@1>" in desc and "<@2>" in desc
        # fmt_duration, the repo-wide clock format — never raw seconds.
        assert "1:01:01" in desc
        assert "2 songs" in desc and "1 song" in desc
        assert "2 plays" in desc and "1 play" in desc
        assert "[Song 1](https://yt.com/v=1)" in desc
        # Rank order is the archive's order, unmodified.
        assert desc.index("<@1>") < desc.index("<@2>")

    async def test_one_empty_section_is_skipped(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # A guild whose plays all predate requester stamping has songs but no
        # attributable listeners.
        music_bot.history_archive = _fake_archive(_board(songs=[_song(1)]))
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        desc = mock_ctx.send.call_args[1]["embed"].description
        assert "**Top songs**" in desc and "**Top listeners**" not in desc

    async def test_empty_board_sends_notice(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.history_archive = _fake_archive(_board())
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "Nothing has been archived yet" in embed.description

    async def test_archive_failure_renders_the_error_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        archive = MagicMock()
        archive.leaderboard = AsyncMock(side_effect=OSError("unreachable"))
        music_bot.history_archive = archive
        music_bot._command_error = AsyncMock()
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        assert mocked(music_bot._command_error).await_args.kwargs["title"] == (
            "Leaderboard unavailable"
        )

    async def test_title_markdown_is_neutralized(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # Brackets break the masked link (escape_markdown does not cover them);
        # the rest is escaped so a title cannot bold or strike its line.
        music_bot.history_archive = _fake_archive(
            _board(songs=[_song(1, title="[Official] **loud** _x_")])
        )
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        desc = mock_ctx.send.call_args[1]["embed"].description
        assert "[Official]" not in desc
        assert "(Official)" in desc
        assert "**loud**" not in desc

    async def test_long_title_is_truncated(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.history_archive = _fake_archive(
            _board(songs=[_song(1, title="x" * 80)])
        )
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        desc = mock_ctx.send.call_args[1]["embed"].description
        # Ellipsized, not silently cut: two songs sharing a 50-char prefix
        # would otherwise render as the same line.
        assert "x" * 49 + "\u2026" in desc
        assert "x" * 50 not in desc

    async def test_blank_title_falls_back_rather_than_rendering_an_empty_link(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # '' is a real archived title (the zero-value convention), and `[](url)`
        # renders as an invisible link with nothing to click.
        music_bot.history_archive = _fake_archive(_board(songs=[_song(1, title="")]))
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        desc = mock_ctx.send.call_args[1]["embed"].description
        assert "[Unknown](https://yt.com/v=1)" in desc

    async def test_both_song_durations_are_labelled(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # A song row carries two clocks — time this server spent on it and the
        # track's own length — and unlabelled they are indistinguishable.
        music_bot.history_archive = _fake_archive(
            _board([_requester(1)], [_song(1, plays=2)])
        )
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        desc = mock_ctx.send.call_args[1]["embed"].description
        assert "6:40 listened · 2 plays · track 3:30" in desc
        assert "1:00:00 listened · 2 songs" in desc

    async def test_departed_requester_renders_the_archived_name(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # Discord renders a mention for a non-member as a raw id, so the name the
        # archive kept is the only thing that still identifies them.
        mock_ctx.guild.get_member = MagicMock(return_value=None)
        music_bot.history_archive = _fake_archive(_board([_requester(1)]))
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        desc = mock_ctx.send.call_args[1]["embed"].description
        assert "user1" in desc and "<@1>" not in desc

    async def test_archived_name_cannot_inject_markdown(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.guild.get_member = MagicMock(return_value=None)
        leader = RequesterLeader(
            requester_id=1,
            requester_name="**boss**\nfake line",
            plays=1,
            played_secs=10,
        )
        music_bot.history_archive = _fake_archive(_board([leader]))
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        desc = mock_ctx.send.call_args[1]["embed"].description
        listeners = desc.split("**Top listeners**\n", 1)[1]
        assert listeners.count("\n") == 0
        assert "\\*\\*boss\\*\\*" in listeners

    async def test_title_control_characters_cannot_break_the_line(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # A newline inside a masked-link label ends the link there and renders
        # the rest of the markdown — `](url)` included — as its own line.
        music_bot.history_archive = _fake_archive(
            _board(songs=[_song(1, title="Real\n**Top songs**\r\x07 fake")])
        )
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        desc = mock_ctx.send.call_args[1]["embed"].description
        songs = desc.split("**Top songs**\n", 1)[1]
        assert songs.count("\n") == 0
        assert not any(c in songs for c in "\n\r\x07")

    async def test_title_is_capped_before_it_is_escaped(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # Escaping first and cutting after can split a `\*` pair and leave a
        # trailing backslash that eats the character following the label.
        music_bot.history_archive = _fake_archive(
            _board(songs=[_song(1, title="*" * 80)])
        )
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        desc = mock_ctx.send.call_args[1]["embed"].description
        # 49 asterisks plus the ellipsis survive the cap, each escaped — so no
        # unescaped asterisk anywhere to open a bold run, and no dangling
        # backslash where the cut landed.
        assert "\\*" * 49 + "\u2026" in desc
        assert "\\*" * 50 not in desc

    async def test_link_host_is_shown_next_to_the_label(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # Label and target both come from the archive, so nothing else stops a
        # played song's title from naming a destination it does not go to.
        music_bot.history_archive = _fake_archive(
            _board(
                songs=[
                    _song(1, title="youtube.com/watch", url="https://www.evil.test/x")
                ]
            )
        )
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        desc = mock_ctx.send.call_args[1]["embed"].description
        # www. stripped, and the host is outside the label the title controls.
        assert "](https://www.evil.test/x) `evil.test`" in desc

    @pytest.mark.parametrize(
        "url",
        [
            "https://yt.com/a(b)",
            "https://yt.com/a b",
            "https://yt.com/" + "x" * 200,
            "https://yt.com/a\nb",
            "",
        ],
        ids=["parens", "space", "too-long", "newline", "empty"],
    )
    async def test_unlinkable_urls_fall_back_to_a_plain_title(
        self, music_bot: MusicBot, mock_ctx: MagicMock, url: str
    ) -> None:
        # A paren, whitespace or control character ends masked-link markdown
        # early, leaving the rest of the URL as visible text. An empty URL would
        # render `[Song 1]()`, a broken link with nowhere to go.
        music_bot.history_archive = _fake_archive(_board(songs=[_song(1, url=url)]))
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        desc = mock_ctx.send.call_args[1]["embed"].description
        assert "](" not in desc
        assert "Song 1" in desc

    async def test_full_board_stays_inside_the_description_limit(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        # Ten of each, adversarially sized: this is why the sections live in the
        # description (1024/field) rather than in embed fields.
        music_bot.history_archive = _fake_archive(
            _board(
                [_requester(222222222222222222 + i) for i in range(10)],
                [
                    _song(i, title="*" * 80, url="https://yt.com/" + "y" * 130)
                    for i in range(10)
                ],
            )
        )
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        assert len(mock_ctx.send.call_args[1]["embed"].description) < 4096

    async def test_out_of_range_days_is_refused_without_querying(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        archive = _fake_archive(_board([_requester(1)]))
        music_bot.history_archive = archive
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags(days=leaderboard.MAX_DAYS + 1)
        )
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "--days must be between" in embed.description
        archive.leaderboard.assert_not_awaited()

    async def test_default_window_is_all_time(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        archive = _fake_archive(_board([_requester(1)]))
        music_bot.history_archive = archive
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        assert archive.leaderboard.await_args.kwargs["since_epoch"] == 0.0
        # Named, not blank: FlagConverter drops `--days=7`/`--days`/`7` to the
        # default silently, so an unnamed title would read as the user's window.
        assert mock_ctx.send.call_args[1]["embed"].title == "🏆 Leaderboard — all time"

    async def test_days_flag_passes_a_rolling_cutoff(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        archive = _fake_archive(_board([_requester(1)]))
        music_bot.history_archive = archive
        with patch("src.musicbot.time.time", return_value=1_000_000.0):
            await command_callback(MusicBot.leaderboard)(
                music_bot, mock_ctx, flags=_lb_flags(days=30)
            )
        assert archive.leaderboard.await_args.kwargs["since_epoch"] == (
            1_000_000.0 - 30 * 86400
        )
        assert mock_ctx.send.call_args[1]["embed"].title == (
            "🏆 Leaderboard — last 30 days"
        )

    async def test_windowed_empty_board_names_the_window(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.history_archive = _fake_archive(_board())
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags(days=7)
        )
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "last 7 days" in embed.description

    @pytest.mark.parametrize(
        "exc, secret",
        [
            (
                ConnectionRefusedError(
                    "[Errno 61] Connect call failed ('10.0.0.9', 5432)"
                ),
                "10.0.0.9",
            ),
            (
                SchemaVersionError(
                    "play_history schema is at version 1, this build needs 2. "
                    "Run 'just db-migrate' against the archive database."
                ),
                "db-migrate",
            ),
            (
                RuntimeError('password authentication failed for user "musicbot"'),
                "musicbot",
            ),
        ],
    )
    async def test_archive_failures_never_reach_the_channel_verbatim(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        exc: Exception,
        secret: str,
    ) -> None:
        """This is the only command whose exceptions come from infrastructure,
        so _command_error's default detail would publish the archive's address
        or an operator runbook to whoever ran it. -ping reduces the same class
        for the same reason (ping._error_detail)."""
        archive = MagicMock()
        archive.leaderboard = AsyncMock(side_effect=exc)
        music_bot.history_archive = archive
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        embed = mock_ctx.send.call_args[1]["embed"]
        assert embed.title == "Leaderboard unavailable"
        assert secret not in embed.description
        assert type(exc).__name__ not in embed.description
        assert "could not be reached" in embed.description


class TestLinkHost:
    """The host chip beside a masked link. Its input is an archived URL, so it
    has to survive whatever a guild member once managed to play."""

    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://www.youtube.com/watch?v=1", "youtube.com"),
            ("https://soundcloud.com/a/b", "soundcloud.com"),
            ("not a url", ""),
            ("", ""),
            # urlsplit raises "Invalid IPv6 URL" on an unbalanced bracket rather
            # than returning None, and this runs inside embed rendering — an
            # uncaught one would fail the whole board over a single row.
            ("https://[oops", ""),
        ],
    )
    def test_host(self, url: str, expected: str) -> None:
        assert leaderboard._link_host(url) == expected


class TestLeaderboardCache:
    """The 60s Redis cache. It is what bounds Postgres to one aggregate pass per
    guild per window per minute — max_concurrency only serializes, it does not
    limit rate."""

    async def test_second_call_is_served_from_the_cache(
        self, music_bot: MusicBot, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        music_bot.redis = fake_redis
        archive = _fake_archive(_board([_requester(1)]))
        music_bot.history_archive = archive
        for _ in range(2):
            await command_callback(MusicBot.leaderboard)(
                music_bot, mock_ctx, flags=_lb_flags()
            )
        archive.leaderboard.assert_awaited_once()
        assert await fake_redis.get(_lb_key(mock_ctx.guild.id, 0)) is not None

    async def test_windows_are_cached_separately(
        self, music_bot: MusicBot, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        music_bot.redis = fake_redis
        archive = _fake_archive(_board([_requester(1)]))
        music_bot.history_archive = archive
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags(days=30)
        )
        assert archive.leaderboard.await_count == 2
        gid = mock_ctx.guild.id
        assert await fake_redis.get(_lb_key(gid, 0)) is not None
        assert await fake_redis.get(_lb_key(gid, 30)) is not None

    async def test_an_empty_board_is_cached_and_served(
        self, music_bot: MusicBot, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        # The regression this guards: treating an empty board as a cache MISS —
        # by skipping the cache_set for one, or by having the codec answer None
        # instead of an empty Leaderboard. Either way an idle guild re-queries
        # Postgres on every invocation. (Not `if not board:` — Leaderboard has no
        # __bool__, so an instance is always truthy; `board is None` is the check
        # that decides, which is why the codec's contract is documented as
        # "None means malformed, never empty".)
        music_bot.redis = fake_redis
        archive = _fake_archive(_board())
        music_bot.history_archive = archive
        for _ in range(2):
            await command_callback(MusicBot.leaderboard)(
                music_bot, mock_ctx, flags=_lb_flags()
            )
        archive.leaderboard.assert_awaited_once()
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "Nothing has been archived yet" in embed.description

    @pytest.mark.parametrize(
        "blob",
        [b"not json", b'{"requesters": 3}', b"[]"],
        ids=["junk", "wrong", "list"],
    )
    async def test_a_malformed_blob_falls_through_to_the_archive(
        self, music_bot: MusicBot, mock_ctx: MagicMock, fake_redis: Redis, blob: bytes
    ) -> None:
        music_bot.redis = fake_redis
        archive = _fake_archive(_board([_requester(1)]))
        music_bot.history_archive = archive
        await fake_redis.set(_lb_key(mock_ctx.guild.id, 0), blob)
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        archive.leaderboard.assert_awaited_once()

    async def test_cached_rows_round_trip(
        self, music_bot: MusicBot, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        music_bot.redis = fake_redis
        board = _board([_requester(222222222222222222)], [_song(1)])
        music_bot.history_archive = _fake_archive(board)
        first = await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        del first
        rendered = mock_ctx.send.call_args[1]["embed"].description
        mock_ctx.send.reset_mock()
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        # Byte-identical off the cache: snowflake ids survive orjson as ints.
        assert mock_ctx.send.call_args[1]["embed"].description == rendered

    async def test_without_redis_every_call_queries(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        music_bot.redis = None
        archive = _fake_archive(_board([_requester(1)]))
        music_bot.history_archive = archive
        for _ in range(2):
            await command_callback(MusicBot.leaderboard)(
                music_bot, mock_ctx, flags=_lb_flags()
            )
        assert archive.leaderboard.await_count == 2

    async def test_entry_expires_after_the_documented_ttl(
        self, music_bot: MusicBot, mock_ctx: MagicMock, fake_redis: Redis
    ) -> None:
        # The TTL is the whole rate limit, and it is also what keeps the key a
        # legitimate volatile-lru eviction candidate. An entry written without
        # one would pin a board forever and join the non-evictable set.
        music_bot.redis = fake_redis
        music_bot.history_archive = _fake_archive(_board([_requester(1)]))
        await command_callback(MusicBot.leaderboard)(
            music_bot, mock_ctx, flags=_lb_flags()
        )
        ttl = await fake_redis.ttl(_lb_key(mock_ctx.guild.id, 0))
        assert 0 < ttl <= leaderboard.CACHE_TTL_SECS

    def test_key_carries_the_row_limit_and_a_version(self) -> None:
        # Raising leaderboard.TOP_N must not render a short board off an entry
        # the previous limit wrote, and a shape change must not decode into a
        # valid-looking board — the codec defaults missing fields rather than
        # rejecting them.
        key = leaderboard.cache_key(7, 30, 10)
        assert key == "leaderboard:v1:7:30:10"
        assert leaderboard.cache_key(7, 30, 25) != key

    def test_codec_caps_an_oversized_cached_board(self) -> None:
        # The entry is decoded before anything checks its size, so nothing else
        # stops another build's (or anything else's) larger value from rendering
        # more rows than the command promises.
        raw = leaderboard.to_cache(
            _board(
                [_requester(i) for i in range(1, 30)],
                [_song(i) for i in range(30)],
            )
        )
        board = leaderboard.from_cache(raw, top_n=leaderboard.TOP_N)
        assert board is not None
        assert len(board.requesters) == leaderboard.TOP_N
        assert len(board.songs) == leaderboard.TOP_N


class TestLeaderboardMetadata:
    def test_aliases_and_category(self) -> None:
        assert set(MusicBot.leaderboard.aliases) == {"lb", "top"}
        assert (MusicBot.leaderboard.extras or {}).get("category") == "Queue"

    def test_help_copy_names_the_archive_caveat(self) -> None:
        # The command reads a tier the operator opts into, and it lags song end
        # by the drain — both belong in the copy rather than in a support thread.
        note = (MusicBot.leaderboard.extras or {}).get("note", "")
        assert "archive" in note

    def test_help_copy_matches_the_row_limit(self) -> None:
        # The copy promises a specific number of rows and the query enforces it;
        # raising one without the other quietly makes the help wrong.
        copy = f"{MusicBot.leaderboard.help} {MusicBot.leaderboard.brief}"
        assert str(leaderboard.TOP_N) in copy

    def test_days_flag_defaults_to_all_time(self) -> None:
        # The flags class is what the dispatcher actually constructs; every other
        # test hands the body a stand-in, so nothing else would notice a renamed
        # flag or a changed default.
        assert LeaderboardFlags.__commands_flags__["days"].default == 0

    async def test_days_flag_parses_from_a_real_invocation(
        self, mock_ctx: MagicMock
    ) -> None:
        flags = await LeaderboardFlags.convert(mock_ctx, "--days 30")
        assert flags.days == 30
        assert "moment to appear" in (MusicBot.leaderboard.help or "")
