"""Tests for `-remove` (src/commands/remove.py) and its label helpers."""

from unittest.mock import AsyncMock, MagicMock

import discord

from src.guild_queue import QueueItem, RemoveMode, RemoveOutcome
from src.musicbot import MusicBot
from src.commands.remove import _echo, _removed_label
from src.sources import YTSource
from src.util import EMBED_FIELD_LIMIT
from src.youtube import QueueObject
from tests.helpers import command_callback, mocked


def _removed_song(n: int, query_source: str = "") -> QueueObject:
    """A stand-in for what queue_remove hands back — the reply reads query_source
    off these to name how it matched."""
    return QueueObject(
        f"https://yt.com/v={n}", f"Song {n}", MagicMock(), query_source=query_source
    )


class TestEchoIsSafeInAnEmbed:
    """`-remove`'s argument is user text echoed back into an embed, and Discord
    renders markdown in descriptions and field values. Escaping it is not
    cosmetic: unescaped, any member can make the bot post a styled masked link
    under its own name, with no queue state required."""

    def test_a_masked_link_cannot_form(self) -> None:
        """Asserted on the OUTCOME, not on the escape: what matters is that no
        `](` pair survives to pick a link's label and destination. Escaping alone
        does not cover it — brackets are not in escape_markdown's set."""
        attack = "[Free Discord Nitro](https://evil.example/phish)"
        out = _echo(attack)
        assert "[" not in out and "]" not in out

    def test_markdown_riding_behind_a_url_is_neutralized(self) -> None:
        """escape_markdown defaults to ignore_links=True, which passes any http(s)
        URL through UNTOUCHED — so an attack prefixed with a bare link reaches the
        embed verbatim unless safe_label overrides that default."""
        attack = "https://x.com/`[FREE NITRO](https://evil.example/phish)"
        out = _echo(attack)
        assert "[" not in out and "]" not in out
        assert "`" not in out

    def test_emphasis_behind_a_url_is_escaped(self) -> None:
        """What `ignore_links=False` still buys, now that the brackets are
        neutralized outright: escape_markdown's URL exemption covers the WHOLE
        token, so emphasis after a scheme renders styled unless the flag is off.
        Pinned separately because the masked-link tests above pass either way."""
        out = _echo("https://x.com/**bold**_em_")
        assert "\\*\\*" in out
        assert "\\_" in out

    def test_a_backtick_cannot_close_the_code_span(self) -> None:
        """Two call sites wrap this in a code span, and Discord gives a backslash
        NO meaning inside one — so an ESCAPED backtick still closes the span and
        renders everything after it. The backtick has to go, not be escaped."""
        out = _echo("foo` **bold** `bar")
        assert "`" not in out
        assert "\\*\\*" in out

    def test_control_characters_cannot_end_the_line_early(self) -> None:
        """A control character truncates the rendered line, hiding whatever the
        needle put after it."""
        assert _echo("a\x00b\x1fc\x7fd") == "a b c d"

    def test_the_echo_is_bounded_well_inside_the_field_cap(self) -> None:
        """Discord 400s the whole send past 1024 chars in a field value, and
        escaping can double the length. The removal has already committed by then,
        so the user sees "Command failed" for a removal that happened. `*`, not
        `x`: escaping leaves `x` alone and would not exercise the doubling."""
        assert len(_echo("*" * 5000)) <= 1024

    def test_an_ordinary_needle_is_unchanged_apart_from_the_span(self) -> None:
        assert _echo("never gonna give you up") == "never gonna give you up"


class TestRemovedLabelNamesEveryItemType:
    """The Songs field exists because one argument can now take out a whole
    playlist and there is no undo. `YTSource` has no `.title` at all, so reaching
    for it rendered every unresolved Spotify-playlist track as `?` — the exact
    case the field was added for, and the one the -remove help now advertises."""

    def test_a_resolved_song_uses_its_title(self, mock_author: MagicMock) -> None:
        item = QueueObject("https://yt.com/v=1", "Real Title", mock_author)
        assert _removed_label(item) == "Real Title"

    def test_an_unresolved_search_uses_its_search_text(self) -> None:
        item = YTSource(ytsearch="ytsearch:Artist - Song", process=True)
        assert _removed_label(item) == "Artist - Song"

    def test_an_unresolved_link_falls_back_to_the_url(self) -> None:
        item = YTSource(url="https://yt.com/v=2", process=True)
        assert _removed_label(item) == "https://yt.com/v=2"


class TestRemoveReplyStaysInsideDiscordsCaps:
    """Every field of the `-remove` reply is built from a list the USER sizes —
    the removed songs and their positions — and the send happens AFTER
    queue_remove() has already mutated memory and Redis. So an over-length field
    is not cosmetic: Discord 400s the whole send, `_command_error` reports
    "Command failed", and the user is told nothing happened to a queue that has
    already been irreversibly changed. Asserted on the ASSEMBLED embed, since ten
    individually-capped echoes still share one field."""

    @staticmethod
    def _fields(mock_ctx: MagicMock) -> list[discord.embeds.EmbedProxy]:
        return list(mock_ctx.send.await_args_list[0][1]["embed"].fields)

    async def _run(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        *,
        removed: list[QueueItem],
        positions: list[int],
    ) -> None:
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=removed, positions=positions, mode=RemoveMode.RESOLVED
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)
        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://yt.com/v=0"
        )

    async def test_ten_long_titles_fit_the_songs_field(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """99 characters is INSIDE YouTube's own 100-char title limit, so ten
        ordinary songs overflow the 1024-char field with no crafted content."""
        songs: list[QueueItem] = [
            QueueObject(f"https://yt.com/v={i}", "A" * 99, MagicMock())
            for i in range(10)
        ]
        await self._run(
            music_bot, mock_ctx, removed=songs, positions=list(range(1, 11))
        )
        for field in self._fields(mock_ctx):
            assert len(field.value or "") <= EMBED_FIELD_LIMIT, field.name

    async def test_a_markdown_heavy_title_cannot_blow_the_field(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Escaping roughly doubles a title of pure markdown characters, which is
        the shape a hostile uploader picks."""
        songs: list[QueueItem] = [
            QueueObject(f"https://yt.com/v={i}", "*" * 200, MagicMock())
            for i in range(10)
        ]
        await self._run(
            music_bot, mock_ctx, removed=songs, positions=list(range(1, 11))
        )
        for field in self._fields(mock_ctx):
            assert len(field.value or "") <= EMBED_FIELD_LIMIT, field.name

    async def test_a_playlists_worth_of_positions_fits(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """One `-remove <playlist link>` drops every track the link added. A raw
        join passes 1024 characters at 227 positions — well inside what a real
        playlist holds."""
        songs: list[QueueItem] = [_removed_song(i) for i in range(240)]
        await self._run(
            music_bot, mock_ctx, removed=songs, positions=list(range(1, 241))
        )
        fields = {f.name: f.value or "" for f in self._fields(mock_ctx)}
        positions_field = next(v for k, v in fields.items() if k and "removed" in k)
        assert len(positions_field) <= EMBED_FIELD_LIMIT
        # The count is still honest about what went, even though the list is cut.
        assert "180 more" in positions_field

    async def test_the_whole_embed_stays_under_the_total_cap(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Discord caps an embed at 6000 characters across every part, so three
        fields each legal on their own can still fail together."""
        songs: list[QueueItem] = [
            QueueObject(f"https://yt.com/v={i}", "*" * 200, MagicMock())
            for i in range(240)
        ]
        await self._run(
            music_bot, mock_ctx, removed=songs, positions=list(range(1, 241))
        )
        assert len(mock_ctx.send.await_args_list[0][1]["embed"]) <= 6000


class TestRemoveCommand:
    async def test_a_failure_becomes_a_command_error(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Every other command wraps its body; -remove was the one that did not, so
        a raise escaped to discord.py's generic handler — logged server-side, and
        answered with nothing the user could act on or quote. _command_error is
        what renders the embed and puts the trace id in its footer."""
        mp = MagicMock()
        mp.queue_remove = AsyncMock(side_effect=RuntimeError("queue exploded"))
        mp.wait_for_restore = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot._command_error = AsyncMock()

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://yt.com/watch?v=abc"
        )

        music_bot._command_error.assert_awaited_once()
        recorded = mocked(music_bot._command_error).await_args
        assert recorded is not None
        # The real exception, not one manufactured by the handler — that is what
        # puts a usable type and message in the log and on the span.
        assert isinstance(recorded.args[1], RuntimeError)

    async def test_no_url_sends_usage_message(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        await command_callback(MusicBot.remove)(music_bot, mock_ctx, needle=None)

        mock_ctx.send.assert_awaited_once()
        msg = mock_ctx.send.call_args.kwargs["embed"].description
        assert "-remove" in msg

    async def test_no_match_sends_not_found_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(removed=[], positions=[])
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://yt.com/watch?v=notfound"
        )

        mock_ctx.send.assert_awaited_once()
        embed = mock_ctx.send.call_args[1]["embed"]
        assert "No queued songs found" in embed.description

    async def test_match_sends_removal_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[_removed_song(i) for i in range(1)],
                positions=[2],
                mode=RemoveMode.RESOLVED,
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://yt.com/watch?v=abc"
        )

        calls = mock_ctx.send.await_args_list
        # First call: removal embed
        first_kwargs = calls[0][1]
        assert "embed" in first_kwargs
        removal_embed = first_kwargs["embed"]
        assert "Removed" in removal_embed.title

    async def test_match_sends_updated_queue_embed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        queue_embed = discord.Embed(title="Queue")
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[_removed_song(i) for i in range(1)],
                positions=[1],
                mode=RemoveMode.RESOLVED,
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=queue_embed)
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://yt.com/watch?v=abc"
        )

        calls = mock_ctx.send.await_args_list
        assert len(calls) == 2
        second_kwargs = calls[1][1]
        assert "embed" in second_kwargs
        assert second_kwargs["embed"] is queue_embed

    async def test_match_adds_trash_reaction(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[_removed_song(i) for i in range(1)],
                positions=[1],
                mode=RemoveMode.RESOLVED,
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://yt.com/watch?v=abc"
        )

        mock_ctx.message.add_reaction.assert_awaited_once_with("🗑️")

    async def test_an_origin_match_explains_itself(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """One argument removing eight songs needs a reason on screen, or it reads
        as the bot having removed more than it was asked to."""
        album = "https://open.spotify.com/album/abc123"
        removed: list[QueueItem] = [_removed_song(i, "spotify.com") for i in range(8)]
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=removed,
                positions=list(range(1, 9)),
                mode=RemoveMode.ORIGIN,
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(music_bot, mock_ctx, needle=album)

        fields = {
            f.name: f.value for f in mock_ctx.send.await_args_list[0][1]["embed"].fields
        }
        assert (
            fields["Matched"] == f"{album} — the spotify.com link you queued them with"
        )

    async def test_a_search_match_is_quoted_not_linked(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Search text is not a URL — angle-bracketing it would render as a broken
        link, and "them" would be wrong for the single song it took."""
        song = _removed_song(1, "search")
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[song], positions=[2], mode=RemoveMode.ORIGIN
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="never gonna give you up"
        )

        fields = {
            f.name: f.value for f in mock_ctx.send.await_args_list[0][1]["embed"].fields
        }
        assert fields["Matched"] == (
            "never gonna give you up — the search you queued it with"
        )

    async def test_removal_embed_names_what_it_matched(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[_removed_song(i) for i in range(1)],
                positions=[3],
                mode=RemoveMode.RESOLVED,
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)
        url = "https://yt.com/watch?v=abc"

        await command_callback(MusicBot.remove)(music_bot, mock_ctx, needle=url)

        removal_embed = mock_ctx.send.await_args_list[0][1]["embed"]
        fields = {f.name: f.value for f in removal_embed.fields}
        # Escaped, and deliberately NOT wrapped in a code span: escaping inside
        # one renders the backslashes literally, and a bare URL auto-links —
        # which the angle-bracket form this replaced also did.
        assert fields["Matched"] == url

    async def test_removal_embed_shows_positions(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[_removed_song(i) for i in range(2)],
                positions=[1, 4],
                mode=RemoveMode.RESOLVED,
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://yt.com/watch?v=abc"
        )

        removal_embed = mock_ctx.send.await_args_list[0][1]["embed"]
        field_values = [f.value for f in removal_embed.fields]
        assert any("1" in v and "4" in v for v in field_values)

    async def test_removal_embed_color_is_orange(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = MagicMock()
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[_removed_song(i) for i in range(1)],
                positions=[1],
                mode=RemoveMode.RESOLVED,
            )
        )
        mp.wait_for_restore = AsyncMock(return_value=True)
        mp.queue_embed = MagicMock(return_value=discord.Embed(title="Queue"))
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.remove)(
            music_bot, mock_ctx, needle="https://yt.com/watch?v=abc"
        )

        removal_embed = mock_ctx.send.await_args_list[0][1]["embed"]
        assert removal_embed.colour == discord.Color.orange()
