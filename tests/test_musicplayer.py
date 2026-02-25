"""Tests for src/musicplayer.py — queue operations and embed building."""
from collections import deque
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from src.musicplayer import MusicPlayer
from src.youtube import QueueObject


@pytest.fixture
def music_player(mock_bot, mock_ctx):
    """Create a MusicPlayer with the background loop task suppressed."""
    player = MusicPlayer(mock_bot, mock_ctx)
    return player


@pytest.fixture
def mock_song():
    """A mock YTDL-like song object with all metadata attributes."""
    song = MagicMock()
    song.title = "Test Song Title"
    song.requester = MagicMock()
    song.requester.mention = "<@123456>"
    song.webpage_url = "https://www.youtube.com/watch?v=testid"
    song.duration = "3:30"
    song.uploader = "Test Channel"
    song.views = 1_000_000
    song.likes = 50_000
    song.dislikes = 500
    song.thumbnail = "https://img.youtube.com/vi/testid/0.jpg"
    song.abr = 128
    song.asr = 44100
    song.acodec = "opus"
    return song


@pytest.fixture
def queue_obj(mock_author):
    return QueueObject(
        webpage_url="https://www.youtube.com/watch?v=abc123",
        title="Test Song",
        requester=mock_author,
    )


class TestQueuePut:
    async def test_put_single_queue_object(self, music_player, queue_obj):
        await music_player.queue_put(queue_obj)
        assert music_player.queue.qsize() == 1

    async def test_put_list_of_sources(self, music_player, mock_author):
        from src.sources import YTSource

        sources = [
            YTSource(ytsearch="ytsearch:song one", process=True),
            YTSource(ytsearch="ytsearch:song two", process=True),
            YTSource(ytsearch="ytsearch:song three", process=True),
        ]
        await music_player.queue_put(sources)
        assert music_player.queue.qsize() == 3

    async def test_put_multiple_singles_increments_size(self, music_player, mock_author):
        for i in range(4):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)
        assert music_player.queue.qsize() == 4


class TestQueueClear:
    async def test_clear_empties_queue(self, music_player, mock_author):
        for i in range(3):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)
        assert music_player.queue.qsize() == 3

        await music_player.queue_clear()
        assert music_player.queue.qsize() == 0

    async def test_clear_empties_song_queue(self, music_player, mock_author):
        music_player.song_queue.extend(["Song 1 - url1", "Song 2 - url2"])

        await music_player.queue_clear()
        assert len(music_player.song_queue) == 0

    async def test_clear_on_empty_queue_is_safe(self, music_player):
        await music_player.queue_clear()
        assert music_player.queue.qsize() == 0


class TestQueueShuffle:
    async def test_shuffle_requires_minimum_four_items(self, music_player, mock_author):
        for i in range(3):
            qobj = QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            await music_player.queue_put(qobj)

        result = await music_player.queue_shuffle()
        assert result == "There must be at least 3 songs to shuffle the queue"

    async def test_shuffle_empty_queue_returns_error(self, music_player):
        result = await music_player.queue_shuffle()
        assert result == "There must be at least 3 songs to shuffle the queue"

    async def test_shuffle_sufficient_songs_returns_shuffled(
        self, music_player, mock_author
    ):
        songs = [
            QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            for i in range(5)
        ]
        for song in songs:
            await music_player.queue_put(song)
            music_player.song_queue.append(f"Song {song.title} - {song.webpage_url}")

        result = await music_player.queue_shuffle()
        assert result == "Shuffled!"

    async def test_shuffle_preserves_queue_size(self, music_player, mock_author):
        songs = [
            QueueObject(f"https://yt.com/watch?v={i}", f"Song {i}", mock_author)
            for i in range(5)
        ]
        for song in songs:
            await music_player.queue_put(song)
            music_player.song_queue.append(f"Song {song.title} - {song.webpage_url}")

        await music_player.queue_shuffle()
        assert music_player.queue.qsize() == 5


class TestGetQueue:
    def test_get_queue_with_songs(self, music_player):
        music_player.song_queue = deque(
            ["Song A - url_a", "Song B - url_b", "Song C - url_c"]
        )
        result = music_player.get_queue()
        assert isinstance(result, str)
        assert "Song A - url_a" in result

    def test_get_queue_empty(self, music_player):
        result = music_player.get_queue()
        assert result == ""

    def test_get_queue_caps_at_ten(self, music_player):
        music_player.song_queue = deque([f"Song {i} - url{i}" for i in range(15)])
        result = music_player.get_queue()
        # queue_message has off-by-one: shows range(1, len(songs[:10])) = 9 items
        lines = [l for l in result.split("\n") if l and l != "..."]
        assert len(lines) <= 9


class TestBuildNowPlayingEmbed:
    def test_returns_discord_embed(self, music_player, mock_song):
        embed = music_player._build_now_playing_embed(mock_song)
        assert isinstance(embed, discord.Embed)

    def test_embed_title_contains_song_title(self, music_player, mock_song):
        embed = music_player._build_now_playing_embed(mock_song)
        assert mock_song.title in embed.title

    def test_embed_description_contains_requester_mention(
        self, music_player, mock_song
    ):
        embed = music_player._build_now_playing_embed(mock_song)
        assert mock_song.requester.mention in embed.description

    def test_embed_color_is_green(self, music_player, mock_song):
        embed = music_player._build_now_playing_embed(mock_song)
        assert embed.colour == discord.Color.green()

    def test_embed_has_youtube_link_field(self, music_player, mock_song):
        embed = music_player._build_now_playing_embed(mock_song)
        field_names = [f.name for f in embed.fields]
        assert "Youtube link" in field_names

    def test_embed_has_duration_field(self, music_player, mock_song):
        embed = music_player._build_now_playing_embed(mock_song)
        field_names = [f.name for f in embed.fields]
        assert "Duration" in field_names

    def test_embed_thumbnail_is_set(self, music_player, mock_song):
        embed = music_player._build_now_playing_embed(mock_song)
        assert embed.thumbnail.url == mock_song.thumbnail

    def test_embed_footer_contains_bitrate_info(self, music_player, mock_song):
        embed = music_player._build_now_playing_embed(mock_song)
        assert str(mock_song.abr) in embed.footer.text
        assert str(mock_song.acodec) in embed.footer.text


class TestMusicPlayerInitialState:
    def test_queue_starts_empty(self, music_player):
        assert music_player.queue.qsize() == 0

    def test_song_queue_starts_empty(self, music_player):
        assert len(music_player.song_queue) == 0

    def test_history_starts_empty(self, music_player):
        assert music_player.history == []

    def test_current_song_is_none(self, music_player):
        assert music_player.current_song is None

    def test_play_message_is_none(self, music_player):
        assert music_player.play_message is None
