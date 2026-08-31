"""One module per command: `src/commands/<command>.py`, each exposing `run()`.

The cog keeps what discord.py owns — registration, converters, checks, the span —
and the try/except that renders the failure embed. Everything past that is here.
**Raising is the contract**: `_command_error` runs inside the caller's `except`,
and its `exc_info` only captures a live traceback from inside the handler.

What a body is handed, narrowest first: resolved data (a `GuildRedisStore`, an
`ArchiveReader`, a `GuildHistory`); the guild's `MusicPlayer`, which
cog_before_invoke has already built, so passing it costs a dict lookup; or the cog
itself, under a TYPE_CHECKING guard, for the two things only it can do — reach the
player REGISTRY, and run another command through discord.py.
"""
