"""One module per command: `src/commands/<command>.py`, each exposing `run()`.
The cog keeps what discord.py owns (registration, converters, checks, the span)
and the try/except that renders the failure embed; a body RAISES, because
`_command_error`'s exc_info only captures a traceback inside that handler. A
body is handed resolved data, the guild's `MusicPlayer`, or — under a
TYPE_CHECKING guard — the cog, for the player registry and ctx.invoke."""
