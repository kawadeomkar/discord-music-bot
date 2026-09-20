---
paths:
  - "src/commands/*.py"
  - "src/musicbot.py"
  - "src/help.py"
  - "tests/commands/*.py"
  - "tests/test_help.py"
---

# Commands: registration, the body's own module, and the help copy

Every command is registered on the `MusicBot` cog and implemented in its own
module under `src/commands/`. The cog keeps what discord.py owns — registration,
converters, checks, cooldowns — and one try/except; nothing else about a command
lives there. `paths` covers every command module, so this loads whenever one is
touched.

## Recipes

**Add a command**: method on `MusicBot` with `@commands.command(name=..., aliases=...,
brief=..., usage=..., help=..., extras={"category": ..., "examples": [...], "note": ...})`;
add `@commands.before_invoke(validate_commands)` if it needs the author in voice; open a
span with `@_tracer.start_as_current_span("bot.<name>")`; every reply an embed; list it
in help.py's `CATEGORY_COMMANDS`; tests in `tests/commands/test_<command>.py`.

**The body belongs in the command's own module, not on the cog.** The cog keeps only
what discord.py owns — registration, converters, checks, cooldowns — and one
`try: await <module>.run(...) except Exception as e: await self._command_error(...)`.
`run()` takes `ctx`, the flags, and whatever the cog RESOLVES for it (`redis`,
`archive`, a `MusicPlayer` from `get_mp`), so the module never reaches back into
`MusicBot` and has no import edge to musicbot.py — or the COG itself, under a
`TYPE_CHECKING` guard, for the two things only it can do: reach the player registry,
and run another command through discord.py. **`run()` must not swallow**: the
`except` has to be the caller's, because `_command_error` logs with `exc_info=True`
and that only captures the live traceback from inside the handler. **Every command is
on this pattern**; musicbot.py holds no command logic at all. musicbot.py imports each
module as `<command>_cmd` — the bare name would be the module inside the like-named
cog method, which reads as the method and is not.
