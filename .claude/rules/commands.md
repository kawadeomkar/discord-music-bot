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

## Which argument mechanism a command takes

Three exist. The choice is decided by one property — **whether the command takes free
text alongside its options** — and picking wrong is not a style question, it changes
what the command can accept.

| The command takes | Use | Today |
|---|---|---|
| nothing, or one positional — consume-rest included | discord.py's own parameters | 14 commands; `-volume <0-100>`, `-remove <needle>` |
| options only, no free text | `commands.FlagConverter` | `-history`, `-analytics`, `-leaderboard` |
| free text AND options | `play_placement._PLAY_OPTIONS` + `split_play_args` | `-play`, `-playnow`, `-playnext` |
| a grammar of its own (scopes, bounds, suggestions) | a module registry | `-settings`, `-debug` |

**`FlagConverter` cannot take a command with free text.** It requires a `name:` prefix
for every value, so `-play never gonna give you up` would have to be written
`-play query: never gonna give you up`, and even then a flag before the query swallows
the rest of the line into itself (`--ts 1:32 query: …` yields `ts="1:32 query: …"`). A
flag spelled inside the text is lifted out and raises `BadFlagArgument`. It fits
`-history --limit 5` precisely because there is nothing there for a flag to swallow.

**`-play`'s query is not ordinary free text**: it is stamped as the entry's `origin`,
which is what `-remove` matches on. An option lifted out of mid-line, or a `name:`
prefix left in, persists a value the user never typed and the song becomes unremovable.
That is why the grammar reads only a LEADING RUN of options and hands everything from
the first non-option token through verbatim.

**Do not reach for `argparse`.** Measured against this grammar it gets four of ten cases
wrong: a repeated option is silently accepted (last wins), a near-miss like `-now` is
`unrecognized arguments` with no did-you-mean, and `--nowhere man` — a search — is
refused. `shlex.split` raises on `don't stop me now`. `parser.error()` calls
`sys.exit()`, `-h` writes to stdout, and `allow_abbrev` silently matches `--time` to
`--timestamp`. Everything it would not do — the value parser, the near-miss regex,
repeat detection, the search-not-refusal fallback — is what the 173 lines actually are;
what it adds is mutual exclusion, which `_PlayOption.field` already gives for free.

**Adding an option to `-play`** is one `_PlayOption` entry: its canonical `name`, the
`spellings` a user may type, the `PlayArgs` field it sets, and either a `constant` it
stands for or a `read` that parses its value — never both, since `read` is what decides
whether the next token is consumed. Parsing, refusals and the did-you-mean all follow;
`usage=` on the command is separate and still hand-written, because help.py wraps the
command-list heading at `_WIDTH = 48`. See docs/ARCHITECTURE.md#the-play-flag-grammar.

**A fourth command needing free text plus options** is the trigger to lift the registry
out of `play_placement.py` into its own module. Until then it has one caller and the
generality would be speculative.
