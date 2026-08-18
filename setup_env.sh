#!/usr/bin/env bash
# Bootstrap .env and generate POSTGRES_PASSWORD.
#
#   ./setup_env.sh            # create/patch .env, generate the password if unset
#   ./setup_env.sh --force    # regenerate even if a password is already set
#
# Creates .env from .env.example when it doesn't exist, then writes a random
# 128-bit POSTGRES_PASSWORD. Hex output is deliberate: the value is interpolated
# into a DSN (postgresql://user:PASS@host/db) by docker-compose.yml, and hex has
# no characters that would need URL-escaping there.
#
# Idempotent by default, and that is a safety property rather than politeness:
# Postgres reads POSTGRES_PASSWORD only when it initializes an empty data
# directory. An existing postgres-data volume keeps whatever password it was
# created with, so silently rewriting .env would leave the bot unable to
# authenticate against its own database. --force overrides and warns about
# exactly that.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# _env_value lives here — one parser for .env files across the build scripts and
# this one, so the quoting/last-wins rules can't drift between them.
source ./build_common.sh

ENV_FILE=".env"
TEMPLATE=".env.example"
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; $d'; exit 0 ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 64 ;;
    esac
done

die() { printf 'Error: %s\n' "$1" >&2; shift; for l in "$@"; do printf '       %s\n' "$l" >&2; done; exit 1; }

_random_hex() {
    # 16 bytes = 128 bits. openssl first (present on essentially every dev
    # machine), /dev/urandom as the fallback for a stripped container. Both are
    # CSPRNGs — do NOT substitute $RANDOM or a timestamp; that makes the
    # password guessable.
    if command -v openssl >/dev/null 2>&1; then
        openssl rand -hex 16
    elif [ -r /dev/urandom ]; then
        head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n'
    else
        die "no source of randomness found (need openssl or /dev/urandom)."
    fi
}

# Replace the first NAME= line (commented or not) in place so the template's
# surrounding comments keep their meaning; append only when the key is absent.
_set_env_var() {
    local name="$1" value="$2" file="$3" tmp
    tmp="$(mktemp)"
    if grep -qE "^[[:space:]]*#?[[:space:]]*(export[[:space:]]+)?${name}=" "$file"; then
        awk -v n="$name" -v v="$value" '
            !replaced && $0 ~ "^[[:space:]]*#?[[:space:]]*(export[[:space:]]+)?" n "=" {
                print n "=" v; replaced = 1; next
            }
            { print }
        ' "$file" > "$tmp"
    else
        cp "$file" "$tmp"
        printf '%s=%s\n' "$name" "$value" >> "$tmp"
    fi
    # Preserve the original OWNER mode across the rewrite, then strip group and
    # other unconditionally. `chmod --reference` is GNU-only and fails on macOS,
    # this project's primary dev platform, so the mode is read with stat.
    #
    # Both spellings are tried AND THE RESULT IS VALIDATED, because a plain
    # `bsd || gnu` fallback does not work here: on GNU coreutils `stat -f` is a
    # valid option meaning "display FILESYSTEM status", so the BSD attempt
    # SUCCEEDS on Linux and prints a multi-line filesystem dump. The `||` never
    # fires and chmod is handed that dump, which fails with `invalid mode` and
    # exits 1 — after .env has already been created, so the run leaves a file
    # with no generated password behind it. That is every Linux run of this
    # script, including on the production host the README points at. Checking
    # that the answer is octal is what makes the fallback real rather than
    # decorative.
    #
    # The `go-rwx` is the point, and it is a narrowing of what this used to do.
    # Preserving the mode verbatim is right for a .env this script created (600)
    # and wrong for one the operator made by hand — `cp .env.example .env` lands
    # at 644 under the usual umask, and this function is how the generated
    # POSTGRES_PASSWORD gets written into it. Faithfully preserving 644 meant
    # minting a fresh credential and immediately publishing it to every account
    # on the box. Tightening never loses anything: a stricter original (400, or
    # 600) survives untouched, because go-rwx only ever clears bits.
    if [ -f "$file" ]; then
        mode="$(stat -f '%Lp' "$file" 2>/dev/null || true)"
        case "$mode" in '' | *[!0-7]*) mode="$(stat -c '%a' "$file" 2>/dev/null || true)" ;; esac
        case "$mode" in '' | *[!0-7]*) mode='' ;; esac
        if [ -n "$mode" ]; then chmod "$mode" "$tmp"; else chmod 600 "$tmp"; fi
    fi
    chmod go-rwx "$tmp"
    mv "$tmp" "$file"
}

# ── 1. ensure .env exists ────────────────────────────────────────────────────
if [ ! -f "$ENV_FILE" ]; then
    [ -f "$TEMPLATE" ] || die "neither $ENV_FILE nor $TEMPLATE exists — nothing to bootstrap from."
    cp "$TEMPLATE" "$ENV_FILE"
    chmod 600 "$ENV_FILE" 2>/dev/null || true
    echo "Created $ENV_FILE from $TEMPLATE."
    CREATED=1
else
    CREATED=0
fi

# ── 2. generate + write (idempotent unless --force) ──────────────────────────
existing="$(_env_value POSTGRES_PASSWORD "$ENV_FILE")"
if [ -n "$existing" ] && [ "$FORCE" -eq 0 ]; then
    echo "POSTGRES_PASSWORD is already set in $ENV_FILE — leaving it untouched."
    echo "  (Regenerate with --force. Only do that if no postgres-data volume has"
    echo "   been initialized yet: Postgres reads this only on first init, so an"
    echo "   existing volume would keep the old password and reject the new one.)"
else
    if [ -n "$existing" ] && [ "$FORCE" -eq 1 ]; then
        echo "WARNING: overwriting an existing POSTGRES_PASSWORD (--force)."
        echo "  If a postgres-data volume already exists it was initialized with the"
        echo "  OLD password and will reject this one. Either change it in place with"
        echo "    docker compose exec postgres psql -U musicbot -c \"ALTER USER musicbot PASSWORD '<new>'\""
        echo "  or recreate the volume:"
        echo "    docker compose down && docker volume rm discord-music-bot_postgres-data"
    fi
    password="$(_random_hex)"
    _set_env_var POSTGRES_PASSWORD "$password" "$ENV_FILE"
    # Never echo any part of the credential, not even a masked prefix — that
    # puts it in terminal scrollback and CI logs. It is in $ENV_FILE.
    echo "Set POSTGRES_PASSWORD in $ENV_FILE (32 random hex characters)."
fi

# ── 3. next steps ────────────────────────────────────────────────────────────
# Advisory, not a gate: nothing this script writes depends on DISCORD_TOKEN, so
# an unset token is a reminder rather than an error.
token="$(_env_value DISCORD_TOKEN "$ENV_FILE")"
placeholder=""
[ -f "$TEMPLATE" ] && placeholder="$(_env_value DISCORD_TOKEN "$TEMPLATE")"
if [ -z "$token" ] || { [ -n "$placeholder" ] && [ "$token" = "$placeholder" ]; }; then
    echo "Note: DISCORD_TOKEN in $ENV_FILE is still unset or the template placeholder."
    echo "  The bot will not start until you replace it with your real bot token."
fi
[ "$CREATED" -eq 1 ] && echo "Remember to fill in the other required values in $ENV_FILE."
exit 0
