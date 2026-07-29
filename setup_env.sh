#!/usr/bin/env bash
# Bootstrap .env and generate POSTGRES_PASSWORD.
#
#   ./setup_env.sh            # create/patch .env, generate the password if unset
#   ./setup_env.sh --force    # regenerate even if a password is already set
#
# Creates .env from .env.example when it doesn't exist, then writes a random
# 128-bit POSTGRES_PASSWORD.
#
# The password used to be DERIVED as sha256("<domain>:" + DISCORD_TOKEN), which
# saved managing one secret at the cost of tying two unrelated ones together:
# the database credential became a function of the Discord token, so anyone who
# obtained the token could compute it, rotating the token silently invalidated
# a live database credential, and one leaked secret compromised two systems.
# Generating instead removes the coupling entirely — the password now depends
# on nothing else, and DISCORD_TOKEN no longer has to be set before this runs.
#
# Hex output is deliberate — the password is interpolated into a DSN
# (postgresql://user:PASS@host/db) by docker-compose.yml, and hex has no
# characters that would need URL-escaping there.
#
# IDEMPOTENT BY DEFAULT, and that is a safety property, not politeness:
# Postgres only reads POSTGRES_PASSWORD when it initializes an empty data
# directory. An existing postgres-data volume keeps whatever password it was
# created with, so silently rewriting .env would leave the bot unable to
# authenticate against its own database. An already-set password is left alone
# unless you pass --force (and --force warns about exactly this).
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
    # 16 bytes = 128 bits, hex-encoded. openssl first because it is the one
    # tool present on essentially every developer machine; /dev/urandom is the
    # fallback for a stripped container. Both are CSPRNGs — do NOT substitute
    # $RANDOM or a timestamp, which would make the password guessable.
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
    # Preserve the original mode (a .env is often 600); mktemp creates 600 anyway,
    # but an existing stricter/looser mode should survive the rewrite.
    #
    # `chmod --reference` is GNU-only — on macOS, this project's primary dev
    # platform, it always failed and silently fell through to 600, so the
    # preserve-the-mode intent never actually applied there. Read the mode with
    # stat instead, trying the BSD spelling first and then the GNU one.
    if [ -f "$file" ]; then
        mode="$(stat -f '%Lp' "$file" 2>/dev/null || stat -c '%a' "$file" 2>/dev/null || echo '')"
        if [ -n "$mode" ]; then chmod "$mode" "$tmp"; else chmod 600 "$tmp"; fi
    fi
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
    # No prefix, not even a masked one: printing the first 8 hex characters put
    # a third of a 32-character credential into terminal scrollback and CI logs,
    # which is the thing this comment used to claim it was avoiding. The value
    # is in $ENV_FILE; nothing here needs to echo any part of it.
    echo "Set POSTGRES_PASSWORD in $ENV_FILE (32 random hex characters)."
fi

# ── 3. next steps ────────────────────────────────────────────────────────────
# Advisory, not a gate: nothing this script writes depends on DISCORD_TOKEN any
# more, so an unset token is a reminder rather than an error.
token="$(_env_value DISCORD_TOKEN "$ENV_FILE")"
placeholder=""
[ -f "$TEMPLATE" ] && placeholder="$(_env_value DISCORD_TOKEN "$TEMPLATE")"
if [ -z "$token" ] || { [ -n "$placeholder" ] && [ "$token" = "$placeholder" ]; }; then
    echo "Note: DISCORD_TOKEN in $ENV_FILE is still unset or the template placeholder."
    echo "  The bot will not start until you replace it with your real bot token."
fi
[ "$CREATED" -eq 1 ] && echo "Remember to fill in the other required values in $ENV_FILE."
exit 0
