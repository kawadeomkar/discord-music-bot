#!/usr/bin/env bash
# Bootstrap .env and derive POSTGRES_PASSWORD from DISCORD_TOKEN.
#
#   ./setup_env.sh            # create/patch .env, derive the password if unset
#   ./setup_env.sh --force    # re-derive even if a password is already set
#
# Creates .env from .env.example when it doesn't exist, then writes a
# POSTGRES_PASSWORD derived as sha256("<domain>:" + DISCORD_TOKEN). Deriving
# rather than prompting means the archive needs no extra secret to manage: the
# one credential you already had to configure produces it.
#
# Hex output is deliberate — the password is interpolated into a DSN
# (postgresql://user:PASS@host/db) by docker-compose.yml, and hex has no
# characters that would need URL-escaping there.
#
# IDEMPOTENT BY DEFAULT, and that is a safety property, not politeness: the
# password is a function of DISCORD_TOKEN, so rotating the token would derive a
# DIFFERENT password. Postgres only reads POSTGRES_PASSWORD when it initializes
# an empty data directory — an existing postgres-data volume keeps the old
# credential, and a silently-rewritten .env would leave the bot unable to
# authenticate against its own database. So an already-set password is left
# alone unless you pass --force (and --force warns about exactly this).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# _env_value lives here — one parser for .env files across the build scripts and
# this one, so the quoting/last-wins rules can't drift between them.
source ./build_common.sh

ENV_FILE=".env"
TEMPLATE=".env.example"
# Domain separation: the stored value is not a bare sha256 of the token, so this
# password and the token are not interchangeable if one is ever seen in isolation.
DOMAIN="discord-music-bot:postgres"
FORCE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE=1; shift ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//; $d'; exit 0 ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 64 ;;
    esac
done

die() { printf 'Error: %s\n' "$1" >&2; shift; for l in "$@"; do printf '       %s\n' "$l" >&2; done; exit 1; }

_sha256_hex() {
    # printf '%s' (no trailing newline) so the digest is stable across tools.
    if command -v sha256sum >/dev/null 2>&1; then
        printf '%s' "$1" | sha256sum | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        printf '%s' "$1" | shasum -a 256 | awk '{print $1}'
    elif command -v openssl >/dev/null 2>&1; then
        printf '%s' "$1" | openssl dgst -sha256 | awk '{print $NF}'
    else
        die "no sha256 tool found (need sha256sum, shasum, or openssl)."
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
    if [ -f "$file" ]; then chmod --reference="$file" "$tmp" 2>/dev/null || chmod 600 "$tmp"; fi
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

# ── 2. DISCORD_TOKEN must be real ────────────────────────────────────────────
token="$(_env_value DISCORD_TOKEN "$ENV_FILE")"
# The "unchanged default" is whatever the template ships, read from the template
# rather than hardcoded here, so the two can never drift apart.
placeholder=""
[ -f "$TEMPLATE" ] && placeholder="$(_env_value DISCORD_TOKEN "$TEMPLATE")"

if [ -z "$token" ]; then
    die "DISCORD_TOKEN is not set in $ENV_FILE." \
        "At minimum, DISCORD_TOKEN must be populated — POSTGRES_PASSWORD is" \
        "derived from it, so there is nothing to derive from until you set it." \
        "Edit $ENV_FILE, then re-run ./setup_env.sh"
fi

if [ -n "$placeholder" ] && [ "$token" = "$placeholder" ]; then
    die "DISCORD_TOKEN is still the placeholder from $TEMPLATE ('$placeholder')." \
        "At minimum, DISCORD_TOKEN must be populated with your real bot token —" \
        "POSTGRES_PASSWORD is derived from it, and deriving from the shared" \
        "placeholder would give every checkout the same database password." \
        "Edit $ENV_FILE, then re-run ./setup_env.sh"
fi

# ── 3. derive + write (idempotent unless --force) ────────────────────────────
existing="$(_env_value POSTGRES_PASSWORD "$ENV_FILE")"
if [ -n "$existing" ] && [ "$FORCE" -eq 0 ]; then
    echo "POSTGRES_PASSWORD is already set in $ENV_FILE — leaving it untouched."
    echo "  (Re-derive with --force. Only do that if no postgres-data volume has"
    echo "   been initialized yet: Postgres reads this only on first init, so an"
    echo "   existing volume would keep the old password and reject the new one.)"
else
    if [ -n "$existing" ] && [ "$FORCE" -eq 1 ]; then
        echo "WARNING: overwriting an existing POSTGRES_PASSWORD (--force)."
        echo "  If a postgres-data volume already exists it was initialized with the"
        echo "  OLD password and will reject this one. Recreate it with:"
        echo "    docker compose down && docker volume rm discord-music-bot_postgres-data"
    fi
    password="$(_sha256_hex "${DOMAIN}:${token}")"
    _set_env_var POSTGRES_PASSWORD "$password" "$ENV_FILE"
    # Masked: the full value is in $ENV_FILE; no reason to put a credential in
    # terminal scrollback or CI logs.
    echo "Set POSTGRES_PASSWORD in $ENV_FILE (sha256-derived, ${password:0:8}…)."
fi

# ── 4. next step ─────────────────────────────────────────────────────────────
[ "$CREATED" -eq 1 ] && echo "Remember to fill in the other required values in $ENV_FILE."
exit 0
