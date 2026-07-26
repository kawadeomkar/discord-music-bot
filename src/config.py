import os
import subprocess
from enum import Enum

# The deploy environment, tagged onto every log line and OTel resource.
#
# Read from the environment with a plain default. It used to be derived by
# running `git rev-parse` at MODULE IMPORT time, which was wrong twice over: a
# production process's behaviour should not depend on a git binary and a repo
# being present, and the fallback path raised a RuntimeWarning that pyproject's
# `filterwarnings = ["error", ...]` promoted to an exception — so in any
# detached checkout (which includes every `git worktree add --detach`) the whole
# test suite died at collection with a message about git branch detection. CI
# escaped only because ci.yml sets ENVIRONMENT explicitly.
#
# Branch inference survives as a dev-only convenience, but it is opt-in and
# lazy: main() calls infer_environment_from_git() when ENVIRONMENT is unset,
# where a subprocess is a reasonable thing to run and a failure can just be
# logged.
ENVIRONMENT: str = os.environ.get("ENVIRONMENT") or "development"


def infer_environment_from_git() -> str | None:
    """Best-effort deploy-environment name from the current git branch.

    Returns None when the branch cannot be determined — no repo, no git binary,
    or a detached HEAD (`git rev-parse --abbrev-ref HEAD` prints "HEAD"), which
    is a normal state for a worktree and not worth warning about. Never raises,
    and is never called at import time.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    branch = result.stdout.strip()
    if result.returncode != 0 or not branch or branch == "HEAD":
        return None
    return "production" if branch == "main" else branch.replace("/", "-")[:50]


NOW_PLAYING_UPDATE_INTERVAL_SECS: float = float(
    os.environ.get("NOW_PLAYING_UPDATE_INTERVAL_SECS", "3.0")
)


def spotify_enabled() -> bool:
    """True when Spotify-link support should be active — i.e. both Spotify
    credentials are present in the environment.

    Read at call time (not cached as a module constant) so the value tracks
    the live environment: tests monkeypatch these vars per case, and the bot
    process only ever reads them once at startup, so there is no hot path to
    optimise. Presence, not validity, is the gate here — credentials that are
    set but wrong still count as "enabled" and fail at the first Spotify API
    call with a clear error, rather than silently disabling the feature.
    """
    return bool(
        os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET")
    )


# Spotify ID for Rick Astley — "Never Gonna Give You Up". A well-known, permanent
# public track used purely as a probe: on startup, if Spotify credentials are
# present, the bot fetches this track once to confirm the credentials actually
# authenticate against the live API (see MusicBot.cog_load). Any real track ID
# would do; this one is memorable and unlikely to ever be removed.
# Source URL: https://open.spotify.com/track/4PTG3Z6ehGkBFwjybzWkR8
SPOTIFY_TEST_TRACK_ID = "4PTG3Z6ehGkBFwjybzWkR8"


class SpotifyStatus(Enum):
    """Runtime state of the Spotify source, resolved once at startup.

    `spotify_enabled()` answers only "are credentials present?". This goes a step
    further and records whether those credentials actually work, so both the
    play-time gate (`MusicBot._require_spotify`) and `-ping` can tell a user
    *why* Spotify links are unavailable.
    """

    # No credentials configured — Spotify support was never turned on.
    DISABLED = "disabled"
    # Credentials are present but Spotify rejected them (or the startup probe
    # failed): links are declined with an "invalid credentials" message.
    INVALID = "invalid"
    # Credentials are present and validated against the live API at startup.
    ENABLED = "enabled"
