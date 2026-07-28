import os
import subprocess
import warnings
from enum import Enum


def _git_branch() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        branch = result.stdout.strip()
        if result.returncode == 0 and branch and branch != "HEAD":
            return branch
    except Exception:
        pass
    # TODO: A detached HEAD turns this advisory warning into a hard test failure.
    # `git rev-parse --abbrev-ref HEAD` prints "HEAD" when detached (including every
    # `git worktree add --detach`), and pyproject's `filterwarnings = ["error", ...]`
    # promotes this RuntimeWarning to an import-time exception, killing the suite at
    # collection with an error about git branch detection rather than anything about
    # the tests. CI escapes only because ci.yml sets ENVIRONMENT explicitly.
    # Fix: skip the warning when the checkout is legitimately detached, or default
    # ENVIRONMENT for the test session in tests/conftest.py.
    warnings.warn(
        "Could not detect git branch; defaulting ENVIRONMENT to 'development'",
        RuntimeWarning,
        stacklevel=2,
    )
    return "development"


def _parse() -> str:
    raw = os.environ.get("ENVIRONMENT")
    if raw is not None:
        return raw
    branch = _git_branch()
    return "production" if branch == "main" else branch.replace("/", "-")[:50]


ENVIRONMENT: str = _parse()

NOW_PLAYING_UPDATE_INTERVAL_SECS: float = float(
    os.environ.get("NOW_PLAYING_UPDATE_INTERVAL_SECS", "3.0")
)


def spotify_enabled() -> bool:
    """True when both Spotify credentials are present in the environment.

    Read at call time, not cached, so it tracks the live environment (tests
    monkeypatch per case; the bot reads it once at startup, so there is no hot
    path). Gates on presence, not validity: credentials that are set but wrong
    still count as enabled and fail loudly at the first API call.
    """
    return bool(
        os.environ.get("SPOTIFY_CLIENT_ID") and os.environ.get("SPOTIFY_CLIENT_SECRET")
    )


# Startup credential probe (MusicBot.cog_load) fetches this one track to confirm
# the credentials authenticate. Any real ID would do; this one is permanent and
# memorable — Rick Astley, "Never Gonna Give You Up".
# https://open.spotify.com/track/4PTG3Z6ehGkBFwjybzWkR8
SPOTIFY_TEST_TRACK_ID = "4PTG3Z6ehGkBFwjybzWkR8"


class SpotifyStatus(Enum):
    """Runtime state of the Spotify source, resolved once at startup.

    Where `spotify_enabled()` answers only "are credentials present?", this
    records whether they actually work, so `MusicBot._require_spotify` and
    `-ping` can tell a user *why* Spotify links are unavailable.
    """

    # No credentials configured.
    DISABLED = "disabled"
    # Present but rejected (or the probe failed): links declined as invalid.
    INVALID = "invalid"
    # Present and validated against the live API at startup.
    ENABLED = "enabled"
