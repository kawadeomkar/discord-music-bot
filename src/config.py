import os
import subprocess
import warnings


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
    # `git rev-parse --abbrev-ref HEAD` prints "HEAD" in any detached checkout — which
    # includes every `git worktree add --detach` — so this branch is taken and the
    # RuntimeWarning is raised at import time. pyproject's `filterwarnings = ["error",
    # ...]` promotes it to an exception, so the whole suite dies at collection with a
    # message about git branch detection rather than anything about the tests. CI is
    # unaffected only because ci.yml sets ENVIRONMENT explicitly, short-circuiting
    # _parse() before this is ever reached; a developer running pytest from a worktree
    # gets no such rescue. Found when a review agent had to export ENVIRONMENT by hand
    # just to run the suite in a detached worktree.
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
