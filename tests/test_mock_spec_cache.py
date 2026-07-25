"""Parity net for tests/mock_spec_cache.py.

The cache patches private `unittest.mock` internals, so it is only trustworthy for
as long as a patched mock is indistinguishable from an unpatched one. `install()`
runs at `tests/conftest.py` import and cannot be undone in-process, so the
authoritative check (`test_matches_unpatched_interpreter`) re-runs the same
snapshot in a subprocess that never installs the patch and diffs the two. The
rest are cheap in-process invariants that name the specific behaviours the cache
must not lose.
"""

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, NonCallableMagicMock, create_autospec

import discord
import pytest

from tests import mock_spec_cache

REPO_ROOT = Path(__file__).resolve().parent.parent

SPEC_CLASSES = [
    discord.Guild,
    discord.Message,
    discord.TextChannel,
    discord.Member,
    discord.VoiceClient,
    asyncio.Task,
]
FACTORIES = [MagicMock, AsyncMock, NonCallableMagicMock]


def _snapshot(mock: Any) -> dict[str, Any]:
    """Everything the cache is responsible for reproducing, as plain JSON data."""
    d = mock.__dict__
    spec_class = d["_spec_class"]
    return {
        "magics": sorted(n for n in type(mock).__dict__ if n.startswith("__")),
        "mro": [c.__name__ for c in type(mock).__mro__],
        "spec_class": None if spec_class is None else spec_class.__name__,
        "spec_set": d["_spec_set"],
        "methods": None if d["_mock_methods"] is None else sorted(d["_mock_methods"]),
        "asyncs": sorted(d["_spec_asyncs"]),
        "signature": str(d["_spec_signature"]),
        "repr": repr(mock).split(" id=")[0],
    }


def snapshot_all() -> dict[str, dict[str, Any]]:
    """Snapshot every spec/factory combination. Imported by the subprocess too."""
    out: dict[str, dict[str, Any]] = {}
    for cls in SPEC_CLASSES:
        for factory in FACTORIES:
            key = f"{cls.__name__}|{factory.__name__}"
            out[f"{key}|spec"] = _snapshot(factory(spec=cls))
            out[f"{key}|spec_set"] = _snapshot(factory(spec_set=cls))
    # Paths that must fall through to the untouched original.
    out["instance_spec"] = _snapshot(MagicMock(spec=discord.Object(id=1)))
    out["list_spec"] = _snapshot(MagicMock(spec=["alpha", "beta"]))
    out["no_spec"] = _snapshot(MagicMock())
    return out


# Runs in a pristine interpreter: importing this module does not install the
# patch (only conftest does), so the child sees stock unittest.mock.
_CHILD = """
import json, sys
sys.path.insert(0, sys.argv[1])
from tests.test_mock_spec_cache import snapshot_all
sys.stdout.write(json.dumps(snapshot_all()))
"""


class TestUnpatchedParity:
    def test_matches_unpatched_interpreter(self) -> None:
        """A patched mock is indistinguishable from one built by stock CPython.

        This is the whole justification for patching stdlib internals. If a
        CPython release changes what `_mock_add_spec` produces, or the seeding in
        `MagicMixin.__init__` stops lining up with `_mock_set_magics`, this is
        what catches it.
        """
        assert mock_spec_cache._installed, "conftest should have installed the cache"

        proc = subprocess.run(
            [sys.executable, "-c", _CHILD, str(REPO_ROOT)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            # Pin PYTHONPATH rather than inheriting: an ambient entry can shadow
            # the `tests` package and make the child snapshot the wrong code.
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
            timeout=120,
        )
        assert proc.returncode == 0, f"unpatched child failed:\n{proc.stderr}"

        unpatched = json.loads(proc.stdout)
        patched = snapshot_all()

        assert set(patched) == set(unpatched)
        differing = {k for k in patched if patched[k] != unpatched[k]}
        assert not differing, (
            "patched mocks diverge from stock CPython for: "
            f"{sorted(differing)}\n"
            + "\n".join(
                f"  {k}: patched={patched[k]} unpatched={unpatched[k]}"
                for k in sorted(differing)
            )
        )


class TestSpecSemanticsPreserved:
    @pytest.mark.parametrize("factory", FACTORIES)
    def test_isinstance_still_works(self, factory: Any) -> None:
        assert isinstance(factory(spec=discord.Guild), discord.Guild)

    @pytest.mark.parametrize("factory", FACTORIES)
    def test_off_spec_attribute_access_raises(self, factory: Any) -> None:
        with pytest.raises(AttributeError):
            factory(spec=discord.Guild).not_a_guild_attribute

    @pytest.mark.parametrize("factory", FACTORIES)
    def test_spec_set_rejects_off_spec_assignment(self, factory: Any) -> None:
        strict = factory(spec_set=discord.Guild)
        with pytest.raises(AttributeError):
            strict.not_a_guild_attribute = 1
        # spec= alone must stay permissive — the cache reads spec_set from the
        # live argument, not from the shared per-class payload.
        lenient = factory(spec=discord.Guild)
        lenient.not_a_guild_attribute = 1
        assert lenient.not_a_guild_attribute == 1

    def test_coroutine_attributes_become_async_mocks(self) -> None:
        guild = MagicMock(spec=discord.Guild)
        assert isinstance(guild.kick, AsyncMock)
        assert not isinstance(guild.name, AsyncMock)

    def test_call_signature_is_enforced(self) -> None:
        autospecced = create_autospec(discord.Guild)
        with pytest.raises(TypeError):
            autospecced.kick()


class TestMagicMethodsMatchTheSpec:
    """The defect this module was written to fix.

    The `spec_mock` helper it replaces injected `_mock_methods` behind mock's back
    and never ran `_mock_set_magics()`, leaving all 79 magic methods installed
    instead of the 11 `discord.Guild` allows. Operations the real class forbids
    then succeeded silently against the mock.
    """

    @pytest.mark.parametrize("factory", FACTORIES)
    def test_forbidden_magics_are_absent(self, factory: Any) -> None:
        installed = {n for n in type(factory(spec=discord.Guild)).__dict__}
        assert not installed & {"__len__", "__iter__", "__aenter__", "__anext__"}

    @pytest.mark.parametrize("factory", FACTORIES)
    def test_permitted_magics_are_present(self, factory: Any) -> None:
        installed = {n for n in type(factory(spec=discord.Guild)).__dict__}
        assert {"__str__", "__eq__", "__hash__"} <= installed

    def test_len_raises_type_error_not_attribute_error(self) -> None:
        with pytest.raises(TypeError):
            len(MagicMock(spec=discord.Guild))

    def test_iter_raises_type_error(self) -> None:
        with pytest.raises(TypeError):
            iter(MagicMock(spec=discord.Guild))

    def test_spec_that_allows_iteration_keeps_it(self) -> None:
        """asyncio.Task defines __iter__, so it must survive."""
        assert "__iter__" in type(MagicMock(spec=asyncio.Task)).__dict__


class TestCacheBehaviour:
    def test_uncacheable_specs_delegate_to_the_original(self) -> None:
        assert not mock_spec_cache._is_cacheable(["alpha"])
        assert not mock_spec_cache._is_cacheable(discord.Object(id=1))
        assert not mock_spec_cache._is_cacheable(None)
        assert mock_spec_cache._is_cacheable(discord.Guild)

    def test_mock_classes_are_never_cached(self) -> None:
        """Every mock instantiation mints a throwaway class; caching those would
        grow the dict without bound and pin them in memory."""
        throwaway = type(MagicMock())
        assert not mock_spec_cache._is_cacheable(throwaway)
        before = len(mock_spec_cache._CACHE)
        MagicMock(spec=throwaway)
        assert len(mock_spec_cache._CACHE) == before

    def test_repeated_construction_reuses_one_entry(self) -> None:
        before = len(mock_spec_cache._CACHE)
        for _ in range(5):
            MagicMock(spec=discord.Guild)
            AsyncMock(spec=discord.Guild)
        assert len(mock_spec_cache._CACHE) == before

    def test_explicit_eat_self_is_honoured(self) -> None:
        """`_eat_self` is normally left to default from `parent`, but mock's own
        child/autospec paths pass it explicitly. Seeding must use the value given
        rather than re-deriving it, or the cache key drifts from the one
        `_cached_add_spec` uses a moment later."""
        explicit = MagicMock(spec=discord.Guild, _eat_self=True)
        assert isinstance(explicit, discord.Guild)
        assert (
            explicit.__dict__["_mock_methods"]
            == MagicMock(spec=discord.Guild).__dict__["_mock_methods"]
        )

    def test_mock_methods_list_is_not_shared_between_mocks(self) -> None:
        """`_mock_extend_spec_methods` mutates the list in place, so each mock
        needs its own copy of the cached tuple."""
        first = MagicMock(spec=discord.Guild)
        second = MagicMock(spec=discord.Guild)
        assert first.__dict__["_mock_methods"] is not second.__dict__["_mock_methods"]
        first.__dict__["_mock_methods"].append("injected")
        assert "injected" not in second.__dict__["_mock_methods"]

    def test_install_is_idempotent(self) -> None:
        patched = __import__(
            "unittest.mock", fromlist=["NonCallableMock"]
        ).NonCallableMock
        before = patched._mock_add_spec
        mock_spec_cache.install()
        assert patched._mock_add_spec is before
        # Would recurse forever if install() had re-captured the patched function.
        assert isinstance(MagicMock(spec=discord.Guild), discord.Guild)


class TestSignatureGuard:
    def test_rejects_an_unexpected_signature(self) -> None:
        """A CPython release that moves these parameters must fail loudly here
        rather than silently produce subtly different mocks."""

        def wrong(self: Any, spec: Any) -> None: ...

        with pytest.raises(RuntimeError, match="must be updated"):
            mock_spec_cache._assert_signature(wrong, ("self", "spec", "spec_set"))

    def test_accepts_the_current_signatures(self) -> None:
        mock_spec_cache._assert_signature(
            mock_spec_cache._ORIG_ADD_SPEC,
            ("self", "spec", "spec_set", "_spec_as_instance", "_eat_self"),
        )
        mock_spec_cache._assert_signature(
            mock_spec_cache._ORIG_MAGIC_INIT, ("self", "args", "kw")
        )
