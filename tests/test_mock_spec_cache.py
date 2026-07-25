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
import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import (
    AsyncMock,
    MagicMock,
    NonCallableMagicMock,
    NonCallableMock,
    create_autospec,
)

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


@dataclasses.dataclass
class SampleDataclass:
    """`create_autospec` takes a special branch for dataclasses that reaches
    `_mock_extend_spec_methods`, the one in-place mutator of the cached
    `_mock_methods` list."""

    alpha: int = 1
    beta: str = "b"


class EatSelfProbe:
    """Spec target for the cache-key tests. Never mutated, so the session-end
    drift check stays quiet."""

    def method(self) -> None: ...


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


def _child_names(cls: type) -> list[str]:
    """Three deterministic child-attribute names for `cls`."""
    return [name for name in sorted(dir(cls)) if not name.startswith("_")][:3]


def snapshot_all() -> dict[str, dict[str, Any]]:
    """Snapshot every spec/factory combination. Imported by the subprocess too."""
    out: dict[str, dict[str, Any]] = {}
    for cls in SPEC_CLASSES:
        for factory in FACTORIES:
            key = f"{cls.__name__}|{factory.__name__}"
            out[f"{key}|spec"] = _snapshot(factory(spec=cls))
            out[f"{key}|spec_set"] = _snapshot(factory(spec_set=cls))

        # `create_autospec` and `_get_child_mock` build roughly half of this
        # suite's spec'd mocks and have no call site to edit — which is a large
        # part of why the cache patches `unittest.mock` instead of offering a
        # helper. They belong in the net that justifies the patch.
        autospecced = create_autospec(cls)
        out[f"{cls.__name__}|autospec"] = _snapshot(autospecced)
        out[f"{cls.__name__}|autospec_instance"] = _snapshot(
            create_autospec(cls, instance=True)
        )
        out[f"{cls.__name__}|autospec_spec_set"] = _snapshot(
            create_autospec(cls, spec_set=True)
        )
        for name in _child_names(cls):
            out[f"{cls.__name__}|child|{name}"] = _snapshot(getattr(autospecced, name))

    # The dataclass branch of `create_autospec`, which mutates `_mock_methods`
    # in place via `_mock_extend_spec_methods`.
    out["dataclass|autospec"] = _snapshot(create_autospec(SampleDataclass))
    out["dataclass|autospec_instance"] = _snapshot(
        create_autospec(SampleDataclass, instance=True)
    )

    # Paths that must fall through to the untouched original.
    out["instance_spec"] = _snapshot(MagicMock(spec=discord.Object(id=1)))
    out["list_spec"] = _snapshot(MagicMock(spec=["alpha", "beta"]))
    out["no_spec"] = _snapshot(MagicMock())
    return out


@pytest.fixture
def isolated_cache() -> Iterator[None]:
    """Restore the cache after a test that deliberately poisons it.

    Without this, the throwaway classes and mutations below would still be in
    `_CACHE` at session end and would trip `fail_on_stale_mock_spec_cache` in
    tests/conftest.py — which is exactly what that check is for.
    """
    before = {cls: dict(variants) for cls, variants in mock_spec_cache._CACHE.items()}
    yield
    mock_spec_cache._CACHE.clear()
    mock_spec_cache._CACHE.update(before)


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

    def test_explicit_eat_self_reuses_one_cache_entry(
        self, isolated_cache: None
    ) -> None:
        """`_eat_self` is normally left to default from `parent`, but mock's own
        child/autospec paths pass it explicitly. Seeding must use the value given
        rather than re-deriving it, or it caches under one key and
        `_cached_add_spec` re-computes under another a moment later.

        Asserted on the cache keys, not on `_mock_methods`: that field is
        `dir(spec)`, which is identical for every variant, so comparing it would
        pass no matter what the seeding did.
        """
        mock_spec_cache._CACHE.pop(EatSelfProbe, None)

        explicit = MagicMock(spec=EatSelfProbe, _eat_self=True)

        assert isinstance(explicit, EatSelfProbe)
        assert set(mock_spec_cache._CACHE[EatSelfProbe]) == {(False, True)}

    def test_seeding_installs_no_magic_it_then_has_to_remove(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The `MagicMixin.__init__` half of the patch, asserted by its only
        observable effect.

        Upstream runs `_mock_set_magics()` before `_mock_add_spec`, when
        `_mock_methods` is still None — so it installs all 79 magic methods and
        the second call deletes the 68 the spec forbids. Seeding `_mock_methods`
        first makes the first call compute the correct 11 immediately. Nothing
        else distinguishes the two: the end state is identical, which is why the
        parity net cannot see this and every other test stays green when the
        patch is removed. Without it, this counts 68 deletions.
        """
        removed: list[str] = []
        original_delattr = NonCallableMock.__delattr__

        def recording(self: Any, name: str) -> None:
            removed.append(name)
            original_delattr(self, name)

        monkeypatch.setattr(NonCallableMock, "__delattr__", recording)

        MagicMock(spec=discord.Guild)

        assert removed == []

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
            mock_spec_cache._assert_signature(
                wrong,
                (
                    ("self", mock_spec_cache._POS_OR_KW),
                    ("spec", mock_spec_cache._POS_OR_KW),
                    ("spec_set", mock_spec_cache._POS_OR_KW),
                ),
            )

    def test_rejects_a_changed_parameter_kind(self) -> None:
        """Names alone are not enough. `_seeded_magic_init` keeps upstream's
        positional-only `self` so that `MagicMock(self=...)` configures an
        attribute instead of colliding with the receiver; if upstream dropped
        the marker the names would still match."""

        def positional_or_keyword(self: Any, *args: Any, **kw: Any) -> None: ...

        with pytest.raises(RuntimeError, match="must be updated"):
            mock_spec_cache._assert_signature(
                positional_or_keyword,
                (
                    ("self", mock_spec_cache._POS_ONLY),
                    ("args", mock_spec_cache._VAR_POS),
                    ("kw", mock_spec_cache._VAR_KW),
                ),
            )

    def test_accepts_the_current_signatures(self) -> None:
        mock_spec_cache._assert_signature(
            mock_spec_cache._ORIG_ADD_SPEC,
            (
                ("self", mock_spec_cache._POS_OR_KW),
                ("spec", mock_spec_cache._POS_OR_KW),
                ("spec_set", mock_spec_cache._POS_OR_KW),
                ("_spec_as_instance", mock_spec_cache._POS_OR_KW),
                ("_eat_self", mock_spec_cache._POS_OR_KW),
            ),
        )
        mock_spec_cache._assert_signature(
            mock_spec_cache._ORIG_MAGIC_INIT,
            (
                ("self", mock_spec_cache._POS_ONLY),
                ("args", mock_spec_cache._VAR_POS),
                ("kw", mock_spec_cache._VAR_KW),
            ),
        )

    def test_rejects_an_unexpected_dict_key_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The likelier forward-compat break than a signature move: a CPython
        release that writes a sixth field. The cached path replays a fixed set,
        so an unguarded new key would be silently dropped for cached specs while
        the delegated path still set it."""
        original = mock_spec_cache._ORIG_ADD_SPEC

        def writes_an_extra_key(
            self: Any,
            spec: Any,
            spec_set: Any,
            _spec_as_instance: bool = False,
            _eat_self: bool = False,
        ) -> None:
            original(self, spec, spec_set, _spec_as_instance, _eat_self)
            self.__dict__["_spec_hints"] = {}

        monkeypatch.setattr(mock_spec_cache, "_ORIG_ADD_SPEC", writes_an_extra_key)

        with pytest.raises(RuntimeError, match="now writes"):
            mock_spec_cache._recompute(EatSelfProbe, False, False)

    def test_rejects_a_second_import_of_the_module(self) -> None:
        """`install()`'s idempotence flag is a module global, so it only holds
        within one module object. A reload — or an import under a second name —
        would capture this module's patches as its "originals" and make the
        delegate path recurse."""
        with pytest.raises(RuntimeError, match="imported twice"):
            mock_spec_cache._capture(mock_spec_cache._cached_add_spec)


class TestDriftDetection:
    """The cache is keyed on class identity but its payload is derived from the
    class's mutable state. `check_for_drift()` is what converts that from a
    silent wrong-mock into a named session failure (tests/conftest.py calls it)."""

    def test_a_clean_cache_reports_nothing(self) -> None:
        MagicMock(spec=discord.Guild)
        assert mock_spec_cache.check_for_drift() == []

    def test_an_added_attribute_is_reported(self, isolated_cache: None) -> None:
        class Probe:
            alpha = 1

        MagicMock(spec=Probe)
        Probe.beta = 2  # pyright: ignore[reportAttributeAccessIssue]

        drift = mock_spec_cache.check_for_drift()
        assert len(drift) == 1
        assert "Probe" in drift[0]
        assert "_mock_methods: gained ['beta']" in drift[0]

    def test_a_removed_attribute_is_reported(self, isolated_cache: None) -> None:
        class Probe:
            alpha = 1
            beta = 2

        MagicMock(spec=Probe)
        del Probe.beta

        drift = mock_spec_cache.check_for_drift()
        assert len(drift) == 1
        assert "_mock_methods: lost ['beta']" in drift[0]

    def test_a_sync_method_swapped_for_an_async_one_is_reported(
        self, isolated_cache: None
    ) -> None:
        """The `patch.object` case, and the nastiest of the three: `_spec_asyncs`
        is what decides whether a child comes back as an `AsyncMock`, so a stale
        entry yields an un-awaited coroutine or a `TypeError` far from the cause."""

        class Probe:
            def handler(self) -> None: ...

        MagicMock(spec=Probe)

        async def async_handler(self: Any) -> None: ...

        Probe.handler = async_handler  # pyright: ignore[reportAttributeAccessIssue]

        drift = mock_spec_cache.check_for_drift()
        assert len(drift) == 1
        assert "_spec_asyncs: gained ['handler']" in drift[0]

    def test_a_changed_signature_is_reported(self, isolated_cache: None) -> None:
        class Probe:
            def __init__(self, alpha: int = 1) -> None: ...

        MagicMock(spec=Probe)

        def __init__(self: Any, alpha: int = 1, beta: int = 2) -> None: ...

        Probe.__init__ = __init__

        drift = mock_spec_cache.check_for_drift()
        assert len(drift) == 1
        assert "_spec_signature" in drift[0]

    def test_a_self_unequal_signature_is_not_a_false_positive(self) -> None:
        """`MusicContext.__init__` carries three discord.py `MISSING` defaults,
        and `MISSING.__eq__` returns False — so its `inspect.Signature` is not
        equal to itself. Comparing signatures by value rather than by text would
        report drift on every single run."""
        from src.main import MusicContext

        MagicMock(spec=MusicContext)

        assert mock_spec_cache.check_for_drift() == []

    def test_the_check_does_not_itself_touch_the_cache(self) -> None:
        MagicMock(spec=discord.Guild)
        before = {
            cls: dict(variants) for cls, variants in mock_spec_cache._CACHE.items()
        }

        mock_spec_cache.check_for_drift()

        assert {
            cls: dict(variants) for cls, variants in mock_spec_cache._CACHE.items()
        } == before
