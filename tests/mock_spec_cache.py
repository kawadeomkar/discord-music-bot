"""Memoize `unittest.mock`'s spec introspection for the whole test suite.

`MagicMock(spec=discord.Guild)` costs ~1.0ms, and the suite builds ~4700 spec'd
mocks per run. Almost all of that is `NonCallableMock._mock_add_spec`
(`unittest/mock.py`), which walks `dir(spec)` — 221 entries for `discord.Guild` —
calling `inspect.getattr_static`, `inspect.unwrap` and `iscoroutinefunction` on
every attribute:

    spec_list = dir(spec)
    for attr in spec_list:
        static_attr = inspect.getattr_static(spec, attr, None)
        unwrapped_attr = inspect.unwrap(static_attr)
        if iscoroutinefunction(unwrapped_attr):
            _spec_asyncs.append(attr)

The result depends only on `(spec, _spec_as_instance, _eat_self)`, so it is
cached here and replayed into each new mock. The whole suite uses 14 distinct
spec classes, so the cache hits ~99.9% of the time.

Why patch `unittest.mock` rather than offer a `spec_mock(cls)` helper for tests
to call: a helper only speeds up the call sites that remember to use it, and it
has to reproduce by hand everything mock does *after* the introspection. Roughly
half of this suite's spec'd mocks are built by `_get_child_mock` and
`create_autospec`, which have no call site to edit at all. Patching the one
expensive function leaves every downstream step — magic-method setup, `__new__`
base selection, child-mock creation, `spec_set` enforcement — running exactly as
upstream wrote it. See docs/SPEC_MOCK_PLAN.md for the measurements.

Two functions are replaced:

`_mock_add_spec`
    Serves the cached payload instead of re-walking `dir(spec)`. Only class specs
    are cached: `dir()` on an *instance* is instance-dependent, and a list spec
    already skips the walk upstream. Everything else is delegated untouched.

`MagicMixin.__init__`
    Seeds `_mock_methods` before the first `_mock_set_magics()`. Upstream calls
    that method once before `_mock_add_spec` has run — when `_mock_methods` is
    still `None`, so it installs all 79 magic methods — and once after, which
    deletes the 68 the spec forbids. Seeding lets the first call compute the
    correct 11 immediately, so nothing is installed only to be removed. This is
    what makes a *correct* cache cheaper than the naive one it replaces: a spec'd
    mock now costs less than a bare `MagicMock()`.

**A spec class must not be mutated once it has been used as a spec.** The cache
answers from the snapshot it took the first time it saw the class, so a
`patch.object`/`monkeypatch.setattr` on a spec class makes every later mock of it
silently wrong — and, because the entry outlives the `with` block, wrong for the
rest of the process. `check_for_drift()` reports an entry whose class still
differs when the session ends; `tests/conftest.py` is what calls it. A mutation
reverted before then leaves nothing to compare against, so it is invisible there:
`MOCK_SPEC_CACHE_STRICT=1` re-checks each entry where it is served and raises at
that mock instead.

This reads private `unittest.mock` internals, so `install()` asserts the two
signatures it depends on and `_recompute()` asserts the exact set of `__dict__`
keys it replays. tests/test_mock_spec_cache.py holds the parity net, including a
subprocess check against an unpatched interpreter.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable, MutableMapping
from typing import Any
from unittest.mock import MagicMixin, NonCallableMock
from weakref import WeakKeyDictionary


def _capture(func: Callable[..., Any]) -> Callable[..., Any]:
    """Capture an untouched `unittest.mock` function.

    `install()`'s `_installed` flag is a module global, so it only prevents a
    second install *within one module object*. `importlib.reload()`, or importing
    this file under a second name (`mock_spec_cache` as well as
    `tests.mock_spec_cache`), builds a fresh module whose "originals" are the
    previous module's patches — and `install()` would then make the delegate path
    call itself, i.e. a `RecursionError` rather than the loud failure this module
    promises everywhere else.
    """
    if func.__module__ != "unittest.mock":
        raise RuntimeError(
            f"unittest.mock.{func.__qualname__} is already patched by "
            f"{func.__module__!r}. tests/mock_spec_cache.py has been imported "
            "twice (a reload, or an import under a second name); only one module "
            "object may own the patch."
        )
    return func


_ORIG_ADD_SPEC = _capture(NonCallableMock._mock_add_spec)
_ORIG_MAGIC_INIT = _capture(MagicMixin.__init__)

# (spec_signature, mock_methods, spec_asyncs). `_spec_class` is deliberately NOT
# stored: on this path `spec` is a class, so upstream sets `_spec_class = spec`,
# and keeping a copy in the value would strongly reference the key and defeat the
# WeakKeyDictionary below. `_spec_set` is likewise taken from the live argument.
_Payload = tuple[inspect.Signature | None, tuple[str, ...], tuple[str, ...]]
_FIELDS = ("_spec_signature", "_mock_methods", "_spec_asyncs")

# Keyed by spec class, then by the `(_spec_as_instance, _eat_self)` variant — the
# exact inputs `_mock_add_spec` reads. Weak on the class so a test that builds a
# class per parametrized case does not pin every one of them for the session.
_CACHE: MutableMapping[type, dict[tuple[bool, bool], _Payload]] = WeakKeyDictionary()

# Which test filled each entry, so a drift line can bound the search that follows
# it. Kept beside `_CACHE` rather than in the payload: this is read only when
# something has already gone wrong, and a str cannot pin the weak key.
_ORIGINS: MutableMapping[type, dict[tuple[bool, bool], str]] = WeakKeyDictionary()

# Recheck every served entry against a fresh introspection and raise where it is
# used. The session-end check cannot see a mutation that is reverted before it
# runs; this can, because it looks at the moment the cache answers.
_STRICT = os.environ.get("MOCK_SPEC_CACHE_STRICT", "") not in ("", "0")

# Everything `_mock_add_spec` writes. Asserted on every miss so that a CPython
# release which starts writing a sixth key fails loudly instead of having it
# silently dropped on the cached path while the delegated path still sets it.
_EXPECTED_KEYS = frozenset(
    {"_spec_class", "_spec_set", "_spec_signature", "_mock_methods", "_spec_asyncs"}
)

_installed = False


def _recompute(spec: type, spec_as_instance: bool, eat_self: bool) -> _Payload:
    """Run the real introspection for `spec`, bypassing the cache.

    Probes both `spec_set` values because the cache key holds neither: the
    payload is only allowed to depend on `(spec, _spec_as_instance, _eat_self)`,
    and a release where `spec_set` reaches the introspection has to fail here
    rather than serve one arm's answer for both.
    """
    seen: list[_Payload] = []
    for spec_set in (False, True):
        # A bare instance is enough: `_mock_add_spec` only writes to `self.__dict__`.
        probe = NonCallableMock.__new__(NonCallableMock)
        _ORIG_ADD_SPEC(probe, spec, spec_set, spec_as_instance, eat_self)
        d = probe.__dict__

        if d.keys() != _EXPECTED_KEYS:
            raise RuntimeError(
                f"unittest.mock._mock_add_spec now writes {sorted(d)}, expected "
                f"{sorted(_EXPECTED_KEYS)}. tests/mock_spec_cache.py replays a fixed "
                "set of fields and must be updated for this Python version before the "
                "suite can be trusted."
            )
        if d["_spec_class"] is not spec:
            raise RuntimeError(
                f"unittest.mock._mock_add_spec set _spec_class to {d['_spec_class']!r} "
                f"for the class spec {spec!r}. tests/mock_spec_cache.py relies on "
                "these being the same object and must be updated."
            )

        seen.append(
            (
                d["_spec_signature"],
                tuple(d["_mock_methods"]),
                tuple(d["_spec_asyncs"]),
            )
        )

    if _comparable(seen[0]) != _comparable(seen[1]):
        raise RuntimeError(
            f"unittest.mock._mock_add_spec now derives different introspection for "
            f"spec_set=False and spec_set=True on {spec!r}. tests/mock_spec_cache.py "
            "keys the cache on neither and must be updated for this Python version."
        )
    return seen[0]


def _payload(spec: type, spec_as_instance: bool, eat_self: bool) -> _Payload:
    """Return the introspection result for `spec`, computing it at most once."""
    by_variant = _CACHE.get(spec)
    if by_variant is None:
        by_variant = {}
        _CACHE[spec] = by_variant

    variant = (spec_as_instance, eat_self)
    cached = by_variant.get(variant)
    if cached is not None:
        return cached

    computed = _recompute(spec, spec_as_instance, eat_self)
    by_variant[variant] = computed
    _ORIGINS.setdefault(spec, {})[variant] = os.environ.get(
        "PYTEST_CURRENT_TEST", "collection"
    )
    return computed


def _is_cacheable(spec: object) -> bool:
    """Classes only, and never a mock's own per-instance class.

    Every mock instantiation creates a throwaway subclass (`unittest/mock.py`
    builds one in `NonCallableMock.__new__`), so caching those would add an entry
    per mock. They are excluded rather than left to the weak cache because the
    per-instance class stays alive as long as its mock does, which for a
    module-scoped fixture is the whole session.
    """
    return isinstance(spec, type) and not issubclass(spec, NonCallableMock)


def _cached_add_spec(
    self: NonCallableMock,
    spec: Any,
    spec_set: Any,
    _spec_as_instance: bool = False,
    _eat_self: bool = False,
) -> None:
    if not _is_cacheable(spec):
        _ORIG_ADD_SPEC(self, spec, spec_set, _spec_as_instance, _eat_self)
        return

    cached = _payload(spec, _spec_as_instance, _eat_self)
    if _STRICT:
        _verify_now(spec, _spec_as_instance, _eat_self, cached)
    signature, methods, asyncs = cached
    d = self.__dict__
    # Both taken from the live arguments rather than the cache: `_spec_class` is
    # `spec` by construction (asserted in `_recompute`), and the same class can be
    # used with spec= and spec_set=, where only spec_set makes assignment strict.
    d["_spec_class"] = spec
    d["_spec_set"] = spec_set
    d["_spec_signature"] = signature
    # Fresh lists per mock — `_mock_extend_spec_methods` mutates them in place.
    d["_mock_methods"] = list(methods)
    d["_spec_asyncs"] = list(asyncs)


def _resolve_init_spec(args: tuple[Any, ...], kw: dict[str, Any]) -> Any:
    """The spec `NonCallableMock.__init__` will end up using, or None.

    Mirrors upstream's `if spec_set is not None: spec = spec_set`. Only the
    keyword forms and a lone positional spec are recognised; anything more exotic
    returns None, which costs a seeding opportunity but never correctness.
    """
    spec_set = kw.get("spec_set")
    if spec_set is not None:
        return spec_set
    spec = kw.get("spec")
    if spec is not None:
        return spec
    return args[0] if len(args) == 1 else None


def _seeded_magic_init(self: MagicMixin, /, *args: Any, **kw: Any) -> None:
    spec = _resolve_init_spec(args, kw)
    if _is_cacheable(spec):
        eat_self = kw.get("_eat_self")
        if eat_self is None:
            eat_self = kw.get("parent") is not None
        # Only `_mock_methods` is needed here, and it is `dir(spec)` — the same
        # for every (spec_as_instance, eat_self) pair. Matching the key that
        # `_cached_add_spec` will use a moment later just avoids a second entry.
        methods = _payload(spec, kw.get("_spec_as_instance", False), eat_self)[1]
        self.__dict__["_mock_methods"] = list(methods)
    _ORIG_MAGIC_INIT(self, *args, **kw)


def _owner(spec: type, attr: str) -> str:
    """Which class in the MRO supplies `attr`, or "?" once nothing does."""
    for klass in spec.__mro__:
        if attr in vars(klass):
            return klass.__name__
    return "?"


def _describe(spec: type, field: str, cached: Any, fresh: Any) -> str:
    """One human-readable line explaining how a cached field went stale.

    Each attribute carries the class that supplies it. The entry is keyed on the
    spec, but the mutation is often on a base, and a line naming only the spec
    sends the reader grepping for a patch of the subclass that does not exist.
    """
    if isinstance(cached, tuple) and isinstance(fresh, tuple):

        def _named(names: list[str]) -> list[str]:
            return [f"{n} (on {_owner(spec, n)})" for n in names]

        gained = _named(sorted(set(fresh) - set(cached)))
        lost = _named(sorted(set(cached) - set(fresh)))
        changes = []
        if gained:
            changes.append(f"gained {gained}")
        if lost:
            changes.append(f"lost {lost}")
        return f"{field}: {' and '.join(changes) or 'reordered'}"
    return f"{field}: cached {cached!r}, now {fresh!r}"


def _comparable(payload: _Payload) -> tuple[Any, ...]:
    """A payload in a form that compares reliably.

    `_spec_signature` is compared as text because `inspect.Signature.__eq__` is
    not reflexive for every signature: discord.py's `MISSING` sentinel defines
    `__eq__` to return False, so a signature carrying it as a default (three
    parameters of `MusicContext`) is not equal to itself. `str()` still changes
    whenever a parameter is added, removed, renamed, reordered or re-defaulted.
    """
    signature, methods, asyncs = payload
    return (str(signature), methods, asyncs)


def check_for_drift() -> list[str]:
    """Describe every cache entry that no longer matches its spec class.

    The cache is keyed on class *identity* but its payload is derived from the
    class's mutable state, and nothing invalidates it. Mutating a spec class
    after its first spec'd mock therefore poisons every later mock of it —
    an attribute the class gained raises `AttributeError`, one it lost still
    resolves, and a method swapped between sync and async flips `_spec_asyncs`
    the wrong way. All of it is silent, and under `-n 8` it is worker-dependent.

    Recomputing the whole cache costs one `dir()` walk per entry, so the session
    can check at the end that it was never lied to. It compares against the class
    as it stands then, so it sees a mutation that is still in force and not one
    that has been reverted — `MOCK_SPEC_CACHE_STRICT=1` covers the second.
    Returns an empty list when the cache is still true.
    """
    problems: list[str] = []
    # list(): a weak cache can drop entries mid-iteration if a GC runs.
    for spec, by_variant in list(_CACHE.items()):
        for variant, cached in list(by_variant.items()):
            problem = _diff(spec, variant, cached)
            if problem is not None:
                problems.append(problem)
    return problems


def _diff(spec: type, variant: tuple[bool, bool], cached: _Payload) -> str | None:
    """One drift line for a single entry, or None when it still matches."""
    spec_as_instance, eat_self = variant
    fresh = _recompute(spec, spec_as_instance, eat_self)
    old, new = _comparable(cached), _comparable(fresh)
    if old == new:
        return None
    diffs = "; ".join(
        _describe(spec, field, a, b) for field, a, b in zip(_FIELDS, old, new) if a != b
    )
    origin = _ORIGINS.get(spec, {}).get(variant, "an unrecorded point")
    return (
        f"{spec.__module__}.{spec.__qualname__} "
        f"(_spec_as_instance={spec_as_instance}, _eat_self={eat_self}): "
        f"{diffs} [entry first cached during {origin}]"
    )


def _verify_now(
    spec: type, spec_as_instance: bool, eat_self: bool, cached: _Payload
) -> None:
    """Raise where a stale entry is served, naming the test that is reading it."""
    problem = _diff(spec, (spec_as_instance, eat_self), cached)
    if problem is None:
        return
    here = os.environ.get("PYTEST_CURRENT_TEST", "an unrecorded point")
    raise AssertionError(
        f"mock spec cache served stale data for {spec!r} during {here}. "
        f"The spec class was mutated after this entry was cached, so this mock "
        f"does not match the class it specs.\n  {problem}"
    )


def _assert_signature(
    func: Callable[..., Any], expected: tuple[tuple[str, Any], ...]
) -> None:
    """Check a patched function's parameter names *and* kinds.

    Kinds matter: `_seeded_magic_init` keeps the positional-only marker upstream
    uses so that `MagicMock(self=...)` configures an attribute rather than
    colliding with the receiver. A name-only check would not notice it moving.
    """
    actual = tuple(
        (p.name, p.kind) for p in inspect.signature(func).parameters.values()
    )
    if actual != expected:
        raise RuntimeError(
            f"unittest.mock.{func.__qualname__} has signature {actual}, expected "
            f"{expected}. tests/mock_spec_cache.py patches it and must be updated "
            f"for this Python version before the suite can be trusted."
        )


_POS_OR_KW = inspect.Parameter.POSITIONAL_OR_KEYWORD
_POS_ONLY = inspect.Parameter.POSITIONAL_ONLY
_VAR_POS = inspect.Parameter.VAR_POSITIONAL
_VAR_KW = inspect.Parameter.VAR_KEYWORD


def install() -> None:
    """Patch `unittest.mock` in this process. Idempotent."""
    global _installed
    if _installed:
        return

    _assert_signature(
        _ORIG_ADD_SPEC,
        (
            ("self", _POS_OR_KW),
            ("spec", _POS_OR_KW),
            ("spec_set", _POS_OR_KW),
            ("_spec_as_instance", _POS_OR_KW),
            ("_eat_self", _POS_OR_KW),
        ),
    )
    _assert_signature(
        _ORIG_MAGIC_INIT,
        (("self", _POS_ONLY), ("args", _VAR_POS), ("kw", _VAR_KW)),
    )

    NonCallableMock._mock_add_spec = _cached_add_spec
    # Keeping `/` on `_seeded_magic_init` matches upstream, which uses it so that
    # `MagicMock(self=...)` configures an attribute instead of colliding with the
    # receiver. typeshed declares `__init__` without it, hence the mismatch.
    MagicMixin.__init__ = _seeded_magic_init  # pyright: ignore[reportAttributeAccessIssue]
    _installed = True
