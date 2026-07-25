"""Memoize `unittest.mock`'s spec introspection for the whole test suite.

`MagicMock(spec=discord.Guild)` costs ~0.9ms, and the suite builds ~9200 spec'd
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

This reads private `unittest.mock` internals, so `install()` asserts the two
signatures it depends on and fails loudly if a CPython release moves them.
tests/test_mock_spec_cache.py holds the parity net, including a subprocess check
against an unpatched interpreter.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable
from unittest.mock import MagicMixin, NonCallableMock

# The originals, captured once at import. `install()` refuses to run twice, so
# these can never be rebound to an already-patched function.
_ORIG_ADD_SPEC = NonCallableMock._mock_add_spec
_ORIG_MAGIC_INIT = MagicMixin.__init__

# (spec_class, spec_signature, mock_methods, spec_asyncs), keyed by
# (spec, _spec_as_instance, _eat_self) — the exact inputs `_mock_add_spec` reads.
_Payload = tuple[
    type | None, inspect.Signature | None, tuple[str, ...], tuple[str, ...]
]
_CACHE: dict[tuple[type, bool, bool], _Payload] = {}

_installed = False


def _payload(spec: type, spec_as_instance: bool, eat_self: bool) -> _Payload:
    """Return the introspection result for `spec`, computing it at most once."""
    key = (spec, spec_as_instance, eat_self)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached

    # A bare instance is enough: `_mock_add_spec` only writes to `self.__dict__`.
    probe = NonCallableMock.__new__(NonCallableMock)
    _ORIG_ADD_SPEC(probe, spec, False, spec_as_instance, eat_self)
    d = probe.__dict__
    computed: _Payload = (
        d["_spec_class"],
        d["_spec_signature"],
        tuple(d["_mock_methods"]),
        tuple(d["_spec_asyncs"]),
    )
    _CACHE[key] = computed
    return computed


def _is_cacheable(spec: object) -> bool:
    """Classes only, and never a mock's own per-instance class.

    Every mock instantiation creates a throwaway subclass (`unittest/mock.py`
    builds one in `NonCallableMock.__new__`), so caching those would grow the
    dict without bound and pin the classes in memory.
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

    spec_class, signature, methods, asyncs = _payload(
        spec, _spec_as_instance, _eat_self
    )
    d = self.__dict__
    d["_spec_class"] = spec_class
    # Taken from the live argument, never the cache: the same class can be used
    # with spec= and spec_set=, and only spec_set makes assignment strict.
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
        methods = _payload(spec, kw.get("_spec_as_instance", False), eat_self)[2]
        self.__dict__["_mock_methods"] = list(methods)
    _ORIG_MAGIC_INIT(self, *args, **kw)


def _assert_signature(func: Callable[..., Any], expected: tuple[str, ...]) -> None:
    actual = tuple(inspect.signature(func).parameters)
    if actual != expected:
        raise RuntimeError(
            f"unittest.mock.{func.__qualname__} has signature {actual}, expected "
            f"{expected}. tests/mock_spec_cache.py patches it and must be updated "
            f"for this Python version before the suite can be trusted."
        )


def install() -> None:
    """Patch `unittest.mock` in this process. Idempotent."""
    global _installed
    if _installed:
        return

    _assert_signature(
        _ORIG_ADD_SPEC, ("self", "spec", "spec_set", "_spec_as_instance", "_eat_self")
    )
    _assert_signature(_ORIG_MAGIC_INIT, ("self", "args", "kw"))

    NonCallableMock._mock_add_spec = _cached_add_spec
    # Keeping `/` on `_seeded_magic_init` matches upstream, which uses it so that
    # `MagicMock(self=...)` configures an attribute instead of colliding with the
    # receiver. typeshed declares `__init__` without it, hence the mismatch.
    MagicMixin.__init__ = _seeded_magic_init  # pyright: ignore[reportAttributeAccessIssue]
    _installed = True
