"""Real-Redis integration tier for the outbox stream.

Opt-in, with two ways to supply the server — see `redis_url` below:

    just test-redis                  # local: testcontainers, needs Docker
    REDIS_TEST_URL=... just test-redis   # CI: an already-running server

Why this tier exists, when fakeredis executes every stream command the design
uses: **fakeredis is sufficient for behaviour and insufficient for fidelity**,
and the gap is not evenly distributed. Five divergences were reproduced against
`redis:7-alpine` (7.4.9) while designing the transport, and every one of them
fails in the SAFE-LOOKING direction — green unit tests, broken production:

    1  xtrim(approximate=True), redis-py's DEFAULT   fakeredis trims EXACTLY;
                                                     real Redis trims NOTHING on
                                                     a small stream, and reports
                                                     success either way
    2  XAUTOCLAIM completion cursor                  fakeredis returns the
                                                     last-scanned id; real Redis
                                                     returns "0-0"
    3  XINFO GROUPS `lag`                            fakeredis is off by one on a
                                                     fresh group and returns a
                                                     NEGATIVE value after
                                                     deletion; real Redis returns
                                                     nil when unreconcilable
    4  XADD against a LIST key                       fakeredis raises
                                                     AttributeError; real Redis
                                                     raises ResponseError
                                                     (WRONGTYPE)
    5  ref_policy=KEEPREF/DELREF/ACKED               syntax error below Redis 8.2
                                                     on both

Rows 1, 3 and 4 are undetectable without a real server; row 2 is asserted here
so the unit-tier workaround stays honest about what it is working around.

Floor is Redis 7.0, not 6.2. XAUTOCLAIM exists at 6.2, but the 3-element reply
carrying the deleted-ID list — the tombstone behaviour the drainer depends on —
arrived in 7.0; on 6.2 redis-py hands back a literal (None, None) inside the
claimed list and the tombstone is never purged.

Isolation: Redis has no throwaway-database-per-test equivalent to the pg tier's
`raw_pg_dsn`, so each test FLUSHDBs a dedicated numbered database instead. The
tier therefore assumes it owns the server it is pointed at.
"""

import os
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

import pytest
import redis.asyncio as aioredis

from src.guild_state import HistoryEntry
from src.redis_client import (
    HISTORY_OUTBOX_CONSUMER,
    HISTORY_OUTBOX_GROUP,
    HISTORY_OUTBOX_KEY,
    OUTBOX_FIELD,
    GuildRedisStore,
    ack_outbox,
    ensure_outbox_group,
    outbox_depth,
    outbox_pending_below,
    outbox_pending_count,
    read_outbox_new,
    read_outbox_pending,
    reclaim_outbox_stale,
    retire_outbox,
    trim_outbox_below,
)

from tests.helpers import bind_loopback_only, tier_enabled

# REDIS_TEST_URL enables the tier on its own, for the same reason
# POSTGRES_TEST_URL does in the pg tier: a CI job that supplied the server but
# forgot a second flag would skip every test here and report green, and a blind
# job that looks covered is worse than a missing one.
_REDIS_ENABLED = tier_enabled("RUN_REDIS_TESTS", "REDIS_TEST_URL")

pytestmark = [
    pytest.mark.redis,
    pytest.mark.skipif(
        not _REDIS_ENABLED,
        reason="redis tier is opt-in: set RUN_REDIS_TESTS=1 (Docker) or REDIS_TEST_URL",
    ),
]

# Must match the redis service image in .github/workflows/ci.yml's
# redis-integration job AND the one in docker-compose.yml — `just pins` asserts
# all three agree. That is enforcement rather than a "keep these in step"
# comment, and it is live risk: Dependabot bumps the compose tag, `redis:7-alpine`
# FLOATS (7.4.9 today), and an earlier measurement of this very design was taken
# against the wrong server for exactly that reason.
_REDIS_IMAGE = "redis:7-alpine"


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    """A Redis this tier may FLUSHDB.

    Two providers behind one fixture, mirroring the pg tier's admin_dsn.
    REDIS_TEST_URL wins when set: CI runs a GitHub Actions *service container*,
    already pulled and health-checked during job setup, so paying
    testcontainers' pull and reaper cost again inside the job buys nothing.

    Session-scoped and SYNCHRONOUS, yielding a URL rather than a client:
    asyncio_default_fixture_loop_scope is "function", so a connection opened
    here would be bound to a loop that is closed before the next test runs. The
    pg tier documents the same trap.
    """
    external = os.getenv("REDIS_TEST_URL")
    if external:
        yield external
        return

    # Scoped suppression, not a pyproject filterwarnings entry. testcontainers
    # 4.x applies its own deprecated @wait_container_is_ready decorator inside
    # testcontainers.redis at IMPORT time, so under golden rule 11
    # (filterwarnings = ["error"]) the import itself fails. Confining the filter
    # to this one statement keeps every other DeprecationWarning in the suite
    # fatal, which is the property the rule exists for. Removable once
    # testcontainers migrates its own module to the structured wait strategies
    # the warning recommends.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*wait_container_is_ready decorator is deprecated.*",
            category=DeprecationWarning,
        )
        from testcontainers.redis import RedisContainer

    container = RedisContainer(_REDIS_IMAGE)
    bind_loopback_only(container, 6379)
    with container:
        yield (
            f"redis://{container.get_container_host_ip()}"
            f":{container.get_exposed_port(6379)}/0"
        )


@pytest.fixture
async def redis(redis_url: str) -> AsyncIterator[aioredis.Redis]:
    """A flushed client per test. decode_responses=False, as production is."""
    client = aioredis.Redis.from_url(redis_url, decode_responses=False)
    await client.flushdb()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


def _entry(n: int) -> HistoryEntry:
    return HistoryEntry(
        guild_id=42,
        title=f"Song {n}",
        webpage_url=f"https://yt.com/v={n}",
        duration_secs=200,
        played_secs=200,
        requester_id=222222222222222222,
        requester_name=f"user{n}",
        played_at=1000.0 + n,
    )


async def _push(redis: aioredis.Redis, *ns: int) -> None:
    store = GuildRedisStore(redis, guild_id=42)
    for n in ns:
        await store.push_history(_entry(n))


async def _ids(redis: aioredis.Redis) -> list[bytes]:
    entries = cast(
        list[tuple[bytes, dict[bytes, bytes]]],
        await redis.xrange(HISTORY_OUTBOX_KEY),
    )
    return [i for i, _ in entries]


class TestServerFloor:
    async def test_the_server_is_at_least_redis_7(self, redis: aioredis.Redis) -> None:
        # 7.0 is where XAUTOCLAIM's 3-element reply arrived. Below it the
        # tombstone purge silently does not happen, and the failure is a
        # TypeError deep inside a housekeeping sweep rather than anything that
        # points at the version.
        info = cast(dict[str, Any], await redis.info("server"))
        major = int(str(info["redis_version"]).split(".")[0])
        assert major >= 7, (
            f"outbox stream needs Redis 7.0+, got {info['redis_version']}"
        )


class TestTrimActuallyTrims:
    """DIVERGENCE 1 — the one that made this whole tier necessary.

    redis-py defaults `approximate` to True, which trims only to whole
    radix-tree nodes. On a small stream that removes NOTHING, while still
    returning success. fakeredis models approximate trimming as EXACT, so a unit
    test asserting end state passes under both settings and the cap ships
    silently disabled.
    """

    async def test_exact_trim_removes_entries(self, redis: aioredis.Redis) -> None:
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2, 3)
        ids = await _ids(redis)
        assert await trim_outbox_below(redis, ids[2]) == 2
        assert await outbox_depth(redis) == 1

    async def test_the_default_would_have_trimmed_nothing(
        self, redis: aioredis.Redis
    ) -> None:
        # The negative control. This is what src would do if approximate=False
        # were ever dropped — and it is what fakeredis CANNOT show.
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2, 3)
        ids = await _ids(redis)
        assert await redis.xtrim(HISTORY_OUTBOX_KEY, minid=ids[2]) == 0
        assert await outbox_depth(redis) == 3

    async def test_limit_cannot_be_combined_with_exact_trimming(
        self, redis: aioredis.Redis
    ) -> None:
        # Records why the old list implementation's 1000-entry slicing is not
        # ported rather than merely unnecessary: Redis refuses the combination
        # outright, so an incremental exact trim is not expressible.
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2, 3)
        ids = await _ids(redis)
        with pytest.raises(aioredis.ResponseError, match="LIMIT"):
            await redis.xtrim(
                HISTORY_OUTBOX_KEY, minid=ids[2], approximate=False, limit=1
            )


class TestTrimIsBlindToThePel:
    """The defect the cap's ack-before-trim rule exists for.

    XTRIM does not consult the PEL, so a trim that crosses delivered-but-unacked
    entries destroys their bodies and leaves the pending records behind. Those
    are plays a drainer was holding: no Postgres row, no play_history_rejected
    row, and no error naming them.
    """

    async def test_trimming_delivered_entries_leaves_the_pel_intact(
        self, redis: aioredis.Redis
    ) -> None:
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2, 3, 4, 5)
        await read_outbox_new(redis, 10)  # all five delivered, none acked
        ids = await _ids(redis)
        assert await outbox_pending_count(redis) == 5
        await trim_outbox_below(redis, ids[3])
        assert await outbox_depth(redis) == 2  # three bodies destroyed
        assert await outbox_pending_count(redis) == 5  # PEL untouched

    async def test_acking_first_leaves_no_tombstone(
        self, redis: aioredis.Redis
    ) -> None:
        # The cap's actual sequence, end to end against a real server: list the
        # doomed pending IDs, ack them, then trim the bodies in one MINID
        # command.
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2, 3, 4, 5)
        await read_outbox_new(redis, 10)
        ids = await _ids(redis)
        doomed = await outbox_pending_below(redis, ids[3])
        assert doomed == ids[:3]
        await ack_outbox(redis, doomed)
        assert await trim_outbox_below(redis, ids[3]) == 3
        assert await outbox_pending_count(redis) == 2
        # The survivors replay with their payloads; nothing comes back empty.
        replay = await read_outbox_pending(redis, 10)
        assert [e.id for e in replay] == ids[3:]
        assert all(e.wire is not None for e in replay)

    async def test_a_tombstone_replays_with_an_empty_field_map(
        self, redis: aioredis.Redis
    ) -> None:
        # The shape the drainer's null-safe extraction depends on. fakeredis
        # agrees here, which is what makes the P1 path unit-testable — this
        # confirms the agreement rather than assuming it.
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2)
        await read_outbox_new(redis, 10)
        ids = await _ids(redis)
        await redis.xdel(HISTORY_OUTBOX_KEY, ids[0])
        replay = await read_outbox_pending(redis, 10)
        assert [e.wire for e in replay] == [None, _entry(2).to_redis()]

    async def test_a_tombstone_can_be_acked(self, redis: aioredis.Redis) -> None:
        await ensure_outbox_group(redis)
        await _push(redis, 1)
        await read_outbox_new(redis, 10)
        ids = await _ids(redis)
        await redis.xdel(HISTORY_OUTBOX_KEY, ids[0])
        await retire_outbox(redis, [ids[0]])
        assert await outbox_pending_count(redis) == 0


class TestWrongTypeIsAResponseError:
    """DIVERGENCE 4. A pre-R1 LIST at history:outbox must abort startup, and the
    abort turns on the exception CLASS: ensure_outbox_group tolerates only
    BUSYGROUP and re-raises everything else. fakeredis raises AttributeError
    from XADD against a list, so only a real server can prove the class."""

    async def test_group_create_against_a_list_raises_wrongtype(
        self, redis: aioredis.Redis
    ) -> None:
        await redis.lpush(HISTORY_OUTBOX_KEY, b"pre-R1 list entry")
        with pytest.raises(aioredis.ResponseError, match="WRONGTYPE"):
            await ensure_outbox_group(redis)

    async def test_xadd_against_a_list_raises_wrongtype(
        self, redis: aioredis.Redis
    ) -> None:
        # The producer side of the same condition. This is the arm fakeredis
        # cannot carry at all — it raises AttributeError, so a unit test would
        # assert against a class production never produces.
        await redis.lpush(HISTORY_OUTBOX_KEY, b"pre-R1 list entry")
        with pytest.raises(aioredis.ResponseError, match="WRONGTYPE"):
            await redis.xadd(HISTORY_OUTBOX_KEY, {OUTBOX_FIELD: b"x"})

    async def test_the_documented_remedy_works(self, redis: aioredis.Redis) -> None:
        # The upgrade note says: stop the bot, DEL history:outbox, start it.
        await redis.lpush(HISTORY_OUTBOX_KEY, b"pre-R1 list entry")
        await redis.delete(HISTORY_OUTBOX_KEY)
        await ensure_outbox_group(redis)
        await _push(redis, 1)
        assert [e.wire for e in await read_outbox_new(redis, 10)] == [
            _entry(1).to_redis()
        ]


class TestNogroupAfterDelete:
    async def test_deleting_the_key_destroys_the_group(
        self, redis: aioredis.Redis
    ) -> None:
        """The foot-gun the stream introduces in the one operation the upgrade
        note asks operators to perform.

        Under the list transport `DEL history:outbox` merely emptied it and the
        drainer kept working. Here it takes the consumer group with it, XADD
        recreates the key with no group, and every read then fails identically
        forever unless the drain cycle heals it.
        """
        await ensure_outbox_group(redis)
        await _push(redis, 1)
        await redis.delete(HISTORY_OUTBOX_KEY)
        await _push(redis, 2)  # recreates the key, WITHOUT a group
        with pytest.raises(aioredis.ResponseError, match="NOGROUP"):
            await read_outbox_new(redis, 10)
        # And the heal restores service without losing the entry.
        await ensure_outbox_group(redis)
        assert [e.wire for e in await read_outbox_new(redis, 10)] == [
            _entry(2).to_redis()
        ]


class TestGroupBootstrap:
    async def test_id_zero_sees_entries_that_predate_the_group(
        self, redis: aioredis.Redis
    ) -> None:
        # redis-py's xgroup_create defaults to id="$", which would skip every
        # entry already in the stream — permanently, and with no error.
        await _push(redis, 1, 2)
        await ensure_outbox_group(redis)
        assert [e.wire for e in await read_outbox_new(redis, 10)] == [
            _entry(1).to_redis(),
            _entry(2).to_redis(),
        ]

    async def test_repeat_create_does_not_rewind(self, redis: aioredis.Redis) -> None:
        await ensure_outbox_group(redis)
        await _push(redis, 1)
        batch = await read_outbox_new(redis, 10)
        await retire_outbox(redis, [e.id for e in batch])
        await ensure_outbox_group(redis)  # BUSYGROUP
        assert await read_outbox_new(redis, 10) == []
        assert await read_outbox_pending(redis, 10) == []


class TestAutoclaimCursorContract:
    """DIVERGENCE 2. Real Redis signals a completed scan with "0-0"; fakeredis
    returns the last-scanned ID, which fed back as an inclusive start
    re-delivers entries the sweep already counted.

    The sweep therefore terminates on "this pass found nothing new" and counts
    DISTINCT ids, which is correct under both conventions. These tests pin the
    real contract so that workaround stays explicable rather than looking like
    superstition.
    """

    async def test_a_completed_scan_returns_the_zero_cursor(
        self, redis: aioredis.Redis
    ) -> None:
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2)
        await read_outbox_new(redis, 10)
        cursor, claimed, deleted = cast(
            tuple[bytes, list[Any], list[bytes]],
            await redis.xautoclaim(
                HISTORY_OUTBOX_KEY,
                HISTORY_OUTBOX_GROUP,
                HISTORY_OUTBOX_CONSUMER,
                min_idle_time=0,
                start_id="0-0",
                count=10,
            ),
        )
        assert cursor == b"0-0"
        assert len(claimed) == 2
        assert deleted == []

    async def test_the_sweep_counts_each_entry_once(
        self, redis: aioredis.Redis
    ) -> None:
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2, 3)
        await read_outbox_new(redis, 10)
        reclaimed, purged = await reclaim_outbox_stale(
            redis, min_idle_ms=0, count=1, max_passes=20
        )
        assert (reclaimed, purged) == (3, 0)

    async def test_the_sweep_purges_tombstones(self, redis: aioredis.Redis) -> None:
        # Besides XACK this is the only thing that clears a tombstone on
        # Redis 7, which is why the sweep is not optional.
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2)
        await read_outbox_new(redis, 10)
        ids = await _ids(redis)
        await redis.xdel(HISTORY_OUTBOX_KEY, ids[0])
        reclaimed, purged = await reclaim_outbox_stale(
            redis, min_idle_ms=0, count=10, max_passes=20
        )
        assert purged == 1
        assert await outbox_pending_count(redis) == 1

    async def test_min_idle_time_protects_a_live_peers_batch(
        self, redis: aioredis.Redis
    ) -> None:
        await ensure_outbox_group(redis)
        await _push(redis, 1)
        await redis.xreadgroup(
            HISTORY_OUTBOX_GROUP, "live-peer", {HISTORY_OUTBOX_KEY: ">"}
        )
        assert await reclaim_outbox_stale(
            redis, min_idle_ms=600_000, count=10, max_passes=20
        ) == (0, 0)


class TestLagIsUnusable:
    """DIVERGENCE 3, and the reason outbox_depth is XLEN rather than `lag`.

    XINFO GROUPS' lag is Optional: Redis returns nil whenever a deletion leaves
    a gap it cannot reconcile, so no caller can treat it as a number. fakeredis
    returns a number unconditionally — and a NEGATIVE one after deletion — so a
    unit test would assert against values production never produces.

    The precise trigger is narrower than "any XDEL", and the tests below pin the
    boundary rather than the slogan. An earlier draft of this design claimed lag
    went nil after an ordinary drain cycle; measured on 7.4.9, it does not — a
    contiguous prefix removal stays countable, whether it is the drain's own
    XACK+XDEL or the cap's MINID trim. What breaks it is a HOLE: deleting an
    entry ahead of last-delivered-id, which is exactly the shape an operator's
    manual XDEL takes.

    The conclusion is unchanged and does not depend on how often nil happens:
    an Optional gauge cannot back DEPTH_ALARM's threshold comparison, and XLEN
    is unconditional.
    """

    async def _lag(self, redis: aioredis.Redis) -> object:
        groups = cast(
            list[dict[str, Any]], await redis.xinfo_groups(HISTORY_OUTBOX_KEY)
        )
        return groups[0]["lag"]

    async def test_an_ordinary_drain_leaves_lag_countable(
        self, redis: aioredis.Redis
    ) -> None:
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2, 3)
        batch = await read_outbox_new(redis, 2)
        await retire_outbox(redis, [e.id for e in batch])  # XACK + XDEL
        assert await self._lag(redis) == 1

    async def test_the_caps_prefix_trim_also_leaves_it_countable(
        self, redis: aioredis.Redis
    ) -> None:
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2, 3, 4, 5)
        await read_outbox_new(redis, 2)
        ids = await _ids(redis)
        await trim_outbox_below(redis, ids[3])  # crosses the read boundary
        assert await self._lag(redis) == 2

    async def test_an_operator_xdel_ahead_of_the_cursor_makes_it_nil(
        self, redis: aioredis.Redis
    ) -> None:
        # The upgrade note tells operators to touch this key, so this is a
        # reachable state, not a contrived one — and it is the state in which a
        # gauge built on lag would go silent.
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2, 3, 4, 5)
        await read_outbox_new(redis, 2)
        ids = await _ids(redis)
        await redis.xdel(HISTORY_OUTBOX_KEY, ids[4])  # never delivered
        assert await self._lag(redis) is None

    async def test_xlen_plus_xpending_is_the_honest_measure(
        self, redis: aioredis.Redis
    ) -> None:
        # Unconditional, in the same state that just nil'd lag.
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2, 3)
        await read_outbox_new(redis, 2)
        assert await outbox_depth(redis) == 3
        assert await outbox_pending_count(redis) == 2


class TestDisjointDelivery:
    async def test_two_consumers_never_receive_the_same_entry(
        self, redis: aioredis.Redis
    ) -> None:
        # The server-side guarantee that replaced the drainer lease. Asserting
        # it against a real server is the point: fakeredis could implement it
        # differently and nothing else in the suite would notice.
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2, 3, 4)

        async def claim(consumer: str) -> set[bytes]:
            reply = cast(
                list[tuple[bytes, list[tuple[bytes, dict[bytes, bytes]]]]],
                await redis.xreadgroup(
                    HISTORY_OUTBOX_GROUP, consumer, {HISTORY_OUTBOX_KEY: ">"}, count=2
                ),
            )
            return {i for i, _ in reply[0][1]}

        a, b = await claim("a"), await claim("b")
        assert len(a) == 2 and len(b) == 2 and not (a & b)

    async def test_a_shared_name_replays_the_same_pending_set(
        self, redis: aioredis.Redis
    ) -> None:
        # The other half: recovery works because the PEL belongs to the NAME.
        # A successor process reading "0" inherits its predecessor's in-flight
        # batch with no lease and no TTL to wait out.
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2)
        first = await read_outbox_new(redis, 10)
        again = await read_outbox_pending(redis, 10)
        assert [e.id for e in again] == [e.id for e in first]


class TestRetireSemantics:
    async def test_ack_and_delete_are_both_required(
        self, redis: aioredis.Redis
    ) -> None:
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2, 3)
        batch = await read_outbox_new(redis, 10)
        await retire_outbox(redis, [e.id for e in batch[:2]])
        assert await outbox_pending_count(redis) == 1  # XACK ran
        assert await outbox_depth(redis) == 1  # XDEL ran

    async def test_re_settling_is_a_no_op(self, redis: aioredis.Redis) -> None:
        # Why the drain path needs no special no-retry pool: it runs on the
        # application pool with retries ENABLED, and a re-sent settle is inert.
        await ensure_outbox_group(redis)
        await _push(redis, 1)
        batch = await read_outbox_new(redis, 10)
        ids = [e.id for e in batch]
        await retire_outbox(redis, ids)
        await retire_outbox(redis, ids)
        assert await outbox_depth(redis) == 0
        assert await outbox_pending_count(redis) == 0

    async def test_minid_trim_is_idempotent_where_maxlen_is_not(
        self, redis: aioredis.Redis
    ) -> None:
        """MINID names an absolute ID, so a re-send is inert. MAXLEN names a
        LENGTH, so a re-send after concurrent arrivals destroys a SECOND tranche
        of unarchived plays — structurally the same destructive-retry defect the
        positional RPOP had. This asserts both halves, because the claim that
        MINID is safe is only meaningful next to the one that MAXLEN is not.
        """
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2, 3)
        ids = await _ids(redis)
        assert await trim_outbox_below(redis, ids[2]) == 2
        await _push(redis, 4, 5)  # concurrent arrivals, as after a blip
        assert await trim_outbox_below(redis, ids[2]) == 0  # re-send: inert

        assert await redis.xtrim(HISTORY_OUTBOX_KEY, maxlen=1, approximate=False) == 2
        await _push(redis, 6, 7)
        # Same command, same argument, a second destructive effect.
        assert await redis.xtrim(HISTORY_OUTBOX_KEY, maxlen=1, approximate=False) == 2


class TestOutboxKeyIsNonEvictable:
    async def test_the_stream_carries_no_ttl(self, redis: aioredis.Redis) -> None:
        # Golden rule 12, against a real server. An entry here is a play that is
        # not durable in Postgres yet; under volatile-lru, a TTL would make it
        # an eviction candidate and an evicted entry is a play that vanishes
        # with no error and no log line.
        await _push(redis, 1)
        assert await redis.ttl(HISTORY_OUTBOX_KEY) == -1


class TestRefPolicyIsNotAvailableYet:
    async def test_acked_ref_policy_is_rejected_below_redis_8_2(
        self, redis: aioredis.Redis
    ) -> None:
        """DIVERGENCE 5, recorded as a live check rather than a comment.

        Redis 8.2's `XTRIM ... ACKED` refuses to drop unacked entries, which is
        the cap's ack-before-trim rule in one keyword. It is a syntax error on
        the pinned redis:7-alpine and on fakeredis, so it is neither usable nor
        testable today.

        This test is written to FAIL LOUDLY the day compose moves to 8.2+,
        because that is exactly when the hand-rolled rule should collapse into
        the native one — and a silently-still-hand-rolled implementation is how
        that opportunity gets missed.
        """
        info = cast(dict[str, Any], await redis.info("server"))
        version = tuple(int(p) for p in str(info["redis_version"]).split(".")[:2])
        await ensure_outbox_group(redis)
        await _push(redis, 1, 2)
        ids = await _ids(redis)
        if version >= (8, 2):
            pytest.fail(
                f"Redis {info['redis_version']} supports ref_policy=ACKED. The cap's "
                f"hand-rolled ack-before-trim rule (history_archive._enforce_cap, "
                f"redis_client.outbox_pending_below/ack_outbox) exists only because "
                f"this server did not. Collapse it to XTRIM ... ACKED and delete this."
            )
        with pytest.raises(aioredis.ResponseError, match="syntax error"):
            await redis.execute_command(
                "XTRIM", HISTORY_OUTBOX_KEY, "MINID", ids[1], "ACKED"
            )
