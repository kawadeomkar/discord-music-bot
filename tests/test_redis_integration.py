"""Real-Redis integration tier for the outbox stream.

Opt-in; two ways to supply the server (see `redis_url`):

    just test-redis                      # local: testcontainers, needs Docker
    REDIS_TEST_URL=... just test-redis   # CI: an already-running server

fakeredis runs every stream command the design uses, but six behaviours diverge
from `redis:7-alpine` (measured on 7.4.9) and every one fails in the
SAFE-LOOKING direction — green unit tests, broken production:

    1  xtrim(approximate=True), redis-py's default: fakeredis trims exactly;
       real Redis trims NOTHING on a small stream and reports success anyway
    2  XAUTOCLAIM completion cursor: fakeredis returns the last-scanned id,
       real Redis returns "0-0"
    3  XINFO GROUPS `lag`: fakeredis is off by one on a fresh group and goes
       NEGATIVE after deletion; real Redis returns nil when unreconcilable
    4  XADD against a list key: fakeredis raises AttributeError, real Redis
       raises ResponseError (WRONGTYPE)
    5  ref_policy=KEEPREF/DELREF/ACKED: syntax error below Redis 8.2 on both
    6  generated IDs after the stream empties: fakeredis derives `*` from the
       live entries alone, so a drained stream forgets its last ID and reissues
       one the group already delivered; real Redis keeps last_id independently

Rows 1, 3 and 4 are undetectable without a real server; rows 2 and 6 are
asserted here so the unit tier's workarounds stay honest about what they work
around. Row 6 is patched into the fake — see _patch_xadd_monotonic_ids in
tests/conftest.py — because it surfaces as a rare, silent drain of nothing.

Floor is Redis 7.0, not 6.2: XAUTOCLAIM's 3-element reply carrying the
deleted-ID list — the tombstone behaviour the drainer depends on — arrived in
7.0; on 6.2 redis-py hands back (None, None) and the tombstone is never purged.

Isolation: there is no throwaway-database-per-test equivalent to the pg tier's
`raw_pg_dsn`, so each test FLUSHDBs a dedicated numbered database and the tier
assumes it owns the server it is pointed at.
"""

import asyncio
import os
import warnings
from collections.abc import AsyncIterator, Iterator
from typing import Any, cast

import pytest
import redis.asyncio as aioredis

from redis.exceptions import OutOfMemoryError

from src.guild_state import HistoryEntry
from src.redis_client import (
    read_guild_configs,
    HISTORY_CACHE_LIMIT,
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

# Must match the redis image in ci.yml's redis-integration job AND in
# docker-compose.yml — `just pins` asserts all three agree. Live risk, not
# housekeeping: `redis:7-alpine` FLOATS (7.4.9 today), so a compose bump can
# move the server out from under the divergences measured above.
_REDIS_IMAGE = "redis:7-alpine"


@pytest.fixture(scope="session")
def redis_url() -> Iterator[str]:
    """A Redis this tier may FLUSHDB.

    REDIS_TEST_URL wins when set: CI's service container is already pulled and
    health-checked, so testcontainers' pull and reaper cost buys nothing. Yields
    a URL, not a client, and is SYNCHRONOUS — fixture loop scope is "function",
    so a connection opened here would outlive its loop (as in the pg tier).
    """
    external = os.getenv("REDIS_TEST_URL")
    if external:
        yield external
        return

    # Scoped suppression, not a pyproject filterwarnings entry: testcontainers
    # 4.x applies its own deprecated @wait_container_is_ready at IMPORT time, so
    # under golden rule 11 the import itself fails. Confining it here keeps every
    # other DeprecationWarning fatal. Removable when testcontainers migrates.
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


def _id_parts(raw: bytes) -> tuple[int, int]:
    """`ms-seq` as integers — bytes compare lexicographically, not numerically."""
    ts, seq = raw.split(b"-")
    return int(ts), int(seq)


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
    """DIVERGENCE 1, the one that made this tier necessary: redis-py defaults
    `approximate` to True, which trims only whole radix-tree nodes — NOTHING on
    a small stream, still reporting success. fakeredis models it as EXACT, so a
    unit test passes either way and the cap ships silently disabled."""

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
    """The defect the cap's ack-before-trim rule exists for: XTRIM does not
    consult the PEL, so a trim crossing delivered-but-unacked entries destroys
    their bodies and leaves the pending records. Those are plays a drainer was
    holding — no Postgres row, no play_history_rejected row, no error."""

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
    """DIVERGENCE 4. A pre-stream list at history:outbox must abort startup, and the
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

    async def test_outbox_depth_against_a_list_raises_wrongtype(
        self, redis: aioredis.Redis
    ) -> None:
        # The disabled arm of the same condition: setup_hook's leftover-outbox
        # probe calls outbox_depth (XLEN) and downgrades a ResponseError to a
        # warning instead of aborting, so the class matters there too. fakeredis
        # models XLEN-against-a-list correctly; this pins that the server agrees.
        await redis.lpush(HISTORY_OUTBOX_KEY, b"pre-R1 list entry")
        with pytest.raises(aioredis.ResponseError, match="WRONGTYPE"):
            await outbox_depth(redis)

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
        """The foot-gun in the one operation the upgrade note asks operators to
        perform: under the list transport `DEL history:outbox` merely emptied
        it, but here it takes the consumer group with it — XADD recreates the
        key groupless and every read fails until the drain cycle heals it."""
        await ensure_outbox_group(redis)
        await _push(redis, 1)
        await redis.delete(HISTORY_OUTBOX_KEY)
        await _push(redis, 2)  # recreates the key, without a group
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
    returns the last-scanned ID, which fed back as an inclusive start re-delivers
    entries already counted. So the sweep terminates on "this pass found nothing
    new" and counts distinct ids — correct under both conventions."""

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

    XINFO GROUPS' lag is Optional — nil whenever a deletion leaves a gap Redis
    cannot reconcile — so no caller can treat it as a number; fakeredis returns
    one unconditionally, and a NEGATIVE one after deletion. The trigger is
    narrower than "any XDEL" and these tests pin the boundary: on 7.4.9 a
    contiguous prefix removal (the drain's XACK+XDEL, the cap's MINID trim)
    stays countable, while a HOLE ahead of last-delivered-id — the shape an
    operator's manual XDEL takes — nils it. Either way an Optional gauge cannot
    back DEPTH_ALARM's threshold comparison; XLEN is unconditional.
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
        # The other half: recovery works because the PEL belongs to the name.
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
        # application pool with retries enabled, and a re-sent settle is inert.
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
        """MINID names an absolute ID, so a re-send is inert; MAXLEN names a
        length, so a re-send after concurrent arrivals destroys a second tranche
        of unarchived plays. Both halves are asserted — "MINID is safe" only
        means something next to "MAXLEN is not"."""
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


class TestGeneratedIdsOutrankRetiredOnes:
    """Divergence 6, asserted against a real server.

    retire_outbox empties the stream while the group keeps its
    last-delivered-id, so every later push depends on the server issuing an ID
    above it. Real Redis carries last_id on the stream itself; fakeredis derives
    it from the live entries and reissues a retired ID, which `>` then skips.
    """

    async def test_an_emptied_stream_still_issues_a_rising_id(
        self, redis: aioredis.Redis
    ) -> None:
        await ensure_outbox_group(redis)
        await _push(redis, 1)
        first = await read_outbox_new(redis, 10)
        await retire_outbox(redis, [e.id for e in first])
        assert await outbox_depth(redis) == 0

        # Same millisecond as the retired entry, most likely — the case the fake
        # gets wrong. The entry must still be delivered.
        await _push(redis, 2)
        second = await read_outbox_new(redis, 10)
        assert len(second) == 1
        assert _id_parts(second[0].id) > _id_parts(first[0].id)

    async def test_last_id_survives_deleting_every_entry(
        self, redis: aioredis.Redis
    ) -> None:
        await ensure_outbox_group(redis)
        await _push(redis, 1)
        ids = await _ids(redis)
        await redis.xdel(HISTORY_OUTBOX_KEY, ids[0])
        # XINFO field names come back decoded even at decode_responses=False;
        # only the values stay bytes.
        info = cast(dict[str, Any], await redis.xinfo_stream(HISTORY_OUTBOX_KEY))
        assert info["length"] == 0
        assert info["last-generated-id"] == ids[0]


class TestOutboxKeyIsNonEvictable:
    async def test_the_stream_carries_no_ttl(self, redis: aioredis.Redis) -> None:
        # Golden rule 12, against a real server. An entry here is a play that is
        # not durable in Postgres yet; under volatile-lru, a TTL would make it
        # an eviction candidate and an evicted entry is a play that vanishes
        # with no error and no log line.
        await _push(redis, 1)
        assert await redis.ttl(HISTORY_OUTBOX_KEY) == -1


class TestHistoryWritePathAgainstARealServer:
    """Real-server coverage for the push path: LPUSH+LTRIM+PERSIST+conditional
    XADD, plus an OOM recovery. The unit tier's LTRIM assertions run against
    fakeredis, where stream behaviour diverges (see the module docstring) and
    the OOM path is untestable at all — there is no maxmemory concept."""

    async def test_the_list_is_capped_and_never_expires(
        self, redis: aioredis.Redis
    ) -> None:
        # An oversized list left by a build that did not cap — the shape every
        # upgrading deployment has — must come down on the first write and stay
        # PERSISTed. Golden rule 12: bounded by length, never by time.
        store = GuildRedisStore(redis, guild_id=42)
        key = store.history_key()
        for n in range(HISTORY_CACHE_LIMIT + 25):
            await redis.lpush(key, _entry(n).to_redis())
        await redis.expire(key, 3600)  # an inherited TTL from an older build

        await store.push_history(_entry(999))

        assert await redis.llen(key) == HISTORY_CACHE_LIMIT
        assert await redis.ttl(key) == -1  # PERSIST self-healed it
        newest = cast(list[bytes], await redis.lrange(key, 0, 0))
        assert b"Song 999" in newest[0]

    async def test_the_disabled_archive_never_creates_the_outbox(
        self, redis: aioredis.Redis, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The consent gate, against a real server: absent, not just empty.
        monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "false")
        store = GuildRedisStore(redis, guild_id=42)
        for n in range(5):
            await store.push_history(_entry(n))

        assert await redis.exists(HISTORY_OUTBOX_KEY) == 0
        assert await redis.llen(store.history_key()) == 5

    async def test_lpush_is_denyoom_but_a_bare_ltrim_is_not(
        self, redis: aioredis.Redis
    ) -> None:
        """The empirical claim the OOM recovery rests on, checkable only here:
        LPUSH carries Redis' `denyoom` flag and LTRIM does not, so at maxmemory
        the transaction is refused while the one command that could free memory
        is still allowed. The unit test can only pin call order, not this."""
        store = GuildRedisStore(redis, guild_id=42)
        key = store.history_key()
        for n in range(HISTORY_CACHE_LIMIT + 50):
            await redis.lpush(key, _entry(n).to_redis())

        used = cast(dict[str, Any], await redis.info("memory"))["used_memory"]
        await redis.config_set("maxmemory-policy", "noeviction")
        await redis.config_set("maxmemory", str(int(used * 0.9)))
        try:
            with pytest.raises(OutOfMemoryError):
                await redis.lpush(key, _entry(998).to_redis())
            # …and the trim is still permitted, which is what makes the
            # recovery possible at all.
            await redis.ltrim(key, 0, HISTORY_CACHE_LIMIT - 1)
            assert await redis.llen(key) == HISTORY_CACHE_LIMIT
        finally:
            await redis.config_set("maxmemory", "0")
            await redis.config_set("maxmemory-policy", "noeviction")


class TestRefPolicyIsNotAvailableYet:
    async def test_acked_ref_policy_is_rejected_below_redis_8_2(
        self, redis: aioredis.Redis
    ) -> None:
        """DIVERGENCE 5, as a live check rather than a comment. Redis 8.2's
        `XTRIM ... ACKED` is the cap's ack-before-trim rule in one keyword, but
        a syntax error on the pinned redis:7-alpine and on fakeredis. This FAILS
        LOUDLY the day compose moves to 8.2+, so the collapse is not missed."""
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


class TestConfigReadsSurviveTheConnectionCap:
    """The divergence this tier exists for. fakeredis has no connection pool, so a
    per-guild fan-out looks perfect there and fails on a real server the moment a
    bot passes `max_connections` guilds — silently, because @_guild_op turns the
    pool's error into an all-unset config that reads as "this guild never chose"."""

    @staticmethod
    async def _client(redis_url: str) -> tuple[aioredis.Redis, Any]:
        pool = aioredis.ConnectionPool.from_url(
            redis_url, max_connections=20, decode_responses=False
        )
        return aioredis.Redis(connection_pool=pool), pool

    async def test_the_naive_per_guild_fanout_really_does_lose_guilds(
        self, redis_url: str
    ) -> None:
        """The bug, pinned. Not hypothetical and not a timing artefact: the pool
        RAISES `MaxConnectionsError` rather than queueing (that is
        BlockingConnectionPool), and the raise happens before `call_with_retry`, so
        the configured retry never applies. Exactly `max_connections` survive."""
        client, pool = await self._client(redis_url)
        try:
            await client.flushdb()
            ids = list(range(1, 61))
            for guild_id in ids:
                await GuildRedisStore(client, guild_id).set_debug_mode(True)

            naive = await asyncio.gather(
                *(GuildRedisStore(client, g).get_config() for g in ids)
            )

            assert sum(c.debug_mode is True for c in naive) == 20
        finally:
            await client.flushdb()
            await client.aclose()
            await pool.disconnect()

    async def test_read_guild_configs_resolves_every_guild(
        self, redis_url: str
    ) -> None:
        """And the fix, against the same server and the same cap."""
        client, pool = await self._client(redis_url)
        try:
            await client.flushdb()
            ids = list(range(1, 61))
            for guild_id in ids:
                await GuildRedisStore(client, guild_id).set_debug_mode(True)

            configs = await read_guild_configs(client, ids)

            assert sorted(configs) == ids
            assert all(c.debug_mode is True for c in configs.values())
        finally:
            await client.flushdb()
            await client.aclose()
            await pool.disconnect()
