# Slot Ring Buffer

Implement a fixed-capacity packet ring under `environment/app/ring.py`.

Two consumers (`A` and `B`) read independently. A producer writes packets into slots. When the ring is full with respect to the **slowest** consumer, new writes follow the discard policy below.

## Contract

1. Capacity `N` slots, indexed `0 .. N-1`.
2. Each slot holds at most one packet: `{seq: int, payload: bytes}`.
3. `write(seq, payload) -> "ok" | "dropped"`  
   - If both consumers have caught up enough that a free slot exists, write and return `"ok"`.  
   - Else drop the **incoming** packet and return `"dropped"` (never overwrite unread data).
4. `read(consumer) -> packet | None` advances that consumer only.
5. A packet remains readable until **both** consumers have read past its slot.
6. `stats()` returns `{writes_ok, writes_dropped, reads_a, reads_b, occupancy}`.
7. Occupancy is the number of slots still needed by at least one consumer.
8. `seq` values are opaque and may wrap; do not assume monotonic use beyond equality checks in tests.

## Starter

The starter single-consumer ring is wrong for dual cursors and drops the wrong packets. Replace the logic. Keep the public methods.
