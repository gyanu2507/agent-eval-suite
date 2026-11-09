# Calibration: slot-ring-buffer

## Why this task

Public systems-lane sample in the spirit of FIFO / packet-path tasks, stripped to a dual-consumer ring with a drop-incoming policy. No HDL, no private netlist fixtures.

## Pass bar

- Oracle clears all unit tests.
- Starter fails dual-consumer + drop-policy checks.
- Agent implementation must keep unread packets for the slow consumer and never overwrite them.

## Anti-shortcuts

- Capacity edge cases (N=1, N=2) are covered.
- Occupancy is derived from cursors, not a stale size counter the agent can fake in isolation.
- Tests import the module under test and drive the public API only.
- v1.1 replays 500 random write/read sequences (capacity 1 to 6, seq wrapping at 2**16) and compares every return value and `stats()` with the reference. v1.0 had four scripted scenarios, which missed a ring that counts empty reads and one that rejects a wrapped seq. See `redteam/`.
