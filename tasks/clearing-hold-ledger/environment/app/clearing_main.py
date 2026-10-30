#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from clearing.replay import replay_events

ROOT = Path(__file__).resolve().parent
EVENTS = ROOT / "data" / "events.json"
OUT = ROOT / "clearing_state.json"


def main() -> None:
    events = json.loads(EVENTS.read_text())
    state = replay_events(events)
    OUT.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
