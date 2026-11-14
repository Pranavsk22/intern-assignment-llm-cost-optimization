from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from question1.agent import SupportCopilot  # noqa: E402


def main(ticket_id: str = "T-1001") -> None:
    copilot = SupportCopilot()
    copilot.handle_ticket(ticket_id)
    print(f"Ticket {ticket_id}")
    print("| call | input tok | cached tok | uncached tok | output tok | points |")
    print("|---|---:|---:|---:|---:|---:|")
    for c in copilot.session.calls:
        cached = min(c.cached_prefix_tokens, c.input_tokens)
        print(f"| {c.label} | {c.input_tokens} | {cached} | {c.input_tokens - cached} | {c.output_tokens} | {c.points()} |")
    print(f"| **total** | | | | | **{copilot.session.total_points()}** |")


if __name__ == "__main__":
    main(*sys.argv[1:2])
