"""Before ingest: every AGENT_FINISHED in a run must carry its task.json nonce.

Usage: python tools/check_markers.py <run dir>
Prints OK, or one line per mismatch and exits 1. A mismatch is a dispatch error (wrong nonce in the stub), not an agent
outcome: stop and fix the dispatch; do not edit the marker, do not ingest.
"""
import json
import sys
from pathlib import Path


def main(run):
    bad = []
    for m in sorted((Path(run).resolve() / "tasks").glob("*/AGENT_FINISHED")):
        nonce = json.loads((m.parent / "task.json").read_text(encoding="utf-8"))["nonce"]
        # utf-8-sig like the engine's read_text: agents writing via PowerShell may add a byte-order mark
        if m.read_text(encoding="utf-8-sig").strip() != f"finished {nonce}":
            bad.append(m.parent.name)
    for b in bad:
        print(f"NONCE MISMATCH: {b}")
    if bad:
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main(sys.argv[1])
