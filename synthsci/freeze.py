"""freeze.py: write and verify the SHA-256 list of everything that must not change between the freeze
and the Stage 1 runs: seeding tables, configs, prompts, schemas, the engine code, the rule-agent parameters."""
from __future__ import annotations

import sys
from pathlib import Path

from .config import WS_ROOT
from .util import read_text, sha256_file, write_text

FROZEN_GLOBS = ["configs/**/*.yaml", "configs/**/*.csv", "grounding/tables/*.csv", "grounding/tables/*.json",
                "synthsci/**/*.py", "synthsci/prompts/*.md", "synthsci/schemas/*.json"]
FREEZE_FILE = WS_ROOT / "FREEZE.sha256"


def _files() -> list[Path]:
    out = set()
    for g in FROZEN_GLOBS:
        out |= {p for p in WS_ROOT.glob(g) if p.is_file() and "__pycache__" not in p.parts}
    return sorted(out)


def freeze() -> int:
    lines = [f"{sha256_file(p)}  {p.relative_to(WS_ROOT).as_posix()}" for p in _files()]
    write_text(FREEZE_FILE, "\n".join(lines) + "\n")
    print(f"froze {len(lines)} files -> {FREEZE_FILE}")
    return 0


def verify() -> int:
    if not FREEZE_FILE.exists():
        print("no FREEZE.sha256")
        return 2
    bad = []
    seen = set()
    for line in read_text(FREEZE_FILE).splitlines():
        if not line.strip():
            continue
        h, rel = line.split("  ", 1)
        p = WS_ROOT / rel
        seen.add(p)
        if not p.exists():
            bad.append(f"missing {rel}")
        elif sha256_file(p) != h:
            bad.append(f"changed {rel}")
    for p in _files():
        if p not in seen:
            bad.append(f"new file not frozen {p.relative_to(WS_ROOT).as_posix()}")
    if bad:
        print("\n".join(bad))
        print(f"FAIL: {len(bad)} problem(s)")
        return 1
    print("OK: frozen files verify")
    return 0


if __name__ == "__main__":
    sys.exit(verify() if "--verify" in sys.argv else freeze())
