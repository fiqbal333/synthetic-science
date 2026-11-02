"""SHA-256 manifest of run folders ('<sha256>  <path relative to the workspace>'), and its verification (used to show that extending a run never changes the original).

    python tools/runs_manifest.py write  runs-manifest.sha256 runs/S0-150/1 runs/S0-150/2 ...
    python tools/runs_manifest.py verify runs-manifest.sha256 runs/S0-150/1 runs/S0-150/2 ...

`verify` checks every listed file and reports files under the given folders that the manifest does not list.
"""
import json
import sys
from pathlib import Path

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W))
from synthsci.extend import verify_manifest, write_manifest  # noqa: E402


def main(argv) -> int:
    cmd, man, roots = argv[0], Path(argv[1]), [W / r for r in argv[2:]]
    man = man if man.is_absolute() else W / man
    if cmd == "write":
        n = write_manifest(man, roots, W)
        print(f"{n} files -> {man}")
        return 0
    res = verify_manifest(man, W, roots)
    print(json.dumps({k: (v if not isinstance(v, list) else v[:20]) for k, v in res.items()}, indent=2))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
