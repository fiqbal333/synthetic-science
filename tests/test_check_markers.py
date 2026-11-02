"""tools/check_markers.py: a marker with a byte-order mark and CRLF passes (agents writing through PowerShell may add both); a wrong nonce still fails."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(tmp_path, marker_bytes):
    d = tmp_path / "tasks" / "lab_decision-1234567890"
    d.mkdir(parents=True)
    (d / "task.json").write_text(json.dumps({"nonce": "87745f48d03c"}), encoding="utf-8")
    (d / "AGENT_FINISHED").write_bytes(marker_bytes)
    return subprocess.run([sys.executable, str(ROOT / "tools" / "check_markers.py"), str(tmp_path)], capture_output=True, text=True)


def test_marker_with_bom_passes(tmp_path):
    r = _run(tmp_path, b"\xef\xbb\xbffinished 87745f48d03c\r\n")
    assert r.returncode == 0 and "OK" in r.stdout


def test_wrong_nonce_still_fails(tmp_path):
    r = _run(tmp_path, b"\xef\xbb\xbffinished 000000000000\r\n")
    assert r.returncode == 1 and "NONCE MISMATCH" in r.stdout
