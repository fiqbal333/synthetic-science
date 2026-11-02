"""Shared helpers: hashing, LF-normalized writes, BOM-tolerant reads, canonical JSON, JSONL streaming.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Iterator


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return sha256_bytes(s.encode("utf-8"))


def sha256_file(p: os.PathLike | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def seed_from(*parts: Any, bits: int = 63) -> int:
    """Deterministic integer seed from arbitrary parts (used for every RNG stream)."""
    h = sha256_text("|".join(str(p) for p in parts))
    return int(h[:16], 16) & ((1 << bits) - 1)


def short_id(*parts: Any, n: int = 10) -> str:
    return sha256_text("|".join(str(p) for p in parts))[:n]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_default)


def _default(o):
    try:
        import numpy as np
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def read_json(p) -> Any:
    with open(p, encoding="utf-8-sig") as f:
        return json.load(f)


def _replace(tmp: str, p) -> None:
    """Atomic rename with retries: Windows indexers and antivirus briefly lock freshly written files."""
    import time
    for attempt in range(6):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (2 ** attempt))


def write_json(p, obj: Any, indent: int | None = 2) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=indent, sort_keys=True, ensure_ascii=False, default=_default)
        f.write("\n")
    _replace(tmp, p)


def write_text(p, s: str) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(p) + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    _replace(tmp, p)


def read_text(p) -> str:
    with open(p, encoding="utf-8-sig") as f:  # tolerate BOMs written by agents via PowerShell
        return f.read()


def append_jsonl(p, rows: Iterable[dict]) -> int:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "a", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(canonical_json(r))
            f.write("\n")
            n += 1
    return n


def iter_jsonl(p) -> Iterator[dict]:
    if not Path(p).exists():
        return
    with open(p, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(p) -> list[dict]:
    return list(iter_jsonl(p))


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def ensure_dir(p) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_posix(p) -> str:
    return str(p).replace("\\", "/")


def hash_tree(root, include: Iterable[str] | None = None, exclude_dirs: Iterable[str] = ("log",)) -> str:
    """Hash of every file under root (relative path + content), for byte-identity tests."""
    root = Path(root)
    h = hashlib.sha256()
    files = []
    for p in root.rglob("*"):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            if any(rel.split("/")[0] == d for d in exclude_dirs):
                continue
            if include and not any(rel.startswith(i) for i in include):
                continue
            files.append((rel, p))
    for rel, p in sorted(files):
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(sha256_file(p).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()
