"""Leak scanner on every agent-visible document: no design term, hidden-truth word or local file path may reach an agent.

Token lists live in configs/leakscan.yaml; POSIX and Windows absolute paths are caught; an allow-list covers known false positives.

Matching rules: a single-word token matches on word boundaries (letters and digits are word
characters; `_` is a separator, so `theta_fe` still trips `theta` and `bootstrap` no longer trips `trap`), with the
plain inflections s/es/ed/ing/ly/'s allowed on the right. A token that contains a space, hyphen, slash, backslash or
dot is a phrase and matches as a substring. Tokens listed under `prefix:` in leakscan.yaml match any word that starts
with them (`simulat` -> simulation, simulated).
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from .config import CONFIG_DIR, load_yaml

# a drive-letter path (not preceded by a word character, and not a JSON escape such as "terms:\n") or a POSIX home/tmp path
ABS_PATH_RE = re.compile(r"(?:(?<![A-Za-z0-9_])[A-Za-z]:[\\/](?![nrtbf\"'\\\\/])|/(?:home|Users|tmp|mnt|var|opt)/)[^\s`'\"<>|)\]]*")

_WORD = "a-z0-9"
_SUFFIX = r"(?:s|es|ed|ing|ly|'s)?"
_PHRASE_CHARS = " -/\\."


@lru_cache(maxsize=4)
def _lists(path: str) -> tuple[list[str], list[str], list[str], list[str]]:
    cfg = load_yaml(path)
    design = [t.lower() for t in cfg.get("design", [])]
    strict = design + [t.lower() for t in cfg.get("strict", [])]
    allow = [t.lower() for t in cfg.get("allow", [])]
    prefix = [t.lower() for t in cfg.get("prefix", [])]
    unknown = [p for p in prefix if p not in strict]
    if unknown:
        raise ValueError(f"leakscan.yaml: prefix tokens must also be listed under design or strict: {unknown}")
    return design, strict, allow, prefix


def is_phrase(token: str) -> bool:
    return any(ch in token for ch in _PHRASE_CHARS)


@lru_cache(maxsize=16)
def _pattern(tokens: tuple[str, ...], prefixes: tuple[str, ...]) -> re.Pattern:
    parts = []
    for i, t in enumerate(tokens):
        e = re.escape(t)
        if t in prefixes:
            parts.append(f"(?P<t{i}>(?<![{_WORD}]){e})")
        elif is_phrase(t):
            parts.append(f"(?P<t{i}>{e})")
        else:
            parts.append(f"(?P<t{i}>(?<![{_WORD}]){e}{_SUFFIX}(?![a-z]))")
    return re.compile("|".join(parts))


def _hits(text: str, tokens: list[str], allow: list[str], prefixes: list[str] = ()) -> list[str]:
    low = text.lower()
    for a in allow:
        low = low.replace(a, " ")
    pat = _pattern(tuple(tokens), tuple(prefixes))
    found = {tokens[int(m.lastgroup[1:])] for m in pat.finditer(low)}
    return sorted(found)


def agent_allow(config: str | Path | None = None) -> list[str]:
    """Phrases allowed in agent-written text only (e.g. 'bad_control' that an author coined from a reviewer's 'bad
    control'); text we write never gets this list."""
    return [t.lower() for t in load_yaml(str(config or CONFIG_DIR / "leakscan.yaml")).get("agent_allow", [])]


def scan_text(text: str, mode: str = "strict", allowed_paths: list[str] = (), config: str | Path | None = None,
              extra_allow: list[str] = ()) -> list[str]:
    """Return a list of problems (empty = clean). mode 'strict' for text we wrote; 'design' for agent-written text."""
    design, strict, allow, prefix = _lists(str(config or CONFIG_DIR / "leakscan.yaml"))
    allow = allow + [a.lower() for a in extra_allow]
    tokens = strict if mode == "strict" else design
    allowed = [a.replace("\\", "/").lower().rstrip("/") for a in allowed_paths]
    problems = []
    stripped = text
    # absolute paths: those inside the task folder are not leaks and are removed before the token scan (the run folder's
    # own name, e.g. the mode "llm", must not trip the scan); any other absolute path is a problem
    for m in sorted(ABS_PATH_RE.finditer(text), key=lambda m: -m.start()):
        q = m.group(0).replace("\\", "/").rstrip(".,;:").lower()
        if any(q.startswith(a) for a in allowed):
            stripped = stripped[:m.start()] + "<PATH>" + stripped[m.end():]
        else:
            problems.append(f"path outside task: {m.group(0)}")
    problems = [f"{mode} token: {t!r}" for t in _hits(stripped, tokens, allow, prefix)] + problems
    return problems


def scan_file(p: str | Path, mode: str = "strict", allowed_paths: list[str] = ()) -> list[str]:
    from .util import read_text
    return [f"{Path(p).name}: {x}" for x in scan_text(read_text(p), mode=mode, allowed_paths=allowed_paths)]


def scan_dir(d: str | Path, mode: str = "strict", allowed_paths: list[str] = (), suffixes=(".md", ".json", ".txt", ".yaml")) -> list[str]:
    out: list[str] = []
    for p in sorted(Path(d).rglob("*")):
        if p.is_file() and p.suffix in suffixes:
            out.extend(scan_file(p, mode=mode, allowed_paths=allowed_paths))
    return out
