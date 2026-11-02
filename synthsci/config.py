"""Scenario configs, seed layers and run identity.

A run = resolved config + master seed + mode. The resolved config is hashed; the three seed layers
(world, population, run) derive from the master seed unless a layer is pinned explicitly in the config
(truth-fixed and population-fixed Monte Carlo designs).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .util import canonical_json, deep_merge, read_text, seed_from, sha256_text

PKG_ROOT = Path(__file__).resolve().parent
WS_ROOT = PKG_ROOT.parent
CONFIG_DIR = WS_ROOT / "configs"
MODES = ("llm", "mock", "fast")
LAYERS = ("world", "population", "run")


def load_yaml(p) -> dict:
    txt = read_text(p)
    return yaml.safe_load(txt) or {}


def _resolve_path(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    cands = [CONFIG_DIR / name_or_path, CONFIG_DIR / "scenarios" / name_or_path,
             CONFIG_DIR / f"{name_or_path}.yaml", CONFIG_DIR / "scenarios" / f"{name_or_path}.yaml"]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"config not found: {name_or_path}")


def resolve(scenario: str | Path | dict, overrides: dict | None = None) -> dict:
    """Load a scenario YAML, follow its `extends` chain, deep-merge, apply overrides."""
    if isinstance(scenario, dict):
        cfg = copy.deepcopy(scenario)
        chain = []
    else:
        path = _resolve_path(str(scenario))
        cfg = load_yaml(path)
        chain = [path]
    while cfg.get("extends"):
        parent_name = cfg.pop("extends")
        parent_path = _resolve_path(parent_name) if not (chain and (chain[-1].parent / parent_name).exists()) \
            else chain[-1].parent / parent_name
        parent = load_yaml(parent_path)
        chain.append(parent_path)
        child_extends = parent.get("extends")
        cfg = deep_merge(parent, cfg)
        if child_extends:
            cfg["extends"] = child_extends
        else:
            cfg.pop("extends", None)
    cfg.pop("extends", None)
    if overrides:
        cfg = deep_merge(cfg, _expand_dotted(overrides))
    cfg.setdefault("scenario", Path(chain[0]).stem if chain else "inline")
    return cfg


def _expand_dotted(d: dict) -> dict:
    """{'a.b': 1} -> {'a': {'b': 1}}"""
    out: dict = {}
    for k, v in d.items():
        parts = k.split(".")
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out


def config_hash(cfg: dict) -> str:
    """Hash of the resolved config, excluding run-identity fields that are supplied separately."""
    c = copy.deepcopy(cfg)
    for k in ("seed", "mode", "paired_run_dir", "run_dir"):
        c.pop(k, None)
    return sha256_text(canonical_json(c))


def seed_layers(cfg: dict, master_seed: int) -> dict:
    pinned = (cfg.get("seeds") or {})
    out = {}
    for layer in LAYERS:
        if pinned.get(layer) is not None:
            out[layer] = int(pinned[layer])
        else:
            out[layer] = seed_from(master_seed, layer)
    out["master"] = int(master_seed)
    return out


def run_id(cfg: dict, master_seed: int, mode: str) -> str:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    return f"{cfg['scenario']}-s{master_seed}-{mode}-{config_hash(cfg)[:8]}"


def rng(layer_seed: int, *tags: Any) -> np.random.Generator:
    """Independent RNG stream for (layer seed, tags...). Deterministic and order-free."""
    return np.random.Generator(np.random.PCG64(seed_from(layer_seed, *tags)))


def get(cfg: dict, dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def dump_yaml(cfg: dict) -> str:
    return yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True, default_flow_style=False)
