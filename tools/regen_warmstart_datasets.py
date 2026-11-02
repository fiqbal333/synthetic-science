"""Write every warm-start (quarter <= 0) dataset of the Stage 1 paper-world runs to disk, verified against the engine's hash.

Usage: python tools/regen_warmstart_datasets.py [out_root]      (default runs/_warmstart-datasets)
For each of runs/S0-150/{1,2,3}/llm and runs/S0P-150/{1,2}/llm (read only): rebuild the World from the run's resolved config
and seed in a SCRATCH folder (World() writes landscape and population files into its run dir, so it must never be pointed at
a finished run), regenerate each quarter <= 0 dataset in ledger/datasets.jsonl from its recorded dataset seed, and write
data.csv, codebook.md and sample_frame.md to <out_root>/<scenario>/<seed>/<dataset_id>/. The SHA-256 of each data.csv must
equal the hash the engine recorded when the run made it. Writes index.json per run. Nothing in the run folders is touched.
"""
import json
import shutil
import sys
from pathlib import Path

import yaml

W = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(W))
from synthsci.engine import World  # noqa: E402
from synthsci.util import read_json, read_jsonl, read_text, sha256_text  # noqa: E402

RUNS = [("S0-150", 1), ("S0-150", 2), ("S0-150", 3), ("S0P-150", 1), ("S0P-150", 2)]


def main(out_root=None):
    out_root = Path(out_root or W / "runs" / "_warmstart-datasets").resolve()
    scratch = W / "tmp" / "regen-scratch"
    total = bad = 0
    for scen, seed in RUNS:
        rd = W / "runs" / scen / str(seed) / "llm"
        cfg = yaml.safe_load(read_text(rd / "config.resolved.yaml"))
        meta = read_json(rd / "run.json")
        rows = [r for r in read_jsonl(rd / "ledger" / "datasets.jsonl") if int(r.get("quarter", 1)) <= 0]
        sd = scratch / scen / str(seed)
        if sd.exists():
            shutil.rmtree(sd)
        w = World(cfg, meta["seed"], meta["mode"], "mock", sd, None)
        index = []
        for r in rows:
            m = r["meta"]
            dest = out_root / scen / str(seed) / r["dataset_id"]
            res = w.gen.generate(r["question_id"], m["dataset_seed"], r["dataset_id"], upper_half=m["frame"].get("upper_half", False), write_dir=dest)
            on_disk = sha256_text((dest / "data.csv").read_text(encoding="utf-8"))
            ok = res["meta"]["data_sha256"] == m["data_sha256"] == on_disk
            total += 1
            bad += 0 if ok else 1
            index.append({"dataset_id": r["dataset_id"], "quarter": r["quarter"], "question_id": r["question_id"], "who": r.get("who"),
                          "n_rows": res["meta"]["n_rows"], "data_sha256": on_disk, "ledger_sha256": m["data_sha256"], "hash_matches": ok})
        (out_root / scen / str(seed)).mkdir(parents=True, exist_ok=True)
        (out_root / scen / str(seed) / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")
        print(f"{scen} s{seed}: {len(rows)} warm-start datasets written; hash mismatches: {sum(1 for x in index if not x['hash_matches'])}")
    (out_root / "README.md").write_text(
        "# Warm-start datasets of the Stage 1 paper-world runs\n\nRebuilt by `tools/regen_warmstart_datasets.py` from each run's config, "
        "seed and recorded dataset seed. These are the datasets the rule agents analysed in quarters -15 to 0 (the warm start), which "
        "the live runs did not keep as files. Each data.csv's SHA-256 equals the hash the engine recorded in the run's "
        "`ledger/datasets.jsonl` when the run made it (see each `index.json`). The run folders were not modified.\n", encoding="utf-8")
    print(f"total {total}, mismatches {bad}")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
