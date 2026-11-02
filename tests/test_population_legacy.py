"""Population of scientists, lab formation, and the legacy library of earlier published findings."""
import copy

from synthsci import config as C
from synthsci.datasets import DatasetGenerator
from synthsci.legacy import build_legacy
from synthsci.population import annual_step, availability, build_population, form_labs


def test_population_profiles(cfg, tables):
    seeds = C.seed_layers(cfg, 1)
    pop = build_population(cfg, seeds, tables.theory_shares())
    assert len(pop) == 60 and len({p["id"] for p in pop}) == 60
    assert all(p["style"] == "VERIFIED" and p["throughput"] == "T0" for p in pop)
    assert build_population(cfg, seeds, tables.theory_shares()) == pop
    s2 = C.resolve("S2")
    pop2 = build_population(s2, seeds, tables.theory_shares())
    assert 0.6 < sum(p["style"] == "AUTO" for p in pop2) / 60 < 0.95
    assert all((p["style"] == "AUTO") == (p["throughput"] == "T1") for p in pop2)   # correlated assignment


def test_availability_identical_across_regimes(cfg):
    seeds = C.seed_layers(cfg, 1)
    a = [availability(seeds, "A001", q, 0, 0.33, 1.0) for q in range(1, 9)]
    b = [availability(seeds, "A001", q, 0, 0.33, 1.0) for q in range(1, 9)]
    assert a == b


def test_lab_formation(cfg, tables):
    rcfg = C.resolve("R-b")
    seeds = C.seed_layers(rcfg, 1)
    pop = build_population(rcfg, seeds, tables.theory_shares())
    labs = form_labs(rcfg, seeds, pop)
    assert len(labs) == round(60 / 4.4)
    assert sum(l["size"] for l in labs) == 60
    assert len({m for l in labs for m in l["members"]}) == 60
    heads = {l["head"] for l in labs}
    assert all(any(p["id"] == h and p["stage"] in ("full", "associate") for p in pop) for h in heads)


def test_legacy_library(cfg, tables, landscape):
    seeds = C.seed_layers(cfg, 1)
    gen = DatasetGenerator(cfg, tables, landscape)
    legacy = build_legacy(cfg, seeds, landscape, gen)
    assert len(legacy) == 100
    core = [p for p in legacy if p["core"]]
    assert len(core) == 70
    assert all(abs(p["hidden"]["focal_t"]) > 1.96 for p in legacy)
    false_share = sum(1 for p in legacy if p["hypotheses"][0]["verdict"] != "not_supported" and
                      (p["hidden"]["theta"] == 0.0 or (p["hidden"]["theta"] > 0) != (p["hypotheses"][0]["estimate"] > 0))) / len(legacy)
    assert 0.0 <= false_share <= 1.0   # an output of the selection rule, not a planted number
    uncovered = 48 - len({p["question_id"] for p in core})
    assert uncovered >= 0


def test_annual_step_turnover(cfg, tables):
    lcfg = copy.deepcopy(cfg)
    lcfg["horizon_quarters"] = 40
    seeds = C.seed_layers(lcfg, 1)
    pop = build_population(lcfg, seeds, tables.theory_shares())
    for s in pop:
        s["age_years"] = 0
    n0 = len(pop)
    events = []
    for q in range(4, 41, 4):
        res = annual_step(lcfg, seeds, pop, q, {s["id"]: {"publications": 0} for s in pop}, tables.theory_shares(), "paper")
        events += res["events"]
    kinds = {e[1] for e in events}
    assert "tenure_exit" in kinds and any(k.startswith("entered") for k in kinds)
    assert sum(1 for s in pop if s.get("active", True)) >= n0   # constant population by replacement
