"""Dataset generator, planted data traps, trap admissibility, and the leak scan that keeps design terms out of agent packets."""
import numpy as np
import pytest

from synthsci import config as C
from synthsci.datasets import DatasetGenerator
from synthsci.leakscan import scan_text
from synthsci.traps import REGISTRY, applicable_traps, draw_traps
from synthsci.util import seed_from


@pytest.fixture(scope="module")
def gen(cfg, tables, landscape):
    return DatasetGenerator(cfg, tables, landscape)


def test_regenerates_to_same_hash(gen, landscape, tmp_path):
    q = landscape.questions[0]
    a = gen.generate(q["id"], seed_from("t", 1), "D1", write_dir=tmp_path / "a")
    b = gen.generate(q["id"], seed_from("t", 1), "D1", write_dir=tmp_path / "b")
    assert a["meta"]["data_sha256"] == b["meta"]["data_sha256"]
    assert (tmp_path / "a" / "data.csv").read_bytes() == (tmp_path / "b" / "data.csv").read_bytes()
    c = gen.generate(q["id"], seed_from("t", 2), "D2")
    assert c["meta"]["data_sha256"] != a["meta"]["data_sha256"]


def test_admissible_and_codebooks_clean(gen, landscape, tmp_path):
    n_ok = 0
    for i, q in enumerate(landscape.questions[:10]):
        res = gen.generate(q["id"], seed_from("adm", i), f"D{i}", write_dir=tmp_path / f"d{i}")
        n_ok += res["meta"]["admissible"]
        for name in ("codebook.md", "sample_frame.md"):
            assert scan_text((tmp_path / f"d{i}" / name).read_text(encoding="utf-8"), "strict") == []
        h = res["hidden"]
        for t in h["traps"]:
            nv, c = h["oracle"]["naive"][t], h["oracle"]["correct"]
            if t == "duplicates":
                assert nv["n"] > c["n"]
            else:
                assert abs(nv["partial_r"] - c["partial_r"]) > 0 or np.sign(nv["partial_r"]) != np.sign(c["partial_r"])
    assert n_ok == 10


def test_trap_registry_applicability(gen, landscape):
    q_panel = next(q for q in landscape.questions if q["panel"])
    q_cs = next(q for q in landscape.questions if not q["panel"])
    sp, sc = gen.spec_for(q_panel), gen.spec_for(q_cs)
    assert "unit_mismatch" in applicable_traps(sp) and "unit_mismatch" not in applicable_traps(sc)
    assert "selection_on_outcome" not in applicable_traps(sp)
    assert set(REGISTRY) == {"duplicates", "scale_mixture", "reverse_coded", "selection_on_outcome", "bad_control", "miscoded_values",
                             "unit_mismatch", "omitted_standard_control"}


def test_trap_density_declared(gen, landscape):
    q = landscape.questions[0]
    spec = gen.spec_for(q)
    counts = [len(draw_traps(spec, C.rng(i, "t"), {"one": 0.6, "two": 0.1})) for i in range(600)]
    share_any = sum(1 for c in counts if c >= 1) / 600
    assert 0.6 < share_any < 0.8


def test_analytic_matches_full_path_on_target(gen, landscape):
    """Gate 5 (reduced): the analytic draw and the full generator share frame, traps and target."""
    q = landscape.questions[3]
    seed = seed_from("g5", 1)
    full = gen.generate(q["id"], seed, "Dg")
    an = gen.analytic_study(q["id"], seed, False, {}, True)
    assert an["traps"] == full["hidden"]["traps"] or set(an["traps"]) >= set(full["hidden"]["traps"])
    assert an["frame"]["n_target"] == full["meta"]["frame"]["n_target"]
    assert abs(an["hidden"]["target"] - full["hidden"]["target"]) < 1e-9


def test_leak_scanner_catches_planted_cues():
    assert scan_text("The outcome has a scale mixture across sources", "strict")
    assert scan_text("C:\\Users\\someone\\secret\\truth.json", "strict", allowed_paths=[])
    assert scan_text("Rows: 1,200. Missing values are coded as empty cells.", "strict") == []
    assert scan_text('{"code": "for t in terms:\\n    pass"}', "design") == []
