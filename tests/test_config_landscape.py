"""Config resolution, hashing and seed layers, and the research landscape built from the seeding tables."""
from synthsci import config as C
from synthsci.landscape import build_landscape
from synthsci.leakscan import scan_text
from synthsci.util import canonical_json, sha256_text


def test_config_hash_stable_and_sensitive(cfg):
    assert C.config_hash(cfg) == C.config_hash(C.resolve("S0"))
    other = C.resolve("S0", {"population": 61})
    assert C.config_hash(other) != C.config_hash(cfg)
    assert C.run_id(cfg, 1, "mock") != C.run_id(cfg, 2, "mock")


def test_seed_layers_derive_and_pin(cfg):
    s = C.seed_layers(cfg, 7)
    assert set(s) == {"world", "population", "run", "master"}
    assert len({s["world"], s["population"], s["run"]}) == 3
    pinned = C.resolve("S0", {"seeds.world": 123})
    s2 = C.seed_layers(pinned, 7)
    assert s2["world"] == 123 and s2["population"] == s["population"]


def test_scenarios_resolve():
    for name in ("S0", "S1", "S2", "R-a", "R-b", "smoke-P", "smoke-R"):
        c = C.resolve(name)
        assert c["scenario"] == name
        assert c["regime"] in ("paper", "repository")
    assert C.resolve("S2")["population_profile"]["care"]["auto_share"] == 0.8
    assert C.resolve("R-a")["repository"]["variant"] == "a"


def test_landscape_shape(landscape, tables):
    qs = landscape.questions
    assert len(qs) == 48
    fams = {q["family"] for q in qs}
    assert len(fams) == 8
    for f in fams:
        assert sum(1 for q in qs if q["family"] == f) == 6
    assert all(q["iv"] != q["dv"] for q in qs)
    assert len(landscape.fringe) == 150
    # counts quoted in questions come from the pairs table
    for q in qs:
        if q["pair_source"].startswith("top50"):
            assert q["corpus_papers"] == tables.pair_count(q["iv"], q["dv"])


def test_landscape_truth_properties(landscape):
    th = [landscape.truth["questions"][q["id"]]["theta"] for q in landscape.questions]
    assert any(t == 0.0 for t in th) and any(t != 0.0 for t in th)
    for fam, q in landscape.truth["q_T"].items():
        assert 0.55 <= q <= 0.95
    for qid, t in landscape.truth["questions"].items():
        assert t["truth_class"] in ("null", "negligible", "positive", "negative")
        assert 0.05 <= t["confounder"]["r_ux"] <= 0.20


def test_landscape_byte_identical_and_seed_sensitive(cfg, tables):
    a = build_landscape(cfg, C.seed_layers(cfg, 3), tables)
    b = build_landscape(cfg, C.seed_layers(cfg, 3), tables)
    c = build_landscape(cfg, C.seed_layers(cfg, 4), tables)
    assert canonical_json(a.truth) == canonical_json(b.truth)
    assert canonical_json(a.questions) == canonical_json(b.questions)
    assert canonical_json(a.truth) != canonical_json(c.truth)


def test_deid_map_roundtrips(landscape):
    d = landscape.deid_map(12345)
    assert len(set(d["constructs"].values())) == len(d["constructs"])
    assert sorted(d["families"].values()) == sorted(f"Theory {chr(65 + i)}" for i in range(8))
    assert landscape.deid_map(12345) == d and landscape.deid_map(999) != d


def test_agent_visible_landscape_text_passes_scan(landscape):
    for q in landscape.questions[:10] + landscape.fringe[:5]:
        assert scan_text(q["prediction"], "strict") == []


def test_seeding_manifest_hashes(tables):
    m = tables.manifest()
    assert m["name"].startswith("smj")
    assert all(len(v["sha256"]) == 64 for v in m["tables"].values())
