"""Run 3: extremal witnesses and the witness cache.

The two lynchpins RESULTS.md quotes from run 2 are re-derived here from the brute
force alone: the Fano plane (n = 7, pairwise intersections in {1, 3}) and the Steiner
system S(3,4,8) (n = 8, pairwise intersections in {0, 2})."""

import itertools
import json
import random

from mathexp.problems import get_family
from mathexp.truth import TruthCache

A, B = get_family("setfam"), get_family("addfree")
FANO = {"p": 2, "int_abs": [1, 3]}
S348 = {"p": 2, "int_abs": [0, 2]}


def _covers_exactly_once(blocks, t, n):
    for sub in itertools.combinations(range(1, n + 1), t):
        hits = sum(1 for b in blocks if set(sub) <= set(b))
        if hits != 1:
            return False
    return True


def test_fano_plane():
    value, fam = A.truth_witness(FANO, 7, None)
    assert value == 8 == len(fam)
    assert A.check_witness(FANO, 7, fam) == []
    lines = [b for b in fam if len(b) == 3]
    rest = [b for b in fam if len(b) != 3]
    # the 7 lines of the Fano plane: every pair of points on exactly one line
    assert len(lines) == 7 and _covers_exactly_once(lines, 2, 7)
    assert all(len(set(a) & set(b)) == 1 for a, b in itertools.combinations(lines, 2))
    # plus the whole set, which meets every line in 3
    assert rest == [list(range(1, 8))]


def test_steiner_s348():
    value, fam = A.truth_witness(S348, 8, None)
    assert value == 15 == len(fam)
    assert A.check_witness(S348, 8, fam) == []
    blocks = [b for b in fam if len(b) == 4]
    assert len(blocks) == 14 and _covers_exactly_once(blocks, 3, 8)  # S(3,4,8)
    assert [b for b in fam if len(b) != 4] == [[]]  # plus the empty set


def test_check_witness_rejects():
    _, fam = A.truth_witness(FANO, 7, None)
    assert A.check_witness(FANO, 7, fam[:-1] + [[1, 2, 3]])  # [1,2,3] meets [1,2,7] in 2
    _, s = B.truth_witness({"group": "int", "equation": "3ap"}, 9, None)
    assert B.check_witness({"group": "int", "equation": "3ap"}, 9, sorted(set(s) | {5, 3, 4}))


def test_addfree_argmax_matches_value_and_is_free():
    rng = random.Random(3)
    specs = [({"group": "int", "equation": "3ap"}, [8, 12, 15]),
             ({"group": "cyclic", "equation": "sidon"}, [7, 13]),
             ({"group": "cyclic", "equation": "sumfree", "equation2": "3ap"}, [9, 11]),
             ({"group": "int", "equation": {"a": 1, "b": 2, "c": 3}}, [10]),
             ({"group": "fpk", "p": 3, "equation": "3ap"}, [2, 3]),
             ({"group": "fpk", "p": 2, "equation": "sumfree"}, [3])]
    for spec, ns in specs:
        for n in ns:
            value, s = B.truth_witness(spec, n, None)
            assert value == len(s) == B.truth(spec, n, None)
            assert B.check_witness(spec, n, s) == [], (spec, n, s)
    # setfam: random specs, witness size equals f and the witness is valid
    for _ in range(8):
        spec = A.mutate_random({"p": 2, "size_mod": [1], "int_mod": [0]}, rng)
        spec.pop("ns", None)
        value, fam = A.truth_witness(spec, 5, None)
        assert value == len(fam) and A.check_witness(spec, 5, fam) == []


def test_witness_text():
    _, fam = A.truth_witness(FANO, 7, None)
    t = A.witness_text(FANO, 7, fam)
    assert "n = 7" in t and "8 members" in t and "{1, 2, 3, 4, 5, 6, 7}" in t
    spec = {"group": "fpk", "p": 3, "equation": "3ap"}
    _, s = B.truth_witness(spec, 2, None)
    assert "(0, 0)" in B.witness_text(spec, 2, s)


def test_witness_cache_roundtrip(tmp_path):
    path = tmp_path / "truth.json"
    c = TruthCache(path)
    v, secs, cached = c.get(A, FANO, 7)
    assert v == 8 and not cached
    assert c.witness_path.exists()
    # a fresh cache reads the witness from the sidecar without recomputing
    c2 = TruthCache(path)
    calls = []
    orig = A.truth_witness
    try:
        A.truth_witness = lambda *a, **k: calls.append(a) or orig(*a, **k)
        v2, w2 = c2.get_witness(A, FANO, 7)
    finally:
        del A.truth_witness
    assert v2 == 8 and len(w2) == 8 and not calls
    # truth.json stays values-only
    data = json.loads(path.read_text(encoding="utf-8"))
    assert all("witness" not in r for r in data.values())


def test_witness_recomputed_for_old_cache_entries(tmp_path):
    path = tmp_path / "truth.json"
    key = f"{A.problem_id(S348)}:8"
    path.write_text(json.dumps({key: {"value": 15, "seconds": 0.1, "time_limit": 20.0, "family": "setfam"}}),
                    encoding="utf-8")
    c = TruthCache(path)
    v, w = c.get_witness(A, S348, 8)
    assert v == 15 and len(w) == 15
    assert key in TruthCache(path)._load_witnesses()


def test_throttled_save_flushes(tmp_path):
    path = tmp_path / "truth.json"
    c = TruthCache(path, save_every=3600)
    c.get(A, FANO, 5)
    c.get(A, FANO, 6)
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 1  # first save only
    c.flush()
    assert len(json.loads(path.read_text(encoding="utf-8"))) == 2
