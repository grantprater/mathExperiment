import math

import pytest

from mathexp.golf import BINARY_FN, UNARY_FN, GolfTable, apply_vec, evaluate

NS = tuple(range(3, 9))


@pytest.fixture(scope="module")
def table():
    return GolfTable(NS, 6, cache_dir=None)


def seq(fn):
    return [fn(n) for n in NS]


def test_known_sizes(table):
    assert table.golf(seq(lambda n: n)) == (1, "n")
    g, w = table.golf(seq(lambda n: 2 ** (n // 2)))
    assert g == 5 and [evaluate(w, n) for n in NS] == seq(lambda n: 2 ** (n // 2))
    g, w = table.golf(seq(lambda n: 1 + n + n * (n - 1) // 2))  # binom_sum(n, 2)
    assert g == 3 and w == "binom_sum(n, 2)"


def test_unmatched_gets_cap_plus_one(table):
    assert table.golf([7, 1, 8, 2, 8, 1]) == (7, None)


def test_perturbed_last_value_jumps(table):
    regular = seq(lambda n: n)
    r = table.jump(regular)
    assert r["G_prefix"] == r["G_full"] == 1 and r["J"] == 0 and r["predicts_last"]
    bumped = regular[:-1] + [regular[-1] + 5]
    r = table.jump(bumped)
    assert r["G_prefix"] == 1 and r["witness_prefix"] == "n"
    assert r["J"] > 0 and not r["predicts_last"]
    if r["witness_full"]:
        assert [evaluate(r["witness_full"], n) for n in NS] == bumped


def test_shorter_sequence_matches_first_values(table):
    # a sequence cut off by a timeout is golfed on the first len(seq) values of ns
    r = table.jump(seq(lambda n: 2 ** n)[:4])
    assert r["G_full"] == 3 and r["G_prefix"] == 3 and r["J"] == 0


def test_jump_many_matches_single(table):
    seqs = [seq(lambda n: n * n), seq(lambda n: n)[:-1] + [0], [1, 2, 3, 4]]
    assert table.jump_many(seqs) == [table.jump(s) for s in seqs]


def test_invalid_operations_dropped():
    div, mod, pw = BINARY_FN["//"], BINARY_FN["%"], BINARY_FN["**"]
    assert apply_vec(div, (4, 5), (2, 0)) is None  # one bad component kills the vector
    assert apply_vec(mod, (4, 5), (0, 1)) is None
    assert apply_vec(pw, (2, 2), (3, -1)) is None  # negative exponent
    assert apply_vec(pw, (1, 1), (31, 2)) is None  # exponent over 30, even with base 1
    assert apply_vec(pw, (2,), (30,)) is None  # 2**30 > 10**9
    assert apply_vec(BINARY_FN["*"], (40000,), (30000,)) is None  # |value| > 10**9
    assert apply_vec(BINARY_FN["comb"], (5,), (6,)) is None
    assert apply_vec(BINARY_FN["comb"], (201,), (1,)) is None
    assert apply_vec(BINARY_FN["binom_sum"], (5,), (-1,)) is None
    assert apply_vec(UNARY_FN["isqrt"], (-1,)) is None
    assert apply_vec(UNARY_FN["ilog2"], (0,)) is None
    assert apply_vec(UNARY_FN["ilog2"], (8, 9)) == (3, 3)
    assert apply_vec(div, (-7,), (2,)) == (-4,)  # floor division


def test_table_is_consistent():
    """Every entry's witness re-evaluates to its vector with no invalid step, and no
    vector is stored twice: the enumeration really dropped invalid expressions."""
    ns = (1, 2, 3, 4)
    t = GolfTable(ns, 5, cache_dir=None)
    seen = set()
    for key in t.entries:
        w = t.expr(key)
        vals = tuple(evaluate(w, n) for n in ns)
        assert None not in vals, w
        assert vals not in seen
        seen.add(vals)
    # n - 1 is 0 at n = 1, so n // (n - 1) never appears as a witness
    assert all("// (n - 1)" not in t.expr(k) for k in t.entries)


def test_disk_cache(tmp_path):
    a = GolfTable((3, 4, 5, 6), 5, cache_dir=tmp_path)
    b = GolfTable((3, 4, 5, 6), 5, cache_dir=tmp_path)
    assert not a.loaded_from_cache and b.loaded_from_cache
    assert a.entries == b.entries and len(list(tmp_path.glob("*.pkl"))) == 1


def test_occam_one_clean_break():
    ns = tuple(range(1, 9))
    t = GolfTable(ns, 6, cache_dir=None)
    triv = [1000] * len(ns)  # each value costs log2(1001) ~ 10 bits literally
    regular = [2 * n for n in ns]
    assert t.occam(regular, triv)["S"] == 0  # n + n predicts every next value
    bumped = regular[:-1] + [99]
    o = t.occam(bumped, triv)
    assert len(o["breaks"]) == 1
    b = o["breaks"][0]
    assert b["n"] == 8 and b["G"] == 3 and b["actual"] == 99 and b["predicted"] == 16
    assert evaluate(b["witness"], 8) == 16 and b["Lambda"] > 4 * b["G"]
    assert o["S"] == pytest.approx(math.log2(1001))
    # the steps before the break compress and predict correctly
    assert all(st["compresses"] and st["predicted_ok"] for st in o["steps"][:-1])


def test_occam_guard_blocks_coincidental_fits():
    """A tiny-valued sequence: any expression fitting the prefix is no shorter than the
    values it explains (trivial = 1, so Lambda_k = k bits < 4 bits per node)."""
    ns = tuple(range(1, 9))
    t = GolfTable(ns, 6, cache_dir=None)
    seq = [0, 1, 1, 0, 1, 0, 0, 1]
    o = t.occam(seq, [1] * 8)
    assert o["S"] == 0 and not o["breaks"]
    assert not any(st["compresses"] for st in o["steps"])
    # the same fits with expensive values would compress and break
    assert t.occam([1, 1, 1, 1, 1, 7, 1, 1], [1000] * 8)["S"] > 0
