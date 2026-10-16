"""Brute force against known values. If one of these fails, the solver is wrong:
do not edit the expected value."""

import pytest

from mathexp.problems import get_family

A = get_family("setfam")
B = get_family("addfree")
ODDTOWN = {"p": 2, "size_mod": [1], "int_mod": [0]}
EVENTOWN = {"p": 2, "size_mod": [0], "int_mod": [0]}


@pytest.mark.parametrize("n", range(3, 8))
def test_oddtown(n):
    assert A.truth(ODDTOWN, n, 60) == n


@pytest.mark.parametrize("n", range(3, 8))
def test_eventown(n):
    assert A.truth(EVENTOWN, n, 60) == 2 ** (n // 2)


@pytest.mark.parametrize("k,expected", [(1, 2), (2, 4), (3, 9)])
def test_cap_set(k, expected):
    assert B.truth({"group": "fpk", "p": 3, "equation": "3ap"}, k, 60) == expected


@pytest.mark.parametrize("n", range(4, 13, 2))
def test_sumfree_cyclic_even(n):
    assert B.truth({"group": "cyclic", "p": 3, "equation": "sumfree"}, n, 60) == n // 2


def test_three_ap_free_integers():
    expected = [1, 2, 2, 3, 4, 4, 4, 4, 5, 5, 6, 6, 7, 8]
    got = [B.truth({"group": "int", "p": 3, "equation": "3ap"}, n, 60) for n in range(1, 15)]
    assert got == expected


@pytest.mark.parametrize("n,expected", [(4, 3), (7, 4), (12, 5)])
def test_sidon_integers(n, expected):
    assert B.truth({"group": "int", "p": 3, "equation": "sidon"}, n, 60) == expected


def test_trivial_bounds():
    assert A.trivial_bound(ODDTOWN, 5) == 16
    assert B.trivial_bound({"group": "fpk", "p": 3, "equation": "3ap"}, 3) == 27
    assert B.trivial_bound({"group": "int", "p": 3, "equation": "3ap"}, 9) == 9


def test_render_is_neutral():
    text = A.render(ODDTOWN)
    assert text == ("Let [n] = {1,…,n}. Consider families F of distinct subsets of [n] such that every "
                    "member A has |A| ≡ 1 (mod 2), and any two distinct members A ≠ B have "
                    "|A ∩ B| ≡ 0 (mod 2). Let f(n) be the largest possible |F|.")
    assert "town" not in text.lower()


def test_problem_id_ignores_ns_and_order():
    a = dict(ODDTOWN, ns=[3, 4])
    b = {"int_mod": [0], "size_mod": [1], "p": 2, "ns": [5]}
    assert A.problem_id(a) == A.problem_id(b)
    assert A.problem_id(ODDTOWN) != A.problem_id(EVENTOWN)
    # p is irrelevant for int/cyclic groups
    assert B.problem_id({"group": "int", "p": 3, "equation": "3ap"}) == \
        B.problem_id({"group": "int", "p": 5, "equation": "3ap"})


def test_timeout_returns_none():
    # Sidon in [1..36] needs far more than 1024 search nodes
    assert B.truth({"group": "int", "p": 3, "equation": "sidon"}, 36, 0.001) is None


# ids recorded by run 1 (store/a4-*/problems.jsonl), before equation2 existed
RUN1_IDS = [({"group": "fpk", "p": 3, "equation": "3ap"}, "5844ffafe50d"),
            ({"group": "cyclic", "p": 3, "equation": "sumfree"}, "93f362bec5e1"),
            ({"group": "int", "p": 3, "equation": "sidon"}, "f7cb64d70afa"),
            ({"group": "fpk", "p": 5, "equation": {"a": 1, "b": 1, "c": 2}}, "f0826f82acbc")]


@pytest.mark.parametrize("spec,pid", RUN1_IDS)
def test_family_b_ids_stable_without_equation2(spec, pid):
    assert B.problem_id(spec) == pid
    assert B.problem_id({**spec, "equation2": None}) == pid


def test_equation2():
    two = {"group": "int", "equation": "sumfree", "equation2": "3ap"}
    assert B.validate(two) == []
    assert B.problem_id(two) == B.problem_id({"group": "int", "equation": "3ap", "equation2": "sumfree"})
    assert B.problem_id(two) != B.problem_id({"group": "int", "equation": "3ap"})
    assert B.validate({**two, "equation2": "sumfree"})  # must differ from equation
    assert B.validate({**two, "equation2": {"a": 1, "b": 0, "c": 1}})
    text = B.render(two)
    assert "x + y = z" in text and "x + y = 2z" in text
    # avoiding both equations: brute force against an explicit check for small n
    from itertools import combinations

    def ok(A):
        S = set(A)
        sumfree = not any(x + y in S for x in A for y in A)
        ap = not any(x != y and (x + y) % 2 == 0 and (x + y) // 2 in S for x in A for y in A)
        return sumfree and ap
    for n in range(4, 11):
        best = max(k for k in range(n + 1) if any(ok(A) for A in combinations(range(1, n + 1), k)))
        assert B.truth(two, n, 60) == best


def test_statement_length():
    assert A.statement_length(ODDTOWN) == 6  # the PROTOCOL example
    assert A.statement_length({"p": 3, "int_abs": [1]}) == 2  # p dropped without a mod condition
    assert A.statement_length({"p": 2, "int_abs": [0, 2], "int_min": 1}) == 5
    assert B.statement_length({"group": "int", "p": 3, "equation": "3ap"}) == 4
    assert B.statement_length({"group": "fpk", "p": 3, "equation": {"a": 1, "b": 2, "c": 3}}) == 8
    assert B.statement_length({"group": "cyclic", "equation": "sidon", "equation2": {"a": 1, "b": 1, "c": 3}}) == 8
