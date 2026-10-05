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
