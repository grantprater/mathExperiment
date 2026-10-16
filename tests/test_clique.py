import random

from mathexp.problems.clique import max_clique, naive_max_clique


def _random_graph(rng, n, p):
    adj = [0] * n
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                adj[i] |= 1 << j
                adj[j] |= 1 << i
    return adj


def test_against_naive_search():
    rng = random.Random(7)
    for trial in range(60):
        n = rng.randint(0, 14)
        adj = _random_graph(rng, n, rng.choice([0.2, 0.5, 0.8]))
        size, verts = max_clique(adj)
        assert size == naive_max_clique(adj), (trial, n)
        assert len(verts) == size
        for a in verts:
            for b in verts:
                assert a == b or adj[a] >> b & 1


def test_complete_and_empty():
    n = 9
    full = [((1 << n) - 1) & ~(1 << v) for v in range(n)]
    assert max_clique(full)[0] == n
    assert max_clique([0] * 5)[0] == 1
    assert max_clique([])[0] == 0
