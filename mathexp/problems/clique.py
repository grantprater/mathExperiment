"""Maximum clique on bitset graphs: pure-Python branch and bound.

Tomita-style (MCQ/MCS family): vertices are renumbered by non-increasing
degree, and each node of the search greedily colours the candidate set; the
colour number of a candidate is an upper bound on the clique that can still be
built from it, which prunes the branch. Adjacency rows are Python ints used as
bitsets.
"""

from __future__ import annotations

import time


class Timeout(Exception):
    """Raised when the search runs past its deadline."""


def max_clique(adj: list[int], time_limit: float | None = None) -> tuple[int, list[int]]:
    """Return (size, vertices) of a maximum clique.

    ``adj[v]`` is a bitmask of the neighbours of ``v`` (no self-loops). Raises
    :class:`Timeout` if ``time_limit`` seconds pass first.
    """
    n = len(adj)
    if n == 0:
        return 0, []
    deadline = None if time_limit is None else time.monotonic() + time_limit

    # Renumber: highest degree gets bit 0, so the low-bit-first colouring
    # below sees the dense vertices first.
    order = sorted(range(n), key=lambda v: -adj[v].bit_count())
    pos = {v: i for i, v in enumerate(order)}
    radj = [0] * n
    for v in range(n):
        row = adj[v]
        r = 0
        while row:
            low = row & -row
            r |= 1 << pos[low.bit_length() - 1]
            row ^= low
        radj[pos[v]] = r

    best_size = 0
    best_set: list[int] = []
    nodes = 0

    def colour_sort(P: int) -> tuple[list[int], list[int]]:
        verts: list[int] = []
        cols: list[int] = []
        U = P
        k = 0
        while U:
            k += 1
            Q = U
            while Q:
                low = Q & -Q
                v = low.bit_length() - 1
                U &= ~low
                Q &= ~low
                Q &= ~radj[v]
                verts.append(v)
                cols.append(k)
        return verts, cols

    def expand(R: list[int], P: int) -> None:
        nonlocal best_size, best_set, nodes
        nodes += 1
        if deadline is not None and (nodes & 1023) == 0 and time.monotonic() > deadline:
            raise Timeout()
        verts, cols = colour_sort(P)
        for i in range(len(verts) - 1, -1, -1):
            if len(R) + cols[i] <= best_size:
                return
            v = verts[i]
            R.append(v)
            NP = P & radj[v]
            if NP:
                expand(R, NP)
            elif len(R) > best_size:
                best_size = len(R)
                best_set = list(R)
            R.pop()
            P &= ~(1 << v)

    expand([], (1 << n) - 1)
    return best_size, sorted(order[v] for v in best_set)


def naive_max_clique(adj: list[int]) -> int:
    """Exhaustive search over all vertex subsets; for tests only (n <= ~16)."""
    n = len(adj)
    best = 0
    for mask in range(1 << n):
        size = mask.bit_count()
        if size <= best:
            continue
        ok = True
        m = mask
        while m:
            low = m & -m
            v = low.bit_length() - 1
            if (mask & ~low) & ~adj[v]:
                ok = False
                break
            m ^= low
        if ok:
            best = size
    return best
