"""Optimal 2-D assignment, written from scratch.

``linear_sum_assignment`` solves  min sum C[i, col(i)]  over one-to-one
assignments of a rectangular cost matrix with the shortest augmenting path
method (Jonker-Volgenant / Crouse 2016; the same algorithm SciPy uses). It is
O(n^2 m), and the inner loop over columns is vectorized with NumPy.
Forbidden pairs are +inf.

``connected_components`` splits a gated track x detection problem into
independent clusters. With N targets spread over the sky, most clusters
are 1x1, so solving them separately turns one O(N^3) problem into many tiny ones.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class InfeasibleAssignment(ValueError):
    pass


def linear_sum_assignment(cost) -> Tuple[np.ndarray, np.ndarray]:
    C = np.asarray(cost, dtype=float)
    if C.ndim != 2:
        raise ValueError("cost must be 2-D")
    if C.size == 0:
        return np.zeros(0, int), np.zeros(0, int)
    transposed = C.shape[0] > C.shape[1]
    if transposed:
        C = C.T
    nr, nc = C.shape
    if np.isnan(C).any() or np.isneginf(C).any():
        raise ValueError("cost contains NaN or -inf")

    u = np.zeros(nr)          # row duals
    v = np.zeros(nc)          # column duals
    col4row = np.full(nr, -1)
    row4col = np.full(nc, -1)

    for cur in range(nr):
        shortest = np.full(nc, np.inf)
        path = np.full(nc, -1)
        SR = np.zeros(nr, bool)
        SC = np.zeros(nc, bool)
        min_val, i, sink = 0.0, cur, -1
        while sink < 0:
            SR[i] = True
            # Reduced cost of reaching each unvisited column through row i.
            r = min_val + C[i] - u[i] - v
            better = ~SC & (r < shortest)
            path[better] = i
            shortest[better] = r[better]
            cand = np.where(SC, np.inf, shortest)
            j = int(np.argmin(cand))
            min_val = cand[j]
            if not np.isfinite(min_val):
                raise InfeasibleAssignment("no feasible assignment")
            # Prefer an unassigned column among ties: ends the search sooner.
            ties = np.flatnonzero((cand == min_val) & (row4col == -1))
            if ties.size:
                j = int(ties[0])
            SC[j] = True
            if row4col[j] == -1:
                sink = j
            else:
                i = row4col[j]
        # Update duals so all reduced costs stay non-negative.
        u[cur] += min_val
        others = SR.copy()
        others[cur] = False
        u[others] += min_val - shortest[col4row[others]]
        v[SC] -= min_val - shortest[SC]
        # Augment along the path back to `cur`.
        j = sink
        while True:
            i = path[j]
            row4col[j] = i
            col4row[i], j = j, col4row[i]
            if i == cur:
                break

    rows = np.arange(nr)
    if transposed:
        order = np.argsort(col4row)
        return col4row[order], rows[order]
    return rows, col4row


def connected_components(gate: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Clusters of a boolean (n_tracks, n_dets) gate matrix.

    Returns a list of (track_indices, det_indices). Tracks with no gated
    detections come back as singleton clusters with an empty det array.
    """
    nt, nd = gate.shape
    parent = list(range(nt + nd))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for t, d in zip(*np.nonzero(gate)):
        ra, rb = find(int(t)), find(nt + int(d))
        if ra != rb:
            parent[ra] = rb
    groups = {}
    for t in range(nt):
        groups.setdefault(find(t), ([], []))[0].append(t)
    for d in range(nd):
        root = find(nt + d)
        if root in groups:
            groups[root][1].append(d)
    return [(np.array(t, int), np.array(d, int)) for t, d in groups.values()]
