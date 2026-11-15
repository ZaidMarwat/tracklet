"""Tracking performance metrics.

* OSPA (Schuhmacher et al. 2008): a true metric between finite sets that
  folds localization error and cardinality error (missed or false targets)
  into one number, with cutoff c.
* CLEAR MOT (Bernardin & Stiefelhagen 2008): MOTA = 1 - (misses + false
  positives + ID switches) / ground-truth objects; MOTP = mean distance of
  matched pairs. Correspondences from the previous scan are kept while they
  stay within threshold, so an ID switch is counted only when a target is
  genuinely re-assigned to a different track.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .assignment import linear_sum_assignment


def ospa(X: np.ndarray, Y: np.ndarray, c: float = 500.0, p: float = 2.0) -> float:
    X, Y = np.atleast_2d(X).reshape(-1, 2), np.atleast_2d(Y).reshape(-1, 2)
    m, n = len(X), len(Y)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return c
    if m > n:
        X, Y, m, n = Y, X, n, m
    D = np.minimum(np.linalg.norm(X[:, None] - Y[None], axis=-1), c) ** p
    r, col = linear_sum_assignment(D)
    return float(((D[r, col].sum() + c**p * (n - m)) / n) ** (1 / p))


@dataclass
class MOTAccumulator:
    threshold: float = 500.0
    gt: int = 0
    misses: int = 0
    false_positives: int = 0
    id_switches: int = 0
    matches: int = 0
    dist_sum: float = 0.0
    sq_sum: float = 0.0
    last_match: Dict[int, int] = field(default_factory=dict)   # truth -> track (ever)
    ospa_series: List[float] = field(default_factory=list)

    def update(self, truths: Dict[int, np.ndarray], tracks: Dict[int, np.ndarray]) -> None:
        self.gt += len(truths)
        self.ospa_series.append(ospa(np.array(list(truths.values())), np.array(list(tracks.values()))))
        pairs: Dict[int, int] = {}
        # 1. keep previous correspondences that are still valid
        for g, t in self.last_match.items():
            if g in truths and t in tracks and np.linalg.norm(truths[g] - tracks[t]) < self.threshold:
                if t not in pairs.values():
                    pairs[g] = t
        # 2. optimally match the rest
        g_rest = [g for g in truths if g not in pairs]
        t_rest = [t for t in tracks if t not in pairs.values()]
        if g_rest and t_rest:
            D = np.array([[np.linalg.norm(truths[g] - tracks[t]) for t in t_rest] for g in g_rest])
            C = np.where(D < self.threshold, D, 1e9)
            for r, c in zip(*linear_sum_assignment(C)):
                if C[r, c] < 1e9:
                    pairs[g_rest[r]] = t_rest[c]
        for g, t in pairs.items():
            if g in self.last_match and self.last_match[g] != t:
                self.id_switches += 1
            self.last_match[g] = t
            d = float(np.linalg.norm(truths[g] - tracks[t]))
            self.dist_sum += d
            self.sq_sum += d * d
        self.matches += len(pairs)
        self.misses += len(truths) - len(pairs)
        self.false_positives += len(tracks) - len(pairs)

    def summary(self) -> Dict[str, float]:
        return {
            "MOTA": 1 - (self.misses + self.false_positives + self.id_switches) / max(1, self.gt),
            "MOTP_m": self.dist_sum / max(1, self.matches),
            "RMSE_m": float(np.sqrt(self.sq_sum / max(1, self.matches))),
            "misses": self.misses,
            "false_pos": self.false_positives,
            "id_switches": self.id_switches,
            "mean_OSPA_m": float(np.mean(self.ospa_series)) if self.ospa_series else 0.0,
        }
