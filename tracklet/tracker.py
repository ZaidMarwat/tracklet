"""Multi-target tracker: predict -> gate -> associate -> update -> manage.

Data association is Global Nearest Neighbour (GNN): per scan, find the
one-to-one track/detection assignment that minimizes total Mahalanobis
distance, with a "missed detection" option for every track, solved exactly
with the Hungarian/JV solver in `assignment.py`. For comparison, the greedy
mode takes the closest gated pair first, then the next closest, and so on.

Track lifecycle:
  TENTATIVE --(M hits in the first N scans)--> CONFIRMED --(K consecutive misses)--> DELETED
  TENTATIVE --(can no longer reach M of N)--> DELETED

Association is a two-stage cascade. Confirmed tracks pick detections first,
then tentative tracks take what's left. Without this, a fresh tentative
track spawned on clutter can steal a detection from an established track.
"""

from __future__ import annotations

import enum
import itertools
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from .assignment import connected_components, linear_sum_assignment
from .filters import IMM
from .motion import OM, PX, PY, VX, VY, ConstantVelocity, CoordinatedTurn
from .sensor import Radar


class TrackStatus(enum.Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


@dataclass
class TrackerConfig:
    dt: float = 1.0
    motion: str = "imm"              # "imm" (CV + CT) or "cv"
    association: str = "gnn"         # "gnn" (optimal) or "greedy"
    clustering: bool = True          # solve independent gate clusters separately
    gate: float = 13.82              # chi-square, 2 dof, P = 0.999
    confirm_m: int = 3
    confirm_n: int = 4
    delete_misses: int = 4
    init_speed_std: float = 120.0    # m/s, unknown velocity at birth
    init_omega_std: float = 0.03     # rad/s
    q_cv: float = 1.0
    q_ct: float = 4.0
    q_omega: float = 4e-3
    p_switch: float = 0.1            # IMM per-scan mode switch probability
    max_pos_std: float = 2_000.0     # delete a track whose position is this uncertain


@dataclass
class Track:
    id: int
    filter: IMM
    born: int
    status: TrackStatus = TrackStatus.TENTATIVE
    hits: Deque[bool] = field(default_factory=lambda: deque(maxlen=64))
    consecutive_misses: int = 0
    history: List[Tuple[int, float, float]] = field(default_factory=list)

    @property
    def position(self) -> np.ndarray:
        x = self.filter.x
        return np.array([x[PX], x[PY]])

    @property
    def mode_probabilities(self) -> np.ndarray:
        return self.filter.mu


class Tracker:
    def __init__(self, radar: Radar, cfg: Optional[TrackerConfig] = None):
        self.radar = radar
        self.cfg = cfg or TrackerConfig()
        self.tracks: List[Track] = []
        self.dead: List[Track] = []
        self._ids = itertools.count(1)
        self.scan_index = 0
        if self.cfg.motion == "imm":
            self.models = [ConstantVelocity(self.cfg.q_cv), CoordinatedTurn(self.cfg.q_ct, self.cfg.q_omega)]
            p = self.cfg.p_switch
            self.transition = np.array([[1 - p, p], [p, 1 - p]])
        elif self.cfg.motion == "cv":
            self.models = [ConstantVelocity(self.cfg.q_cv)]
            self.transition = np.eye(1)
        else:
            raise ValueError(self.cfg.motion)

    # ------------------------------------------------------------------ API
    def step(self, Z: np.ndarray) -> List[Track]:
        """Process one scan of detections Z (k x [range, bearing]); return confirmed tracks."""
        k = self.scan_index
        for t in self.tracks:
            t.filter.predict(self.cfg.dt)

        unused = np.ones(len(Z), bool)
        assigned: Dict[int, int] = {}
        for status in (TrackStatus.CONFIRMED, TrackStatus.TENTATIVE):
            group = [t for t in self.tracks if t.status is status]
            det_idx = np.flatnonzero(unused)
            for ti, di in self._associate(group, Z[det_idx]).items():
                assigned[group[ti].id] = det_idx[di]
                unused[det_idx[di]] = False

        for t in self.tracks:
            if t.id in assigned:
                t.filter.update(Z[assigned[t.id]], self.radar)
                t.hits.append(True)
                t.consecutive_misses = 0
            else:
                t.hits.append(False)
                t.consecutive_misses += 1
            self._manage(t)
            if t.status is not TrackStatus.DELETED:
                p = t.position
                t.history.append((k, float(p[0]), float(p[1])))

        self.dead += [t for t in self.tracks if t.status is TrackStatus.DELETED]
        self.tracks = [t for t in self.tracks if t.status is not TrackStatus.DELETED]
        for d in np.flatnonzero(unused):
            self.tracks.append(self._birth(Z[d], k))
        self.scan_index += 1
        return self.confirmed()

    def confirmed(self) -> List[Track]:
        return [t for t in self.tracks if t.status is TrackStatus.CONFIRMED]

    # ---------------------------------------------------------- internals
    def _associate(self, tracks: List[Track], Z: np.ndarray) -> Dict[int, int]:
        """Return {track index -> detection index} for one group of tracks."""
        if not tracks or len(Z) == 0:
            return {}
        d2 = np.full((len(tracks), len(Z)), np.inf)
        for i, t in enumerate(tracks):
            zhat, S = t.filter.innovation(self.radar)
            Y = self.radar.residual(Z.copy(), zhat)
            d2[i] = np.einsum("ij,jk,ik->i", Y, np.linalg.inv(S), Y)
        gate = d2 < self.cfg.gate
        result: Dict[int, int] = {}
        if self.cfg.association == "greedy":
            pairs = sorted(zip(d2[gate], *np.nonzero(gate)))
            used_t, used_d = set(), set()
            for _, ti, di in pairs:
                if ti not in used_t and di not in used_d:
                    result[int(ti)] = int(di)
                    used_t.add(ti)
                    used_d.add(di)
            return result
        clusters = (connected_components(gate) if self.cfg.clustering
                    else [(np.arange(len(tracks)), np.arange(len(Z)))])
        for t_idx, d_idx in clusters:
            if d_idx.size == 0:
                continue
            nt, nd = len(t_idx), len(d_idx)
            # [ distances | diagonal of "missed" costs ]: every track may stay unassigned.
            C = np.full((nt, nd + nt), np.inf)
            sub = d2[np.ix_(t_idx, d_idx)]
            C[:, :nd] = np.where(sub < self.cfg.gate, sub, np.inf)
            C[np.arange(nt), nd + np.arange(nt)] = self.cfg.gate
            rows, cols = linear_sum_assignment(C)
            for r, c in zip(rows, cols):
                if c < nd:
                    result[int(t_idx[r])] = int(d_idx[c])
        return result

    def _birth(self, z: np.ndarray, k: int) -> Track:
        pos, Ppos = self.radar.to_cartesian(z)
        c = self.cfg
        x0 = np.array([pos[0], 0.0, pos[1], 0.0, 0.0])
        P0 = np.diag([0, c.init_speed_std**2, 0, c.init_speed_std**2, c.init_omega_std**2]).astype(float)
        P0[np.ix_([PX, PY], [PX, PY])] = Ppos
        f = IMM(self.models, self.transition, x0, P0)
        t = Track(next(self._ids), f, born=k)
        t.hits.append(True)
        t.history.append((k, float(pos[0]), float(pos[1])))
        return t

    def _manage(self, t: Track) -> None:
        c = self.cfg
        P = t.filter.P
        if np.sqrt(max(P[PX, PX], P[PY, PY])) > c.max_pos_std:
            t.status = TrackStatus.DELETED
            return
        if t.status is TrackStatus.TENTATIVE:
            window = list(t.hits)[: c.confirm_n]
            n_hits = sum(window)
            if n_hits >= c.confirm_m:
                t.status = TrackStatus.CONFIRMED
            elif n_hits + (c.confirm_n - len(window)) < c.confirm_m:
                t.status = TrackStatus.DELETED  # can no longer reach M of N
        elif t.status is TrackStatus.CONFIRMED and t.consecutive_misses >= c.delete_misses:
            t.status = TrackStatus.DELETED


__all__ = ["Tracker", "TrackerConfig", "Track", "TrackStatus", "VX", "VY", "OM"]
