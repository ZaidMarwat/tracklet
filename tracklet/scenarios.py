"""Deterministic ground-truth scenarios.

Each target follows a piecewise flight plan of (duration_s, turn_rate) legs,
propagated exactly with the coordinated-turn equations. turn_rate 0 is a
straight leg.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .motion import CoordinatedTurn
from .sensor import Radar

Leg = Tuple[float, float]  # (duration seconds, omega rad/s)


@dataclass
class Scenario:
    name: str
    dt: float
    n_scans: int
    radar: Radar
    # truth[k] = list of (target_id, state) present at scan k
    truth: List[List[Tuple[int, np.ndarray]]] = field(default_factory=list)
    maneuvering: Dict[int, np.ndarray] = field(default_factory=dict)  # id -> bool per scan

    def truth_positions(self, k: int) -> Dict[int, np.ndarray]:
        return {tid: np.array([x[0], x[2]]) for tid, x in self.truth[k]}


def _fly(x0: np.ndarray, legs: List[Leg], dt: float, n_scans: int, start: int = 0):
    ct = CoordinatedTurn()
    states, turning = [None] * n_scans, np.zeros(n_scans, bool)
    plan = []
    for dur, w in legs:
        plan += [w] * int(round(dur / dt))
    x = x0.astype(float).copy()
    for k in range(start, n_scans):
        i = k - start
        if i >= len(plan):
            break
        x[4] = plan[i]
        states[k] = x.copy()
        turning[k] = plan[i] != 0
        x = ct.f(x, dt)
    return states, turning


def _state(px, py, speed, heading_deg):
    h = np.deg2rad(heading_deg)
    return np.array([px, speed * np.cos(h), py, speed * np.sin(h), 0.0])


def _build(name, dt, n_scans, radar, plans) -> Scenario:
    sc = Scenario(name, dt, n_scans, radar, [[] for _ in range(n_scans)])
    for tid, (x0, legs, start) in enumerate(plans, start=1):
        states, turning = _fly(x0, legs, dt, n_scans, start)
        sc.maneuvering[tid] = turning
        for k, s in enumerate(states):
            if s is not None:
                sc.truth[k].append((tid, s))
    return sc


def crossing(radar: Radar | None = None) -> Scenario:
    """Two pairs of targets whose paths cross at shallow and steep angles. Association stress."""
    radar = radar or Radar()
    plans = [
        (_state(6000, -2000, 150, 15), [(90, 0)], 0),
        (_state(6000, 2000, 150, -15), [(90, 0)], 0),
        (_state(10000, -9000, 180, 90), [(90, 0)], 0),
        (_state(4000, -3000, 180, 0), [(90, 0)], 0),
    ]
    return _build("crossing", 1.0, 90, radar, plans)


def maneuvering(radar: Radar | None = None) -> Scenario:
    """Targets alternating straight legs and 2-3 g coordinated turns. Filter stress."""
    radar = radar or Radar()
    w = 0.12  # rad/s: at 220 m/s that is 26 m/s^2 (~2.7 g)
    plans = [
        (_state(4000, -6000, 220, 60), [(20, 0), (25, w), (20, 0), (25, -w), (30, 0)], 0),
        (_state(14000, 4000, 200, 200), [(25, 0), (20, -w), (25, 0), (30, w), (20, 0)], 0),
        (_state(8000, 9000, 240, -100), [(15, 0), (26, w), (30, 0), (26, -w), (23, 0)], 0),
    ]
    return _build("maneuvering", 1.0, 120, radar, plans)


def formation(spacing: float = 150.0, radar: Radar | None = None) -> Scenario:
    """Four aircraft in tight line-abreast formation that fan out. Their gates
    overlap for most of the run, so detection-to-track assignment is genuinely
    ambiguous: this is where optimal (GNN) association beats greedy."""
    radar = radar or Radar(clutter_rate=40.0)
    plans = []
    for i, w in enumerate([0.0, 0.02, -0.02, 0.04]):
        plans.append((_state(6000, -3000 + i * spacing, 200, 30), [(30, 0), (20, w), (40, 0)], 0))
    return _build("formation", 1.0, 90, radar, plans)


def dense(n_targets: int = 40, seed: int = 0, radar: Radar | None = None, n_scans: int = 60) -> Scenario:
    """Many targets with random headings and gentle turns, heavy clutter. Scale stress."""
    radar = radar or Radar(clutter_rate=40.0)
    rng = np.random.default_rng(seed)
    plans = []
    for _ in range(n_targets):
        r = rng.uniform(4000, 14000)
        th = rng.uniform(-np.pi, np.pi)
        speed = rng.uniform(60, 250)
        x0 = _state(r * np.cos(th), r * np.sin(th), speed, rng.uniform(0, 360))
        legs = []
        while sum(d for d, _ in legs) < n_scans:
            legs.append((float(rng.integers(10, 30)), float(rng.choice([0, 0, 0.03, -0.03]))))
        plans.append((x0, legs, int(rng.integers(0, 10))))
    return _build(f"dense-{n_targets}", 1.0, n_scans, radar, plans)


SCENARIOS = {"crossing": crossing, "maneuvering": maneuvering, "formation": formation, "dense": dense}
