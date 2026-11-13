"""2-D surveillance radar: measures range and bearing, misses targets with
probability 1-Pd, and reports Poisson-distributed clutter uniform over the
range-bearing cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .motion import PX, PY


def wrap(a):
    """Wrap an angle (or array of angles) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


@dataclass(frozen=True)
class Radar:
    sigma_r: float = 15.0                    # m
    sigma_theta: float = np.deg2rad(0.3)     # rad
    pd: float = 0.9
    clutter_rate: float = 10.0               # expected false alarms per scan
    r_min: float = 500.0
    r_max: float = 20_000.0

    @property
    def R(self) -> np.ndarray:
        return np.diag([self.sigma_r**2, self.sigma_theta**2])

    # --------------------------------------------------- measurement model
    @staticmethod
    def h(x: np.ndarray) -> np.ndarray:
        return np.array([np.hypot(x[PX], x[PY]), np.arctan2(x[PY], x[PX])])

    @staticmethod
    def H(x: np.ndarray) -> np.ndarray:
        px, py = x[PX], x[PY]
        r2 = max(px * px + py * py, 1e-9)
        r = np.sqrt(r2)
        J = np.zeros((2, 5))
        J[0, PX], J[0, PY] = px / r, py / r
        J[1, PX], J[1, PY] = -py / r2, px / r2
        return J

    @staticmethod
    def residual(z: np.ndarray, zhat: np.ndarray) -> np.ndarray:
        """z - zhat, with the bearing difference wrapped (so 179 deg vs -179 deg is 2 deg)."""
        d = z - zhat
        d[..., 1] = wrap(d[..., 1])
        return d

    def to_cartesian(self, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Position and its covariance from one detection (linearized conversion)."""
        r, th = z
        pos = np.array([r * np.cos(th), r * np.sin(th)])
        J = np.array([[np.cos(th), -r * np.sin(th)], [np.sin(th), r * np.cos(th)]])
        return pos, J @ self.R @ J.T

    # ---------------------------------------------------------- simulation
    def scan(self, truths: List[Tuple[int, np.ndarray]], rng: np.random.Generator):
        """Return (Z, labels): Z is (k, 2) [range, bearing]; labels are truth ids or -1 for clutter."""
        Z, labels = [], []
        for tid, x in truths:
            z = self.h(x)
            if not (self.r_min <= z[0] <= self.r_max) or rng.random() > self.pd:
                continue
            z = z + rng.normal(0, [self.sigma_r, self.sigma_theta])
            z[1] = wrap(z[1])
            Z.append(z)
            labels.append(tid)
        for _ in range(rng.poisson(self.clutter_rate)):
            Z.append(np.array([rng.uniform(self.r_min, self.r_max), rng.uniform(-np.pi, np.pi)]))
            labels.append(-1)
        if not Z:
            return np.zeros((0, 2)), np.zeros(0, dtype=int)
        order = rng.permutation(len(Z))  # the tracker must not rely on ordering
        return np.array(Z)[order], np.array(labels)[order]
