"""Motion models over the state  x = [px, vx, py, vy, omega]  (m, m/s, rad/s).

Both models share the 5-D state so an IMM can mix them directly:
* CV (constant velocity): straight-line motion, drives omega to 0.
* CT (coordinated turn): constant speed, constant turn rate omega. It is
  nonlinear in omega, so the EKF uses its analytic Jacobian.

Process noise is the discretized continuous white-noise acceleration model:
for each axis  Q = q * [[T^3/3, T^2/2], [T^2/2, T]].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PX, VX, PY, VY, OM = range(5)
_SMALL = 1e-4  # below this |omega| we use Taylor expansions to avoid 0/0


def _accel_block(T: float, q: float) -> np.ndarray:
    return q * np.array([[T**3 / 3, T**2 / 2], [T**2 / 2, T]])


def _place_accel(Q: np.ndarray, T: float, q: float) -> None:
    b = _accel_block(T, q)
    Q[np.ix_([PX, VX], [PX, VX])] = b
    Q[np.ix_([PY, VY], [PY, VY])] = b


@dataclass(frozen=True)
class ConstantVelocity:
    q: float = 2.0  # accel PSD, m^2/s^3
    name: str = "CV"

    def f(self, x: np.ndarray, T: float) -> np.ndarray:
        return self.F(x, T) @ x

    def F(self, x: np.ndarray, T: float) -> np.ndarray:
        F = np.eye(5)
        F[PX, VX] = T
        F[PY, VY] = T
        F[OM, OM] = 0.0  # a straight-flying target has no turn rate
        return F

    def Q(self, T: float) -> np.ndarray:
        Q = np.zeros((5, 5))
        _place_accel(Q, T, self.q)
        Q[OM, OM] = 1e-8
        return Q


@dataclass(frozen=True)
class CoordinatedTurn:
    q: float = 1.0          # residual accel PSD, m^2/s^3
    q_omega: float = 1e-4   # turn-rate random walk PSD, rad^2/s^3
    name: str = "CT"

    @staticmethod
    def _terms(w: float, T: float):
        """sin(wT)/w, (1-cos(wT))/w and their d/dw, stable near w = 0."""
        if abs(w) < _SMALL:
            s_w = T - (w**2) * T**3 / 6
            c_w = w * T**2 / 2 - (w**3) * T**4 / 24
            ds = -w * T**3 / 3
            dc = T**2 / 2 - (w**2) * T**4 / 8
        else:
            a = w * T
            S, C = np.sin(a), np.cos(a)
            s_w = S / w
            c_w = (1 - C) / w
            ds = (T * C * w - S) / w**2
            dc = (T * S * w - (1 - C)) / w**2
        return s_w, c_w, ds, dc

    def f(self, x: np.ndarray, T: float) -> np.ndarray:
        px, vx, py, vy, w = x
        s_w, c_w, _, _ = self._terms(w, T)
        S, C = np.sin(w * T), np.cos(w * T)
        return np.array([
            px + s_w * vx - c_w * vy,
            C * vx - S * vy,
            py + c_w * vx + s_w * vy,
            S * vx + C * vy,
            w,
        ])

    def F(self, x: np.ndarray, T: float) -> np.ndarray:
        _, vx, _, vy, w = x
        s_w, c_w, ds, dc = self._terms(w, T)
        S, C = np.sin(w * T), np.cos(w * T)
        F = np.eye(5)
        F[PX, VX], F[PX, VY], F[PX, OM] = s_w, -c_w, ds * vx - dc * vy
        F[VX, VX], F[VX, VY], F[VX, OM] = C, -S, -T * S * vx - T * C * vy
        F[PY, VX], F[PY, VY], F[PY, OM] = c_w, s_w, dc * vx + ds * vy
        F[VY, VX], F[VY, VY], F[VY, OM] = S, C, T * C * vx - T * S * vy
        F[PX, PX] = F[PY, PY] = F[OM, OM] = 1.0
        return F

    def Q(self, T: float) -> np.ndarray:
        Q = np.zeros((5, 5))
        _place_accel(Q, T, self.q)
        Q[OM, OM] = self.q_omega * T
        return Q
