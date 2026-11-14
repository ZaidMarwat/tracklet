"""Interacting Multiple Model (IMM) estimator with EKF mode-matched filters.

A single-model IMM *is* an EKF, so the tracker uses this one class for
both the CV-only baseline and the CV+CT IMM.

One IMM cycle (Bar-Shalom, Li & Kirubarajan, ch. 11):
  1. Mixing: each model j starts from a probability-weighted blend of every
     model's estimate, weighted by the chance the target just switched i -> j.
  2. Mode-matched prediction and update (EKF per model).
  3. Mode probability update from each model's measurement likelihood.
  4. Output: moment-matched combination of the model estimates.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .sensor import Radar


class IMM:
    def __init__(self, models: Sequence, transition: np.ndarray, x0: np.ndarray, P0: np.ndarray,
                 mu0: Sequence[float] | None = None):
        self.models = list(models)
        M = len(self.models)
        self.Pi = np.asarray(transition, float)
        assert self.Pi.shape == (M, M) and np.allclose(self.Pi.sum(1), 1)
        self.mu = np.full(M, 1.0 / M) if mu0 is None else np.asarray(mu0, float)
        self.xs: List[np.ndarray] = [x0.copy() for _ in range(M)]
        self.Ps: List[np.ndarray] = [P0.copy() for _ in range(M)]

    # ------------------------------------------------------------ outputs
    @property
    def x(self) -> np.ndarray:
        return sum(m * x for m, x in zip(self.mu, self.xs))

    @property
    def P(self) -> np.ndarray:
        x = self.x
        return sum(m * (P + np.outer(xj - x, xj - x)) for m, xj, P in zip(self.mu, self.xs, self.Ps))

    # --------------------------------------------------------- prediction
    def predict(self, T: float) -> None:
        M = len(self.models)
        c = self.Pi.T @ self.mu                       # predicted mode probabilities
        c = np.maximum(c, 1e-300)
        W = self.Pi * self.mu[:, None] / c[None, :]   # W[i, j] = P(was i | now j)
        mixed_x, mixed_P = [], []
        for j in range(M):
            x0 = sum(W[i, j] * self.xs[i] for i in range(M))
            P0 = sum(W[i, j] * (self.Ps[i] + np.outer(self.xs[i] - x0, self.xs[i] - x0)) for i in range(M))
            mixed_x.append(x0)
            mixed_P.append(P0)
        for j, model in enumerate(self.models):
            F = model.F(mixed_x[j], T)
            self.xs[j] = model.f(mixed_x[j], T)
            P = F @ mixed_P[j] @ F.T + model.Q(T)
            self.Ps[j] = 0.5 * (P + P.T)
        self.mu = c / c.sum()

    # ------------------------------------------------------------- gating
    def innovation(self, radar: Radar):
        """Predicted measurement and innovation covariance of the combined estimate."""
        x, P = self.x, self.P
        H = radar.H(x)
        return radar.h(x), H @ P @ H.T + radar.R

    # ------------------------------------------------------------- update
    def update(self, z: np.ndarray, radar: Radar) -> None:
        M = len(self.models)
        lik = np.zeros(M)
        for j in range(M):
            x, P = self.xs[j], self.Ps[j]
            H = radar.H(x)
            S = H @ P @ H.T + radar.R
            y = radar.residual(z, radar.h(x))
            Sinv = np.linalg.inv(S)
            K = P @ H.T @ Sinv
            I_KH = np.eye(5) - K @ H
            self.xs[j] = x + K @ y
            # Joseph form: stays symmetric positive-definite under rounding.
            P = I_KH @ P @ I_KH.T + K @ radar.R @ K.T
            self.Ps[j] = 0.5 * (P + P.T)
            d2 = float(y @ Sinv @ y)
            lik[j] = np.exp(-0.5 * d2) / np.sqrt(np.linalg.det(2 * np.pi * S))
        mu = self.mu * lik
        s = mu.sum()
        # If every model finds z wildly unlikely, keep the prior rather than divide by ~0.
        self.mu = mu / s if s > 1e-300 else self.mu
