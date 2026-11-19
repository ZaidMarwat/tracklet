"""Run: python -m unittest discover -s tests -v"""

import itertools
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tracklet.assignment import InfeasibleAssignment, connected_components, linear_sum_assignment  # noqa: E402
from tracklet.filters import IMM  # noqa: E402
from tracklet.metrics import MOTAccumulator, ospa  # noqa: E402
from tracklet.motion import ConstantVelocity, CoordinatedTurn  # noqa: E402
from tracklet.run import run  # noqa: E402
from tracklet.scenarios import _build, _state, crossing, maneuvering  # noqa: E402
from tracklet.sensor import Radar, wrap  # noqa: E402
from tracklet.tracker import Tracker, TrackerConfig  # noqa: E402


def brute_force(C):
    C = np.asarray(C, float)
    transposed = C.shape[0] > C.shape[1]
    if transposed:
        C = C.T
    best = np.inf
    for cols in itertools.permutations(range(C.shape[1]), C.shape[0]):
        best = min(best, sum(C[i, j] for i, j in enumerate(cols)))
    return best


class TestAssignment(unittest.TestCase):
    def test_matches_brute_force_on_random_rectangular_matrices(self):
        rng = np.random.default_rng(0)
        for trial in range(300):
            n, m = rng.integers(1, 6), rng.integers(1, 6)
            C = rng.uniform(0, 10, (n, m)).round(rng.integers(0, 3))  # rounding => many ties
            if trial % 3 == 0:
                C[rng.random((n, m)) < 0.3] = np.inf
            expected = brute_force(C)
            if not np.isfinite(expected):
                with self.assertRaises(InfeasibleAssignment):
                    linear_sum_assignment(C)
                continue
            r, c = linear_sum_assignment(C)
            self.assertEqual(len(r), min(n, m))
            self.assertEqual(len(set(r)), len(r))
            self.assertEqual(len(set(c)), len(c))
            self.assertAlmostEqual(C[r, c].sum(), expected, places=9)

    def test_negative_costs_and_empty(self):
        r, c = linear_sum_assignment(np.array([[-5.0, 1.0], [2.0, -3.0]]))
        self.assertEqual(list(c), [0, 1])
        r, c = linear_sum_assignment(np.zeros((0, 3)))
        self.assertEqual(len(r), 0)

    def test_clustering_is_exact(self):
        rng = np.random.default_rng(1)
        n = 60
        T = rng.uniform(0, 3000, (n, 2))
        D = T + rng.normal(0, 50, (n, 2))
        d2 = ((T[:, None] - D[None]) ** 2).sum(-1) / 80.0**2
        gate = d2 < 9
        C = np.full((n, 2 * n), np.inf)
        C[:, :n] = np.where(gate, d2, np.inf)
        C[np.arange(n), n + np.arange(n)] = 9.0
        r, c = linear_sum_assignment(C)
        total = 0.0
        clusters = connected_components(gate)
        self.assertGreater(len(clusters), 1)
        for ti, di in clusters:
            sub = np.full((len(ti), len(di) + len(ti)), np.inf)
            sub[:, : len(di)] = np.where(gate[np.ix_(ti, di)], d2[np.ix_(ti, di)], np.inf)
            sub[np.arange(len(ti)), len(di) + np.arange(len(ti))] = 9.0
            rr, cc = linear_sum_assignment(sub)
            total += sub[rr, cc].sum()
        self.assertAlmostEqual(total, C[r, c].sum(), places=6)


class TestMotion(unittest.TestCase):
    def test_ct_reduces_to_cv_as_omega_goes_to_zero(self):
        x = np.array([100.0, 30.0, -50.0, 20.0, 1e-7])
        cv = ConstantVelocity().f(x, 2.0)
        ct = CoordinatedTurn().f(x, 2.0)
        np.testing.assert_allclose(ct[:4], cv[:4], atol=1e-4)  # true rotation is w*T = 2e-7 rad

    def test_ct_full_circle_returns_home(self):
        w = 0.1
        x = np.array([0.0, 100.0, 0.0, 0.0, w])
        ct = CoordinatedTurn()
        steps = 100
        T = 2 * np.pi / w / steps
        for _ in range(steps):
            x = ct.f(x, T)
        np.testing.assert_allclose(x[[0, 2]], [0, 0], atol=1e-6)
        self.assertAlmostEqual(np.hypot(x[1], x[3]), 100.0, places=6)  # speed preserved

    def test_ct_jacobian_matches_finite_differences(self):
        ct = CoordinatedTurn()
        for w in (0.15, -0.07, 3e-5, 0.0):
            x = np.array([1000.0, 120.0, -400.0, -60.0, w])
            J = ct.F(x, 1.5)
            num = np.zeros((5, 5))
            for i in range(5):
                e = np.zeros(5)
                e[i] = 1e-6 if i == 4 else 1e-3
                num[:, i] = (ct.f(x + e, 1.5) - ct.f(x - e, 1.5)) / (2 * e[i])
            np.testing.assert_allclose(J, num, rtol=1e-4, atol=1e-4)

    def test_radar_jacobian_and_angle_wrap(self):
        radar = Radar()
        x = np.array([-5000.0, 0, 10.0, 0, 0])  # bearing ~ +180 deg
        J = radar.H(x)
        num = np.zeros((2, 5))
        for i in (0, 2):
            e = np.zeros(5)
            e[i] = 1e-3
            num[:, i] = (radar.h(x + e) - radar.h(x - e)) / 2e-3
        np.testing.assert_allclose(J, num, atol=1e-8)
        d = radar.residual(np.array([0.0, np.pi - 0.01]), np.array([0.0, -np.pi + 0.01]))
        self.assertAlmostEqual(d[1], -0.02, places=9)
        self.assertAlmostEqual(float(wrap(3 * np.pi)), -np.pi, places=9)


class TestFilterConsistency(unittest.TestCase):
    def test_ekf_is_statistically_consistent(self):
        """Normalized estimation error squared (NEES) of a CV-EKF on a CV target
        should average to the state dimension (4) when the filter's noise
        model is right; a mis-tuned or buggy filter lands far outside."""
        radar = Radar(sigma_r=20.0, sigma_theta=np.deg2rad(0.2))
        q = 2.0
        model = ConstantVelocity(q)
        rng = np.random.default_rng(7)
        nees = []
        for run_i in range(60):
            x = np.array([8000.0, -120.0, 3000.0, 80.0, 0.0])
            P0 = np.diag([100.0**2, 50.0**2, 100.0**2, 50.0**2, 1e-6])
            est = IMM([model], np.eye(1), x + rng.multivariate_normal(np.zeros(5), P0), P0)
            Qc = model.Q(1.0)[:4, :4]
            for k in range(40):
                x = model.f(x, 1.0)
                x[:4] += rng.multivariate_normal(np.zeros(4), Qc)
                est.predict(1.0)
                z = radar.h(x) + rng.normal(0, [radar.sigma_r, radar.sigma_theta])
                est.update(z, radar)
                if k >= 10:
                    e = (est.x - x)[:4]
                    nees.append(e @ np.linalg.inv(est.P[:4, :4]) @ e)
        mean = float(np.mean(nees))
        # 95% two-sided chi-square band for the mean of 1800 samples of chi2(4) ~ 4 +/- 0.13.
        self.assertGreater(mean, 3.6, mean)
        self.assertLess(mean, 4.5, mean)

    def test_imm_mixing_preserves_combined_estimate(self):
        """Mixing only re-distributes probability mass between models, so with a
        zero-length prediction the combined mean must not move, while each
        model's own estimate is pulled toward the other's."""
        Pi = np.array([[0.9, 0.1], [0.3, 0.7]])
        P0 = np.eye(5) * 100.0
        est = IMM([ConstantVelocity(), ConstantVelocity()], Pi, np.zeros(5), P0, mu0=[0.8, 0.2])
        est.xs[0] = np.array([0.0, 10.0, 0.0, 0.0, 0.0])
        est.xs[1] = np.array([300.0, 10.0, 90.0, 5.0, 0.0])
        before, xb = est.x.copy(), est.xs[1].copy()
        est.predict(0.0)
        np.testing.assert_allclose(est.x, before, atol=1e-9)
        np.testing.assert_allclose(est.mu, Pi.T @ [0.8, 0.2])
        self.assertLess(np.linalg.norm(est.xs[1] - est.xs[0]), np.linalg.norm(xb))

    def test_imm_switches_mode_during_turn(self):
        sc = maneuvering(Radar(clutter_rate=0.0, pd=1.0))
        tr = Tracker(sc.radar, TrackerConfig())
        rng = np.random.default_rng(0)
        ct_prob_turning, ct_prob_straight = [], []
        for k in range(sc.n_scans):
            Z, _ = sc.radar.scan(sc.truth[k], rng)
            tr.step(Z)
            for t in tr.confirmed():
                tid = min(sc.truth_positions(k).items(), key=lambda kv: np.linalg.norm(kv[1] - t.position))[0]
                (ct_prob_turning if sc.maneuvering[tid][k] else ct_prob_straight).append(t.mode_probabilities[1])
        self.assertGreater(np.mean(ct_prob_turning), np.mean(ct_prob_straight) + 0.25)


class TestTracker(unittest.TestCase):
    def test_clean_crossing_is_tracked_perfectly(self):
        sc = crossing(Radar(clutter_rate=0.0, pd=1.0))
        s = run(sc, TrackerConfig(), seed=0).summary
        self.assertEqual(s["id_switches"], 0)
        self.assertEqual(s["false_pos"], 0)
        self.assertLess(s["RMSE_m"], 40)
        # Only misses allowed: the confirmation delay (3 of 4) for each of the 4 targets.
        self.assertLessEqual(s["misses"], 4 * 2)

    def test_clutter_rarely_confirms_false_tracks(self):
        sc = _build("empty", 1.0, 100, Radar(clutter_rate=30.0), [])
        tr = Tracker(sc.radar, TrackerConfig())
        rng = np.random.default_rng(3)
        confirmed_ever = set()
        for k in range(sc.n_scans):
            Z, _ = sc.radar.scan([], rng)
            confirmed_ever |= {t.id for t in tr.step(Z)}
        # 3000 clutter points; the 3-of-4 rule should confirm only a handful.
        self.assertLess(len(confirmed_ever), 15)

    def test_target_across_the_bearing_discontinuity(self):
        # Flies through bearing +/-180 deg, where naive angle subtraction breaks.
        plans = [(_state(-9000, 2000, 150, -90), [(30, 0)], 0)]
        sc = _build("wrap", 1.0, 30, Radar(clutter_rate=0.0, pd=1.0), plans)
        s = run(sc, TrackerConfig(), seed=0).summary
        self.assertEqual(s["id_switches"], 0)
        self.assertLess(s["RMSE_m"], 60)

    def test_gnn_beats_greedy_in_tight_formation(self):
        from tracklet.scenarios import formation

        sc = formation(150.0)
        g = [run(sc, TrackerConfig(association="gnn"), seed=s).summary for s in range(4)]
        n = [run(sc, TrackerConfig(association="greedy"), seed=s).summary for s in range(4)]
        self.assertLess(np.mean([r["id_switches"] for r in g]), np.mean([r["id_switches"] for r in n]))
        self.assertLess(np.mean([r["RMSE_m"] for r in g]), np.mean([r["RMSE_m"] for r in n]))


class TestMetrics(unittest.TestCase):
    def test_ospa_properties(self):
        X = np.array([[0.0, 0.0], [100.0, 0.0]])
        self.assertEqual(ospa(X, X), 0.0)
        self.assertEqual(ospa(X, np.zeros((0, 2)), c=500), 500)
        self.assertAlmostEqual(ospa(X, X[[1, 0]] + [3, 4]), 5.0)  # order-invariant
        Y = np.array([[0.0, 0.0], [100.0, 0.0], [5000.0, 0.0]])
        self.assertAlmostEqual(ospa(X, Y, c=300, p=2), np.sqrt(300**2 / 3))

    def test_mot_counts_id_switch(self):
        acc = MOTAccumulator(threshold=100)
        acc.update({1: np.array([0.0, 0.0])}, {7: np.array([5.0, 0.0])})
        acc.update({1: np.array([10.0, 0.0])}, {8: np.array([12.0, 0.0])})  # track 7 gone, 8 takes over
        s = acc.summary()
        self.assertEqual(s["id_switches"], 1)
        self.assertEqual(s["misses"], 0)
        self.assertAlmostEqual(s["MOTA"], 0.5)


if __name__ == "__main__":
    unittest.main()
