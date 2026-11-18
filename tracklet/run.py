"""Run a scenario through the tracker, score it, and optionally render an SVG.

    python -m tracklet.run --scenario maneuvering --seed 3 --svg maneuvering.svg
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .metrics import MOTAccumulator
from .scenarios import SCENARIOS, Scenario
from .tracker import Tracker, TrackerConfig


@dataclass
class RunResult:
    summary: Dict[str, float]
    scan_ms: List[float]
    detections: List[np.ndarray] = field(default_factory=list)
    tracker: Tracker | None = None
    maneuver_sq: float = 0.0
    maneuver_n: int = 0
    straight_sq: float = 0.0
    straight_n: int = 0

    @property
    def maneuver_rmse(self) -> float:
        return float(np.sqrt(self.maneuver_sq / max(1, self.maneuver_n)))

    @property
    def straight_rmse(self) -> float:
        return float(np.sqrt(self.straight_sq / max(1, self.straight_n)))


def run(scenario: Scenario, cfg: TrackerConfig, seed: int = 0, keep: bool = False) -> RunResult:
    rng = np.random.default_rng(seed)
    tracker = Tracker(scenario.radar, cfg)
    acc = MOTAccumulator()
    scan_ms, dets = [], []
    man_sq, man_n, str_sq, str_n = 0.0, 0, 0.0, 0
    for k in range(scenario.n_scans):
        Z, _ = scenario.radar.scan(scenario.truth[k], rng)
        t0 = time.perf_counter()
        confirmed = tracker.step(Z)
        scan_ms.append((time.perf_counter() - t0) * 1e3)
        truths = scenario.truth_positions(k)
        tracks = {t.id: t.position for t in confirmed}
        acc.update(truths, tracks)
        # Split position error by whether the target is turning right now.
        for g, t in acc.last_match.items():
            if g in truths and t in tracks:
                d = float(np.linalg.norm(truths[g] - tracks[t]))
                if d >= acc.threshold:
                    continue
                if scenario.maneuvering[g][k]:
                    man_sq += d * d
                    man_n += 1
                else:
                    str_sq += d * d
                    str_n += 1
        if keep:
            dets.append(Z)
    s = acc.summary()
    s["mean_scan_ms"] = float(np.mean(scan_ms))
    s["p99_scan_ms"] = float(np.percentile(scan_ms, 99))
    r = RunResult(s, scan_ms, dets, tracker if keep else None, man_sq, man_n, str_sq, str_n)
    s["maneuver_RMSE_m"] = r.maneuver_rmse
    s["straight_RMSE_m"] = r.straight_rmse
    return r


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Run one tracking scenario")
    ap.add_argument("--scenario", choices=sorted(SCENARIOS), default="maneuvering")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--motion", choices=["imm", "cv"], default="imm")
    ap.add_argument("--association", choices=["gnn", "greedy"], default="gnn")
    ap.add_argument("--svg", help="write a plot of truth, detections and tracks")
    args = ap.parse_args(argv)

    sc = SCENARIOS[args.scenario]()
    res = run(sc, TrackerConfig(motion=args.motion, association=args.association), args.seed, keep=bool(args.svg))
    print(f"scenario={sc.name} seed={args.seed} motion={args.motion} association={args.association}")
    for k, v in res.summary.items():
        print(f"  {k:>16}: {v:.3f}" if isinstance(v, float) else f"  {k:>16}: {v}")
    if args.svg:
        from .svg import render

        with open(args.svg, "w") as f:
            f.write(render(sc, res, title=f"{sc.name} · {args.motion.upper()} · {args.association.upper()}"))
        print(f"wrote {args.svg}")


if __name__ == "__main__":
    main()
