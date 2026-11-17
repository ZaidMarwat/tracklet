"""Monte Carlo experiments behind the README tables and figures.

    python -m tracklet.experiments            # full (several minutes)
    python -m tracklet.experiments --quick
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import numpy as np

from .run import run
import time

from .assignment import connected_components, linear_sum_assignment
from .scenarios import crossing, dense, formation, maneuvering
from .sensor import Radar
from .svg import render
from .tracker import TrackerConfig


def mc(scenario, cfg: TrackerConfig, seeds: int) -> Dict[str, float]:
    rows = [run(scenario, cfg, seed=s).summary for s in range(seeds)]
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def table(title: str, cols: List[str], rows: Dict[str, Dict[str, float]], fmt: Dict[str, str]) -> None:
    print(f"\n### {title}\n")
    print("| config | " + " | ".join(cols) + " |")
    print("|---|" + "---|" * len(cols))
    for name, r in rows.items():
        print(f"| {name} | " + " | ".join(fmt.get(c, "{:.2f}").format(r[c]) for c in cols) + " |")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures", default="docs", help="directory for SVG figures ('' to skip)")
    args = ap.parse_args(argv)
    seeds = 3 if args.quick else 10
    f = {"MOTA": "{:.3f}", "id_switches": "{:.1f}", "false_pos": "{:.1f}", "misses": "{:.1f}"}

    # 1. Filter: smooth CV vs agile CV vs IMM on maneuvering targets.
    sc = maneuvering()
    configs = {
        "CV-EKF, low noise (q=1)": TrackerConfig(motion="cv", q_cv=1.0),
        "CV-EKF, high noise (q=1000)": TrackerConfig(motion="cv", q_cv=1000.0),
        "IMM (CV + CT)": TrackerConfig(motion="imm"),
    }
    rows = {name: mc(sc, cfg, seeds) for name, cfg in configs.items()}
    table(f"Filter comparison: 3 targets, 2.7 g turns, Pd 0.9, 10 clutter/scan ({seeds} Monte Carlo runs)",
          ["straight_RMSE_m", "maneuver_RMSE_m", "RMSE_m", "MOTA", "id_switches"], rows, f)

    # 2. Association: greedy vs optimal where gates overlap.
    for title, sc in [
        ("tight formation (150 m spacing) that fans out, 40 clutter/scan", formation(150.0)),
        ("well-separated crossing targets, 60 clutter/scan", crossing(Radar(clutter_rate=60.0, pd=0.85))),
    ]:
        rows = {
            "greedy nearest-neighbour": mc(sc, TrackerConfig(association="greedy"), seeds),
            "GNN (Hungarian, optimal)": mc(sc, TrackerConfig(association="gnn"), seeds),
        }
        table(f"Association: {title} ({seeds} runs)",
              ["MOTA", "id_switches", "false_pos", "RMSE_m"], rows, f)

    # 3. Scale: per-scan latency vs number of targets.
    rows = {}
    sizes = [25, 50, 100] if args.quick else [25, 50, 100, 200]
    for n in sizes:
        sc = dense(n_targets=n, seed=0, radar=Radar(clutter_rate=40.0), n_scans=40)
        r = run(sc, TrackerConfig(), seed=0).summary
        rows[f"{n} targets + 40 clutter"] = r
    table("Scaling: full IMM-GNN tracker, wall time per scan (single thread, 1 s scan period)",
          ["mean_scan_ms", "p99_scan_ms", "MOTA", "id_switches"], rows, f)

    # 4. Assignment solver alone: one global problem vs gate-cluster decomposition.
    print("\n### Assignment solver: global vs clustered (N tracks + N detections at constant density, 300 m gate)\n")
    print("| N | clusters | largest | global solve (ms) | clustered solve (ms) | same cost |")
    print("|---|---|---|---|---|---|")
    rng = np.random.default_rng(0)
    for n in ([200, 1000] if args.quick else [200, 1000, 4000]):
        side = 10_000 * np.sqrt(n / 200)  # airspace grows with the number of targets
        T = rng.uniform(0, side, (n, 2))
        D = T + rng.normal(0, 60, (n, 2))
        d2 = ((T[:, None] - D[None]) ** 2).sum(-1) / 100.0**2
        gate = d2 < 9.0
        C = np.full((n, 2 * n), np.inf)
        C[:, :n] = np.where(gate, d2, np.inf)
        C[np.arange(n), n + np.arange(n)] = 9.0
        t0 = time.perf_counter()
        r, c = linear_sum_assignment(C)
        t_global = (time.perf_counter() - t0) * 1e3
        cost_global = C[r, c].sum()
        t0 = time.perf_counter()
        clusters = connected_components(gate)
        cost_cl = 0.0
        for ti, di in clusters:
            sub = np.full((len(ti), len(di) + len(ti)), np.inf)
            sub[:, : len(di)] = np.where(gate[np.ix_(ti, di)], d2[np.ix_(ti, di)], np.inf)
            sub[np.arange(len(ti)), len(di) + np.arange(len(ti))] = 9.0
            rr, cc = linear_sum_assignment(sub)
            cost_cl += sub[rr, cc].sum()
        t_cl = (time.perf_counter() - t0) * 1e3
        largest = max(len(ti) for ti, _ in clusters)
        print(f"| {n} | {len(clusters)} | {largest} | {t_global:.1f} | {t_cl:.1f} | "
              f"{'yes' if abs(cost_global - cost_cl) < 1e-6 * max(1, cost_global) else 'NO'} |")

    if args.figures:
        os.makedirs(args.figures, exist_ok=True)
        figs = [
            ("maneuvering_imm.svg", maneuvering(), TrackerConfig(motion="imm"), "maneuvering · IMM (CV+CT) · GNN"),
            ("maneuvering_cv.svg", maneuvering(), TrackerConfig(motion="cv"), "maneuvering · CV-EKF (q=1) · GNN"),
            ("formation_gnn.svg", formation(150.0), TrackerConfig(association="gnn"), "formation · IMM · GNN (optimal)"),
            ("formation_greedy.svg", formation(150.0), TrackerConfig(association="greedy"), "formation · IMM · greedy"),
            ("dense_40.svg", dense(40), TrackerConfig(), "40 targets · 40 clutter/scan · IMM · GNN"),
        ]
        for name, sc, cfg, title in figs:
            res = run(sc, cfg, seed=1, keep=True)
            with open(os.path.join(args.figures, name), "w") as fh:
                fh.write(render(sc, res, title=title))
        print(f"\nfigures written to {args.figures}/")


if __name__ == "__main__":
    main()
