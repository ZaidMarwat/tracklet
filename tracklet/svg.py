"""Dependency-free SVG plot: ground truth, raw detections, confirmed tracks."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

PALETTE = ["#2a78d6", "#e1572a", "#1f9e6e", "#b44fc7", "#d4a017", "#d6336c", "#4b5bd6", "#3d9a9a"]


def render(scenario, result, title: str = "", size: int = 760) -> str:
    tracker = result.tracker
    truth_paths = {}
    for k, row in enumerate(scenario.truth):
        for tid, x in row:
            truth_paths.setdefault(tid, []).append((x[0], x[2]))
    dets = []
    for Z in result.detections:
        for r, th in Z:
            dets.append((r * np.cos(th), r * np.sin(th)))
    tracks = [t for t in tracker.tracks + tracker.dead if len(t.history) >= 6 and _was_confirmed(t)]

    pts = [p for path in truth_paths.values() for p in path]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    pad = 1500
    x0, x1, y0, y1 = min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad
    span = max(x1 - x0, y1 - y0)
    m = 40

    def P(x: float, y: float) -> Tuple[float, float]:
        return m + (x - x0) / span * (size - 2 * m), size - m - (y - y0) / span * (size - 2 * m)

    def poly(points: List[Tuple[float, float]]) -> str:
        return " ".join(f"{a:.1f},{b:.1f}" for a, b in (P(x, y) for x, y in points))

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size + 44}" '
           f'font-family="ui-monospace, monospace" font-size="12">',
           f'<rect width="100%" height="100%" fill="#fcfcfb"/>',
           f'<text x="{m}" y="24" font-size="15" fill="#222">{title}</text>']
    for dx, dy in dets:
        if x0 <= dx <= x0 + span and y0 <= dy <= y0 + span:
            a, b = P(dx, dy)
            out.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="1.3" fill="#9aa0a6" opacity="0.55"/>')
    for path in truth_paths.values():
        out.append(f'<polyline points="{poly(path)}" fill="none" stroke="#222" stroke-width="1.2" '
                   f'stroke-dasharray="4 3" opacity="0.7"/>')
    for i, t in enumerate(sorted(tracks, key=lambda t: t.id)):
        c = PALETTE[i % len(PALETTE)]
        hist = [(x, y) for _, x, y in t.history]
        out.append(f'<polyline points="{poly(hist)}" fill="none" stroke="{c}" stroke-width="2.2" opacity="0.9"/>')
        a, b = P(*hist[-1])
        out.append(f'<text x="{a + 5:.1f}" y="{b - 5:.1f}" fill="{c}">T{t.id}</text>')
    s = result.summary
    out.append(f'<text x="{m}" y="{size + 8}" fill="#222">MOTA {s["MOTA"]:.3f} · RMSE {s["RMSE_m"]:.0f} m · '
               f'ID switches {s["id_switches"]} · FP {s["false_pos"]} · {s["mean_scan_ms"]:.1f} ms/scan</text>')
    out.append(f'<text x="{m}" y="{size + 28}" fill="#777">dashed = ground truth · grey dots = detections incl. '
               f'clutter · colour = confirmed track</text>')
    out.append("</svg>")
    return "\n".join(out)


def _was_confirmed(track) -> bool:
    from .tracker import TrackStatus

    return track.status is TrackStatus.CONFIRMED or sum(track.hits) >= 3
