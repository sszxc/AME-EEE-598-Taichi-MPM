"""
Visualize offline MPM trajectory (.npz from ``--offline``).

Run (from repo root):
  ``python src/snowfall_particles/visualize_output.py path/to/snowfall_trajectory.npz``

Only particle positions are shown (no obstacles / tank).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _active_mask(data: np.lib.npyio.NpzFile) -> np.ndarray:
    if "used" in data:
        u = np.asarray(data["used"]).astype(np.int8, copy=False).ravel()
        return u > 0
    return np.ones(int(np.asarray(data["positions"]).shape[1]), dtype=bool)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plotly 3D viewer for trajectory .npz")
    ap.add_argument("npz", type=Path, help="Path to .npz (positions [T,N,3], optional used [N])")
    args = ap.parse_args()
    path = args.npz.expanduser().resolve()
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        import plotly.graph_objects as go
    except ImportError as e:
        print("Error: install plotly, e.g. `pip install plotly`", file=sys.stderr)
        raise SystemExit(1) from e

    with np.load(path, allow_pickle=False) as data:
        pos = np.asarray(data["positions"], dtype=np.float64)
        if pos.ndim != 3 or pos.shape[2] != 3:
            print("Error: expected positions with shape (T, N, 3)", file=sys.stderr)
            sys.exit(1)
        mask = _active_mask(data)
        if not np.any(mask):
            print("Warning: no used particles; plotting all columns.", file=sys.stderr)
            mask = np.ones(pos.shape[1], dtype=bool)

    T = pos.shape[0]
    pts = pos[:, mask, :]
    xmin, ymin, zmin = 0, 0, 0
    xmax, ymax, zmax = 1, 1, 1

    p0 = pts[0]
    frames = [
        go.Frame(
            data=[
                go.Scatter3d(
                    x=pts[t, :, 0],
                    y=pts[t, :, 1],
                    z=pts[t, :, 2],
                    mode="markers",
                    marker=dict(size=2, opacity=0.75, line=dict(width=0)),
                )
            ],
            name=str(t),
        )
        for t in range(T)
    ]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=p0[:, 0],
                y=p0[:, 1],
                z=p0[:, 2],
                mode="markers",
                marker=dict(size=2, opacity=0.75, line=dict(width=0)),
            )
        ],
        frames=frames,
    )

    fig.update_layout(
        title=f"MPM particles ({path.name}) — frame 0 / {T - 1}",
        scene=dict(
            xaxis=dict(title="x", range=[xmin, xmax]),
            yaxis=dict(title="y", range=[ymin, ymax]),
            zaxis=dict(title="z", range=[zmin, zmax]),
            # range 相同但 aspectmode="data" 仍可能随视口变形；manual + 1:1:1 锁定立方体比例
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                up=dict(x=0, y=1, z=0),
            ),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        sliders=[
            dict(
                active=0,
                pad=dict(t=30),
                len=0.92,
                x=0.04,
                y=0,
                steps=[
                    dict(
                        method="animate",
                        args=[
                            [str(t)],
                            dict(
                                frame=dict(duration=0, redraw=True),
                                mode="immediate",
                                transition=dict(duration=0),
                            ),
                        ],
                        label=str(t),
                    )
                    for t in range(T)
                ],
            )
        ],
    )

    fig.show()


if __name__ == "__main__":
    main()
