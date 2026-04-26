"""
Visualize a saved diffmpm3d rollout with Taichi's 3D viewer (GGUI).

Data format (produced by ``src/walking_tree_controller/diffmpm3d.py``):

- Directory layout: ``mpm3d/iter{iter:04d}/{step:04d}.bin`` (steps are 7, 9, 11, ...).
- Each ``.bin`` is a flat ``float32`` array of length ``7 * N`` (N = particle count):
  ``[xs(N), ys(N), zs(N), us(N), vs(N), ws(N), cs(N)]``
  where ``xs/ys/zs`` are positions, ``us/vs/ws`` are velocities, and ``cs`` is the
  per-particle color packed via ``ti.rgb_to_hex((r,g,b))`` (i.e. ``0xRRGGBB``)
  and then stored as float32 (all values <= 0xFFFFFF, which is exactly
  representable in float32).

Usage (from repo root):
    python src/walking_tree_controller/visualize_output.py mpm3d/iter0019
    python src/walking_tree_controller/visualize_output.py mpm3d/iter0019 --fps 30 --loop
    python src/walking_tree_controller/visualize_output.py mpm3d/iter0019 \
        --save-frames out_frames --no-gui
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import taichi as ti

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.camera import FixedLookatCamera


def _list_bin_frames(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.bin"))
    if not files:
        raise FileNotFoundError(f"No .bin frames found in {folder}")
    return files


def _infer_n_particles(frame_path: Path) -> int:
    size = frame_path.stat().st_size
    stride = 7 * 4  # 7 float32 values per particle per frame
    if size % stride != 0:
        raise ValueError(
            f"Unexpected file size {size} for {frame_path}: "
            f"not a multiple of 7 * sizeof(float32)."
        )
    return size // stride


def _load_frame(frame_path: Path, n_particles: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a single frame file and return (positions[N,3], colors[N,3] in [0,1])."""
    raw = np.fromfile(str(frame_path), dtype=np.float32)
    if raw.size != 7 * n_particles:
        raise ValueError(
            f"Frame {frame_path} has {raw.size} floats, expected {7 * n_particles}."
        )
    xs = raw[0 * n_particles: 1 * n_particles]
    ys = raw[1 * n_particles: 2 * n_particles]
    zs = raw[2 * n_particles: 3 * n_particles]
    cs = raw[6 * n_particles: 7 * n_particles]

    positions = np.stack([xs, ys, zs], axis=1).astype(np.float32, copy=False)

    # Unpack 0xRRGGBB stored as float32 -> (R,G,B) in [0,1].
    hex_ints = cs.astype(np.int64, copy=False)  # widen to avoid sign issues
    r = ((hex_ints >> 16) & 0xFF).astype(np.float32) / 255.0
    g = ((hex_ints >> 8) & 0xFF).astype(np.float32) / 255.0
    b = (hex_ints & 0xFF).astype(np.float32) / 255.0
    colors = np.stack([r, g, b], axis=1)
    return positions, colors


def _preload_rollout(
    frames: list[Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Preload all frames while allowing the particle count to change over time.

    Some rollouts write only "active" particles per frame (variable N). For
    visualization we pad each frame to N_max and move unused slots offscreen.
    """
    n_per_frame = np.asarray([_infer_n_particles(fp) for fp in frames], dtype=np.int32)
    n_max = int(n_per_frame.max(initial=0))
    if n_max <= 0:
        raise ValueError("Could not infer a positive particle count from frames.")

    # Allocate padded arrays [T, N_max, 3].
    all_pos = np.empty((len(frames), n_max, 3), dtype=np.float32)
    all_col = np.empty((len(frames), n_max, 3), dtype=np.float32)

    # Fill defaults: move unused particles far away; color doesn't matter then.
    all_pos[...] = np.array([-10.0, -10.0, -10.0], dtype=np.float32)
    all_col[...] = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    for i, fp in enumerate(frames):
        n_i = int(n_per_frame[i])
        pos_i, col_i = _load_frame(fp, n_i)
        all_pos[i, :n_i] = pos_i
        all_col[i, :n_i] = col_i

    return all_pos, all_col, n_per_frame, n_max


def _make_ground_grid_segments(
    *,
    x_range: tuple[float, float] = (0.0, 1.0),
    z_range: tuple[float, float] = (0.0, 1.0),
    y: float = 0.0,
    divisions: int = 10,
) -> np.ndarray:
    """Build the endpoints of a wireframe ground grid on the y = const plane.

    Returns an array of shape ``(2 * num_lines, 3)`` laid out as
    ``[p0, p1, p0, p1, ...]``, ready for ``scene.lines(field, width=...)``.
    """
    x0, x1 = float(x_range[0]), float(x_range[1])
    z0, z1 = float(z_range[0]), float(z_range[1])
    n = max(1, int(divisions))
    xs = np.linspace(x0, x1, n + 1, dtype=np.float32)
    zs = np.linspace(z0, z1, n + 1, dtype=np.float32)

    # Lines parallel to z-axis (one per x tick) + lines parallel to x-axis
    # (one per z tick). Each line contributes 2 endpoints.
    segs = np.empty((2 * (len(xs) + len(zs)), 3), dtype=np.float32)
    i = 0
    for x in xs:
        segs[i] = (x, y, z0); segs[i + 1] = (x, y, z1); i += 2
    for z in zs:
        segs[i] = (x0, y, z); segs[i + 1] = (x1, y, z); i += 2
    return segs


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Play a diffmpm3d rollout folder (mpm3d/iterXXXX) in Taichi's 3D viewer."
    )
    ap.add_argument(
        "folder",
        type=Path,
        help="Folder containing *.bin frames, e.g. mpm3d/iter0019",
    )
    ap.add_argument("--fps", type=float, default=30.0, help="Playback FPS (default: 30).")
    ap.add_argument(
        "--radius",
        type=float,
        default=0.006,
        help="Particle radius in world units (simulation domain is [0,1]^3).",
    )
    ap.add_argument("--loop", action="store_true", help="Loop playback forever.")
    ap.add_argument(
        "--window-size",
        type=int,
        nargs=2,
        default=(960, 960),
        metavar=("W", "H"),
        help="GGUI window size in pixels (default: 960 960).",
    )
    ap.add_argument(
        "--save-frames",
        type=Path,
        default=None,
        help="If set, save one PNG per rendered frame into this folder "
             "(useful for offline ffmpeg encoding).",
    )
    ap.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable interactive window; only useful together with --save-frames "
             "on systems that still need a GGUI context to render.",
    )
    ap.add_argument(
        "--bg",
        type=float,
        nargs=3,
        default=(1.0, 1.0, 1.0),
        metavar=("R", "G", "B"),
        help="Background color in [0,1] (default: white).",
    )
    ap.add_argument(
        "--no-ground",
        action="store_true",
        help="Disable the wireframe ground grid.",
    )
    ap.add_argument(
        "--ground-divisions",
        type=int,
        default=10,
        help="Number of cells per side in the wireframe ground grid (default: 10).",
    )
    ap.add_argument(
        "--ground-y",
        type=float,
        default=0.0,
        help="Y coordinate of the ground plane (default: 0.0, matches the "
             "diffmpm3d floor at j < bound).",
    )
    ap.add_argument(
        "--ground-color",
        type=float,
        nargs=3,
        default=(0.35, 0.35, 0.35),
        metavar=("R", "G", "B"),
        help="Wireframe ground color in [0,1] (default: dark grey).",
    )
    ap.add_argument(
        "--ground-width",
        type=float,
        default=1.0,
        help="Line width (pixels) of the ground grid (default: 1.0).",
    )
    args = ap.parse_args()

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    frames = _list_bin_frames(folder)
    all_pos, all_col, n_per_frame, n_particles = _preload_rollout(frames)
    print(f"[visualize_output] folder    = {folder}")
    print(f"[visualize_output] #frames   = {len(frames)}")
    print(f"[visualize_output] #particles= {n_particles}")
    if int(n_per_frame.min(initial=n_particles)) != int(n_per_frame.max(initial=n_particles)):
        print(
            "[visualize_output] warning   = particle count varies across frames "
            f"(min={int(n_per_frame.min())}, max={int(n_per_frame.max())}); "
            "visualization will pad smaller frames."
        )

    ti.init(arch=ti.gpu)

    pos_field = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
    col_field = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)

    ground_field = None
    ground_color = tuple(float(c) for c in args.ground_color)
    if not args.no_ground:
        ground_segs = _make_ground_grid_segments(
            x_range=(0.0, 1.0),
            z_range=(0.0, 1.0),
            y=float(args.ground_y),
            divisions=int(args.ground_divisions),
        )
        ground_field = ti.Vector.field(3, dtype=ti.f32, shape=ground_segs.shape[0])
        ground_field.from_numpy(ground_segs)

    w_px, h_px = int(args.window_size[0]), int(args.window_size[1])
    window = ti.ui.Window(
        f"diffmpm3d rollout - {folder.name}",
        (w_px, h_px),
        vsync=True,
        show_window=not args.no_gui,
    )
    canvas = window.get_canvas()
    canvas.set_background_color(tuple(float(c) for c in args.bg))
    scene = ti.ui.Scene()
    camera = FixedLookatCamera()

    # Simulation domain is [0,1]^3; pick a pleasant isometric-ish initial view.
    camera.position(1.8, 1.3, 1.8)
    camera.lookat(0.5, 0.25, 0.5)
    camera.up(0.0, 1.0, 0.0)
    camera.fov(45)

    if args.save_frames is not None:
        args.save_frames.mkdir(parents=True, exist_ok=True)

    dt_target = 1.0 / max(1e-3, args.fps)
    frame_idx = 0
    last_advance = time.time()

    while window.running:
        now = time.time()
        if now - last_advance >= dt_target:
            advance = max(1, int((now - last_advance) / dt_target))
            frame_idx += advance
            last_advance = now
            if frame_idx >= len(frames):
                if args.loop:
                    frame_idx %= len(frames)
                else:
                    frame_idx = len(frames) - 1

        pos_field.from_numpy(all_pos[frame_idx])
        col_field.from_numpy(all_col[frame_idx])

        # Mouse + WASD camera navigation (RMB drag orbits around the initial look-at).
        camera.track_user_inputs_fixed_lookat(window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.7, 0.7, 0.7))
        scene.point_light(pos=(2.0, 2.5, 2.0), color=(1.0, 1.0, 1.0))

        scene.particles(
            pos_field,
            radius=float(args.radius),
            per_vertex_color=col_field,
        )
        if ground_field is not None:
            scene.lines(
                ground_field,
                color=ground_color,
                width=float(args.ground_width),
            )
        canvas.scene(scene)

        if args.save_frames is not None:
            out_png = args.save_frames / f"{frame_idx:04d}.png"
            window.save_image(str(out_png))

        window.show()

        # When not looping, stop the main loop as soon as the last frame has been
        # written to disk in headless/save mode; otherwise keep the window open
        # so the user can orbit the final frame.
        if (not args.loop) and frame_idx >= len(frames) - 1:
            if args.save_frames is not None and args.no_gui:
                break


if __name__ == "__main__":
    main()
