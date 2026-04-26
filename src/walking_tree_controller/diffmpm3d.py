import argparse
import os
import shutil
import subprocess
import tempfile
import time
from typing import Optional

import numpy as np
import taichi as ti

import config as cfg
import kernels as kernels
import scene as scene_lib
import viz as viz

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg and ensure `ffmpeg` is on your PATH.\n"
            "- macOS (brew): brew install ffmpeg\n"
            "- conda: conda install -c conda-forge ffmpeg"
        )
    return ffmpeg


def _encode_mp4_from_png_folder(png_folder: str, out_mp4: str, *, fps: int = 30) -> None:
    ffmpeg = _require_ffmpeg()
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)

    # Use glob pattern so filenames don't need to be contiguous (e.g. 0000.png, 0008.png, ...)
    pattern = os.path.join(os.path.abspath(png_folder), "*.png")
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostats",
        "-framerate",
        str(int(fps)),
        "-pattern_type",
        "glob",
        "-i",
        pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        # Ensure even dimensions for H.264 compatibility
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        out_mp4,
    ]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for Python/NumPy/Taichi and procedural scene generation (default: 0).",
    )
    parser.add_argument(
        '--out_interval',
        type=int,
        default=20,
        help='Unified interval (in outer iters) for BOTH visualization and .bin dumping. '
        'Set <=0 to disable both. Default: 20.',
    )
    parser.add_argument('--vis_stride', type=int, default=8)
    parser.add_argument(
        '--out_dir',
        type=str,
        default='outputs',
        help="Output directory root (default: 'outputs').",
    )
    parser.add_argument(
        '--progress',
        action='store_true',
        help='Show a tqdm progress bar for the outer optimization loop (if tqdm is installed).',
    )
    parser.add_argument(
        '--warmup',
        action='store_true',
        help='Run one forward/backward warmup before timing to exclude Taichi first-time JIT overhead.',
    )
    parser.add_argument(
        '--vis_save',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Save rollout as mp4 under mpm3d_vis/ (default: enabled).',
    )
    parser.add_argument(
        '--vis_no_gui',
        action='store_true',
        help='Disable interactive GUI playback (use with --vis_save).',
    )
    parser.add_argument(
        '--dump_mesh',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Also dump marching-cubes mesh frames under iterXXXX_mesh/ (default: disabled).',
    )
    parser.add_argument(
        '--mesh_res',
        type=int,
        default=64,
        help='Voxel grid resolution for marching-cubes mesh dumping (default: 64).',
    )
    parser.add_argument(
        '--mesh_sigma',
        type=float,
        default=1.25,
        help='Gaussian smoothing sigma in voxels for particle density meshing (default: 1.25).',
    )
    parser.add_argument(
        '--mesh_level_ratio',
        type=float,
        default=0.08,
        help='Iso level as a fraction of max density for marching cubes (default: 0.08).',
    )
    options = parser.parse_args()
    out_interval = int(options.out_interval)
    seed = int(options.seed)

    # ---- Seed all RNG sources for determinism ----
    # 1) Python's random (used in Scene.add_rect for non-solid particles)
    try:
        import random as _py_random

        _py_random.seed(seed)
    except Exception:
        pass
    # 2) NumPy global seed (TreeRoot uses default_rng(seed), but keep this too)
    try:
        np.random.seed(seed)
    except Exception:
        pass
    # 3) Taichi random (used by ti.randn in kernels.init)
    if hasattr(ti, "seed"):
        try:
            ti.seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass
    # 4) Propagate to procedural root generator.
    cfg.seed = seed

    # initialization
    scene = scene_lib.Scene()
    scene_lib.build_walking_tree_root(scene)
    os.makedirs(os.path.abspath(options.out_dir), exist_ok=True)
    init_ply_path = os.path.join(options.out_dir, "walking_tree_init.ply")
    viz.export_init_ply(scene, init_ply_path)
    # scene.add_rect(0.4, 0.4, 0.2, 0.1, 0.3, 0.1, -1, 1)
    scene.finalize()
    cfg.allocate_fields()

    kernels.init(
        np.array(scene.x, dtype=np.float32),
        np.array(scene.actuator_id, dtype=np.int32),
        np.array(scene.particle_type, dtype=np.int32),
        np.array(scene.root_id, dtype=np.int32),
        np.array(scene.segment_id, dtype=np.int32),
        np.array(scene.actuator_dir, dtype=np.float32),
    )

    losses = []

    if options.warmup:
        # Exclude first-time Taichi JIT/initialization from per-iter timing.
        ti.ad.clear_all_gradients()
        _ = kernels.forward()
        cfg.loss.grad[None] = 1
        kernels.backward()

    iters_range = range(options.iters)
    pbar: Optional[object] = None
    if options.progress and tqdm is not None:
        pbar = tqdm(iters_range, total=options.iters, dynamic_ncols=True)
        iters_iter = pbar
    else:
        iters_iter = iters_range

    for iter in iters_iter:
        t = time.time()
        ti.ad.clear_all_gradients()
        l = kernels.forward()
        losses.append(l)
        cfg.loss.grad[None] = 1
        kernels.backward()
        per_iter_time = time.time() - t
        if pbar is not None:
            try:
                pbar.set_postfix(loss=float(l), per_iter_s=f"{per_iter_time:.2f}")
            except Exception:
                pass
        else:
            print('i=', iter, 'loss=', l, F' per iter {per_iter_time:.2f}s')
        learning_rate = 30
        kernels.learn(learning_rate)

        # Trigger visualization + bin dump together on a unified cadence.
        # We use (iter + 1) % interval == 0 to match the historical bin-dump timing
        # (previously: iter % 20 == 19).
        if out_interval > 0 and (iter + 1) % out_interval == 0:
            mp4_path = (
                os.path.join(options.out_dir, f"iter{iter:04d}.mp4")
                if options.vis_save
                else None
            )
            print(
                "Visualizing rollout "
                + ("(GUI)" if not options.vis_no_gui else "(no GUI)")
                + (f" + saving to '{mp4_path}'" if mp4_path is not None else "")
                + " ..."
            )
            if mp4_path is None:
                viz.visualize_rollout(
                    iter_idx=iter,
                    stride=max(1, int(options.vis_stride)),
                    save_folder=None,
                    interactive=(not options.vis_no_gui),
                )
            else:
                with tempfile.TemporaryDirectory(
                    prefix=f"mpm3d_vis_iter{iter:04d}_"
                ) as td:
                    viz.visualize_rollout(
                        iter_idx=iter,
                        stride=max(1, int(options.vis_stride)),
                        save_folder=td,
                        interactive=(not options.vis_no_gui),
                    )
                    _encode_mp4_from_png_folder(td, mp4_path, fps=30)

            viz.dump_particles_bin(iter_idx=iter, start_s=7, step_s=2, out_dir=options.out_dir)
            if options.dump_mesh:
                viz.dump_mesh_sequence(
                    iter_idx=iter,
                    start_s=7,
                    step_s=2,
                    out_dir=options.out_dir,
                    mesh_res=int(options.mesh_res),
                    mesh_sigma=float(options.mesh_sigma),
                    mesh_level_ratio=float(options.mesh_level_ratio),
                )

    if plt is not None and options.iters > 0:
        plt.title("Optimization of Initial Velocity")
        plt.ylabel("Loss")
        plt.xlabel("Gradient Descent Iterations")
        plt.plot(losses)
        loss_plot_path = os.path.join(options.out_dir, "loss.png")
        os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)
        plt.savefig(loss_plot_path, dpi=200, bbox_inches="tight")
        # plt.show()


if __name__ == '__main__':
    main()
