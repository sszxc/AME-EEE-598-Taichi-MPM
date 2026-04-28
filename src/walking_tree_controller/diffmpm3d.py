import argparse
import glob
import os
import re
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


_WEIGHTS_RE = re.compile(r"^iter(\d+)_weights\.npz$")


def _save_weights_checkpoint(out_dir: str, iter_idx: int, *, seed: int) -> str:
    path = os.path.join(out_dir, f"iter{iter_idx:04d}_weights.npz")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(
        path,
        weights=cfg.weights.to_numpy(),
        iter=np.int32(iter_idx),
        seed=np.int32(seed),
        steps=np.int32(cfg.steps),
        max_steps=np.int32(cfg.max_steps),
        n_actuators=np.int32(cfg.n_actuators),
        n_sin_waves=np.int32(cfg.n_sin_waves),
        actuation_omega=np.float32(cfg.actuation_omega),
        dt=np.float32(cfg.dt),
    )
    print(f"Saved weights checkpoint to '{path}'")
    return path


def _find_latest_weights_checkpoint(out_dir: str) -> tuple[str, int]:
    candidates: list[tuple[int, float, str]] = []
    for path in glob.glob(os.path.join(out_dir, "iter*_weights.npz")):
        match = _WEIGHTS_RE.match(os.path.basename(path))
        if match is None:
            continue
        candidates.append((int(match.group(1)), os.path.getmtime(path), path))

    if not candidates:
        raise FileNotFoundError(
            f"No weights checkpoints found in '{out_dir}'. "
            "Run training with --out_interval > 0 first."
        )

    iter_idx, _, path = max(candidates, key=lambda item: (item[0], item[1]))
    return path, iter_idx


def _checkpoint_seed_or_default(path: str, default_seed: int) -> int:
    with np.load(path) as data:
        if "seed" not in data:
            return default_seed
        return int(data["seed"].item())


def _load_weights_checkpoint(path: str) -> None:
    with np.load(path) as data:
        weights = np.asarray(data["weights"], dtype=np.float32)

    expected_shape = (cfg.n_actuators, cfg.n_sin_waves)
    if weights.shape != expected_shape:
        raise ValueError(
            f"Checkpoint weights shape {weights.shape} does not match "
            f"current scene shape {expected_shape}. Check the seed/scene config."
        )

    cfg.weights.from_numpy(weights)
    print(f"Loaded weights checkpoint from '{path}'")


def _write_rollout_outputs(iter_idx: int, options, *, output_label: str) -> None:
    mp4_path = (
        os.path.join(options.out_dir, f"{output_label}.mp4")
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
            iter_idx=iter_idx,
            stride=max(1, int(options.vis_stride)),
            save_folder=None,
            interactive=(not options.vis_no_gui),
        )
    else:
        with tempfile.TemporaryDirectory(prefix=f"mpm3d_vis_{output_label}_") as td:
            viz.visualize_rollout(
                iter_idx=iter_idx,
                stride=max(1, int(options.vis_stride)),
                save_folder=td,
                interactive=(not options.vis_no_gui),
            )
            _encode_mp4_from_png_folder(td, mp4_path, fps=30)

    viz.dump_particles_bin(
        iter_idx=iter_idx,
        start_s=7,
        step_s=2,
        out_dir=options.out_dir,
        name=output_label,
    )
    if options.dump_mesh:
        viz.dump_mesh_sequence(
            iter_idx=iter_idx,
            start_s=7,
            step_s=2,
            out_dir=options.out_dir,
            name=output_label,
            mesh_res=int(options.mesh_res),
            mesh_sigma=float(options.mesh_sigma),
            mesh_level_ratio=float(options.mesh_level_ratio),
        )


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
        help='Unified interval (in outer iters) for visualization, weights, and .bin dumping. '
        'Set <=0 to disable all interval outputs. Default: 20.',
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
        '--eval',
        action='store_true',
        help='Load the latest weights checkpoint from --out_dir and run a longer rollout without training.',
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=1024,
        help='Number of simulation steps for --eval rollout (default: 1024).',
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
    eval_weights_path: str | None = None
    eval_iter_idx: int | None = None

    if options.eval:
        if int(options.eval_steps) <= 1:
            raise ValueError("--eval_steps must be > 1.")
        cfg.steps = int(options.eval_steps)
        cfg.max_steps = int(options.eval_steps)
        eval_weights_path, eval_iter_idx = _find_latest_weights_checkpoint(options.out_dir)
        seed = _checkpoint_seed_or_default(eval_weights_path, seed)
        print(
            f"Eval mode: using checkpoint '{eval_weights_path}' "
            f"with seed={seed}, steps={cfg.steps}."
        )

    # ---- Seed all RNG sources for determinism ----
    # 1) Python's random (used in Scene.add_rect for non-solid particles)
    try:
        import random as _py_random

        _py_random.seed(seed)
    except Exception:
        pass
    # 2) NumPy global seed (TreePlant uses default_rng(seed), but keep this too)
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
    scene_lib.build_walking_tree_plant(scene)
    # scene_lib.build_walking_tree_root(scene)
    os.makedirs(os.path.abspath(options.out_dir), exist_ok=True)
    init_ply_path = os.path.join(options.out_dir, "walking_tree_init.ply")
    viz.export_init_ply(scene, init_ply_path)
    # scene.add_rect(0.4, 0.4, 0.2, 0.1, 0.3, 0.1, -1, 1)
    scene.finalize()
    cfg.allocate_fields(enable_gradients=(not options.eval))

    kernels.init(
        np.array(scene.x, dtype=np.float32),
        np.array(scene.actuator_id, dtype=np.int32),
        np.array(scene.particle_type, dtype=np.int32),
        np.array(scene.root_id, dtype=np.int32),
        np.array(scene.segment_id, dtype=np.int32),
        np.array(scene.actuator_dir, dtype=np.float32),
    )

    if options.eval:
        assert eval_weights_path is not None
        assert eval_iter_idx is not None
        _load_weights_checkpoint(eval_weights_path)
        output_label = f"eval_iter{eval_iter_idx:04d}_steps{cfg.steps}"
        _write_rollout_outputs(eval_iter_idx, options, output_label=output_label)
        return

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

        # Trigger visualization, weights, and bin dump together on a unified cadence.
        # We use (iter + 1) % interval == 0 to match the historical bin-dump timing
        # (previously: iter % 20 == 19).
        if out_interval > 0 and (iter + 1) % out_interval == 0:
            output_label = f"iter{iter:04d}"
            _save_weights_checkpoint(options.out_dir, iter, seed=seed)
            _write_rollout_outputs(iter, options, output_label=output_label)

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
