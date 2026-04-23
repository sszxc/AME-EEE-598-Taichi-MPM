"""
MPM 3D demo with mesh obstacles (package path: repo root ``src/snowfall_particles/``).

- Global parameters live in YAML: ``configs/default.yml`` (relative to this file's directory)
- Code is split into submodules: app / mpm / scene / sdf / geometry / ui
- Mesh SDF is built on demand and cached next to the mesh file (``.npz``)

Run (either works; from repo root is recommended):
- ``python src/snowfall_particles/main.py``
- ``cd src && python -m snowfall_particles.main``
- Offline trajectory export: ``python src/snowfall_particles/main.py --offline`` (requires interactive terminal; set ``offline.simulation_duration_seconds`` in YAML)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# This file lives under ``src/snowfall_particles/``; add parent ``src/`` to path so ``snowfall_particles`` resolves as a package
_SRC_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_OFFLINE_NPZ = _REPO_ROOT / "outputs" / "snowfall_trajectory.npz"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import taichi as ti

from snowfall_particles.app import create_app
from snowfall_particles.config import load_config
from snowfall_particles.scene.presets import load_fluid_presets, load_obstacle_presets


def _init_taichi(arch: str):
    if arch == "cpu":
        ti.init(arch=ti.cpu)
    else:
        ti.init(arch=ti.gpu)


def main():
    ap = argparse.ArgumentParser()
    _here = Path(__file__).resolve().parent
    ap.add_argument("--config", default=str(_here / "configs" / "default.yml"))
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Run without GUI and save particle positions to an .npz file (see YAML offline.simulation_duration_seconds).",
    )
    ap.add_argument(
        "--output",
        default=str(_DEFAULT_OFFLINE_NPZ),
        help="Output path for --offline (default: repo outputs/snowfall_trajectory.npz).",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.offline:
        if cfg.offline is None or cfg.offline.simulation_duration_seconds <= 0:
            print(
                "Error: --offline requires YAML `offline.simulation_duration_seconds` as a positive number.",
                file=sys.stderr,
            )
            sys.exit(1)
        if not sys.stdin.isatty():
            print(
                "Error: --offline requires an interactive terminal (stdin) for preset selection.",
                file=sys.stderr,
            )
            sys.exit(1)

        import questionary

        preset_names, _ = load_fluid_presets(cfg.particles)
        obs_names = [p.name for p in load_obstacle_presets(cfg.obstacles)]

        p_choice = questionary.select("Particle preset:", choices=preset_names).ask()
        if p_choice is None:
            sys.exit(0)
        p_idx = preset_names.index(p_choice)

        o_choice = questionary.select("Obstacle preset:", choices=obs_names).ask()
        if o_choice is None:
            sys.exit(0)
        o_idx = obs_names.index(o_choice)

        _init_taichi(cfg.arch)
        app = create_app(cfg, headless=True)
        out_path = Path(args.output).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        app.run_offline_export(
            output_path=out_path,
            duration_sim_s=cfg.offline.simulation_duration_seconds,
            particle_preset_idx=p_idx,
            obstacle_preset_idx=o_idx,
        )
        print(f"Wrote {out_path}", file=sys.stderr)
        return

    _init_taichi(cfg.arch)

    app = create_app(cfg)
    app.run()


if __name__ == "__main__":
    main()
