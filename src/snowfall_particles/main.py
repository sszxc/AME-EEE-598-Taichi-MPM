"""
MPM 3D demo with mesh obstacles (package path: repo root ``src/snowfall_particles/``).

- Global parameters live in YAML: ``configs/default.yml`` (relative to this file's directory)
- Code is split into submodules: app / mpm / scene / sdf / geometry / ui
- Mesh SDF is built on demand and cached next to the mesh file (``.npz``)

Run (either works; from repo root is recommended):
- ``python src/snowfall_particles/main.py``
- ``cd src && python -m snowfall_particles.main``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# This file lives under ``src/snowfall_particles/``; add parent ``src/`` to path so ``snowfall_particles`` resolves as a package
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import taichi as ti

from snowfall_particles.app import create_app
from snowfall_particles.config import load_config


def _init_taichi(arch: str):
    if arch == "cpu":
        ti.init(arch=ti.cpu)
    else:
        ti.init(arch=ti.gpu)


def main():
    ap = argparse.ArgumentParser()
    _here = Path(__file__).resolve().parent
    ap.add_argument("--config", default=str(_here / "configs" / "default.yml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    _init_taichi(cfg.arch)

    app = create_app(cfg)
    app.run()


if __name__ == "__main__":
    main()
