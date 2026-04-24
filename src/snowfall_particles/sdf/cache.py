from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from snowfall_particles.geometry.mesh_io import load_obj_triangles
from snowfall_particles.geometry.transform import transform_mesh_for_preset
from snowfall_particles.sdf.builders import build_sdf_mesh_volume


@dataclass(frozen=True)
class MeshSdfRequest:
    mesh_path: Path
    sdf_res: int
    scale: float
    center: tuple[float, float, float]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    total = 0
    try:
        total = path.stat().st_size
    except Exception:
        total = 0
    with path.open("rb") as f, tqdm(
        total=total if total > 0 else None,
        desc="Hash mesh (sha256)",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        disable=not sys.stderr.isatty(),
    ) as pbar:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
            if total > 0:
                pbar.update(len(chunk))
    return h.hexdigest()


def _request_hash(req: MeshSdfRequest) -> str:
    mesh_hash = _sha256_file(req.mesh_path)
    payload = f"{mesh_hash}|res={req.sdf_res}|scale={req.scale:.9g}|center={req.center[0]:.9g},{req.center[1]:.9g},{req.center[2]:.9g}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def cache_path_for(req: MeshSdfRequest) -> Path:
    key = _request_hash(req)
    # alongside mesh; "np* format": use npz container
    return req.mesh_path.with_name(f"cache_{req.mesh_path.name}.sdf_res{req.sdf_res}.{key}.npz")


def load_cached_sdf(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        phi = data["phi"]
        if phi.dtype != np.float32:
            phi = phi.astype(np.float32)
        return phi
    except Exception:
        return None


def save_cached_sdf(path: Path, *, phi: np.ndarray, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_arr = np.asarray([repr(meta)], dtype=np.str_)
    np.savez_compressed(path, phi=phi.astype(np.float32, copy=False), meta=meta_arr)


def load_or_build_mesh_sdf(req: MeshSdfRequest, *, cache_enabled: bool = True) -> np.ndarray:
    cpath = cache_path_for(req)
    if cache_enabled:
        cached = load_cached_sdf(cpath)
        if cached is not None and cached.shape == (req.sdf_res, req.sdf_res, req.sdf_res):
            return cached

    with tqdm(total=3, desc="Mesh SDF (cache miss)", unit="step", disable=not sys.stderr.isatty()) as pbar:
        verts, faces = load_obj_triangles(str(req.mesh_path))
        pbar.update(1)
        verts = transform_mesh_for_preset(verts, req.scale, req.center)
        pbar.update(1)
        phi = build_sdf_mesh_volume(verts, faces, req.sdf_res)
        pbar.update(1)

    if cache_enabled:
        save_cached_sdf(
            cpath,
            phi=phi,
            meta={
                "mesh": str(req.mesh_path),
                "sdf_res": req.sdf_res,
                "scale": req.scale,
                "center": req.center,
            },
        )
    return phi

