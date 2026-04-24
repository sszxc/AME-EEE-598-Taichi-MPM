from __future__ import annotations

import sys
import numpy as np
from tqdm import tqdm


def build_sdf_box_volume(mn, mx, res: int) -> np.ndarray:
    """Axis-aligned box SDF; inside phi<0, outside phi>0; aligned to grid centers ((i+0.5)/res,...)."""
    mn = np.asarray(mn, dtype=np.float64)
    mx = np.asarray(mx, dtype=np.float64)
    ii = np.arange(res, dtype=np.float64)
    i, j, k = np.meshgrid(ii, ii, ii, indexing="ij")
    p = np.stack([(i + 0.5) / res, (j + 0.5) / res, (k + 0.5) / res], axis=-1)
    c = 0.5 * (mn + mx)
    e = 0.5 * (mx - mn)
    q = np.abs(p - c) - e
    qx, qy, qz = q[..., 0], q[..., 1], q[..., 2]
    outside = np.sqrt(np.maximum(qx, 0.0) ** 2 + np.maximum(qy, 0.0) ** 2 + np.maximum(qz, 0.0) ** 2)
    inside = np.minimum(np.maximum(np.maximum(qx, qy), qz), 0.0)
    return (outside + inside).astype(np.float32)


def build_sdf_mesh_volume(verts, faces, res: int) -> np.ndarray:
    """Approx mesh SDF: voxelized occupancy + EDT; inside phi<0."""
    try:
        import trimesh  # type: ignore
        from scipy import ndimage as ndi  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError("Mesh obstacle SDF needs: pip install trimesh rtree scipy") from e

    with tqdm(total=4, desc=f"Build mesh SDF (res={res})", unit="step", disable=not sys.stderr.isatty()) as pbar:
        mesh = trimesh.Trimesh(
            vertices=np.asarray(verts, dtype=np.float64),
            faces=np.asarray(faces, dtype=np.int64),
            process=True,
        )
        pbar.update(1)

        pitch = 1.0 / float(res)
        voxels = mesh.voxelized(pitch=pitch)
        pbar.update(1)

        ii = np.arange(res, dtype=np.float64)
        i, j, k = np.meshgrid(ii, ii, ii, indexing="ij")
        pts = np.stack([(i + 0.5) / res, (j + 0.5) / res, (k + 0.5) / res], axis=-1).reshape(-1, 3)
        occ = voxels.is_filled(pts).reshape(res, res, res)
        pbar.update(1)

        dist_out = ndi.distance_transform_edt(~occ) * pitch
        dist_in = ndi.distance_transform_edt(occ) * pitch
        pbar.update(1)

    return (dist_out - dist_in).astype(np.float32)

