from __future__ import annotations

import numpy as np


def transform_mesh_for_preset(verts, scale: float, world_center):
    """Center mesh by bbox-center, scale, then translate center to world_center."""
    if len(verts) == 0:
        return verts
    c = (verts.min(axis=0) + verts.max(axis=0)) * 0.5
    verts = (verts - c) * float(scale)
    verts = verts + np.asarray(world_center, dtype=np.float64)
    return verts.astype(np.float64)

