from __future__ import annotations

import numpy as np


def load_obj_triangles(path: str):
    """Load OBJ triangles; supports f v/vt/vn and negative indices."""
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v":
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif tag == "f":
                idxs: list[int] = []
                for corner in parts[1:]:
                    vi = int(corner.split("/")[0])
                    if vi < 0:
                        vi = len(vertices) + vi + 1
                    idxs.append(vi - 1)
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)

