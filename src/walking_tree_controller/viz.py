import os
import colorsys
from typing import Tuple

import numpy as np
import taichi as ti
from tqdm import tqdm

import config as cfg
import kernels as kernels


def _color_for_actuator(act_id: int) -> tuple[int, int, int]:
    if act_id < 0:
        return (120, 120, 120)
    hue = (act_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.72, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


def export_init_ply(scene, out_path: str) -> None:
    points = np.asarray(scene.x, dtype=np.float32)
    actuator_ids = np.asarray(scene.actuator_id, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("scene.x must be shaped [N, 3].")
    if points.shape[0] != actuator_ids.shape[0]:
        raise ValueError("scene.x and scene.actuator_id length mismatch.")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for i in range(points.shape[0]):
            r, g, b = _color_for_actuator(int(actuator_ids[i]))
            f.write(
                f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f} "
                f"{r:d} {g:d} {b:d}\n"
            )
    print(f"Exported init particles to '{out_path}'")


def _colors_for_frame(s: int) -> np.ndarray:
    particle_type_ = cfg.particle_type.to_numpy()
    actuation_ = cfg.actuation.to_numpy()
    actuator_id_ = cfg.actuator_id.to_numpy()
    colors = np.empty(shape=cfg.n_particles, dtype=np.uint32)
    for i in range(cfg.n_particles):
        if particle_type_[i] == 0:
            # fluid
            r, g, b = 0.3, 0.3, 1.0
        else:
            # solid
            if actuator_id_[i] != -1:
                act = float(actuation_[s, actuator_id_[i]]) * 0.5
                r = 0.5 - act
                g = 0.5 - abs(act)
                b = 0.5 + act
            else:
                r, g, b = 0.4, 0.4, 0.4
        colors[i] = ti.rgb_to_hex((r, g, b))
    return colors


def _unpack_hex_colors(hex_colors: np.ndarray) -> np.ndarray:
    """Convert 0xRRGGBB colors to float RGB rows in [0, 1]."""
    hex_ints = hex_colors.astype(np.int64, copy=False)
    r = ((hex_ints >> 16) & 0xFF).astype(np.float32) / 255.0
    g = ((hex_ints >> 8) & 0xFF).astype(np.float32) / 255.0
    b = (hex_ints & 0xFF).astype(np.float32) / 255.0
    return np.stack([r, g, b], axis=1)


def _empty_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = np.zeros((1, 3), dtype=np.float32)
    faces = np.zeros((0, 3), dtype=np.int32)
    colors = np.full((1, 3), 0.5, dtype=np.float32)
    return vertices, faces, colors


def _write_binary_ply(
    out_path: str,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    colors: np.ndarray | None = None,
) -> None:
    """
    Write a little-endian binary PLY triangle mesh with optional per-vertex RGB.

    Vertex layout:
      float x, float y, float z, uchar red, uchar green, uchar blue
    Face layout:
      uchar vertex_count (always 3), int32 i0, int32 i1, int32 i2
    """
    verts = np.asarray(vertices, dtype=np.float32)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must be shaped [V,3], got {verts.shape}.")
    fac = np.asarray(faces, dtype=np.int32)
    if fac.ndim != 2 or fac.shape[1] != 3:
        raise ValueError(f"faces must be shaped [F,3], got {fac.shape}.")

    if colors is None:
        cols_u8 = np.full((verts.shape[0], 3), 127, dtype=np.uint8)
    else:
        cols = np.asarray(colors)
        if cols.shape != verts.shape:
            raise ValueError(f"colors must match vertices shape, got {cols.shape} vs {verts.shape}.")
        if cols.dtype == np.uint8:
            cols_u8 = cols
        else:
            cols_f = np.asarray(cols, dtype=np.float32)
            cols_u8 = np.clip(np.rint(cols_f * 255.0), 0, 255).astype(np.uint8)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {int(verts.shape[0])}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        f"element face {int(fac.shape[0])}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    v_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    v = np.empty((verts.shape[0],), dtype=v_dtype)
    v["x"], v["y"], v["z"] = verts[:, 0], verts[:, 1], verts[:, 2]
    v["red"], v["green"], v["blue"] = cols_u8[:, 0], cols_u8[:, 1], cols_u8[:, 2]

    f_dtype = np.dtype([("n", "u1"), ("i0", "<i4"), ("i1", "<i4"), ("i2", "<i4")])
    f = np.empty((fac.shape[0],), dtype=f_dtype)
    f["n"] = np.uint8(3)
    f["i0"], f["i1"], f["i2"] = fac[:, 0], fac[:, 1], fac[:, 2]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "wb") as fp:
        fp.write(header)
        fp.write(v.tobytes(order="C"))
        fp.write(f.tobytes(order="C"))


def _particles_to_mesh(
    positions: np.ndarray,
    colors: np.ndarray,
    *,
    res: int,
    sigma: float,
    level_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a surface mesh from particles by marching cubes over a smoothed density grid.

    The simulation domain is [0, 1]^3. Colors are transferred from the nearest
    particle to each generated mesh vertex.
    """
    try:
        from scipy import ndimage as ndi  # type: ignore
        from scipy.spatial import cKDTree  # type: ignore
        from skimage import measure  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Mesh dumping needs scikit-image and scipy. Install with "
            "`pip install scikit-image scipy`."
        ) from exc

    res = int(res)
    if res < 4:
        raise ValueError("mesh_res must be >= 4.")
    if sigma < 0:
        raise ValueError("mesh_sigma must be >= 0.")
    if not (0.0 < level_ratio < 1.0):
        raise ValueError("mesh_level_ratio must be in (0, 1).")

    pos = np.asarray(positions, dtype=np.float32)
    col = np.asarray(colors, dtype=np.float32)
    if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] == 0:
        return _empty_mesh()

    pos = np.clip(pos, 0.0, 1.0)
    idx = np.rint(pos * float(res - 1)).astype(np.int32)
    idx = np.clip(idx, 0, res - 1)

    density = np.zeros((res, res, res), dtype=np.float32)
    np.add.at(density, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
    if sigma > 0.0:
        density = ndi.gaussian_filter(density, sigma=float(sigma), mode="constant")

    max_density = float(density.max(initial=0.0))
    if max_density <= 0.0:
        return _empty_mesh()

    spacing = np.array([1.0 / float(res - 1)] * 3, dtype=np.float32)
    padded_density = np.pad(density, 1, mode="constant")
    level = max_density * float(level_ratio)
    try:
        vertices, faces, _, _ = measure.marching_cubes(
            padded_density,
            level=level,
            spacing=tuple(float(x) for x in spacing),
        )
    except ValueError:
        return _empty_mesh()

    vertices = vertices.astype(np.float32, copy=False) - spacing[None, :]
    faces = faces.astype(np.int32, copy=False)

    if col.shape[0] == pos.shape[0] and vertices.shape[0] > 0:
        _, nearest = cKDTree(pos).query(vertices, k=1)
        mesh_colors = col[np.asarray(nearest, dtype=np.int64)].astype(np.float32, copy=False)
    else:
        mesh_colors = np.full((vertices.shape[0], 3), 0.5, dtype=np.float32)

    return vertices, faces, mesh_colors


def dump_mesh_sequence(
    iter_idx: int,
    *,
    start_s: int = 7,
    step_s: int = 2,
    out_dir: str = "outputs",
    name: str | None = None,
    mesh_res: int = 64,
    mesh_sigma: float = 1.25,
    mesh_level_ratio: float = 0.08,
) -> None:
    """
    Dump marching-cubes meshes to `{out_dir}/{name}_mesh/####.ply` (binary).
    """
    print("Writing mesh data to disk...")

    kernels.forward()
    x_ = cfg.x.to_numpy()

    if name is None:
        name = f"iter{iter_idx:04d}"
    folder = os.path.join(out_dir, f"{name}_mesh/")
    os.makedirs(folder, exist_ok=True)

    for s in tqdm(
        range(start_s, cfg.steps, step_s),
        desc="Writing mesh data",
        unit="frame",
    ):
        colors = _unpack_hex_colors(_colors_for_frame(s))
        vertices, faces, mesh_colors = _particles_to_mesh(
            x_[s],
            colors,
            res=int(mesh_res),
            sigma=float(mesh_sigma),
            level_ratio=float(mesh_level_ratio),
        )
        fn = os.path.join(folder, f"{s:04d}.ply")
        _write_binary_ply(fn, vertices=vertices, faces=faces, colors=mesh_colors)
    tqdm.write("Done.")


def _project_isometric_xy(
    pts_3d: np.ndarray,
    *,
    yaw_deg: float = 45.0,
    pitch_deg: float = 35.264389682754654,  # asin(tan(30°)) for classic isometric
    center: np.ndarray | None = None,
    scale: float | None = None,
    margin: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Project Nx3 points to Nx2 using an isometric-like camera:
    - yaw around +y, then pitch around +x
    - output is normalized to [0,1] with a margin

    Returns:
    - pts_2d: (N,2) float32 in [0,1]
    - center_used: (3,) float32
    - scale_used: float
    """
    assert pts_3d.ndim == 2 and pts_3d.shape[1] == 3
    pts = pts_3d.astype(np.float32, copy=False)

    if center is None:
        center_used = pts.mean(axis=0)
    else:
        center_used = center.astype(np.float32, copy=False)

    yaw = np.deg2rad(yaw_deg).astype(np.float32)
    pitch = np.deg2rad(pitch_deg).astype(np.float32)

    cy, sy = float(np.cos(yaw)), float(np.sin(yaw))
    cp, sp = float(np.cos(pitch)), float(np.sin(pitch))

    # Right-handed rotations: yaw about +y, pitch about +x
    R_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    R_x = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    R = R_x @ R_y

    q = (pts - center_used[None, :]) @ R.T  # (N,3)
    p2 = q[:, :2]  # take rotated x-y as screen coords

    if scale is None:
        mn = p2.min(axis=0)
        mx = p2.max(axis=0)
        span = float(max(mx[0] - mn[0], mx[1] - mn[1], 1e-6))
        scale_used = (1.0 - 2.0 * margin) / span
    else:
        scale_used = float(scale)

    p2 = p2 * scale_used
    mn = p2.min(axis=0)
    mx = p2.max(axis=0)
    mid = 0.5 * (mn + mx)
    p2 = p2 - mid[None, :] + 0.5
    p2 = np.clip(p2, 0.0, 1.0)
    return p2.astype(np.float32, copy=False), center_used, scale_used


def visualize_rollout(
    iter_idx: int,
    stride: int = 8,
    save_folder: str | None = None,
    interactive: bool = True,
    window_res: Tuple[int, int] = (640, 640),
) -> None:
    """
    Visualize one full rollout (x-y projection).

    - Always runs forward() once to generate the full trajectory.
    - If interactive=True, opens a GUI window and plays the rollout.
    - If save_folder is not None, saves sampled frames as pngs:
      0000.png, 0008.png, ...
    """
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)

    gui = None
    if interactive:
        gui = ti.GUI(
            f"DiffMPM3D rollout (x-y) | iter {iter_idx}",
            window_res,
            background_color=0xFFFFFF,
        )

    kernels.forward()
    x_np = cfg.x.to_numpy()  # [T, N, 3]

    proj_center = x_np.reshape(-1, 3).mean(axis=0).astype(np.float32, copy=False)
    sample = x_np[:: max(1, int(stride))].reshape(-1, 3)
    _, _, proj_scale = _project_isometric_xy(sample, center=proj_center, scale=None)

    for s in range(0, cfg.steps, stride):
        pts, _, _ = _project_isometric_xy(x_np[s], center=proj_center, scale=proj_scale)
        colors = _colors_for_frame(s)
        if gui is not None:
            if hasattr(gui, "running") and not gui.running:
                break
            gui.circles(pos=pts, color=colors, radius=1.5)
            gui.show()

        if save_folder is not None:
            frame_gui = gui
            if frame_gui is None:
                frame_gui = ti.GUI(
                    f"DiffMPM3D rollout (x-y) | iter {iter_idx}",
                    window_res,
                    background_color=0xFFFFFF,
                )
            frame_gui.circles(pos=pts, color=colors, radius=1.5)
            frame_gui.show(os.path.join(save_folder, f"{s:04d}.png"))


def dump_particles_bin(
    iter_idx: int,
    *,
    start_s: int = 7,
    step_s: int = 2,
    out_dir: str = "outputs",
    name: str | None = None,
) -> None:
    """
    Dump frames to `{out_dir}/{name}_particle/####.bin` with the same binary layout as before.
    """
    print("Writing particle data to disk...")

    kernels.forward()
    x_ = cfg.x.to_numpy()
    v_ = cfg.v.to_numpy()
    particle_type_ = cfg.particle_type.to_numpy()
    actuation_ = cfg.actuation.to_numpy()
    actuator_id_ = cfg.actuator_id.to_numpy()

    if name is None:
        name = f"iter{iter_idx:04d}"
    folder = os.path.join(out_dir, f"{name}_particle/")
    os.makedirs(folder, exist_ok=True)

    for s in tqdm(
        range(start_s, cfg.steps, step_s),
        desc="Writing particle data",
        unit="frame",
    ):
        xs, ys, zs = [], [], []
        us, vs, ws = [], [], []
        cs = []
        for i in range(cfg.n_particles):
            xs.append(x_[s, i][0])
            ys.append(x_[s, i][1])
            zs.append(x_[s, i][2])
            us.append(v_[s, i][0])
            vs.append(v_[s, i][1])
            ws.append(v_[s, i][2])

            if particle_type_[i] == 0:
                r = 0.3
                g = 0.3
                b = 1.0
            else:
                if actuator_id_[i] != -1:
                    act = actuation_[s, actuator_id_[i]] * 0.5
                    r = 0.5 - act
                    g = 0.5 - abs(act)
                    b = 0.5 + act
                else:
                    r, g, b = 0.4, 0.4, 0.4

            cs.append(ti.rgb_to_hex((r, g, b)))

        data = np.array(xs + ys + zs + us + vs + ws + cs, dtype=np.float32)
        fn = os.path.join(folder, f"{s:04d}.bin")
        data.tofile(open(fn, "wb"))
    tqdm.write("Done.")

