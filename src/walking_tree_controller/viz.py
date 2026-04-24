import os
from typing import Tuple

import numpy as np
import taichi as ti

import config as cfg
import kernels as kernels


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


def dump_particles_bin(iter_idx: int, *, start_s: int = 7, step_s: int = 2) -> None:
    """
    Dump frames to `mpm3d/iterXXXX/####.bin` with the same binary layout as before.
    """
    print("Writing particle data to disk...")
    print("(Please be patient)...")

    kernels.forward()
    x_ = cfg.x.to_numpy()
    v_ = cfg.v.to_numpy()
    particle_type_ = cfg.particle_type.to_numpy()
    actuation_ = cfg.actuation.to_numpy()
    actuator_id_ = cfg.actuator_id.to_numpy()

    folder = f"outputs/mpm3d/iter{iter_idx:04d}/"
    os.makedirs(folder, exist_ok=True)

    for s in range(start_s, cfg.steps, step_s):
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
        fn = f"{folder}/{s:04d}.bin"
        data.tofile(open(fn, "wb"))
        print(".", end="")
    print()

