"""
MPM 3D demo with mesh obstacles.
网格障碍的 SDF 由体素占据场 + scipy欧氏距离变换近似（快于逐点 signed_distance）；
非水密网格可能导致内外判断错误，需修复网格或改用更精确距离场。
"""
import os
import time
from math import pi
import numpy as np
import taichi as ti
from taichi.lang.matrix import Vector
from taichi.ui.utils import euler_to_vec, vec_to_euler

ti.init(arch=ti.gpu)


# dim, n_grid, steps, dt = 2, 128, 20, 2e-4
# dim, n_grid, steps, dt = 2, 256, 32, 1e-4
# dim, n_grid, steps, dt = 3, 32, 25, 4e-4
# dim, n_grid, steps, dt = 3, 64, 25, 2e-4
# dim, n_grid, steps, dt = 3, 128, 5, 1e-4
# 空间维度，每条边上的背景网格划分数，每一帧间调用多少次 substep，时间步长
dim, n_grid, steps, dt = 3, 64, 25, 4e-4

# n_particles = n_grid**dim // 2 ** (dim - 1)
n_particles = 1000  # hardcoded

print(f"dim: {dim}")
print(f"n_grid: {n_grid}")
print(f"steps: {steps}")
print(f"dt: {dt}")
print(f"n_particles: {n_particles}")

dx = 1 / n_grid
sdf_res = n_grid  # SDF 体素与 MPM 网格对齐；格心坐标 ((i+0.5)/sdf_res, ...)

p_rho = 1
p_vol = (dx * 0.5) ** 2
p_mass = p_vol * p_rho
GRAVITY = [0, -9.8, 0]
bound = 2
E = 1000  # Young's modulus
nu = 0.2  #  Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

# Obstacle AABB in world [0,1]^3（与下方 box mesh 一致；同步 preset 时仍写入，可作粗界）
obs_min = ti.Vector.field(3, dtype=float, shape=())
obs_max = ti.Vector.field(3, dtype=float, shape=())
obs_phi = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))


def load_obj_triangles(path):
    """Load OBJ：三角面；支持 f v/vt/vn。"""
    vertices = []
    faces = []
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
                idxs = []
                for corner in parts[1:]:
                    vi = int(corner.split("/")[0])
                    if vi < 0:
                        vi = len(vertices) + vi + 1
                    idxs.append(vi - 1)
                for i in range(1, len(idxs) - 1):
                    faces.append([idxs[0], idxs[i], idxs[i + 1]])
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int32)


def transform_mesh_for_preset(verts, scale, world_center):
    """模型包围盒中心对齐到原点后按 scale 缩放，再把该点平移到 world_center（与盒子障碍无关）。"""
    if len(verts) == 0:
        return verts
    c = (verts.min(axis=0) + verts.max(axis=0)) * 0.5
    verts = (verts - c) * float(scale)
    verts = verts + np.asarray(world_center, dtype=np.float64)
    return verts.astype(np.float64)


def build_sdf_box_volume(mn, mx, res):
    """轴对齐盒有符号距离：内部 phi<0，外部 phi>0（与 MPM 格心 (i+0.5)/res 对齐）。"""
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


def build_sdf_mesh_volume(verts, faces, res):
    """三角网格近似 SDF：voxelized(pitch=1/res) + is_filled + EDT，内部 phi<0。"""
    try:
        import trimesh
        from scipy import ndimage as ndi
    except ImportError as e:
        raise ImportError("Mesh obstacle SDF needs: pip install trimesh rtree scipy") from e
    mesh = trimesh.Trimesh(
        vertices=np.asarray(verts, dtype=np.float64),
        faces=np.asarray(faces, dtype=np.int64),
        process=True,
    )
    pitch = 1.0 / float(res)
    print(f"  voxelizing mesh for SDF (pitch={pitch:.4f})...")
    voxels = mesh.voxelized(pitch=pitch)
    ii = np.arange(res, dtype=np.float64)
    i, j, k = np.meshgrid(ii, ii, ii, indexing="ij")
    pts = np.stack([(i + 0.5) / res, (j + 0.5) / res, (k + 0.5) / res], axis=-1).reshape(-1, 3)
    occ = voxels.is_filled(pts).reshape(res, res, res)
    dist_out = ndi.distance_transform_edt(~occ) * pitch
    dist_in = ndi.distance_transform_edt(occ) * pitch
    return (dist_out - dist_in).astype(np.float32)


def build_obstacle_box_fields(cfg=None):
    """轴对齐盒子，底面 y=0；尺寸与 xz 中心由 preset cfg 指定。"""
    cfg = cfg or {}
    half_xz = float(cfg.get("half_xz", 0.2))
    h = float(cfg.get("height", 0.4))
    cx, cz = cfg.get("center_xz", (0.5, 0.5))
    cx, cz = float(cx), float(cz)
    mn = np.array([cx - half_xz, 0.0, cz - half_xz], dtype=np.float32)
    mx = np.array([cx + half_xz, h, cz + half_xz], dtype=np.float32)
    x0, y0, z0 = mn
    x1, y1, z1 = mx
    corners = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )
    tris = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int32,
    )
    obs_v = ti.Vector.field(3, float, shape=8)
    obs_f = ti.field(dtype=ti.i32, shape=36)
    # 必须用 RGBA 且 alpha=1，否则与半透明水箱同一批绘制时易被当成透明
    obs_c = ti.Vector.field(4, float, shape=8)
    obs_v.from_numpy(corners)
    obs_f.from_numpy(tris.reshape(-1))
    rgba = np.array([0.85, 0.52, 0.22, 1.0], dtype=np.float32)
    obs_c.from_numpy(np.tile(rgba, (8, 1)))
    mn_out = np.array([x0, y0, z0], dtype=np.float64)
    mx_out = np.array([x1, y1, z1], dtype=np.float64)
    return obs_v, obs_f, obs_c, mn_out, mx_out


def build_obstacle_mesh_fields(path, scale, world_center):
    """加载 OBJ，按 preset 的 scale / world_center（包围盒中心）变换后写入 Taichi。"""
    verts, faces = load_obj_triangles(path)
    if len(verts) == 0 or len(faces) == 0:
        raise RuntimeError(f"No geometry in {path}")
    verts = transform_mesh_for_preset(verts, scale, world_center)
    mn_out = verts.min(axis=0)
    mx_out = verts.max(axis=0)
    n_v = len(verts)
    obs_v = ti.Vector.field(3, dtype=float, shape=n_v)
    obs_f = ti.field(dtype=ti.i32, shape=len(faces) * 3)
    obs_c = ti.Vector.field(4, dtype=float, shape=n_v)
    obs_v.from_numpy(verts.astype(np.float32))
    obs_f.from_numpy(faces.reshape(-1))
    rgba = np.array([0.85, 0.52, 0.22, 1.0], dtype=np.float32)
    obs_c.from_numpy(np.tile(rgba, (n_v, 1)))
    return obs_v, obs_f, obs_c, mn_out, mx_out


# 与流体 preset 类似：障碍几何由列表选择；mesh 项使用「包围盒中心」与 scale，不依赖 cube 参数。
# old_main：居中后最长边约 2.0，scale=0.11 时最长边约 0.22，center y=0.05 时底面贴近 y=0（与旧脚本观感接近）。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OBSTACLE_PRESETS = [
    {
        "name": "Cube",
        "kind": "box",
        "half_xz": 0.2,
        "height": 0.4,
        "center_xz": (0.5, 0.5),
    },
    {
        "name": "Sphere",
        "kind": "mesh",
        "path": os.path.join(_SCRIPT_DIR, "assets", "sphere.obj"),
        # Same mesh pipeline as old_main: bbox-center -> scale -> translate to center.
        # You may want to tweak scale/center depending on the sphere.obj source units.
        "scale": 0.3,
        "center": (0.5, 0.2, 0.5),
    },
    {
        "name": "Old Main",
        "kind": "mesh",
        "path": os.path.join(_SCRIPT_DIR, "assets", "old_main_Meshy.obj"),
        "scale": 0.4,
        "center": (0.5, 0.2, 0.5),
    },
]

class FixedLookatCamera(ti.ui.Camera):
    """Orbit camera: mouse drag rotates around a fixed look-at; WASD/QE still translate."""

    def track_user_inputs_fixed_lookat(
        self,
        window,
        movement_speed: float = 1.0,
        yaw_speed: float = 2.0,
        pitch_speed: float = 2.0,
        hold_key=None,
        fixed_lookat=None,
    ):
        if fixed_lookat is not None:
            self.fixed_lookat = Vector(
                [float(fixed_lookat[0]), float(fixed_lookat[1]), float(fixed_lookat[2])]
            )
        elif not hasattr(self, "fixed_lookat"):
            self.fixed_lookat = Vector([float(self.curr_lookat[i]) for i in range(3)])

        front = (self.fixed_lookat - self.curr_position).normalized()
        position_change = Vector([0.0, 0.0, 0.0])
        left = self.curr_up.cross(front)
        up = self.curr_up

        if self.last_time is None:
            self.last_time = time.perf_counter_ns()
        time_elapsed = (time.perf_counter_ns() - self.last_time) * 1e-9
        self.last_time = time.perf_counter_ns()

        movement_speed *= time_elapsed * 60.0
        if window.is_pressed("w"):
            position_change += front * movement_speed
        if window.is_pressed("s"):
            position_change -= front * movement_speed
        if window.is_pressed("a"):
            position_change += left * movement_speed
        if window.is_pressed("d"):
            position_change -= left * movement_speed
        if window.is_pressed("e"):
            position_change += up * movement_speed
        if window.is_pressed("q"):
            position_change -= up * movement_speed
        self.position(*(self.curr_position + position_change))

        curr_mouse_x, curr_mouse_y = window.get_cursor_pos()

        if (hold_key is None) or window.is_pressed(hold_key):
            if (self.last_mouse_x is None) or (self.last_mouse_y is None):
                self.last_mouse_x, self.last_mouse_y = curr_mouse_x, curr_mouse_y
            dx = curr_mouse_x - self.last_mouse_x
            dy = curr_mouse_y - self.last_mouse_y

            to_lookat = self.fixed_lookat - self.curr_position
            distance = to_lookat.norm()
            if distance > 1e-6:
                yaw, pitch = vec_to_euler(to_lookat.normalized())

                yaw -= dx * yaw_speed * time_elapsed * 60.0
                pitch += dy * pitch_speed * time_elapsed * 60.0

                pitch_limit = pi / 2 * 0.99
                if pitch > pitch_limit:
                    pitch = pitch_limit
                elif pitch < -pitch_limit:
                    pitch = -pitch_limit

                orbit_front = euler_to_vec(yaw, pitch)
                self.position(*(self.fixed_lookat - orbit_front * distance))

        self.lookat(*self.fixed_lookat)
        self.last_mouse_x, self.last_mouse_y = curr_mouse_x, curr_mouse_y


def build_all_obstacle_geometries():
    out = []
    for p in OBSTACLE_PRESETS:
        if p["kind"] == "box":
            out.append(build_obstacle_box_fields(p))
        elif p["kind"] == "mesh":
            out.append(build_obstacle_mesh_fields(p["path"], p["scale"], p["center"]))
        else:
            raise ValueError(f"Unknown obstacle kind: {p.get('kind')}")
    return out


def build_unit_cube_tank_fields():
    """[0,1]^3 tank: 8 verts, 12 triangles, RGBA per vertex."""
    # corners: z=0 bottom, z=1 top; y up in sim
    corners = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=np.float32,
    )
    # 12 triangles (CCW from outside)
    tris = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # z=0
            [4, 5, 6],
            [4, 6, 7],  # z=1
            [0, 1, 5],
            [0, 5, 4],  # y=0
            [2, 3, 7],
            [2, 7, 6],  # y=1
            [0, 4, 7],
            [0, 7, 3],  # x=0
            [1, 2, 6],
            [1, 6, 5],  # x=1
        ],
        dtype=np.int32,
    )
    rgba = np.array([0.2, 0.5, 0.9, 0.18], dtype=np.float32)
    cols = np.tile(rgba, (8, 1))
    tank_v = ti.Vector.field(3, float, shape=8)
    tank_f = ti.field(int, shape=36)
    tank_c = ti.Vector.field(4, float, shape=8)
    tank_v.from_numpy(corners)
    tank_f.from_numpy(tris.reshape(-1))
    tank_c.from_numpy(cols)
    return tank_v, tank_f, tank_c


_obstacle_geoms = build_all_obstacle_geometries()
curr_obstacle_preset_id = 0


def build_all_obstacle_sdfs():
    phis = []
    for idx, preset in enumerate(OBSTACLE_PRESETS):
        mn, mx = _obstacle_geoms[idx][3], _obstacle_geoms[idx][4]
        if preset["kind"] == "box":
            phis.append(build_sdf_box_volume(mn, mx, sdf_res))
        elif preset["kind"] == "mesh":
            verts, faces = load_obj_triangles(preset["path"])
            verts = transform_mesh_for_preset(verts, preset["scale"], preset["center"])
            phis.append(build_sdf_mesh_volume(verts, faces, sdf_res))
        else:
            raise ValueError(f"Unknown obstacle kind: {preset.get('kind')}")
    return phis


obstacle_sdf_numpy = build_all_obstacle_sdfs()


def sync_obstacle_aabb():
    """当前障碍 preset 的 AABB 写入 obs_min/max（调试用 / 粗界）。"""
    mn, mx = _obstacle_geoms[curr_obstacle_preset_id][3], _obstacle_geoms[curr_obstacle_preset_id][4]
    obs_min[None] = ti.Vector([float(mn[0]), float(mn[1]), float(mn[2])])
    obs_max[None] = ti.Vector([float(mx[0]), float(mx[1]), float(mx[2])])


def sync_obstacle_collision():
    """同步 AABB 与当前 preset 的 SDF 体。"""
    sync_obstacle_aabb()
    obs_phi.from_numpy(obstacle_sdf_numpy[curr_obstacle_preset_id])


sync_obstacle_collision()

tank_vertices, tank_indices, tank_colors = build_unit_cube_tank_fields()

F_x = ti.Vector.field(dim, float, n_particles)
F_v = ti.Vector.field(dim, float, n_particles)
F_C = ti.Matrix.field(dim, dim, float, n_particles)
F_dg = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)  # deformation gradient
F_Jp = ti.field(float, n_particles)

F_colors = ti.Vector.field(4, float, n_particles)
F_colors_random = ti.Vector.field(4, float, n_particles)
F_materials = ti.field(int, n_particles)
F_grid_v = ti.Vector.field(dim, float, (n_grid,) * dim)
F_grid_m = ti.field(float, (n_grid,) * dim)
F_used = ti.field(int, n_particles)

neighbour = (3,) * dim

WATER = 0
JELLY = 1
SNOW = 2


@ti.func
def obs_phi_sample_trilinear(p):
    """三线性采样 obs_phi；内部负、外部正。将 p 限制在格心覆盖域内避免越界。"""
    R = float(n_grid)
    h = 0.5 / R
    pc0 = ti.min(ti.max(p[0], h), 1.0 - h)
    pc1 = ti.min(ti.max(p[1], h), 1.0 - h)
    pc2 = ti.min(ti.max(p[2], h), 1.0 - h)
    u0 = pc0 * R - 0.5
    u1 = pc1 * R - 0.5
    u2 = pc2 * R - 0.5
    i0 = ti.cast(ti.floor(u0), ti.i32)
    j0 = ti.cast(ti.floor(u1), ti.i32)
    k0 = ti.cast(ti.floor(u2), ti.i32)
    f0 = u0 - ti.cast(i0, ti.f32)
    f1 = u1 - ti.cast(j0, ti.f32)
    f2 = u2 - ti.cast(k0, ti.f32)
    i1 = ti.min(i0 + 1, n_grid - 1)
    j1 = ti.min(j0 + 1, n_grid - 1)
    k1 = ti.min(k0 + 1, n_grid - 1)
    i0 = ti.max(0, ti.min(i0, n_grid - 1))
    j0 = ti.max(0, ti.min(j0, n_grid - 1))
    k0 = ti.max(0, ti.min(k0, n_grid - 1))

    c000 = obs_phi[i0, j0, k0]
    c001 = obs_phi[i0, j0, k1]
    c010 = obs_phi[i0, j1, k0]
    c011 = obs_phi[i0, j1, k1]
    c100 = obs_phi[i1, j0, k0]
    c101 = obs_phi[i1, j0, k1]
    c110 = obs_phi[i1, j1, k0]
    c111 = obs_phi[i1, j1, k1]
    c00 = c000 * (1.0 - f2) + c001 * f2
    c01 = c010 * (1.0 - f2) + c011 * f2
    c10 = c100 * (1.0 - f2) + c101 * f2
    c11 = c110 * (1.0 - f2) + c111 * f2
    c0 = c00 * (1.0 - f1) + c01 * f1
    c1 = c10 * (1.0 - f1) + c11 * f1
    return c0 * (1.0 - f0) + c1 * f0


@ti.func
def obs_phi_grad_trilinear(p):
    hh = 0.5 / float(n_grid)
    e0 = ti.Vector([hh, 0.0, 0.0])
    e1 = ti.Vector([0.0, hh, 0.0])
    e2 = ti.Vector([0.0, 0.0, hh])
    g0 = (obs_phi_sample_trilinear(p + e0) - obs_phi_sample_trilinear(p - e0)) / (2.0 * hh)
    g1 = (obs_phi_sample_trilinear(p + e1) - obs_phi_sample_trilinear(p - e1)) / (2.0 * hh)
    g2 = (obs_phi_sample_trilinear(p + e2) - obs_phi_sample_trilinear(p - e2)) / (2.0 * hh)
    return ti.Vector([g0, g1, g2])


@ti.kernel
def substep(g_x: float, g_y: float, g_z: float):
    for I in ti.grouped(F_grid_m):
        F_grid_v[I] = ti.zero(F_grid_v[I])
        F_grid_m[I] = 0
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        F_dg[p] = (ti.Matrix.identity(float, 3) + dt * F_C[p]) @ F_dg[p]  # deformation gradient update
        # Hardening coefficient: snow gets harder when compressed
        h = ti.exp(10 * (1.0 - F_Jp[p]))
        if F_materials[p] == JELLY:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if F_materials[p] == WATER:  # liquid
            mu = 0.0

        U, sig, V = ti.svd(F_dg[p])
        J = 1.0
        for d in ti.static(range(3)):
            new_sig = sig[d, d]
            if F_materials[p] == SNOW:  # Snow
                new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity
            F_Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if F_materials[p] == WATER:
            # Reset deformation gradient to avoid numerical instability
            new_F = ti.Matrix.identity(float, 3)
            new_F[0, 0] = J
            F_dg[p] = new_F
        elif F_materials[p] == SNOW:
            # Reconstruct elastic deformation gradient after plasticity
            F_dg[p] = U @ sig @ V.transpose()
        stress = 2 * mu * (F_dg[p] - U @ V.transpose()) @ F_dg[p].transpose() + ti.Matrix.identity(
            float, 3
        ) * la * J * (J - 1)
        stress = (-dt * p_vol * 4) * stress / dx**2
        affine = stress + p_mass * F_C[p]

        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            F_grid_v[base + offset] += weight * (p_mass * F_v[p] + affine @ dpos)
            F_grid_m[base + offset] += weight * p_mass
    for I in ti.grouped(F_grid_m):
        if F_grid_m[I] > 0:
            F_grid_v[I] /= F_grid_m[I]
        F_grid_v[I] += dt * ti.Vector([g_x, g_y, g_z])
        cond = (I < bound) & (F_grid_v[I] < 0) | (I > n_grid - bound) & (F_grid_v[I] > 0)
        F_grid_v[I] = ti.select(cond, 0, F_grid_v[I])
        phi = obs_phi[I[0], I[1], I[2]]
        F_grid_v[I] = ti.select(phi < 0, ti.Vector([0.0, 0.0, 0.0]), F_grid_v[I])
    ti.loop_config(block_dim=n_grid)
    for p in F_x:
        if F_used[p] == 0:
            continue
        Xp = F_x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.zero(F_v[p])
        new_C = ti.zero(F_C[p])
        for offset in ti.static(ti.grouped(ti.ndrange(*neighbour))):
            dpos = (offset - fx) * dx
            weight = 1.0
            for i in ti.static(range(dim)):
                weight *= w[offset[i]][i]
            g_v = F_grid_v[base + offset]
            new_v += weight * g_v
            new_C += 4 * weight * g_v.outer_product(dpos) / dx**2
        F_v[p] = new_v
        F_x[p] += dt * F_v[p]
        F_C[p] = new_C


@ti.kernel
def resolve_particles_obstacle():
    eps = 1e-4 / n_grid
    for p in F_x:
        if F_used[p] == 0:
            continue
        x = F_x[p]
        phi_p = obs_phi_sample_trilinear(x)
        if phi_p < 0:
            g = obs_phi_grad_trilinear(x)
            gl = g.norm()
            n = ti.Vector([0.0, 1.0, 0.0])
            if gl > 1e-8:
                n = g / gl
            F_x[p] = x + (-phi_p + eps) * n
            vn = F_v[p].dot(n)
            F_v[p] -= ti.min(vn, 0.0) * n


class CubeVolume:
    def __init__(self, minimum, size, material):
        self.minimum = minimum
        self.size = size
        self.volume = self.size.x * self.size.y * self.size.z
        self.material = material


@ti.kernel
def init_cube_vol(
    first_par: int,
    last_par: int,
    x_begin: float,
    y_begin: float,
    z_begin: float,
    x_size: float,
    y_size: float,
    z_size: float,
    material: int,
):
    for i in range(first_par, last_par):
        F_x[i] = ti.Vector([ti.random() for i in range(dim)]) * ti.Vector([x_size, y_size, z_size]) + ti.Vector(
            [x_begin, y_begin, z_begin]
        )
        F_Jp[i] = 1
        F_dg[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_v[i] = ti.Vector([0.0, 0.0, 0.0])
        F_materials[i] = material
        F_colors_random[i] = ti.Vector([ti.random(), ti.random(), ti.random(), ti.random()])
        F_used[i] = 1


@ti.kernel
def set_all_unused():
    for p in F_used:
        F_used[p] = 0
        # basically throw them away so they aren't rendered
        F_x[p] = ti.Vector([533799.0, 533799.0, 533799.0])
        F_Jp[p] = 1
        F_dg[p] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        F_C[p] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        F_v[p] = ti.Vector([0.0, 0.0, 0.0])


def init_vols(vols):
    set_all_unused()
    total_vol = 0
    for v in vols:
        total_vol += v.volume

    next_p = 0
    for i, v in enumerate(vols):
        v = vols[i]
        if isinstance(v, CubeVolume):
            par_count = int(v.volume / total_vol * n_particles)
            if i == len(vols) - 1:  # this is the last volume, so use all remaining particles
                par_count = n_particles - next_p
            init_cube_vol(next_p, next_p + par_count, *v.minimum, *v.size, v.material)
            next_p += par_count
        else:
            raise Exception("???")


@ti.kernel
def set_color_by_material(mat_color: ti.types.ndarray()):
    for i in range(n_particles):
        mat = F_materials[i]
        F_colors[i] = ti.Vector([mat_color[mat, 0], mat_color[mat, 1], mat_color[mat, 2], 1.0])


print("Loading presets...this might take a minute")

presets = [
    [
        CubeVolume(ti.Vector([0.55, 0.45, 0.55]), ti.Vector([0.4, 0.5, 0.4]), WATER),
    ],
    [
        CubeVolume(ti.Vector([0.05, 0.05, 0.05]), ti.Vector([0.3, 0.4, 0.3]), WATER),
        CubeVolume(ti.Vector([0.65, 0.05, 0.65]), ti.Vector([0.3, 0.4, 0.3]), WATER),
    ],
    [
        CubeVolume(ti.Vector([0.2, 0.5, 0.2]), ti.Vector([0.6, 0.4, 0.6]), SNOW),
    ],
    [
        CubeVolume(ti.Vector([0.6, 0.05, 0.6]), ti.Vector([0.25, 0.25, 0.25]), WATER),
        CubeVolume(ti.Vector([0.35, 0.35, 0.35]), ti.Vector([0.25, 0.25, 0.25]), SNOW),
        CubeVolume(ti.Vector([0.05, 0.6, 0.05]), ti.Vector([0.25, 0.25, 0.25]), JELLY),
    ],
]
preset_names = [
    "Single Dam Break",
    "Double Dam Break",
    "Snow",
    "Water Snow Jelly",
]

curr_preset_id = 0

paused = False

use_random_colors = False
particles_radius = 0.01

material_colors = [(0.1, 0.6, 0.9), (0.93, 0.33, 0.23), (1.0, 1.0, 1.0)]


def init():
    global paused
    init_vols(presets[curr_preset_id])


init()

res = (1280, 720)
window = ti.ui.Window("Real MPM 3D", res, vsync=True)

canvas = window.get_canvas()
gui = window.get_gui()
scene = window.get_scene()
camera = FixedLookatCamera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.5, 0.5)
camera.fov(55)


def show_options():
    global use_random_colors
    global paused
    global particles_radius
    global curr_preset_id
    global curr_obstacle_preset_id

    with gui.sub_window("Presets", 0.05, 0.1, 0.2, 0.15) as w:
        old_preset = curr_preset_id
        for i in range(len(presets)):
            if w.checkbox(preset_names[i], curr_preset_id == i):
                curr_preset_id = i
        if curr_preset_id != old_preset:
            init()
            paused = True

    with gui.sub_window("Obstacle", 0.05, 0.26, 0.2, 0.12) as w:
        old_obs = curr_obstacle_preset_id
        for i in range(len(OBSTACLE_PRESETS)):
            name = OBSTACLE_PRESETS[i]["name"]
            if w.checkbox(name, curr_obstacle_preset_id == i):
                curr_obstacle_preset_id = i
        if curr_obstacle_preset_id != old_obs:
            sync_obstacle_collision()

    with gui.sub_window("Gravity", 0.05, 0.4, 0.2, 0.1) as w:
        GRAVITY[0] = w.slider_float("x", GRAVITY[0], -10, 10)
        GRAVITY[1] = w.slider_float("y", GRAVITY[1], -10, 10)
        GRAVITY[2] = w.slider_float("z", GRAVITY[2], -10, 10)

    with gui.sub_window("Options", 0.05, 0.55, 0.2, 0.35) as w:
        use_random_colors = w.checkbox("use_random_colors", use_random_colors)
        if not use_random_colors:
            material_colors[WATER] = w.color_edit_3("water color", material_colors[WATER])
            material_colors[SNOW] = w.color_edit_3("snow color", material_colors[SNOW])
            material_colors[JELLY] = w.color_edit_3("jelly color", material_colors[JELLY])
            set_color_by_material(np.array(material_colors, dtype=np.float32))
        particles_radius = w.slider_float("particles radius ", particles_radius, 0, 0.1)
        if w.button("restart"):
            init()
        if paused:
            if w.button("Continue"):
                paused = False
        else:
            if w.button("Pause"):
                paused = True


def render():
    camera.track_user_inputs_fixed_lookat(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    # 非零环境光：否则仅靠点光时背光面几乎全黑，容易被误认为“没上色/透明”
    scene.ambient_light((0.18, 0.18, 0.18))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.75, 0.75, 0.75))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.75, 0.75, 0.75))
    # 自下侧补光，照亮障碍物朝下的面
    scene.point_light(pos=(0.5, 0.05, 0.5), color=(0.35, 0.35, 0.35))

    # 先画不透明障碍，再粒子，最后半透明水箱，避免透明 pass 影响不透明深度/混合
    ov, ofi, oc, _, __ = _obstacle_geoms[curr_obstacle_preset_id]
    scene.mesh(ov, indices=ofi, per_vertex_color=oc, two_sided=True)
    colors_used = F_colors_random if use_random_colors else F_colors
    scene.particles(F_x, per_vertex_color=colors_used, radius=particles_radius)
    scene.mesh(
        tank_vertices,
        indices=tank_indices,
        per_vertex_color=tank_colors,
        two_sided=True,
    )

    canvas.scene(scene)


def main():
    frame_id = 0

    while window.running:
        frame_id += 1
        frame_id = frame_id % 256

        if not paused:
            for _ in range(steps):
                substep(*GRAVITY)
                resolve_particles_obstacle()

        render()
        show_options()
        window.show()


if __name__ == "__main__":
    main()
