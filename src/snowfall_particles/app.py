from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import taichi as ti
from tqdm import tqdm

from snowfall_particles.config import Config
from snowfall_particles.mpm.solver import MPMSolver, MaterialParams
from snowfall_particles.scene.presets import CubeVolume, load_fluid_presets, load_obstacle_presets
from snowfall_particles.sdf.builders import build_sdf_box_volume
from snowfall_particles.sdf.cache import MeshSdfRequest, load_or_build_mesh_sdf
from snowfall_particles.ui.camera import FixedLookatCamera


def build_unit_cube_tank_fields():
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
    rgba = np.array([0.2, 0.5, 0.9, 0.18], dtype=np.float32)
    cols = np.tile(rgba, (8, 1))
    tank_v = ti.Vector.field(3, float, shape=8)
    tank_f = ti.field(int, shape=36)
    tank_c = ti.Vector.field(4, float, shape=8)
    tank_v.from_numpy(corners)
    tank_f.from_numpy(tris.reshape(-1))
    tank_c.from_numpy(cols)
    return tank_v, tank_f, tank_c


def build_obstacle_box_fields(half_xz: float, height: float, center_xz: tuple[float, float]):
    cx, cz = float(center_xz[0]), float(center_xz[1])
    mn = np.array([cx - half_xz, 0.0, cz - half_xz], dtype=np.float32)
    mx = np.array([cx + half_xz, height, cz + half_xz], dtype=np.float32)
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
    obs_c = ti.Vector.field(4, float, shape=8)
    obs_v.from_numpy(corners)
    obs_f.from_numpy(tris.reshape(-1))
    rgba = np.array([0.85, 0.52, 0.22, 1.0], dtype=np.float32)
    obs_c.from_numpy(np.tile(rgba, (8, 1)))
    return obs_v, obs_f, obs_c, mn.astype(np.float64), mx.astype(np.float64)


def build_obstacle_mesh_fields(mesh_path: Path, scale: float, center: tuple[float, float, float]):
    from geometry.mesh_io import load_obj_triangles
    from geometry.transform import transform_mesh_for_preset

    verts, faces = load_obj_triangles(str(mesh_path))
    if len(verts) == 0 or len(faces) == 0:
        raise RuntimeError(f"No geometry in {mesh_path}")
    verts = transform_mesh_for_preset(verts, scale, center)
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


@dataclass
class ObstacleRuntime:
    preset_name: str
    kind: str
    geom: tuple  # (v,f,c,mn,mx)
    sdf_numpy: np.ndarray | None = None


class MpmApp:
    def __init__(self, cfg: Config, *, headless: bool = False):
        self.cfg = cfg
        self.headless = bool(headless)
        mat = MaterialParams(p_rho=cfg.material.p_rho, E=cfg.material.E, nu=cfg.material.nu)
        self.solver = MPMSolver(
            dim=cfg.sim.dim,
            n_grid=cfg.sim.n_grid,
            steps=cfg.sim.steps,
            dt=cfg.sim.dt,
            n_particles=cfg.sim.n_particles,
            gravity=cfg.sim.gravity,
            bound=cfg.sim.bound,
            material=mat,
        )
        self.sdf_res = int(cfg.sdf.res)
        self.cache_enabled = bool(cfg.sdf.cache_enabled)

        self.material_colors = [(0.1, 0.6, 0.9), (0.93, 0.33, 0.23), (1.0, 1.0, 1.0)]
        self.use_random_colors = False
        self.paused = False
        self.particles_radius = float(cfg.render.particles_radius)

        self.preset_names, self.presets = load_fluid_presets(cfg.particles)
        self.curr_preset_id = 0

        self.obstacle_presets = load_obstacle_presets(cfg.obstacles)
        self.curr_obstacle_preset_id = 0
        self._obstacles: list[ObstacleRuntime] = self._build_obstacle_geometries()

        if self.headless:
            self.tank_vertices = self.tank_indices = self.tank_colors = None
            self.window = self.canvas = self.gui = self.scene = None
            self.camera = None
        else:
            self.tank_vertices, self.tank_indices, self.tank_colors = build_unit_cube_tank_fields()
            self._init_window()
            self._init_scene()
            self.init_sim()
            self.sync_obstacle_collision()

    def _init_window(self):
        res = self.cfg.render.resolution
        self.window = ti.ui.Window("Real MPM 3D", res, vsync=bool(self.cfg.render.vsync))
        self.canvas = self.window.get_canvas()
        self.gui = self.window.get_gui()
        self.scene = self.window.get_scene()

    def _init_scene(self):
        self.camera = FixedLookatCamera()
        pos = self.cfg.render.camera.position
        look = self.cfg.render.camera.lookat
        self.camera.position(*pos)
        self.camera.lookat(*look)
        self.camera.fov(float(self.cfg.render.camera.fov))

    def _build_obstacle_geometries(self) -> list[ObstacleRuntime]:
        out: list[ObstacleRuntime] = []
        for p in self.obstacle_presets:
            if p.kind == "box":
                geom = build_obstacle_box_fields(float(p.half_xz), float(p.height), (p.center_xz[0], p.center_xz[1]))  # type: ignore[index]
            else:
                mesh_path = (self.cfg.base_dir / str(p.path)).resolve()
                geom = build_obstacle_mesh_fields(mesh_path, float(p.scale), (p.center[0], p.center[1], p.center[2]))  # type: ignore[index]
            out.append(ObstacleRuntime(preset_name=p.name, kind=p.kind, geom=geom))
        return out

    def _ensure_obstacle_sdf(self, obs: ObstacleRuntime, preset_idx: int) -> np.ndarray:
        if obs.sdf_numpy is not None:
            return obs.sdf_numpy
        preset = self.obstacle_presets[preset_idx]
        mn, mx = obs.geom[3], obs.geom[4]
        if preset.kind == "box":
            phi = build_sdf_box_volume(mn, mx, self.sdf_res)
        else:
            mesh_path = (self.cfg.base_dir / str(preset.path)).resolve()
            req = MeshSdfRequest(
                mesh_path=mesh_path,
                sdf_res=self.sdf_res,
                scale=float(preset.scale),
                center=(float(preset.center[0]), float(preset.center[1]), float(preset.center[2])),
            )
            phi = load_or_build_mesh_sdf(req, cache_enabled=self.cache_enabled)
        obs.sdf_numpy = phi
        return phi

    def sync_obstacle_aabb(self):
        mn, mx = self._obstacles[self.curr_obstacle_preset_id].geom[3], self._obstacles[self.curr_obstacle_preset_id].geom[4]
        self.solver.obs_min[None] = ti.Vector([float(mn[0]), float(mn[1]), float(mn[2])])
        self.solver.obs_max[None] = ti.Vector([float(mx[0]), float(mx[1]), float(mx[2])])

    def sync_obstacle_collision(self):
        self.sync_obstacle_aabb()
        obs = self._obstacles[self.curr_obstacle_preset_id]
        phi = self._ensure_obstacle_sdf(obs, self.curr_obstacle_preset_id)
        self.solver.obs_phi.from_numpy(phi)

    def init_sim(self):
        self.solver.set_all_unused()

        vols = self.presets[self.curr_preset_id]
        total_vol = sum(v.volume for v in vols) if vols else 1.0
        runtime_vols = []

        material_map = {"water": MPMSolver.WATER, "jelly": MPMSolver.JELLY, "snow": MPMSolver.SNOW}
        for v in vols:
            runtime_vols.append(
                {
                    "minimum": (float(v.minimum[0]), float(v.minimum[1]), float(v.minimum[2])),
                    "size": (float(v.size[0]), float(v.size[1]), float(v.size[2])),
                    "volume": float(v.volume),
                    "material_id": int(material_map[v.material]),
                }
            )
        self.solver.init_vols(runtime_vols)

    def show_options(self):
        with self.gui.sub_window("Presets", 0.05, 0.1, 0.2, 0.15) as w:
            old_preset = self.curr_preset_id
            for i in range(len(self.presets)):
                if w.checkbox(self.preset_names[i], self.curr_preset_id == i):
                    self.curr_preset_id = i
            if self.curr_preset_id != old_preset:
                self.init_sim()
                self.paused = True

        with self.gui.sub_window("Obstacle", 0.05, 0.26, 0.2, 0.12) as w:
            old_obs = self.curr_obstacle_preset_id
            for i in range(len(self.obstacle_presets)):
                if w.checkbox(self.obstacle_presets[i].name, self.curr_obstacle_preset_id == i):
                    self.curr_obstacle_preset_id = i
            if self.curr_obstacle_preset_id != old_obs:
                self.sync_obstacle_collision()

        with self.gui.sub_window("Gravity", 0.05, 0.4, 0.2, 0.1) as w:
            self.solver.gravity[0] = w.slider_float("x", self.solver.gravity[0], -10, 10)
            self.solver.gravity[1] = w.slider_float("y", self.solver.gravity[1], -10, 10)
            self.solver.gravity[2] = w.slider_float("z", self.solver.gravity[2], -10, 10)

        with self.gui.sub_window("Options", 0.05, 0.55, 0.2, 0.35) as w:
            self.use_random_colors = w.checkbox("use_random_colors", self.use_random_colors)
            if not self.use_random_colors:
                self.material_colors[MPMSolver.WATER] = w.color_edit_3("water color", self.material_colors[MPMSolver.WATER])
                self.material_colors[MPMSolver.SNOW] = w.color_edit_3("snow color", self.material_colors[MPMSolver.SNOW])
                self.material_colors[MPMSolver.JELLY] = w.color_edit_3("jelly color", self.material_colors[MPMSolver.JELLY])
                self.solver.set_color_by_material(np.array(self.material_colors, dtype=np.float32))
            self.particles_radius = w.slider_float("particles radius ", self.particles_radius, 0, 0.1)
            if w.button("restart"):
                self.init_sim()
            if self.paused:
                if w.button("Continue"):
                    self.paused = False
            else:
                if w.button("Pause"):
                    self.paused = True

    def render(self):
        self.camera.track_user_inputs_fixed_lookat(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)

        self.scene.ambient_light(self.cfg.render.ambient_light)
        for pl in self.cfg.render.point_lights:
            self.scene.point_light(pos=pl["pos"], color=pl["color"])

        ov, ofi, oc, _, __ = self._obstacles[self.curr_obstacle_preset_id].geom
        self.scene.mesh(ov, indices=ofi, per_vertex_color=oc, two_sided=True)
        colors_used = self.solver.F_colors_random if self.use_random_colors else self.solver.F_colors
        self.scene.particles(self.solver.F_x, per_vertex_color=colors_used, radius=self.particles_radius)
        self.scene.mesh(
            self.tank_vertices,
            indices=self.tank_indices,
            per_vertex_color=self.tank_colors,
            two_sided=True,
        )
        self.canvas.scene(self.scene)

    def run(self):
        if self.headless:
            raise RuntimeError("run() requires GUI mode (headless=False)")
        while self.window.running:
            if not self.paused:
                self.solver.step_frame()
            self.render()
            self.show_options()
            self.window.show()

    def run_offline_export(
        self,
        *,
        output_path: Path,
        duration_sim_s: float,
        particle_preset_idx: int,
        obstacle_preset_idx: int,
    ) -> None:
        if not self.headless:
            raise RuntimeError("run_offline_export requires headless=True")
        self.curr_preset_id = int(particle_preset_idx)
        self.curr_obstacle_preset_id = int(obstacle_preset_idx)
        self.init_sim()
        self.sync_obstacle_collision()

        dt_frame = float(self.cfg.sim.steps) * float(self.cfg.sim.dt)
        n_particles = int(self.cfg.sim.n_particles)
        max_steps = max(1, int(math.ceil(duration_sim_s / dt_frame)))
        buf_len = max_steps + 1
        positions = np.empty((buf_len, n_particles, 3), dtype=np.float32)

        used = self.solver.F_used.to_numpy().astype(np.int8, copy=False)
        positions[0] = self.solver.F_x.to_numpy().astype(np.float32, copy=False)

        row = 1
        with tqdm(
            total=max_steps,
            desc="Offline export",
            unit="frame",
            file=sys.stderr,
        ) as pbar:
            # Fixed max_steps iterations (do not gate on float accum vs duration):
            # floating-point drift can leave accum < duration after max_steps steps
            # and cause an extra write past positions.shape[0]-1.
            for _ in range(max_steps):
                self.solver.step_frame()
                positions[row] = self.solver.F_x.to_numpy().astype(np.float32, copy=False)
                row += 1
                pbar.update(1)

        positions_out = positions[:row].copy()
        particle_name = self.preset_names[self.curr_preset_id]
        obstacle_name = self.obstacle_presets[self.curr_obstacle_preset_id].name

        np.savez_compressed(
            output_path,
            positions=positions_out,
            used=used,
            dt_frame=np.float32(dt_frame),
            simulation_duration_seconds=np.float32(duration_sim_s),
            particle_preset=np.array(particle_name),
            obstacle_preset=np.array(obstacle_name),
        )


def create_app(cfg: Config, *, headless: bool = False) -> MpmApp:
    return MpmApp(cfg, headless=headless)

