from dataclasses import dataclass
from math import cos, radians, sin
from typing import Any

import numpy as np
import taichi as ti


@dataclass(frozen=True)
class MaterialParams:
    p_rho: float
    E: float
    nu: float


@dataclass(frozen=True)
class ParticleMotionParams:
    lateral_force_probability: float
    lateral_force_angle_degrees: float
    lateral_force_min: float
    lateral_force_max: float
    max_fall_speed: float


@ti.data_oriented
class MPMSolver:
    WATER = 0
    JELLY = 1
    SNOW = 2

    def __init__(
        self,
        *,
        dim: int,
        n_grid: int,
        steps: int,
        dt: float,
        n_particles: int,
        max_particles: int | None = None,
        gravity: tuple[float, float, float],
        bound: int,
        material: MaterialParams,
        particle_motion: ParticleMotionParams | None = None,
    ):
        self.dim = int(dim)
        self.n_grid = int(n_grid)
        self.steps = int(steps)
        self.dt = float(dt)
        self.initial_n_particles = int(n_particles)
        self.n_particles = max(self.initial_n_particles, int(max_particles or n_particles))
        self.gravity = [float(gravity[0]), float(gravity[1]), float(gravity[2])]
        self.bound = int(bound)
        if particle_motion is None:
            particle_motion = ParticleMotionParams(0.0, 0.0, 0.0, 0.0, 0.0)
        self.lateral_force_probability = float(particle_motion.lateral_force_probability)
        self.lateral_force_min = float(particle_motion.lateral_force_min)
        self.lateral_force_max = float(particle_motion.lateral_force_max)
        self.max_fall_speed = float(particle_motion.max_fall_speed)
        lateral_force_angle = radians(float(particle_motion.lateral_force_angle_degrees))
        self.lateral_force_dir_x = cos(lateral_force_angle)
        self.lateral_force_dir_z = sin(lateral_force_angle)

        self.dx = 1.0 / float(self.n_grid)
        self.p_rho = float(material.p_rho)
        self.p_vol = (self.dx * 0.5) ** 2
        self.p_mass = self.p_vol * self.p_rho

        E = float(material.E)
        nu = float(material.nu)
        self.mu_0 = E / (2 * (1 + nu))
        self.lambda_0 = E * nu / ((1 + nu) * (1 - 2 * nu))

        self._alloc_fields()

    def _alloc_fields(self):
        dim = self.dim
        n_grid = self.n_grid
        n_particles = self.n_particles

        self.obs_min = ti.Vector.field(3, dtype=float, shape=())
        self.obs_max = ti.Vector.field(3, dtype=float, shape=())
        self.obs_phi = ti.field(dtype=ti.f32, shape=(n_grid, n_grid, n_grid))

        self.F_x = ti.Vector.field(dim, float, n_particles)
        self.F_v = ti.Vector.field(dim, float, n_particles)
        self.F_C = ti.Matrix.field(dim, dim, float, n_particles)
        self.F_dg = ti.Matrix.field(3, 3, dtype=float, shape=n_particles)
        self.F_Jp = ti.field(float, n_particles)

        self.F_colors = ti.Vector.field(4, float, n_particles)
        self.F_colors_random = ti.Vector.field(4, float, n_particles)
        self.F_materials = ti.field(int, n_particles)
        self.F_grid_v = ti.Vector.field(dim, float, (n_grid,) * dim)
        self.F_grid_m = ti.field(float, (n_grid,) * dim)
        self.F_used = ti.field(int, n_particles)
        self.F_next_particle = ti.field(dtype=ti.i32, shape=())

        self.neighbour = (3,) * dim

        # Per-frame stats (computed on GPU/CPU via kernel; cheap to read from Python).
        self.S_active = ti.field(dtype=ti.i32, shape=())
        self.S_speed_rms = ti.field(dtype=ti.f32, shape=())
        self.S_speed_max = ti.field(dtype=ti.f32, shape=())
        self.S_y_min = ti.field(dtype=ti.f32, shape=())
        self.S_y_max = ti.field(dtype=ti.f32, shape=())

    @ti.func
    def obs_phi_sample_trilinear(self, p):
        R = float(self.n_grid)
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
        i1 = ti.min(i0 + 1, self.n_grid - 1)
        j1 = ti.min(j0 + 1, self.n_grid - 1)
        k1 = ti.min(k0 + 1, self.n_grid - 1)
        i0 = ti.max(0, ti.min(i0, self.n_grid - 1))
        j0 = ti.max(0, ti.min(j0, self.n_grid - 1))
        k0 = ti.max(0, ti.min(k0, self.n_grid - 1))

        c000 = self.obs_phi[i0, j0, k0]
        c001 = self.obs_phi[i0, j0, k1]
        c010 = self.obs_phi[i0, j1, k0]
        c011 = self.obs_phi[i0, j1, k1]
        c100 = self.obs_phi[i1, j0, k0]
        c101 = self.obs_phi[i1, j0, k1]
        c110 = self.obs_phi[i1, j1, k0]
        c111 = self.obs_phi[i1, j1, k1]
        c00 = c000 * (1.0 - f2) + c001 * f2
        c01 = c010 * (1.0 - f2) + c011 * f2
        c10 = c100 * (1.0 - f2) + c101 * f2
        c11 = c110 * (1.0 - f2) + c111 * f2
        c0 = c00 * (1.0 - f1) + c01 * f1
        c1 = c10 * (1.0 - f1) + c11 * f1
        return c0 * (1.0 - f0) + c1 * f0

    @ti.func
    def obs_phi_grad_trilinear(self, p):
        hh = 0.5 / float(self.n_grid)
        e0 = ti.Vector([hh, 0.0, 0.0])
        e1 = ti.Vector([0.0, hh, 0.0])
        e2 = ti.Vector([0.0, 0.0, hh])
        g0 = (self.obs_phi_sample_trilinear(p + e0) - self.obs_phi_sample_trilinear(p - e0)) / (2.0 * hh)
        g1 = (self.obs_phi_sample_trilinear(p + e1) - self.obs_phi_sample_trilinear(p - e1)) / (2.0 * hh)
        g2 = (self.obs_phi_sample_trilinear(p + e2) - self.obs_phi_sample_trilinear(p - e2)) / (2.0 * hh)
        return ti.Vector([g0, g1, g2])

    @ti.kernel
    def substep(
        self,
        g_x: ti.f32,
        g_y: ti.f32,
        g_z: ti.f32,
        lateral_force_probability: ti.f32,
        lateral_force_dir_x: ti.f32,
        lateral_force_dir_z: ti.f32,
        lateral_force_min: ti.f32,
        lateral_force_max: ti.f32,
        max_fall_speed: ti.f32,
    ):
        for I in ti.grouped(self.F_grid_m):
            self.F_grid_v[I] = ti.zero(self.F_grid_v[I])
            self.F_grid_m[I] = 0
        ti.loop_config(block_dim=self.n_grid)
        for p in self.F_x:
            if self.F_used[p] == 0:
                continue
            Xp = self.F_x[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            self.F_dg[p] = (ti.Matrix.identity(float, 3) + self.dt * self.F_C[p]) @ self.F_dg[p]
            h = ti.exp(10 * (1.0 - self.F_Jp[p]))
            if self.F_materials[p] == self.JELLY:
                h = 0.3
            mu, la = self.mu_0 * h, self.lambda_0 * h
            if self.F_materials[p] == self.WATER:
                mu = 0.0

            U, sig, V = ti.svd(self.F_dg[p])
            J = 1.0
            for d in ti.static(range(3)):
                new_sig = sig[d, d]
                if self.F_materials[p] == self.SNOW:
                    new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)
                self.F_Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if self.F_materials[p] == self.WATER:
                new_F = ti.Matrix.identity(float, 3)
                new_F[0, 0] = J
                self.F_dg[p] = new_F
            elif self.F_materials[p] == self.SNOW:
                self.F_dg[p] = U @ sig @ V.transpose()

            stress = 2 * mu * (self.F_dg[p] - U @ V.transpose()) @ self.F_dg[p].transpose() + ti.Matrix.identity(
                float, 3
            ) * la * J * (J - 1)
            stress = (-self.dt * self.p_vol * 4) * stress / self.dx**2
            affine = stress + self.p_mass * self.F_C[p]

            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                self.F_grid_v[base + offset] += weight * (self.p_mass * self.F_v[p] + affine @ dpos)
                self.F_grid_m[base + offset] += weight * self.p_mass
        for I in ti.grouped(self.F_grid_m):
            if self.F_grid_m[I] > 0:
                self.F_grid_v[I] /= self.F_grid_m[I]
            self.F_grid_v[I] += self.dt * ti.Vector([g_x, g_y, g_z])
            cond = (I < self.bound) & (self.F_grid_v[I] < 0) | (I > self.n_grid - self.bound) & (self.F_grid_v[I] > 0)
            self.F_grid_v[I] = ti.select(cond, 0, self.F_grid_v[I])
            phi = self.obs_phi[I[0], I[1], I[2]]
            self.F_grid_v[I] = ti.select(phi < 0, ti.Vector([0.0, 0.0, 0.0]), self.F_grid_v[I])
        ti.loop_config(block_dim=self.n_grid)
        for p in self.F_x:
            if self.F_used[p] == 0:
                continue
            Xp = self.F_x[p] / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.zero(self.F_v[p])
            new_C = ti.zero(self.F_C[p])
            for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                dpos = (offset - fx) * self.dx
                weight = 1.0
                for i in ti.static(range(self.dim)):
                    weight *= w[offset[i]][i]
                g_v = self.F_grid_v[base + offset]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) / self.dx**2
            if lateral_force_probability > 0.0 and lateral_force_max > 0.0 and ti.random() < lateral_force_probability:
                force_mag = lateral_force_min + ti.random() * (lateral_force_max - lateral_force_min)

                _lateral_force_dir_x = lateral_force_dir_x
                _lateral_force_dir_z = lateral_force_dir_z
                if ti.random() < 1.0:  # whether to use truly random direction or the configured one
                    lateral_force_angle = ti.random() * 2 * np.pi
                    _lateral_force_dir_x = ti.cos(lateral_force_angle)
                    _lateral_force_dir_z = ti.sin(lateral_force_angle)

                new_v += self.dt * ti.Vector([_lateral_force_dir_x * force_mag, 0.0, _lateral_force_dir_z * force_mag])
            if max_fall_speed > 0.0:
                new_v[1] = ti.max(new_v[1], -max_fall_speed)
            self.F_v[p] = new_v
            self.F_x[p] += self.dt * self.F_v[p]
            self.F_C[p] = new_C

    @ti.kernel
    def resolve_particles_obstacle(self):
        eps = 1e-4 / self.n_grid
        for p in self.F_x:
            if self.F_used[p] == 0:
                continue
            x = self.F_x[p]
            phi_p = self.obs_phi_sample_trilinear(x)
            if phi_p < 0:
                g = self.obs_phi_grad_trilinear(x)
                gl = g.norm()
                n = ti.Vector([0.0, 1.0, 0.0])
                if gl > 1e-8:
                    n = g / gl
                self.F_x[p] = x + (-phi_p + eps) * n
                vn = self.F_v[p].dot(n)
                self.F_v[p] -= ti.min(vn, 0.0) * n

    @ti.kernel
    def set_color_by_material(self, mat_color: ti.types.ndarray()):
        for i in range(self.n_particles):
            mat = self.F_materials[i]
            self.F_colors[i] = ti.Vector([mat_color[mat, 0], mat_color[mat, 1], mat_color[mat, 2], 1.0])

    @ti.kernel
    def init_cube_vol(
        self,
        first_par: ti.i32,
        last_par: ti.i32,
        x_begin: ti.f32,
        y_begin: ti.f32,
        z_begin: ti.f32,
        x_size: ti.f32,
        y_size: ti.f32,
        z_size: ti.f32,
        material: ti.i32,
    ):
        for i in range(first_par, last_par):
            self.F_x[i] = ti.Vector([ti.random() for _ in range(self.dim)]) * ti.Vector([x_size, y_size, z_size]) + ti.Vector(
                [x_begin, y_begin, z_begin]
            )
            self.F_Jp[i] = 1
            self.F_dg[i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.F_C[i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            self.F_v[i] = ti.Vector([0.0, 0.0, 0.0])
            self.F_materials[i] = material
            self.F_colors_random[i] = ti.Vector([ti.random(), ti.random(), ti.random(), ti.random()])
            self.F_used[i] = 1

    @ti.kernel
    def set_all_unused(self):
        for p in self.F_used:
            self.F_used[p] = 0
            self.F_x[p] = ti.Vector([533799.0, 533799.0, 533799.0])
            self.F_Jp[p] = 1
            self.F_dg[p] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.F_C[p] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
            self.F_v[p] = ti.Vector([0.0, 0.0, 0.0])

    def init_vols(self, vols: list[dict[str, Any]]):
        self.set_all_unused()
        total_vol = 0.0
        for v in vols:
            total_vol += float(v["volume"])
        if total_vol <= 0.0:
            self.F_next_particle[None] = 0
            return

        next_p = 0
        for i, v in enumerate(vols):
            par_count = int(float(v["volume"]) / total_vol * self.initial_n_particles)
            if i == len(vols) - 1:
                par_count = self.initial_n_particles - next_p
            mn = v["minimum"]
            sz = v["size"]
            self.init_cube_vol(next_p, next_p + par_count, mn[0], mn[1], mn[2], sz[0], sz[1], sz[2], int(v["material_id"]))
            next_p += par_count
        self.F_next_particle[None] = next_p

    def spawn_vols(self, vols: list[dict[str, Any]], spawn_count: int):
        spawn_count = max(0, min(int(spawn_count), self.n_particles - int(self.F_next_particle[None])))
        if spawn_count == 0:
            return

        total_vol = 0.0
        for v in vols:
            total_vol += float(v["volume"])
        if total_vol <= 0.0:
            return

        next_p = int(self.F_next_particle[None])
        spawned = 0
        for i, v in enumerate(vols):
            par_count = int(float(v["volume"]) / total_vol * spawn_count)
            if i == len(vols) - 1:
                par_count = spawn_count - spawned
            mn = v["minimum"]
            sz = v["size"]
            self.init_cube_vol(next_p, next_p + par_count, mn[0], mn[1], mn[2], sz[0], sz[1], sz[2], int(v["material_id"]))
            next_p += par_count
            spawned += par_count
        self.F_next_particle[None] = next_p

    def step_frame(self):
        for _ in range(self.steps):
            self.substep(
                self.gravity[0],
                self.gravity[1],
                self.gravity[2],
                self.lateral_force_probability,
                self.lateral_force_dir_x,
                self.lateral_force_dir_z,
                self.lateral_force_min,
                self.lateral_force_max,
                self.max_fall_speed,
            )
            self.resolve_particles_obstacle()

    @ti.kernel
    def compute_frame_stats(self):
        active = 0
        sum_v2 = 0.0
        vmax = 0.0
        y_min = 1.0e9
        y_max = -1.0e9

        for p in self.F_x:
            if self.F_used[p] == 0:
                continue
            active += 1
            v = self.F_v[p]
            v2 = v.dot(v)
            sum_v2 += v2
            sp = ti.sqrt(v2)
            vmax = ti.max(vmax, sp)
            y = self.F_x[p][1]
            y_min = ti.min(y_min, y)
            y_max = ti.max(y_max, y)

        self.S_active[None] = active
        if active > 0:
            self.S_speed_rms[None] = ti.sqrt(sum_v2 / ti.cast(active, ti.f32))
            self.S_speed_max[None] = vmax
            self.S_y_min[None] = y_min
            self.S_y_max[None] = y_max
        else:
            self.S_speed_rms[None] = 0.0
            self.S_speed_max[None] = 0.0
            self.S_y_min[None] = 0.0
            self.S_y_max[None] = 0.0

