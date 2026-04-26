import math
from typing import Callable

import taichi as ti

# ---- Taichi init ----
real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True, device_memory_GB=3.5)

# Optional global seed hook for procedural scene generation.
# Entry points may set this before building the scene.
seed: int | None = None

# ---- Global sim parameters (will be overwritten by scene finalize) ----
dim = 3
# this will be overwritten
n_particles = 0
n_solid_particles = 0
n_actuators = 0

n_grid = 32
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 2e-3
p_vol = 1
E = 10
# TODO: update
mu = E
la = E
max_steps = 1024
steps = 1024
gravity = 10
target = [0.8, 0.2, 0.2]
use_apic = False

# ---- Actuation parameters ----
n_sin_waves = 4
actuation_omega = 20
act_strength = 5

# ---- Visualization ----
visualize_resolution = 256

# ---- Field factories ----
scalar: Callable[[], ti.Field] = lambda: ti.field(dtype=real)
vec: Callable[[], ti.MatrixField] = lambda: ti.Vector.field(dim, dtype=real)
mat: Callable[[], ti.MatrixField] = lambda: ti.Matrix.field(dim, dim, dtype=real)

# ---- Fields ----
actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
root_id = ti.field(ti.i32)
segment_id = ti.field(ti.i32)
actuator_dir = ti.Vector.field(3, dtype=real)

x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

screen = ti.Vector.field(3, dtype=real)
loss = scalar()

weights = scalar()
bias = scalar()
x_avg = vec()

actuation = scalar()


def allocate_fields() -> None:
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)

    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_actuators).place(actuator_dir)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type, root_id, segment_id)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ijk, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, x_avg)
    ti.root.dense(ti.ij, (visualize_resolution, visualize_resolution)).place(screen)

    ti.root.lazy_grad()

