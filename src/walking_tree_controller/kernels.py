import math

import taichi as ti

import config as cfg


@ti.func
def zero_vec():
    return [0.0, 0.0, 0.0]


@ti.func
def zero_matrix():
    return [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]


@ti.kernel
def clear_grid():
    for i, j, k in cfg.grid_m_in:
        cfg.grid_v_in[i, j, k] = [0, 0, 0]
        cfg.grid_m_in[i, j, k] = 0
        if ti.static(cfg.enable_grad):
            cfg.grid_v_in.grad[i, j, k] = [0, 0, 0]
            cfg.grid_m_in.grad[i, j, k] = 0
            cfg.grid_v_out.grad[i, j, k] = [0, 0, 0]


@ti.kernel
def clear_particle_grad():
    # for all time steps and all particles
    for f, i in cfg.x:
        cfg.x.grad[f, i] = zero_vec()
        cfg.v.grad[f, i] = zero_vec()
        cfg.C.grad[f, i] = zero_matrix()
        cfg.F.grad[f, i] = zero_matrix()


@ti.kernel
def clear_actuation_grad():
    for t, i in cfg.actuation:
        cfg.actuation[t, i] = 0.0


@ti.kernel
def p2g(f: ti.i32):
    for p in range(0, cfg.n_particles):
        base = ti.cast(cfg.x[f, p] * cfg.inv_dx - 0.5, ti.i32)
        fx = cfg.x[f, p] * cfg.inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_F = (ti.Matrix.diag(dim=cfg.dim, val=1) + cfg.dt * cfg.C[f, p]) @ cfg.F[f, p]
        J = (new_F).determinant()
        if cfg.particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            # TODO: need pow(x, 1/3)
            new_F = ti.Matrix([[sqrtJ, 0, 0], [0, sqrtJ, 0], [0, 0, 1]])

        cfg.F[f + 1, p] = new_F

        act_id = cfg.actuator_id[p]

        act = cfg.actuation[f, ti.max(0, act_id)] * cfg.act_strength
        if act_id == -1:
            act = 0.0

        A = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]) * act
        cauchy = ti.Matrix(zero_matrix())
        mass = 0.0
        ident = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        if cfg.particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix(ident) * (J - 1) * cfg.E
        else:
            mass = 1
            local_mu = cfg.trunk_mu
            local_la = cfg.trunk_la
            if cfg.root_id[p] >= 0:
                local_mu = cfg.root_mu
                local_la = cfg.root_la
            cauchy = local_mu * (new_F @ new_F.transpose()) + ti.Matrix(ident) * (
                local_la * ti.log(J) - local_mu
            )
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(cfg.dt * cfg.p_vol * 4 * cfg.inv_dx * cfg.inv_dx) * cauchy
        affine = stress + mass * cfg.C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    offset = ti.Vector([i, j, k])
                    dpos = (ti.cast(ti.Vector([i, j, k]), cfg.real) - fx) * cfg.dx
                    weight = w[i][0] * w[j][1] * w[k][2]
                    ti.atomic_add(
                        cfg.grid_v_in[base + offset],
                        weight * (mass * cfg.v[f, p] + affine @ dpos),
                    )
                    ti.atomic_add(cfg.grid_m_in[base + offset], weight * mass)


bound = 3
coeff = 1.5


@ti.kernel
def grid_op():
    for i, j, k in cfg.grid_m_in:
        inv_m = 1 / (cfg.grid_m_in[i, j, k] + 1e-10)
        v_out = inv_m * cfg.grid_v_in[i, j, k]
        v_out[1] -= cfg.dt * cfg.gravity

        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0
        if i > cfg.n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0

        if k < bound and v_out[2] < 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0
        if k > cfg.n_grid - bound and v_out[2] > 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0

        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0
            normal = ti.Vector([0.0, 1.0, 0.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                    v_out[2] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                            v_out[2] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > cfg.n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0
            v_out[2] = 0

        cfg.grid_v_out[i, j, k] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(0, cfg.n_particles):
        base = ti.cast(cfg.x[f, p] * cfg.inv_dx - 0.5, ti.i32)
        fx = cfg.x[f, p] * cfg.inv_dx - ti.cast(base, cfg.real)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector(zero_vec())
        new_C = ti.Matrix(zero_matrix())

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                for k in ti.static(range(3)):
                    dpos = ti.cast(ti.Vector([i, j, k]), cfg.real) - fx
                    g_v = cfg.grid_v_out[base[0] + i, base[1] + j, base[2] + k]
                    weight = w[i][0] * w[j][1] * w[k][2]
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) * cfg.inv_dx

        cfg.v[f + 1, p] = new_v
        cfg.x[f + 1, p] = cfg.x[f, p] + cfg.dt * cfg.v[f + 1, p]
        cfg.C[f + 1, p] = new_C


@ti.kernel
def compute_actuation(t: ti.i32):
    for i in range(cfg.n_actuators):
        act = 0.0
        for j in ti.static(range(cfg.n_sin_waves)):
            act += cfg.weights[i, j] * ti.sin(cfg.actuation_omega * t * cfg.dt + 2 * math.pi / cfg.n_sin_waves * j)
        # act += cfg.bias[i]
        cfg.actuation[t, i] = ti.tanh(act)


@ti.kernel
def compute_x_avg():
    for i in range(cfg.n_particles):
        contrib = 0.0
        if cfg.particle_type[i] == 1:
            contrib = 1.0 / cfg.n_solid_particles
        ti.atomic_add(cfg.x_avg[None], contrib * cfg.x[cfg.steps - 1, i])


@ti.kernel
def compute_loss():
    dist = cfg.x_avg[None][0]
    cfg.loss[None] = -dist


def forward(total_steps: int | None = None):
    if total_steps is None:
        total_steps = cfg.steps

    for s in range(total_steps - 1):
        clear_grid()
        compute_actuation(s)
        p2g(s)
        grid_op()
        g2p(s)

    cfg.x_avg[None] = [0, 0, 0]
    compute_x_avg()
    compute_loss()
    return cfg.loss[None]


def backward():
    clear_particle_grad()

    compute_loss.grad()
    compute_x_avg.grad()
    for s in reversed(range(cfg.steps - 1)):
        # Since we do not store the grid history (to save space), we redo p2g and grid op
        clear_grid()
        p2g(s)
        grid_op()

        g2p.grad(s)
        grid_op.grad()
        p2g.grad(s)
        compute_actuation.grad(s)


@ti.kernel
def learn(learning_rate: ti.template()):
    for i, j in ti.ndrange(cfg.n_actuators, cfg.n_sin_waves):
        cfg.weights[i, j] -= learning_rate * cfg.weights.grad[i, j]

    # for i in range(cfg.n_actuators):
    #     cfg.bias[i] -= learning_rate * cfg.bias.grad[i]


@ti.kernel
def init(
    x_: ti.types.ndarray(element_dim=1),
    actuator_id_arr: ti.types.ndarray(),
    particle_type_arr: ti.types.ndarray(),
    root_id_arr: ti.types.ndarray(),
    segment_id_arr: ti.types.ndarray(),
    actuator_dir_arr: ti.types.ndarray(element_dim=1),
):
    for i, j in ti.ndrange(cfg.n_actuators, cfg.n_sin_waves):
        cfg.weights[i, j] = ti.randn() * 0.01

    for i in range(cfg.n_particles):
        cfg.x[0, i] = x_[i]
        cfg.F[0, i] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        cfg.actuator_id[i] = actuator_id_arr[i]
        cfg.particle_type[i] = particle_type_arr[i]
        cfg.root_id[i] = root_id_arr[i]
        cfg.segment_id[i] = segment_id_arr[i]

    for i in range(cfg.n_actuators):
        cfg.actuator_dir[i] = actuator_dir_arr[i]

