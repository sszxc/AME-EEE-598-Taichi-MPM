import random

import config as cfg


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0
        self.offset_z = 0
        self.num_actuators = 0

    def new_actuator(self):
        self.num_actuators += 1
        cfg.n_actuators = self.num_actuators
        return self.num_actuators - 1

    def add_rect(self, x, y, z, w, h, d, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1

        density = 2
        w_count = int(w / cfg.dx * density)
        h_count = int(h / cfg.dx * density)
        d_count = int(d / cfg.dx * density)
        real_dx = w / w_count
        real_dy = h / h_count
        real_dz = d / d_count

        if ptype == 1:
            for i in range(w_count):
                for j in range(h_count):
                    for k in range(d_count):
                        self.x.append(
                            [
                                x + (i + 0.5) * real_dx + self.offset_x,
                                y + (j + 0.5) * real_dy + self.offset_y,
                                z + (k + 0.5) * real_dz + self.offset_z,
                            ]
                        )
                        self.actuator_id.append(actuation)
                        self.particle_type.append(ptype)
                        self.n_particles += 1
                        self.n_solid_particles += int(ptype == 1)
        else:
            for i in range(w_count):
                for j in range(h_count):
                    for k in range(d_count):
                        self.x.append(
                            [
                                x + random.random() * w + self.offset_x,
                                y + random.random() * h + self.offset_y,
                                z + random.random() * d + self.offset_z,
                            ]
                        )
                        self.actuator_id.append(actuation)
                        self.particle_type.append(ptype)
                        self.n_particles += 1
                        self.n_solid_particles += int(ptype == 1)

    def set_offset(self, x, y, z):
        self.offset_x = x
        self.offset_y = y
        self.offset_z = z

    def finalize(self):
        cfg.n_particles = self.n_particles
        cfg.n_solid_particles = max(self.n_solid_particles, 1)
        print("n_particles", cfg.n_particles)
        print("n_solid", cfg.n_solid_particles)

    def set_n_actuators(self, n_act):
        cfg.n_actuators = n_act


def robot(scene: Scene):
    block_size = 0.1  # change the block_size to 0.05 if run out of GPU memory
    scene.set_offset(0.1, 0.05, 0.3)

    def add_leg(x, y, z):
        for i in range(4):
            scene.add_rect(
                x + block_size / 2 * (i // 2),
                y + 0.7 * block_size / 2 * (i % 2),
                z,
                block_size / 2,
                0.7 * block_size / 2,
                block_size,
                scene.new_actuator(),
            )

    for i in range(4):
        add_leg(i // 2 * block_size * 2, 0.0, i % 2 * block_size * 2)
    for i in range(3):
        scene.add_rect(
            block_size * i,
            0,
            block_size,
            block_size,
            block_size * 0.7,
            block_size,
            -1,
            1,
        )

