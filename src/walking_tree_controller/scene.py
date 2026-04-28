import random

import math
import numpy as np

import config as cfg


class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.root_id = []
        self.segment_id = []
        self.offset_x = 0
        self.offset_y = 0
        self.offset_z = 0
        self.num_actuators = 0
        self.actuator_dir = []

    def new_actuator(self):
        self.num_actuators += 1
        cfg.n_actuators = self.num_actuators
        self.actuator_dir.append([0.0, 0.0, 1.0])
        return self.num_actuators - 1

    def set_actuator_direction(self, actuator_id, direction):
        norm = math.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
        if norm < 1e-8:
            self.actuator_dir[actuator_id] = [0.0, 0.0, 1.0]
            return
        self.actuator_dir[actuator_id] = [
            direction[0] / norm,
            direction[1] / norm,
            direction[2] / norm,
        ]

    def _append_particle(self, pos, actuation, ptype, root_id=-1, segment_id=-1):
        self.x.append(pos)
        self.actuator_id.append(actuation)
        self.particle_type.append(ptype)
        self.root_id.append(root_id)
        self.segment_id.append(segment_id)
        self.n_particles += 1
        self.n_solid_particles += int(ptype == 1)

    def add_rect(self, x, y, z, w, h, d, actuation, ptype=1, root_id=-1, segment_id=-1):
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
                        self._append_particle(
                            [
                                x + (i + 0.5) * real_dx + self.offset_x,
                                y + (j + 0.5) * real_dy + self.offset_y,
                                z + (k + 0.5) * real_dz + self.offset_z,
                            ],
                            actuation,
                            ptype,
                            root_id=root_id,
                            segment_id=segment_id,
                        )
        else:
            for i in range(w_count):
                for j in range(h_count):
                    for k in range(d_count):
                        self._append_particle(
                            [
                                x + random.random() * w + self.offset_x,
                                y + random.random() * h + self.offset_y,
                                z + random.random() * d + self.offset_z,
                            ],
                            actuation,
                            ptype,
                            root_id=root_id,
                            segment_id=segment_id,
                        )

    def set_offset(self, x, y, z):
        self.offset_x = x
        self.offset_y = y
        self.offset_z = z

    def finalize(self):
        assert len(self.x) == self.n_particles
        assert len(self.actuator_id) == self.n_particles
        assert len(self.particle_type) == self.n_particles
        assert len(self.root_id) == self.n_particles
        assert len(self.segment_id) == self.n_particles
        assert len(self.actuator_dir) == self.num_actuators
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


def _safe_normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return (v / n).astype(np.float32, copy=False)


def _eval_bezier(p0, p1, p2, p3, t):
    omt = 1.0 - t
    return (
        (omt ** 3) * p0
        + 3.0 * (omt ** 2) * t * p1
        + 3.0 * omt * (t ** 2) * p2
        + (t ** 3) * p3
    )


class TreePlant:
    def __init__(self):
        self.trunk_base = np.array([0.0, 0.3, 0.0], dtype=np.float32)
        self.world_offset = np.array([0.5, 0.0, 0.5], dtype=np.float32)
        self.num_roots = 5
        self.segments_per_root = 1
        self.root_length_range = (0.24, 0.42)
        self.base_spawn_radius = 0.018
        self.start_radius = 0.08
        self.end_radius = 0.04
        self.curve_jitter = 0.05
        # Keep the root tips above the MPM ground boundary layer (bound=3, dx=1/64 -> ~0.0469).
        self.ground_y = 0.13
        self.disk_density = 2.0
        self.particle_jitter = 0.0012
        self.seed = 0
        self.trunk_height = 0.5
        self.trunk_base_radius = 0.09
        self.trunk_tip_radius = 0.04
        self.branch_depth = 0  # 2
        self.branch_length_decay = 0.58
        self.branch_radius_decay = 0.48
        self.branches_per_level = 3
        self._rng = None

    def _local_frame(self, tangent):
        tangent = _safe_normalize(tangent)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        up_proj = world_up - np.dot(world_up, tangent) * tangent
        if np.linalg.norm(up_proj) < 1e-6:
            up_proj = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        up_axis = _safe_normalize(up_proj)
        side_axis = _safe_normalize(np.cross(tangent, up_axis))
        return tangent, up_axis, side_axis

    def _build_root_curve(self, root_idx, rng):
        angle = 2.0 * math.pi * (root_idx / self.num_roots) + rng.uniform(-0.18, 0.18)
        outward = np.array([math.cos(angle), 0.0, math.sin(angle)], dtype=np.float32)
        length = rng.uniform(*self.root_length_range)

        p0 = self.trunk_base + outward * self.base_spawn_radius
        p0[1] += rng.uniform(-0.005, 0.005)
        p3 = self.trunk_base + outward * length
        p3[1] = self.ground_y + rng.uniform(0.004, 0.012)

        c1 = p0 + outward * (0.30 * length)
        c2 = p0 + outward * (0.68 * length)
        c1[1] -= 0.06 + rng.uniform(-0.01, 0.01)
        c2[1] -= 0.12 + rng.uniform(-0.015, 0.015)
        for c in (c1, c2):
            c[0] += rng.uniform(-self.curve_jitter, self.curve_jitter)
            c[2] += rng.uniform(-self.curve_jitter, self.curve_jitter)

        ts = np.linspace(0.0, 1.0, self.segments_per_root + 1, dtype=np.float32)
        curve = np.stack([_eval_bezier(p0, c1, c2, p3, float(t)) for t in ts], axis=0)
        curve[:, 1] = np.maximum(curve[:, 1], self.ground_y + 0.003)
        return curve.astype(np.float32, copy=False)

    def _sample_tapered_segment(
        self,
        scene,
        p0,
        p1,
        r0,
        r1,
        actuation,
        root_id=-1,
        segment_id=-1,
        min_y=None,
    ):
        seg_vec = p1 - p0
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len < 1e-7:
            return

        rng = self._rng
        _, up_axis, side_axis = self._local_frame(seg_vec)
        n_along = max(2, int(math.ceil(seg_len / max(1e-6, cfg.dx * 0.8))))
        for k in range(n_along):
            alpha = (k + 0.5) / n_along
            center = (1.0 - alpha) * p0 + alpha * p1
            radius = (1.0 - alpha) * r0 + alpha * r1
            area = math.pi * radius * radius
            n_cross = max(8, int(area / max(1e-8, cfg.dx * cfg.dx) * self.disk_density))

            for _ in range(n_cross):
                rr = radius * math.sqrt(float(rng.random()))
                theta = 2.0 * math.pi * float(rng.random())
                u = rr * math.cos(theta)
                v = rr * math.sin(theta)
                local_offset = up_axis * u + side_axis * v
                jitter = (rng.random(3, dtype=np.float32) * 2.0 - 1.0) * self.particle_jitter
                pos = center + local_offset + jitter
                if min_y is not None:
                    pos[1] = max(float(pos[1]), min_y)

                act = actuation
                if callable(actuation):
                    act = actuation(u, v)
                scene._append_particle(
                    [
                        float(pos[0]) + scene.offset_x,
                        float(pos[1]) + scene.offset_y,
                        float(pos[2]) + scene.offset_z,
                    ],
                    act,
                    1,
                    root_id=root_id,
                    segment_id=segment_id,
                )

    def _build_roots(self, scene):
        rng = self._rng
        for root_idx in range(self.num_roots):
            curve = self._build_root_curve(root_idx, rng)
            for seg_idx in range(self.segments_per_root):
                p0 = curve[seg_idx]
                p1 = curve[seg_idx + 1]
                seg_vec = p1 - p0
                seg_len = float(np.linalg.norm(seg_vec))
                if seg_len < 1e-7:
                    continue

                _, up_axis, side_axis = self._local_frame(seg_vec)
                actuator_up = scene.new_actuator()
                actuator_down = scene.new_actuator()
                actuator_right = scene.new_actuator()
                actuator_left = scene.new_actuator()
                scene.set_actuator_direction(actuator_up, up_axis.tolist())
                scene.set_actuator_direction(actuator_down, (-up_axis).tolist())
                scene.set_actuator_direction(actuator_right, side_axis.tolist())
                scene.set_actuator_direction(actuator_left, (-side_axis).tolist())

                t0 = seg_idx / self.segments_per_root
                t1 = (seg_idx + 1) / self.segments_per_root
                r0 = (1.0 - t0) * self.start_radius + t0 * self.end_radius
                r1 = (1.0 - t1) * self.start_radius + t1 * self.end_radius

                def root_actuation(u, v):
                    # Assign each particle to the dominant directional lobe
                    # of the root cross-section.
                    if abs(u) >= abs(v):
                        return actuator_up if u >= 0.0 else actuator_down
                    return actuator_right if v >= 0.0 else actuator_left

                self._sample_tapered_segment(
                    scene,
                    p0,
                    p1,
                    r0,
                    r1,
                    root_actuation,
                    root_id=root_idx,
                    segment_id=seg_idx,
                    min_y=self.ground_y,
                )

    def _build_trunk(self, scene):
        p0 = self.trunk_base
        p1 = self.trunk_base + np.array([0.0, self.trunk_height, 0.0], dtype=np.float32)
        self._sample_tapered_segment(
            scene,
            p0,
            p1,
            self.trunk_base_radius,
            self.trunk_tip_radius,
            -1,
        )

    def _branch_direction(self, level, branch_idx, rng):
        angle = 2.0 * math.pi * (branch_idx / max(1, self.branches_per_level))
        angle += level * 0.73 + rng.uniform(-0.28, 0.28)
        horizontal = np.array([math.cos(angle), 0.0, math.sin(angle)], dtype=np.float32)
        upward = 0.55 - 0.08 * level + rng.uniform(-0.08, 0.08)
        return _safe_normalize(horizontal + np.array([0.0, upward, 0.0], dtype=np.float32))

    def _build_branch_level(self, scene, start, parent_dir, level, length, radius):
        if level >= self.branch_depth or length <= 1e-5 or radius <= 1e-5:
            return

        rng = self._rng
        for branch_idx in range(self.branches_per_level):
            blend = self._branch_direction(level, branch_idx, rng)
            direction = _safe_normalize(parent_dir * 0.35 + blend * 0.65)
            end = start + direction * length
            end[1] = max(float(end[1]), self.ground_y + 0.02)
            tip_radius = radius * self.branch_radius_decay
            self._sample_tapered_segment(scene, start, end, radius, tip_radius, -1)
            self._build_branch_level(
                scene,
                end,
                direction,
                level + 1,
                length * self.branch_length_decay,
                tip_radius,
            )

    def _build_branches(self, scene):
        if self.branch_depth <= 0:
            return

        rng = self._rng
        trunk_top = self.trunk_base + np.array([0.0, self.trunk_height, 0.0], dtype=np.float32)
        start_height = self.trunk_height * 0.55
        base_length = self.trunk_height * 0.42
        base_radius = self.trunk_tip_radius * 0.72
        for i in range(self.branches_per_level):
            height_alpha = 0.0 if self.branches_per_level == 1 else i / (self.branches_per_level - 1)
            start = self.trunk_base + np.array(
                [0.0, start_height + height_alpha * (self.trunk_height * 0.32), 0.0],
                dtype=np.float32,
            )
            direction = self._branch_direction(0, i, rng)
            end = start + direction * base_length
            end[1] = min(float(end[1]), float(trunk_top[1] + base_length * 0.35))
            self._sample_tapered_segment(
                scene,
                start,
                end,
                base_radius,
                base_radius * self.branch_radius_decay,
                -1,
            )
            self._build_branch_level(
                scene,
                end,
                direction,
                1,
                base_length * self.branch_length_decay,
                base_radius * self.branch_radius_decay,
            )

    def _begin_population(self, scene):
        self._rng = np.random.default_rng(self.seed)
        scene.set_offset(
            float(self.world_offset[0]),
            float(self.world_offset[1]),
            float(self.world_offset[2]),
        )

    def populate_roots(self, scene):
        self._begin_population(scene)
        self._build_roots(scene)

    def populate_scene(self, scene):
        self._begin_population(scene)
        self._build_roots(scene)
        self._build_trunk(scene)
        self._build_branches(scene)


def _apply_procedural_seed(plant):
    # Allow external control for deterministic procedural generation.
    # (Also see diffmpm3d.py which seeds Python/NumPy/Taichi.)
    if hasattr(cfg, "seed") and cfg.seed is not None:
        try:
            plant.seed = int(cfg.seed)
        except Exception:
            pass
    return plant


def build_walking_tree_root(scene: Scene):
    root_system = _apply_procedural_seed(TreePlant())
    root_system.populate_roots(scene)
    return root_system


def build_walking_tree_plant(scene: Scene):
    plant = _apply_procedural_seed(TreePlant())
    plant.populate_scene(scene)
    return plant

