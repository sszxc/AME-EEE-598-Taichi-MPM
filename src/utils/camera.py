from __future__ import annotations

import time
from math import pi

import taichi as ti
from taichi.lang.matrix import Vector
from taichi.ui.utils import euler_to_vec, vec_to_euler

__all__ = ["FixedLookatCamera"]


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
