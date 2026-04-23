from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover
    yaml = None  # type: ignore
    _yaml_import_error = e


TaichiArch = Literal["gpu", "cpu"]

# Repo root: ``config.py`` is under ``src/snowfall_particles/``; go up two levels to sit beside ``src/`` and ``assets/``; YAML mesh paths resolve relative to this directory
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class CameraConfig:
    position: tuple[float, float, float]
    lookat: tuple[float, float, float]
    fov: float


@dataclass(frozen=True)
class RenderConfig:
    resolution: tuple[int, int]
    vsync: bool
    particles_radius: float
    camera: CameraConfig
    ambient_light: tuple[float, float, float]
    point_lights: list[dict[str, tuple[float, float, float]]]


@dataclass(frozen=True)
class MaterialConfig:
    p_rho: float
    E: float
    nu: float


@dataclass(frozen=True)
class SimConfig:
    dim: int
    n_grid: int
    steps: int
    dt: float
    n_particles: int
    gravity: tuple[float, float, float]
    bound: int


@dataclass(frozen=True)
class SdfConfig:
    res: int
    cache_enabled: bool


@dataclass(frozen=True)
class OfflineConfig:
    """Used with ``--offline``; see ``offline.simulation_duration_seconds`` in YAML."""

    simulation_duration_seconds: float


@dataclass(frozen=True)
class Config:
    arch: TaichiArch
    sim: SimConfig
    material: MaterialConfig
    sdf: SdfConfig
    render: RenderConfig
    particles: dict[str, Any]
    obstacles: dict[str, Any]
    base_dir: Path
    offline: OfflineConfig | None


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _as_tuple3(x: Any, *, name: str) -> tuple[float, float, float]:
    _require(isinstance(x, (list, tuple)) and len(x) == 3, f"{name} must be a 3-tuple/list")
    return (float(x[0]), float(x[1]), float(x[2]))


def _as_tuple2i(x: Any, *, name: str) -> tuple[int, int]:
    _require(isinstance(x, (list, tuple)) and len(x) == 2, f"{name} must be a 2-tuple/list")
    return (int(x[0]), int(x[1]))


def load_config(path: str | Path) -> Config:
    if yaml is None:  # pragma: no cover
        raise ImportError("Missing dependency: pyyaml") from _yaml_import_error

    p = Path(path).expanduser().resolve()
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    _require(isinstance(data, dict), "config root must be a mapping")

    taichi = data.get("taichi", {}) or {}
    arch = str(taichi.get("arch", "gpu"))
    _require(arch in ("gpu", "cpu"), "taichi.arch must be gpu|cpu")

    sim = data.get("sim", {}) or {}
    sim_cfg = SimConfig(
        dim=int(sim.get("dim", 3)),
        n_grid=int(sim.get("n_grid", 128)),
        steps=int(sim.get("steps", 25)),
        dt=float(sim.get("dt", 4e-4)),
        n_particles=int(sim.get("n_particles", 1000)),
        gravity=_as_tuple3(sim.get("gravity", (0.0, -9.8, 0.0)), name="sim.gravity"),
        bound=int(sim.get("bound", 3)),
    )

    material = data.get("material", {}) or {}
    mat_cfg = MaterialConfig(
        p_rho=float(material.get("p_rho", 1.0)),
        E=float(material.get("E", 1000.0)),
        nu=float(material.get("nu", 0.2)),
    )

    sdf = data.get("sdf", {}) or {}
    sdf_cfg = SdfConfig(
        res=int(sdf.get("res", sim_cfg.n_grid)),
        cache_enabled=bool(sdf.get("cache_enabled", True)),
    )

    render = data.get("render", {}) or {}
    cam = render.get("camera", {}) or {}
    camera_cfg = CameraConfig(
        position=_as_tuple3(cam.get("position", (0.5, 1.0, 1.95)), name="render.camera.position"),
        lookat=_as_tuple3(cam.get("lookat", (0.5, 0.5, 0.5)), name="render.camera.lookat"),
        fov=float(cam.get("fov", 55)),
    )
    lights = render.get("lights", {}) or {}
    ambient = _as_tuple3(lights.get("ambient", (0.18, 0.18, 0.18)), name="render.lights.ambient")
    point_lights = lights.get("point_lights", []) or []
    _require(isinstance(point_lights, list), "render.lights.point_lights must be a list")
    render_cfg = RenderConfig(
        resolution=_as_tuple2i(render.get("resolution", (1280, 720)), name="render.resolution"),
        vsync=bool(render.get("vsync", True)),
        particles_radius=float(render.get("particles_radius", 0.01)),
        camera=camera_cfg,
        ambient_light=ambient,
        point_lights=[
            {"pos": _as_tuple3(pl.get("pos"), name="render.lights.point_lights[].pos"),
             "color": _as_tuple3(pl.get("color"), name="render.lights.point_lights[].color")}
            for pl in point_lights
        ],
    )

    particles = data.get("particles", {}) or {}
    obstacles = data.get("obstacles", {}) or {}

    offline_raw = data.get("offline")
    offline_cfg: OfflineConfig | None = None
    if isinstance(offline_raw, dict):
        dur = offline_raw.get("simulation_duration_seconds")
        if dur is not None:
            offline_cfg = OfflineConfig(simulation_duration_seconds=float(dur))

    return Config(
        arch=arch,  # type: ignore[arg-type]
        sim=sim_cfg,
        material=mat_cfg,
        sdf=sdf_cfg,
        render=render_cfg,
        particles=particles,
        obstacles=obstacles,
        base_dir=_PROJECT_ROOT,
        offline=offline_cfg,
    )

