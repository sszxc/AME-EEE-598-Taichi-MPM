from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import taichi as ti


MaterialName = Literal["water", "jelly", "snow"]


@dataclass(frozen=True)
class CubeVolume:
    minimum: tuple[float, float, float]
    size: tuple[float, float, float]
    material: MaterialName

    @property
    def volume(self) -> float:
        return float(self.size[0] * self.size[1] * self.size[2])


def _to_vec3(x) -> ti.Vector:
    return ti.Vector([float(x[0]), float(x[1]), float(x[2])])


def _parse_cube_volumes(vols_cfg: list) -> list[CubeVolume]:
    vols: list[CubeVolume] = []
    for v in vols_cfg:
        if (v or {}).get("kind") != "cube":
            raise ValueError("Only cube volumes supported for now")
        material = str(v.get("material", "water"))
        if material not in ("water", "jelly", "snow"):
            raise ValueError(f"Unknown material: {material}")
        vols.append(
            CubeVolume(
                minimum=(float(v["minimum"][0]), float(v["minimum"][1]), float(v["minimum"][2])),
                size=(float(v["size"][0]), float(v["size"][1]), float(v["size"][2])),
                material=material,  # type: ignore[arg-type]
            )
        )
    return vols


def load_fluid_presets(cfg: dict[str, Any]) -> tuple[list[str], list[list[CubeVolume]]]:
    items = (cfg.get("presets", []) or [])
    if not isinstance(items, list):
        raise ValueError("particles.presets must be a list")
    names: list[str] = []
    presets: list[list[CubeVolume]] = []
    for p in items:
        vols_cfg = (p or {}).get("volumes", []) or []
        if not isinstance(vols_cfg, list):
            raise ValueError("Each particles.presets[] entry must have a volumes list")
        names.append(str((p or {}).get("name", f"preset_{len(names)}")))
        presets.append(_parse_cube_volumes(vols_cfg))
    return names, presets


ObstacleKind = Literal["box", "mesh"]


@dataclass(frozen=True)
class ObstaclePreset:
    name: str
    kind: ObstacleKind
    # box
    half_xz: float | None = None
    height: float | None = None
    center_xz: tuple[float, float] | None = None
    # mesh
    path: str | None = None
    scale: float | None = None
    center: tuple[float, float, float] | None = None


def load_obstacle_presets(cfg: dict[str, Any]) -> list[ObstaclePreset]:
    presets = (cfg.get("presets", []) or [])
    if not isinstance(presets, list):
        raise ValueError("obstacles.presets must be a list")
    out: list[ObstaclePreset] = []
    for p in presets:
        kind = str((p or {}).get("kind"))
        name = str((p or {}).get("name", kind))
        if kind not in ("box", "mesh"):
            raise ValueError(f"Unknown obstacle kind: {kind}")
        if kind == "box":
            cx, cz = (p or {}).get("center_xz", (0.5, 0.5))
            out.append(
                ObstaclePreset(
                    name=name,
                    kind="box",
                    half_xz=float(p.get("half_xz", 0.2)),
                    height=float(p.get("height", 0.4)),
                    center_xz=(float(cx), float(cz)),
                )
            )
        else:
            c = (p or {}).get("center", (0.5, 0.2, 0.5))
            out.append(
                ObstaclePreset(
                    name=name,
                    kind="mesh",
                    path=str(p.get("path")),
                    scale=float(p.get("scale", 1.0)),
                    center=(float(c[0]), float(c[1]), float(c[2])),
                )
            )
    return out


def cubevolume_to_taichi(v: CubeVolume):
    return _to_vec3(v.minimum), _to_vec3(v.size)

