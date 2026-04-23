#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SUPPORTED_INPUT_EXTS = {".mp4", ".mov"}


def _eprint(msg: str) -> None:
    print(msg, file=sys.stderr)


def _require_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg and ensure `ffmpeg` is on your PATH.\n"
            "- macOS (brew): brew install ffmpeg\n"
            "- conda: conda install -c conda-forge ffmpeg"
        )
    return ffmpeg


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".gif")


def _validate_input(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if path.suffix.lower() not in SUPPORTED_INPUT_EXTS:
        exts = ", ".join(sorted(SUPPORTED_INPUT_EXTS))
        raise ValueError(f"Supported input extensions: {exts}; got: {path.suffix}")


def _build_vf(fps: int, width: int | None) -> str:
    # palettegen/paletteuse inputs: keep fps + scale + pixel format consistent
    # When width is set, scale by width and preserve aspect ratio; prefer even width to avoid encoder warnings
    parts = [f"fps={fps}"]
    if width is not None:
        w = int(width)
        if w <= 0:
            raise ValueError("--width must be positive")
        if w % 2 == 1:
            w += 1
        parts.append(f"scale={w}:-1:flags=lanczos")
    parts.append("format=rgb24")
    return ",".join(parts)


def video_to_gif(
    input_path: Path,
    output_path: Path,
    *,
    fps: int = 15,
    width: int | None = 720,
    colors: int = 256,
    dither: str = "bayer",
    bayer_scale: int = 5,
    start: float | None = None,
    duration: float | None = None,
    loop: int = 0,
) -> None:
    _validate_input(input_path)
    ffmpeg = _require_ffmpeg()

    if fps <= 0:
        raise ValueError("--fps must be positive")
    if not (2 <= colors <= 256):
        raise ValueError("--colors must be between 2 and 256")
    if bayer_scale < 0:
        raise ValueError("--bayer-scale cannot be negative")
    if start is not None and start < 0:
        raise ValueError("--start cannot be negative")
    if duration is not None and duration <= 0:
        raise ValueError("--duration must be positive")
    if loop < -1:
        raise ValueError("--loop must be -1 or >= 0 (0 means infinite loop)")
    allowed_dither = {"bayer", "none", "floyd_steinberg", "sierra2", "sierra2_4a"}
    if dither not in allowed_dither:
        raise ValueError(f"--dither must be one of: {', '.join(sorted(allowed_dither))}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = _build_vf(fps=fps, width=width)

    # palettegen/paletteuse for better colors and smaller files
    with tempfile.TemporaryDirectory(prefix="video2gif_") as td:
        palette = Path(td) / "palette.png"

        # Putting -ss before -i enables faster (approximate) seek; for exact cuts, place -ss after -i
        # Here we favor speed: -ss (input seek) + optional -t
        common_in: list[str] = []
        if start is not None:
            common_in += ["-ss", str(start)]
        common_in += ["-i", str(input_path)]
        if duration is not None:
            common_in += ["-t", str(duration)]

        cmd_palette = [
            ffmpeg,
            "-y",
            *common_in,
            "-vf",
            f"{vf},palettegen=max_colors={colors}:stats_mode=diff",
            str(palette),
        ]

        if dither == "bayer":
            paletteuse = f"paletteuse=dither=bayer:bayer_scale={bayer_scale}"
        else:
            paletteuse = f"paletteuse=dither={dither}"

        cmd_gif = [
            ffmpeg,
            "-y",
            *common_in,
            "-i",
            str(palette),
            "-lavfi",
            f"{vf}[x];[x][1:v]{paletteuse}",
            "-loop",
            str(loop),
            str(output_path),
        ]

        subprocess.run(cmd_palette, check=True)
        subprocess.run(cmd_gif, check=True)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="video2gif",
        description="Convert mp4/mov to GIF (requires ffmpeg).",
    )
    p.add_argument("input", type=str, help="Input video path (.mp4 or .mov)")
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output GIF path (default: same basename as input with .gif)",
    )
    p.add_argument("--fps", type=int, default=15, help="Output GIF frame rate (default: 15)")
    p.add_argument(
        "--width",
        type=int,
        default=720,
        help="Output width in pixels (default: 720; use 0 to keep source size)",
    )
    p.add_argument(
        "--colors",
        type=int,
        default=256,
        help="Palette size 2–256 (default: 256; smaller often means smaller file)",
    )
    p.add_argument(
        "--dither",
        type=str,
        default="bayer",
        choices=["bayer", "none", "floyd_steinberg", "sierra2", "sierra2_4a"],
        help="Dither mode (default: bayer; none is often smaller but bandier)",
    )
    p.add_argument(
        "--bayer-scale",
        type=int,
        default=5,
        help="Bayer dither strength (only when dither=bayer, default: 5)",
    )
    p.add_argument("--start", type=float, default=None, help="Start time in seconds")
    p.add_argument("--duration", type=float, default=None, help="Clip duration in seconds")
    p.add_argument(
        "--loop",
        type=int,
        default=0,
        help="GIF loop count: 0 infinite, -1 no loop, N loops N times (default: 0)",
    )
    return p.parse_args(argv)


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    in_path = Path(args.input).expanduser().resolve()
    out_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else _default_output_path(in_path)
    )

    width = args.width
    if width == 0:
        width = None

    try:
        video_to_gif(
            in_path,
            out_path,
            fps=args.fps,
            width=width,
            colors=args.colors,
            dither=args.dither,
            bayer_scale=args.bayer_scale,
            start=args.start,
            duration=args.duration,
            loop=args.loop,
        )
    except subprocess.CalledProcessError as e:
        _eprint("ffmpeg failed. Command:")
        _eprint(" ".join(map(str, e.cmd)))
        return int(e.returncode) if e.returncode is not None else 1
    except Exception as e:
        _eprint(str(e))
        return 1

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
