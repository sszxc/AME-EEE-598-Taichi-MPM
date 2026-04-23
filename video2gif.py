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
            "未找到 ffmpeg。请先安装 ffmpeg，并确保命令 `ffmpeg` 在 PATH 中可用。\n"
            "- macOS (brew): brew install ffmpeg\n"
            "- conda: conda install -c conda-forge ffmpeg"
        )
    return ffmpeg


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".gif")


def _validate_input(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在：{path}")
    if path.suffix.lower() not in SUPPORTED_INPUT_EXTS:
        exts = ", ".join(sorted(SUPPORTED_INPUT_EXTS))
        raise ValueError(f"仅支持输入格式：{exts}，当前：{path.suffix}")


def _build_vf(fps: int, width: int | None) -> str:
    # palettegen/paletteuse 的输入建议统一 fps + scale + 格式化像素
    # width 给定时，按宽度缩放并保持纵横比，宽度建议为偶数以避免某些编码器警告
    parts = [f"fps={fps}"]
    if width is not None:
        w = int(width)
        if w <= 0:
            raise ValueError("--width 必须为正数")
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
        raise ValueError("--fps 必须为正数")
    if not (2 <= colors <= 256):
        raise ValueError("--colors 需在 2~256 之间")
    if bayer_scale < 0:
        raise ValueError("--bayer-scale 不能为负数")
    if start is not None and start < 0:
        raise ValueError("--start 不能为负数")
    if duration is not None and duration <= 0:
        raise ValueError("--duration 必须为正数")
    if loop < -1:
        raise ValueError("--loop 需为 -1 或 >= 0（0 表示无限循环）")
    allowed_dither = {"bayer", "none", "floyd_steinberg", "sierra2", "sierra2_4a"}
    if dither not in allowed_dither:
        raise ValueError(f"--dither 仅支持：{', '.join(sorted(allowed_dither))}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    vf = _build_vf(fps=fps, width=width)

    # 使用 palettegen/paletteuse 获得更好的色彩质量与更小体积
    with tempfile.TemporaryDirectory(prefix="video2gif_") as td:
        palette = Path(td) / "palette.png"

        # 允许 -ss 放在 -i 前做更快的 seek（近似），但对精确截取可把 -ss 放到 -i 后
        # 这里优先速度：-ss (input seek) + 可选 -t
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
        description="将 mp4/mov 视频转换为 GIF（依赖 ffmpeg）。",
    )
    p.add_argument("input", type=str, help="输入视频路径（.mp4 或 .mov）")
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出 GIF 路径（默认：与输入同名 .gif）",
    )
    p.add_argument("--fps", type=int, default=15, help="输出 GIF 帧率（默认：15）")
    p.add_argument(
        "--width",
        type=int,
        default=720,
        help="输出宽度（默认：720；设为 0 可保持原尺寸）",
    )
    p.add_argument(
        "--colors",
        type=int,
        default=256,
        help="调色板颜色数（2~256，默认：256；更小通常更省体积）",
    )
    p.add_argument(
        "--dither",
        type=str,
        default="bayer",
        choices=["bayer", "none", "floyd_steinberg", "sierra2", "sierra2_4a"],
        help="抖动方式（默认：bayer；none 通常更小但更易色带）",
    )
    p.add_argument(
        "--bayer-scale",
        type=int,
        default=5,
        help="bayer 抖动强度（仅 dither=bayer 生效，默认：5）",
    )
    p.add_argument("--start", type=float, default=None, help="起始时间（秒）")
    p.add_argument("--duration", type=float, default=None, help="截取时长（秒）")
    p.add_argument(
        "--loop",
        type=int,
        default=0,
        help="GIF 循环次数：0 无限循环，-1 不循环，N 循环 N 次（默认：0）",
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
        _eprint("ffmpeg 执行失败。命令如下：")
        _eprint(" ".join(map(str, e.cmd)))
        return int(e.returncode) if e.returncode is not None else 1
    except Exception as e:
        _eprint(str(e))
        return 1

    print(f"已生成：{out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
