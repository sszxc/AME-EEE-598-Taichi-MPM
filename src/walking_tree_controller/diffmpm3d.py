import argparse
import time
from typing import Optional

import numpy as np
import taichi as ti

import config as cfg
import kernels as kernels
import scene as scene_lib
import viz as viz

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--vis_interval', type=int, default=10)
    parser.add_argument('--vis_stride', type=int, default=8)
    parser.add_argument(
        '--progress',
        action='store_true',
        help='Show a tqdm progress bar for the outer optimization loop (if tqdm is installed).',
    )
    parser.add_argument(
        '--warmup',
        action='store_true',
        help='Run one forward/backward warmup before timing to exclude Taichi first-time JIT overhead.',
    )
    parser.add_argument(
        '--vis_save',
        action='store_true',
        help='Also save rollout frames as pngs under mpm3d_vis/.',
    )
    parser.add_argument(
        '--vis_no_gui',
        action='store_true',
        help='Disable interactive GUI playback (use with --vis_save).',
    )
    options = parser.parse_args()

    # initialization
    scene = scene_lib.Scene()
    scene_lib.robot(scene)
    # scene.add_rect(0.4, 0.4, 0.2, 0.1, 0.3, 0.1, -1, 1)
    scene.finalize()
    cfg.allocate_fields()

    kernels.init(
        np.array(scene.x, dtype=np.float32),
        np.array(scene.actuator_id, dtype=np.int32),
        np.array(scene.particle_type, dtype=np.int32),
    )

    losses = []

    if options.warmup:
        # Exclude first-time Taichi JIT/initialization from per-iter timing.
        ti.ad.clear_all_gradients()
        _ = kernels.forward()
        cfg.loss.grad[None] = 1
        kernels.backward()

    iters_range = range(options.iters)
    pbar: Optional[object] = None
    if options.progress and tqdm is not None:
        pbar = tqdm(iters_range, total=options.iters, dynamic_ncols=True)
        iters_iter = pbar
    else:
        iters_iter = iters_range

    for iter in iters_iter:
        t = time.time()
        ti.ad.clear_all_gradients()
        l = kernels.forward()
        losses.append(l)
        cfg.loss.grad[None] = 1
        kernels.backward()
        per_iter_time = time.time() - t
        if pbar is not None:
            try:
                pbar.set_postfix(loss=float(l), per_iter_s=f"{per_iter_time:.2f}")
            except Exception:
                pass
        else:
            print('i=', iter, 'loss=', l, F' per iter {per_iter_time:.2f}s')
        learning_rate = 30
        kernels.learn(learning_rate)

        if options.vis_interval > 0 and iter % options.vis_interval == 0:
            out_dir = f"outputs/mpm3d_vis/iter{iter:04d}" if options.vis_save else None
            print(
                "Visualizing rollout "
                + ("(GUI)" if not options.vis_no_gui else "(no GUI)")
                + (f" + saving to '{out_dir}'" if out_dir is not None else "")
                + " ..."
            )
            viz.visualize_rollout(
                iter_idx=iter,
                stride=max(1, int(options.vis_stride)),
                save_folder=out_dir,
                interactive=(not options.vis_no_gui),
            )

        if iter % 20 == 19:
            viz.dump_particles_bin(iter_idx=iter, start_s=7, step_s=2)

    if plt is not None:
        plt.title("Optimization of Initial Velocity")
        plt.ylabel("Loss")
        plt.xlabel("Gradient Descent Iterations")
        plt.plot(losses)
        plt.show()
    else:
        print("matplotlib 未安装，跳过 loss 曲线绘制。")


if __name__ == '__main__':
    main()
