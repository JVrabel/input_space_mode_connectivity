"""
Section 4.3 — Mode connectivity in untrained (randomly initialized) models.
Reproduces Figure 7 from the paper.

Two class-optimal synthetic inputs are generated via feature visualization
by optimization (FVO) on an untrained ResNet-18. The barrier is bypassed
in two rounds, producing a 4-segment low-loss path A→B'S1→B'→B'S2→C.

Usage:
    python sec4_3_untrained_model.py
    python sec4_3_untrained_model.py --target_class 3
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import (
    load_resnet18_untrained,
    linear_interpolation, compute_losses, find_barrier,
    optimize_point, feature_visualization,
    plot_three_rounds,
)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# FVO parameters (Sec 4.3, Fig. 7)
FVO_LR = 0.05
FVO_ITERS = 8192
FVO_WD = 1e-7
FVO_EARLY_STOP = 0.0005
FVO_NOISE_SCALE = 0.01
FVO_PENALTY_WEIGHT = 2.5e-7  # 0.00000025

# Midpoint / barrier optimization parameters
OPT_LR = 0.05
OPT_ITERS = 4096
OPT_WD = 1e-7
OPT_EARLY_STOP = 0.0005


def run(target_class=1, seed=0, N=200):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    class_name = CIFAR10_CLASSES[target_class]

    # Model created before seeding (paper Sec 4.3, Fig 7)
    model = load_resnet18_untrained(device)

    # Seed once, then generate both tensors sequentially (no re-seeding)
    torch.manual_seed(seed)

    print(f"FVO: input A (class '{class_name}', with hf penalty)...")
    x_a, la = feature_visualization(
        model, target_class,
        lr=FVO_LR, iterations=FVO_ITERS, loss_threshold=FVO_EARLY_STOP,
        lambda_hf=FVO_PENALTY_WEIGHT, noise_scale=FVO_NOISE_SCALE,
        weight_decay=FVO_WD, device=device, verbose=True,
    )
    print(f"  loss A = {la:.6f}")

    print(f"FVO: input C (class '{class_name}', no penalty)...")
    x_c, lc = feature_visualization(
        model, target_class,
        lr=FVO_LR, iterations=FVO_ITERS, loss_threshold=FVO_EARLY_STOP,
        lambda_hf=0.0, noise_scale=FVO_NOISE_SCALE,
        weight_decay=FVO_WD, device=device, verbose=True,
    )
    print(f"  loss C = {lc:.6f}")

    # Primary interpolation
    print(f"Primary interpolation ({N} steps)...")
    alphas_p, pts_p = linear_interpolation(x_a, x_c, steps=N)
    losses_p = compute_losses(model, pts_p, target_class)
    idx_p, _, bp = find_barrier(alphas_p, pts_p, losses_p)
    print(f"Primary barrier: loss={losses_p[idx_p]:.4f}")

    # Round 1: free (unconstrained) optimization of primary barrier
    print("Round 1: optimizing primary barrier...")
    b_prime, bl = optimize_point(
        model, bp, target_class,
        lr=OPT_LR, iterations=OPT_ITERS,
        weight_decay=OPT_WD, loss_threshold=OPT_EARLY_STOP,
        verbose=True,
    )
    print(f"B' loss: {bl:.5f}")

    a1, p1 = linear_interpolation(x_a, b_prime, steps=N)
    a2, p2 = linear_interpolation(b_prime, x_c, steps=N)
    l1 = compute_losses(model, p1, target_class)
    l2 = compute_losses(model, p2, target_class)

    # Round 2: optimize each secondary barrier
    def bypass_seg(start, end, losses_seg, seg_name=""):
        _, pts = linear_interpolation(start, end, steps=N)
        i_max = int(np.argmax(losses_seg))
        print(f"  {seg_name} barrier loss={losses_seg[i_max]:.4f}")
        opt, ol = optimize_point(
            model, pts[i_max], target_class,
            lr=OPT_LR, iterations=OPT_ITERS,
            weight_decay=OPT_WD, loss_threshold=OPT_EARLY_STOP,
        )
        print(f"  optimized to {ol:.5f}")
        _, pa = linear_interpolation(start, opt, steps=N)
        _, pb = linear_interpolation(opt, end, steps=N)
        return compute_losses(model, pa, target_class), compute_losses(model, pb, target_class)

    print("Round 2: secondary barriers...")
    l1_1, l1_2 = bypass_seg(x_a, b_prime, l1, "A→B'")
    l2_1, l2_2 = bypass_seg(b_prime, x_c, l2, "B'→C")

    plot_three_rounds(
        alphas_p, losses_p,
        [a1, a2], [l1, l2],
        [None]*4, [l1_1, l1_2, l2_1, l2_2],
        title=f"Untrained ResNet-18 — class '{class_name}' (Sec 4.3, Fig. 7)",
    )
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--target_class", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--N", type=int, default=200)
    args = p.parse_args()
    run(**vars(args))
