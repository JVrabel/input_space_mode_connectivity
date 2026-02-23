"""
Section 4.2 — Adversarial attacks and barrier statistics.
Reproduces Figures 4–6 from the paper.

Part A: Creates an adversarial example from a source image of a different
        class, shows the barrier between the real image and the adversarial
        example, and visualizes both.

Part B: Computes (or loads pre-computed) barrier statistics — within-class
        vs real-adversarial pairs across all 1000 ImageNet classes — and
        plots boxplots of barrier heights and gaps (Figure 6).

Usage:
    python sec4_2_adversarial_attacks.py
    python sec4_2_adversarial_attacks.py --compute --stat_tensors ./data_sel/stat_tensors
    python sec4_2_adversarial_attacks.py --target_class 574 --source_class 763
"""

import argparse
import csv
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils import (
    load_googlenet, load_saved_tensors,
    linear_interpolation, compute_losses, find_barrier,
    plot_loss_curve, plot_images, unnormalize,
    _high_freq_penalty,
)

IMAGENET_CLASSES = {
    955: "jackfruit", 954: "banana", 574: "golf ball",
    409: "analog clock", 1: "goldfish", 71: "scorpion",
    763: "revolver",
}


# ── Adversarial attack ───────────────────────────────────────────────────────

def create_adversarial(model, source_tensor, target_class,
                       lr=0.005, iterations=512, lambda_mse=0.1,
                       lambda_hf=1e-8, early_stop_loss=None, verbose=False):
    """Optimize a source image to minimize CE for target_class (targeted attack).

    Returns (adversarial_tensor, final_loss).
    """
    device = next(model.parameters()).device
    carrier = source_tensor.detach().clone().to(device)
    x = source_tensor.detach().clone().to(device).requires_grad_(True)
    optimizer = optim.Adam([x], lr=lr)
    target = torch.tensor([target_class], dtype=torch.long, device=device)

    model.eval()
    for i in range(iterations):
        optimizer.zero_grad()
        ce = F.cross_entropy(model(x), target)
        dev = torch.mean((x - carrier) ** 2)
        hf = _high_freq_penalty(x - carrier)
        total = ce + lambda_mse * dev + lambda_hf * hf
        total.backward()
        optimizer.step()
        if verbose and i % 100 == 0:
            print(f"  iter {i:4d}  CE={ce.item():.5f}")
        if early_stop_loss is not None and i > 32 and ce.item() < early_stop_loss:
            break

    return x.detach(), ce.item()


# ── Single-pair demo (Figures 4–5) ───────────────────────────────────────────

def run_single_attack(target_class=574, source_class=763,
                      tensor_dir="./data_sel/saved_tensors"):
    """Part A: single adversarial attack demonstration (Figures 4–5)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_googlenet(device)

    target_name = IMAGENET_CLASSES.get(target_class, str(target_class))
    source_name = IMAGENET_CLASSES.get(source_class, str(source_class))
    print(f"Target class: {target_name} ({target_class})")
    print(f"Source class: {source_name} ({source_class})")

    # Real ImageNet images: golf ball (sel_1) and revolver (sel_1)
    x_a = load_saved_tensors(tensor_dir, target_class, (1,), device)[0]
    x_k = load_saved_tensors(tensor_dir, source_class, (1,), device)[0]

    # Create adversarial example K' from K
    print("Creating adversarial example K'...")
    x_k_adv, adv_loss = create_adversarial(
        model, x_k, target_class,
        lr=0.005, iterations=1024,
        lambda_mse=0.1, lambda_hf=1e-7, verbose=True,
    )
    print(f"Adversarial loss: {adv_loss:.5f}")

    # Show: A, K, K'
    loss_a = compute_losses(model, x_a.unsqueeze(0) if x_a.dim() == 3 else x_a, target_class)[0]
    loss_k = compute_losses(model, x_k.unsqueeze(0) if x_k.dim() == 3 else x_k, target_class)[0]
    plot_images(
        [x_a, x_k, x_k_adv],
        labels=[f"A ({target_name})", f"K ({source_name})", f"K' (adversarial)"],
        losses=[loss_a, loss_k, adv_loss],
    )
    plt.suptitle("Adversarial attack (Sec 4.2, Fig. 4)", fontsize=14)
    plt.show()

    # Barrier between A and K'
    print("Computing barrier A → K'...")
    alphas, points = linear_interpolation(x_a, x_k_adv, steps=250)
    losses = compute_losses(model, points, target_class)
    idx, alpha_max, _ = find_barrier(alphas, points, losses)
    print(f"Barrier at α={alpha_max:.3f}, loss={losses[idx]:.4f}")

    plot_loss_curve(alphas, losses,
                    title=f"Barrier: real ({target_name}) → adversarial")
    plt.show()


# ── Barrier statistics computation ───────────────────────────────────────────

def _load_stat_tensors(tensor_dir, class_label):
    """Load the per-class tensor bundle from stat_tensors/.

    Each file is a list of (tensor, filename) tuples.
    """
    path = os.path.join(tensor_dir, f"class_{class_label}_tensors_with_names.pt")
    return torch.load(path, map_location="cpu", weights_only=False)


def _ce_single(model, tensor, target_class, device):
    """Cross-entropy of a single image tensor for target_class."""
    x = tensor.to(device)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    with torch.no_grad():
        return F.cross_entropy(model(x), torch.tensor([target_class], device=device)).item()


def _interpolation_max_loss(model, t1, t2, target_class, device, steps=250):
    """Max CE along a linear interpolation between two tensors."""
    t1, t2 = t1.to(device), t2.to(device)
    if t1.dim() == 3:
        t1 = t1.unsqueeze(0)
    if t2.dim() == 3:
        t2 = t2.unsqueeze(0)
    target = torch.tensor([target_class], device=device)
    max_loss = 0.0
    with torch.no_grad():
        for alpha in torch.linspace(0, 1, steps, device=device):
            mixed = (1 - alpha) * t1 + alpha * t2
            loss = F.cross_entropy(model(mixed), target).item()
            if loss > max_loss:
                max_loss = loss
    return max_loss


def compute_within_class_barriers(model, stat_tensor_dir, classes=range(1000),
                                  pairs_per_class=7, interp_steps=250,
                                  output_csv=None):
    """Compute within-class interpolation barriers for all classes.

    For each class, consecutive pairs of real images are interpolated and
    the maximum cross-entropy along the path is recorded.

    Returns a DataFrame with columns:
        Class, Loss1, Loss2, Max Interpolated Loss
    """
    device = next(model.parameters()).device
    rows = []

    for cls in classes:
        try:
            data = _load_stat_tensors(stat_tensor_dir, cls)
        except FileNotFoundError:
            continue
        for p in range(pairs_per_class):
            idx1, idx2 = p * 2, p * 2 + 1
            if idx2 >= len(data):
                break
            t1, name1 = data[idx1]
            t2, name2 = data[idx2]
            loss1 = _ce_single(model, t1, cls, device)
            loss2 = _ce_single(model, t2, cls, device)
            max_loss = _interpolation_max_loss(model, t1, t2, cls, device, interp_steps)
            rows.append({"Class": cls, "Loss1": loss1, "Loss2": loss2,
                         "Max Interpolated Loss": max_loss})
        print(f"  within-class: class {cls} done ({len(rows)} pairs total)")

    df = pd.DataFrame(rows)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved within-class barriers to {output_csv}")
    return df


def compute_adversarial_barriers(model, stat_tensor_dir, classes=range(1000),
                                 pairs_per_class=7, interp_steps=250,
                                 attack_iters=512, attack_lr=0.005,
                                 attack_lambda_mse=0.1, attack_lambda_hf=1e-8,
                                 output_csv=None):
    """Compute cross-class (adversarial) interpolation barriers.

    For each class, a real image from a randomly chosen other class is
    adversarially optimized toward the target class. The barrier between
    the real target-class image and the adversarial image is recorded.

    Returns a DataFrame with columns:
        Class, Loss1, Loss2, Max Interpolated Loss
    """
    device = next(model.parameters()).device
    all_classes = list(classes)
    rows = []

    for cls in all_classes:
        try:
            data_cls = _load_stat_tensors(stat_tensor_dir, cls)
        except FileNotFoundError:
            continue
        for p in range(pairs_per_class):
            idx = p * 2
            if idx >= len(data_cls):
                break
            t1, _ = data_cls[idx]

            other_cls = random.choice([c for c in all_classes if c != cls])
            try:
                data_other = _load_stat_tensors(stat_tensor_dir, other_cls)
            except FileNotFoundError:
                continue
            if idx >= len(data_other):
                continue
            t2, _ = data_other[idx]

            loss1 = _ce_single(model, t1, cls, device)

            t2_adv, adv_loss = create_adversarial(
                model, t2, cls,
                lr=attack_lr, iterations=attack_iters,
                lambda_mse=attack_lambda_mse, lambda_hf=attack_lambda_hf,
                early_stop_loss=loss1,
            )
            loss2 = _ce_single(model, t2_adv, cls, device)
            max_loss = _interpolation_max_loss(model, t1, t2_adv, cls, device, interp_steps)
            rows.append({"Class": cls, "Loss1": loss1, "Loss2": loss2,
                         "Max Interpolated Loss": max_loss})

        print(f"  adversarial: class {cls} done ({len(rows)} pairs total)")

    df = pd.DataFrame(rows)
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"Saved adversarial barriers to {output_csv}")
    return df


def postprocess_barriers(df, drop_top_n=2):
    """Drop the top-N pairs with largest |Loss1 - Loss2| per class."""
    df = df.copy()
    df["_abs_diff"] = (df["Loss1"] - df["Loss2"]).abs()

    kept_indices = []
    for _, group in df.groupby("Class"):
        n_keep = max(len(group) - drop_top_n, 0)
        kept_indices.extend(group.nsmallest(n_keep, "_abs_diff").index.tolist())

    out = df.loc[kept_indices].reset_index(drop=True)
    return out.drop(columns="_abs_diff")


# ── Plotting / loading statistics (Figure 6) ─────────────────────────────────

def plot_statistics(within, cross, title="Barrier height statistics (Sec 4.2, Fig. 6)"):
    """Boxplot of within-class vs adversarial barrier statistics."""
    within = within.copy()
    cross = cross.copy()
    within["Diff"] = within["Max Interpolated Loss"] - within[["Loss1", "Loss2"]].max(axis=1)
    cross["Diff"] = cross["Max Interpolated Loss"] - cross[["Loss1", "Loss2"]].max(axis=1)

    data = [within["Max Interpolated Loss"], cross["Max Interpolated Loss"],
            within["Diff"], cross["Diff"]]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    positions = [1, 1.5, 2, 2.5]
    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.25)

    colors = ["royalblue", "firebrick", "royalblue", "firebrick"]
    for box, c in zip(bp["boxes"], colors):
        box.set(facecolor=c, edgecolor="black", linewidth=1.5)
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.5)

    ax.set_xticks([1.25, 2.25])
    ax.set_xticklabels(["Barrier max", "Barrier gap"], fontsize=12)
    ax.set_ylabel("Loss value", fontsize=14)
    ax.set_title(title, fontsize=14)

    legend_elements = [
        Patch(facecolor="royalblue", edgecolor="black", label="Within-class"),
        Patch(facecolor="firebrick", edgecolor="black", label="Adversarial attack"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=12)
    ax.grid(False)
    plt.tight_layout()
    plt.show()

    labels = ["Within-class barrier max", "Adversarial barrier max",
              "Within-class gap", "Adversarial gap"]
    for label, d in zip(labels, data):
        print(f"{label}:  mean={d.mean():.2f},  median={d.median():.2f}")


def run_statistics(data_dir="../data", compute=False, stat_tensor_dir=None):
    """Part B: barrier height statistics across 1000 classes (Figure 6).

    If compute=True, runs the full computation from stat_tensors/.
    Otherwise loads pre-computed CSVs from data_dir.
    """
    within_path = os.path.join(data_dir, "interpolation_processed.csv")
    cross_path = os.path.join(data_dir, "cross_class_interpolation_processed.csv")

    if compute:
        stat_tensor_dir = stat_tensor_dir or os.path.join(data_dir, "stat_tensors")
        print(f"Computing barrier statistics from {stat_tensor_dir} ...")
        model = load_googlenet(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        within_raw = compute_within_class_barriers(model, stat_tensor_dir)
        within = postprocess_barriers(within_raw)
        within.to_csv(within_path, index=False)
        print(f"Saved {within_path}")

        cross_raw = compute_adversarial_barriers(model, stat_tensor_dir)
        cross = postprocess_barriers(cross_raw)
        cross.to_csv(cross_path, index=False)
        print(f"Saved {cross_path}")
    else:
        print(f"Loading {within_path}")
        within = pd.read_csv(within_path)
        print(f"Loading {cross_path}")
        cross = pd.read_csv(cross_path)

    plot_statistics(within, cross)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target_class", type=int, default=574)
    p.add_argument("--source_class", type=int, default=763)
    p.add_argument("--tensor_dir", default="./data_sel/saved_tensors")
    p.add_argument("--data_dir", default="./data_sel")
    p.add_argument("--skip_attack", action="store_true",
                   help="Skip single attack demo, only show statistics")
    p.add_argument("--compute", action="store_true",
                   help="Compute statistics from scratch instead of loading CSVs")
    p.add_argument("--stat_tensors", default=None,
                   help="Path to stat_tensors/ (default: <data_dir>/stat_tensors)")
    args = p.parse_args()

    if not args.skip_attack:
        run_single_attack(args.target_class, args.source_class, args.tensor_dir)
    run_statistics(args.data_dir, compute=args.compute, stat_tensor_dir=args.stat_tensors)


if __name__ == "__main__":
    main()
