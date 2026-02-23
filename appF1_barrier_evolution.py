"""
Appendix F.1 — Barrier evolution during training.
Reproduces Figure 10 from the paper:
  - Single-pair FVO visualization (loss curve + two optimal inputs)
  - Averaged primary loss curves across batch checkpoints
  - 2-panel: averaged curves + max loss vs batch ID

Two modes:
  quick  — 5 batches, 1 run/class, 3 classes (~2 min on GPU). Sanity check.
  full   — 30 batches, 5 runs/class, 10 classes (~30 min on GPU). Paper quality.

Usage:
    python appF1_barrier_evolution.py --mode quick
    python appF1_barrier_evolution.py --mode full
    python appF1_barrier_evolution.py --mode full --save_dir ./batch_results
"""

import argparse
import ast
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import _high_freq_penalty, CIFAR10_MEAN, CIFAR10_STD


# ── FVO (feature visualization by optimization) ─────────────────────────────

def fvo_input(model, target_class, input_shape=(3, 32, 32), device=None,
              use_penalty=True, max_iterations=4096, lr=0.05,
              weight_decay=1e-7, penalty_weight=2.5e-7,
              early_stop=0.005, input_scale=0.01):
    """Optimize a random input to minimize CE for target_class.

    Tracks best tensor (by post-step CE).
    Returns (best_tensor, final_loss, loss_trace).
    """
    device = device or next(model.parameters()).device
    x = (torch.randn((1,) + input_shape, device=device) * input_scale).requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr, weight_decay=weight_decay)
    target = torch.tensor([target_class], device=device)

    best_loss = float("inf")
    best_tensor = None
    loss_trace = []

    for i in range(max_iterations):
        optimizer.zero_grad()
        ce = F.cross_entropy(model(x), target)
        total = ce + penalty_weight * _high_freq_penalty(x) if use_penalty else ce
        total.backward()
        optimizer.step()

        with torch.no_grad():
            post_loss = F.cross_entropy(model(x), target).item()
        if post_loss < best_loss:
            best_loss = post_loss
            best_tensor = x.detach().clone()
        loss_trace.append(post_loss)

        if post_loss < early_stop:
            break

    with torch.no_grad():
        final_loss = F.cross_entropy(model(best_tensor), target).item()
    return best_tensor, final_loss, loss_trace


def interpolation_curve(model, t1, t2, target_class, steps=200, device=None):
    """Per-point CE along a linear interpolation. Returns (alphas, losses_list)."""
    device = device or next(model.parameters()).device
    t1, t2 = t1.to(device), t2.to(device)
    if t1.dim() == 4:
        t1 = t1.squeeze(0)
    if t2.dim() == 4:
        t2 = t2.squeeze(0)

    alphas = torch.linspace(0, 1, steps, device=device)
    pts = torch.stack([(1 - a) * t1 + a * t2 for a in alphas])

    with torch.no_grad():
        logits = model(pts)
    targets = torch.full((steps,), target_class, dtype=torch.long, device=device)
    losses = nn.CrossEntropyLoss(reduction="none")(logits, targets)
    return alphas.cpu().numpy(), losses.cpu().numpy().tolist()


# ── Model setup ──────────────────────────────────────────────────────────────

def make_model(device):
    """Torchvision ResNet-18 adapted for CIFAR-10 (matching original experiment)."""
    model = models.resnet18(weights=None, num_classes=10)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(device)


# ── Plot 1: single-pair FVO visualization (App. F.1, Fig. 10) ───────────────

def plot_single_pair(model, target_class=1, device=None):
    """Generate two FVO inputs, interpolate, show loss curve + both images."""
    device = device or next(model.parameters()).device
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"FVO input 1 (with penalty), class {target_class}...")
    t1, loss1, _ = fvo_input(model, target_class, device=device, use_penalty=True)
    print(f"FVO input 2 (no penalty), class {target_class}...")
    t2, loss2, _ = fvo_input(model, target_class, device=device, use_penalty=False)

    alphas, losses = interpolation_curve(model, t1, t2, target_class, device=device)

    def _vis(tensor):
        img = tensor.detach().squeeze().cpu().numpy().transpose(1, 2, 0)
        return (img - img.min()) / (img.max() - img.min())

    fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    axs[0].plot(alphas, losses, color="tab:blue", linewidth=2, label=r"$A \to C$")
    axs[0].set_xlabel(r"Interpolation coefficient ($\alpha$)", fontsize=12)
    axs[0].set_ylabel("Loss", fontsize=12)
    axs[0].set_title("Loss Curve", fontsize=13)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].imshow(_vis(t1), interpolation="none")
    axs[1].set_title(f"Optimal input 1, Loss: {loss1:.4f}", fontsize=12)
    axs[1].axis("off")

    axs[2].imshow(_vis(t2), interpolation="none")
    axs[2].set_title(f"Optimal input 2, Loss: {loss2:.4f}", fontsize=12)
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()
    return fig


# ── Plot 2: averaged loss curves (App. F.1, Fig. 10 left) ───────────────────

def plot_averaged_curves(results, batches_to_plot=None):
    """Averaged primary loss curves with std shading for selected batches."""
    batch_ids = sorted(results["batch_idx"].unique())
    if batches_to_plot is None:
        n = len(batch_ids)
        indices = np.linspace(0, n - 1, min(5, n), dtype=int)
        batches_to_plot = [batch_ids[i] for i in indices]

    colors = ["tab:blue", "darkorange", "mediumorchid", "c", "brown"]
    plt.figure(figsize=(10, 6))

    for idx, batch in enumerate(batches_to_plot):
        batch_df = results[results["batch_idx"] == batch]
        curves = np.array([_parse_curve(r.primary_loss_curve) for r in batch_df.itertuples()])
        if len(curves) == 0:
            continue
        mean_c = curves.mean(axis=0)
        std_c = curves.std(axis=0)
        c = colors[idx % len(colors)]
        plt.plot(mean_c, label=f"Batch {batch}", alpha=0.7, color=c)
        plt.fill_between(range(len(mean_c)),
                         mean_c - std_c, mean_c + std_c, alpha=0.1, color=c)

    plt.title("Averaged Primary Loss Curves with Std Deviation", fontsize=13)
    plt.xlabel("Interpolation Points", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ── Plot 3: 2-panel (App. F.1, Fig. 10 right) ───────────────────────────────

def plot_barrier_evolution(results, batches_to_plot=None, metric="max"):
    """Two-panel: averaged loss curves (left) and metric vs batch ID (right)."""
    batch_ids = sorted(results["batch_idx"].unique())
    if batches_to_plot is None:
        n = len(batch_ids)
        indices = np.linspace(0, n - 1, min(5, n), dtype=int)
        batches_to_plot = [batch_ids[i] for i in indices]

    colors = ["tab:blue", "darkorange", "mediumorchid", "c", "brown"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    for idx, batch in enumerate(batches_to_plot):
        batch_df = results[results["batch_idx"] == batch]
        curves = np.array([_parse_curve(r.primary_loss_curve) for r in batch_df.itertuples()])
        if len(curves) == 0:
            continue
        mean_c = curves.mean(axis=0)
        std_c = curves.std(axis=0)
        xs = np.linspace(0, 1, len(mean_c))
        c = colors[idx % len(colors)]
        ax1.plot(xs, mean_c, label=f"Batch {batch}", alpha=0.7, color=c)
        ax1.fill_between(xs, np.maximum(0, mean_c - std_c),
                         mean_c + std_c, alpha=0.1, color=c)

    ax1.set_title("Primary loss curves", fontsize=13)
    ax1.set_xlabel(r"Interpolation coefficient ($\alpha$)", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.legend()

    means, stds = [], []
    for b in batch_ids:
        vals = [_metric(_parse_curve(r.primary_loss_curve), metric)
                for r in results[results["batch_idx"] == b].itertuples()]
        means.append(np.mean(vals))
        stds.append(np.std(vals))

    ax2.errorbar(batch_ids, means, yerr=stds, fmt="-o",
                 ecolor="black", capsize=5, elinewidth=1, capthick=1, alpha=0.7)
    ax2.set_xlabel("Batch ID", fontsize=12)
    ax2.set_ylabel("Max Loss" if metric == "max" else "Integrated Loss", fontsize=12)
    ax2.set_title("Batch dependence of max loss value", fontsize=13)

    fig.suptitle("Barrier evolution during training (App. F.1, Fig. 10)", fontsize=14)
    plt.tight_layout()
    plt.show()
    return fig


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_curve(val):
    if isinstance(val, str):
        return np.array(ast.literal_eval(val))
    return np.asarray(val)


def _metric(curve, kind="max"):
    if kind == "max":
        return np.max(curve)
    elif kind == "area":
        return np.trapezoid(curve)
    raise ValueError(f"Unknown metric: {kind}")


def load_results(folder):
    """Load all CSV files from a batch-results folder into one DataFrame."""
    frames = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(folder, fname))
            df["source_file"] = fname
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ── Main computation ─────────────────────────────────────────────────────────

PRESETS = {
    "quick": dict(num_batches=5,  runs_per_class=1, num_classes=3),
    "full":  dict(num_batches=30, runs_per_class=5, num_classes=10),
}


def run(mode="quick", batch_size=512, train_lr=0.001,
        data_root="./data", save_dir=None):
    """Run the full barrier evolution experiment.

    mode='quick': 5 batches, 1 run/class, 3 classes. Fast sanity check.
    mode='full':  30 batches, 5 runs/class, 10 classes. Paper quality.
    """
    cfg = PRESETS[mode]
    num_batches = cfg["num_batches"]
    runs_per_class = cfg["runs_per_class"]
    num_classes = cfg["num_classes"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {mode} — {num_batches} batches, {runs_per_class} runs/class, "
          f"{num_classes} classes = {runs_per_class * num_classes} pairs/checkpoint")

    # CIFAR-10 with augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN.tolist(), CIFAR10_STD.tolist()),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = make_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_lr)
    criterion = nn.CrossEntropyLoss()

    all_rows = []

    def _measure(batch_idx):
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        for cls in range(num_classes):
            for r in range(runs_per_class):
                t1, l1, tr1 = fvo_input(model, cls, device=device, use_penalty=True)
                t2, l2, tr2 = fvo_input(model, cls, device=device, use_penalty=False)
                _, curve = interpolation_curve(model, t1, t2, cls, steps=200, device=device)
                all_rows.append({
                    "target_class": cls, "run": r, "batch_idx": batch_idx,
                    "final_loss_1": l1, "final_loss_2": l2,
                    "primary_loss_curve": curve, "loss_trace_1": tr1, "loss_trace_2": tr2,
                })
            print(f"  batch {batch_idx}, class {cls}: max={max(curve):.4f}")
        model.train()
        for p in model.parameters():
            p.requires_grad = True

    # --- Plot 1: single-pair visualization on untrained model ---
    print("\n=== Single-pair FVO visualization (untrained model) ===")
    plot_single_pair(model, target_class=1, device=device)

    # --- Batch 0 (untrained) ---
    print("\n=== Measuring batch 0 (untrained) ===")
    _measure(0)

    # --- Train batch by batch ---
    batch_count = 0
    for X, y in trainloader:
        if batch_count >= num_batches:
            break
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        criterion(model(X), y).backward()
        optimizer.step()
        batch_count += 1
        print(f"\n=== Measuring batch {batch_count} ===")
        _measure(batch_count)

    results = pd.DataFrame(all_rows)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for bidx in results["batch_idx"].unique():
            subset = results[results["batch_idx"] == bidx]
            subset.to_csv(os.path.join(save_dir, f"all_results_batch_{bidx}.csv"), index=False)
        print(f"\nSaved results to {save_dir}/")

    # --- Plot 2: averaged loss curves ---
    print("\n=== Averaged loss curves ===")
    plot_averaged_curves(results)

    # --- Plot 3: 2-panel (curves + metric vs batch) ---
    print("\n=== Barrier evolution (2-panel) ===")
    plot_barrier_evolution(results, metric="max")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="quick", choices=["quick", "full"])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--train_lr", type=float, default=0.001)
    p.add_argument("--data_root", default="./data")
    p.add_argument("--save_dir", default=None)
    args = p.parse_args()
    run(**vars(args))


if __name__ == "__main__":
    main()
