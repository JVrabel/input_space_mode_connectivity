"""
Shared utilities for input space mode connectivity experiments.

Contains: model loading, linear interpolation, barrier bypass optimization,
feature visualization by optimization (FVO), and plotting helpers.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image

from models import ResNet18CIFAR


# ── Constants ────────────────────────────────────────────────────────────────

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])
CIFAR10_MEAN  = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD   = np.array([0.2023, 0.1994, 0.2010])

IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
])

CIFAR10_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN.tolist(), CIFAR10_STD.tolist()),
])


# ── Model loading ────────────────────────────────────────────────────────────

def load_googlenet(device=None):
    """Pretrained GoogLeNet (Inception v1) on ImageNet."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_resnet18_imagenet(device=None):
    """Pretrained ResNet-18 on ImageNet."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_vit(device=None):
    """Pretrained ViT-B/16 on ImageNet."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_resnet18_cifar10(weights_path, device=None, num_classes=10):
    """Trained ResNet-18 for CIFAR-10 from a checkpoint."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18CIFAR(num_classes=num_classes).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_resnet18_untrained(device=None, num_classes=10, seed=None):
    """Randomly initialized (torchvision) ResNet-18 for CIFAR-10."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)
    model = models.resnet18(weights=None, num_classes=num_classes).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ── Data loading ─────────────────────────────────────────────────────────────

def load_saved_tensors(tensor_dir, class_id, sel_indices=(0, 1), device=None):
    """Load pre-saved preprocessed ImageNet tensors (class_{id}_sel_{idx}.pt)."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for idx in sel_indices:
        path = os.path.join(tensor_dir, f"class_{class_id}_sel_{idx}.pt")
        out.append(torch.load(path, map_location=device, weights_only=True))
    return out


def get_cifar10_loaders(batch_size=128, data_root="./data", num_workers=2):
    """CIFAR-10 train and test data loaders."""
    train_ds = datasets.CIFAR10(root=data_root, train=True,  download=True,
                                transform=CIFAR10_TRANSFORM)
    test_ds  = datasets.CIFAR10(root=data_root, train=False, download=True,
                                transform=CIFAR10_TRANSFORM)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    return train_loader, test_loader


# ── Interpolation & loss computation ─────────────────────────────────────────

def linear_interpolation(x_a, x_c, steps=1000):
    """Linearly interpolate between two tensors.

    Returns (alphas, points) where points[i] = (1-α_i)*x_a + α_i*x_c.
    """
    x_a = x_a.detach().squeeze(0) if x_a.dim() == 4 else x_a.detach()
    x_c = x_c.detach().squeeze(0) if x_c.dim() == 4 else x_c.detach()
    alphas = torch.linspace(0, 1, steps, device=x_a.device)
    points = torch.zeros((steps, *x_a.shape), device=x_a.device)
    for i, a in enumerate(alphas):
        points[i] = (1 - a) * x_a + a * x_c
    return alphas, points


def compute_losses(model, points, target_class, batch_size=64):
    """Cross-entropy loss at each point along an interpolation path.

    Returns a numpy array of shape (N,).
    """
    device = next(model.parameters()).device
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size].to(device)
            if batch.dim() == 3:
                batch = batch.unsqueeze(1)
            targets = torch.full((len(batch),), target_class,
                                 dtype=torch.long, device=device)
            all_losses.append(loss_fn(model(batch), targets).cpu())
    return torch.cat(all_losses).numpy()


def find_barrier(alphas, points, losses):
    """Index, alpha value, and tensor of the max-loss interpolated point."""
    idx = int(np.argmax(losses))
    return idx, alphas[idx].item(), points[idx]


# ── Barrier bypass optimization ──────────────────────────────────────────────

def _high_freq_penalty(tensor):
    """Total-variation-style smoothness penalty."""
    dx = torch.abs(tensor[..., :, 1:] - tensor[..., :, :-1])
    dy = torch.abs(tensor[..., 1:, :] - tensor[..., :-1, :])
    return dx.sum() + dy.sum()


def bypass_barrier(model, x_a, x_c, barrier_point, target_class,
                   lr=0.005, iterations=1024,
                   lambda_mse=0.1, lambda_hf=1e-8, verbose=False):
    """Optimize barrier point B → B' on the hyperplane orthogonal to AC.

    Implements the simplified mode-connecting procedure (Fort & Jastrzebski, 2019)
    adapted for input space, as described in Section 3.2 of the paper.

    Returns (optimized_point, loss_history).
    """
    device = next(model.parameters()).device
    x_a = x_a.detach().squeeze(0) if x_a.dim() == 4 else x_a.detach()
    x_c = x_c.detach().squeeze(0) if x_c.dim() == 4 else x_c.detach()
    anchor = barrier_point.detach().clone().to(device)
    v = (x_c - x_a).flatten().to(device)

    current = anchor.clone().unsqueeze(0).requires_grad_(True)
    optimizer = torch.optim.Adam([current], lr=lr)
    target = torch.tensor([target_class], dtype=torch.long, device=device)

    loss_history = []
    model.eval()

    for i in range(iterations):
        optimizer.zero_grad()
        ce = F.cross_entropy(model(current), target)
        dev = torch.mean((current.squeeze(0) - anchor) ** 2)
        hf = _high_freq_penalty(current.squeeze(0) - anchor)
        total = ce + lambda_mse * dev + lambda_hf * hf
        total.backward()

        with torch.no_grad():
            if current.grad is not None:
                g = current.grad.flatten()
                current.grad -= (torch.dot(g, v) / torch.dot(v, v)) * v.view_as(current.grad)

        optimizer.step()
        loss_history.append(ce.item())
        if verbose and (i % 200 == 0 or i == iterations - 1):
            print(f"  iter {i:4d}  CE={ce.item():.6f}")

    return current.detach().squeeze(0), loss_history


# ── Feature visualization by optimization (FVO) ─────────────────────────────

def feature_visualization(model, target_class, input_shape=(3, 32, 32),
                          lr=0.05, iterations=4096, loss_threshold=0.0005,
                          noise_scale=0.01, weight_decay=1e-7,
                          lambda_hf=0.0, seed=None, device=None,
                          verbose=False):
    """Generate a synthetic input minimizing cross-entropy for target_class.

    Used in Section 4.3 (untrained models) and Appendix A (trained models).
    Tracks best tensor found during optimization.
    Returns (optimized_input, final_loss).
    """
    device = device or next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)

    x = (torch.randn((1,) + input_shape, device=device) * noise_scale).requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr, weight_decay=weight_decay)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    loss_fn = nn.CrossEntropyLoss()
    model.eval()

    best_loss = float("inf")
    best_tensor = None

    for i in range(iterations):
        optimizer.zero_grad()
        ce = loss_fn(model(x), target)
        loss = ce + lambda_hf * _high_freq_penalty(x) if lambda_hf > 0 else ce
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            post_loss = loss_fn(model(x), target).item()
        if post_loss < best_loss:
            best_loss = post_loss
            best_tensor = x.detach().clone()
        if verbose and i % 100 == 0:
            print(f"  iter {i:4d}  CE={post_loss:.5f}")
        if post_loss < loss_threshold:
            if verbose:
                print(f"  converged at iter {i}, CE={post_loss:.5f}")
            break

    return best_tensor.squeeze(0), best_loss


def optimize_point(model, tensor, target_class, lr=0.05, iterations=4096,
                   weight_decay=1e-7, loss_threshold=0.0005, verbose=False):
    """Freely optimize an existing input tensor to minimize CE loss.

    Used in Section 4.3 for bypassing barriers in untrained models —
    no gradient projection or deviation penalty, just Adam on the CE.
    Returns (optimized_tensor, final_loss).
    """
    device = next(model.parameters()).device
    x = tensor.detach().clone()
    if x.dim() == 3:
        x = x.unsqueeze(0)
    x = x.to(device).requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr, weight_decay=weight_decay)
    target = torch.tensor([target_class], dtype=torch.long, device=device)
    model.eval()

    for i in range(iterations):
        optimizer.zero_grad()
        ce = F.cross_entropy(model(x), target)
        ce.backward()
        optimizer.step()
        if verbose and i % 100 == 0:
            print(f"  iter {i:4d}  CE={ce.item():.5f}")
        if ce.item() < loss_threshold:
            if verbose:
                print(f"  converged at iter {i}, CE={ce.item():.5f}")
            break

    return x.detach().squeeze(0), ce.item()


# ── Plotting ─────────────────────────────────────────────────────────────────

def unnormalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Normalized (C,H,W) tensor → displayable (H,W,C) numpy array."""
    img = tensor.detach().cpu().numpy()
    if img.ndim == 4:
        img = img.squeeze(0)
    img = img.transpose(1, 2, 0)
    img = std * img + mean
    return np.clip(img, 0, 1)


def plot_loss_curve(alphas, losses, title="Cross-Entropy Loss",
                    ax=None, **kwargs):
    """Plot loss vs. interpolation coefficient α."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3), dpi=150)
    a = alphas.cpu().numpy() if torch.is_tensor(alphas) else np.asarray(alphas)
    l = np.asarray(losses)
    ax.plot(a, l, **kwargs)
    ax.set_xlabel(r"Interpolation coefficient ($\alpha$)", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=13)
    return ax


def plot_barrier_comparison(alphas, losses_primary, losses_seg1, losses_seg2,
                            split_alpha, title=""):
    """Side-by-side: primary A→C barrier vs. bypassed A→B'→C."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5), dpi=150)
    a = alphas.cpu().numpy() if torch.is_tensor(alphas) else np.asarray(alphas)

    ax1.plot(a, losses_primary, color="tab:blue", lw=1.5)
    ax1.set_title(r"Primary $A \to C$", fontsize=13)
    ax1.set_xlabel(r"$\alpha$", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)

    n1, n2 = len(losses_seg1), len(losses_seg2)
    ax2.plot(np.linspace(0, split_alpha, n1), losses_seg1,
             color="tab:green", lw=1.5, label=r"$A \to B'$")
    ax2.plot(np.linspace(split_alpha, 1, n2), losses_seg2,
             color="tab:red", lw=1.5, label=r"$B' \to C$")
    ax2.set_title(r"Bypassed $A \to B' \to C$", fontsize=13)
    ax2.set_xlabel(r"$\alpha$", fontsize=12)
    ax2.legend(fontsize=10)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_three_rounds(alphas_primary, losses_primary,
                      alphas_seg, losses_seg,
                      alphas_ternary, losses_ternary,
                      title=""):
    """3-panel loss curve plot for Sec 4.3 / Fig. 7.

    Secondary: x-axis rescaled by primary barrier position.
    Tertiary: x-axis by local maxima positions, normalized to [0,1].
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Primary
    primary_alphas = alphas_primary.cpu().numpy() if torch.is_tensor(alphas_primary) else np.asarray(alphas_primary)
    primary_losses = np.asarray(losses_primary)
    axes[0].plot(primary_alphas, primary_losses, label="A --> C", color="tab:blue", linewidth=2)
    axes[0].set_xlabel("Primary interpolation coefficient ($\\alpha$)", fontsize=14)
    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].set_title("Primary Loss Curve", fontsize=14)
    axes[0].legend(fontsize=9)

    # Secondary: rescale x-axis by primary barrier position
    max_idx_primary = int(np.argmax(primary_losses))
    max_alpha_primary = float(primary_alphas[max_idx_primary])

    l_s = [np.asarray(l) for l in losses_seg]
    x_seg1 = np.linspace(0, max_alpha_primary, len(l_s[0]))
    x_seg2 = np.linspace(max_alpha_primary, 1.0, len(l_s[1]))
    axes[1].plot(x_seg1, l_s[0], label=r"A --> $B^{\prime}$", color="tab:green", linewidth=2)
    axes[1].plot(x_seg2, l_s[1], label=r"$B^{\prime}$ --> C", color="tab:red", linewidth=2)
    axes[1].set_xlabel("Primary interpolation coefficient ($\\alpha$)", fontsize=14)
    axes[1].set_ylabel("Loss", fontsize=14)
    axes[1].set_title("Secondary Loss Curves", fontsize=14)
    axes[1].set_ylim(0.0005, 0.005)
    axes[1].legend(fontsize=9)

    # Tertiary: x-axis by local maxima in secondary, then normalize to [0,1]
    l_t = [np.asarray(l) for l in losses_ternary]
    local_max_idx_1 = int(np.argmax(l_s[0]))
    local_max_alpha_1 = float(x_seg1[local_max_idx_1])
    local_max_idx_2 = int(np.argmax(l_s[1]))
    local_max_alpha_2 = float(x_seg2[local_max_idx_2])

    interval_1 = np.linspace(0, local_max_alpha_1, len(l_t[0]))
    interval_2 = np.linspace(local_max_alpha_1, max_alpha_primary, len(l_t[1]))
    interval_3 = np.linspace(max_alpha_primary, local_max_alpha_2, len(l_t[2]))
    interval_4 = np.linspace(local_max_alpha_2, 1.0, len(l_t[3]))
    x_segments_3 = np.concatenate([interval_1, interval_2, interval_3, interval_4])
    x_segments_3 = (x_segments_3 - x_segments_3.min()) / (x_segments_3.max() - x_segments_3.min())

    n1, n2, n3 = len(l_t[0]), len(l_t[1]), len(l_t[2])
    axes[2].plot(x_segments_3[:n1], l_t[0],
                 label=r"A --> $B_{S1}^{\prime}$", linestyle="--", color="tab:green", linewidth=2)
    axes[2].plot(x_segments_3[n1:n1 + n2], l_t[1],
                 label=r"$B_{S1}^{\prime}$ --> $B^{\prime}$", linestyle="-.", color="limegreen", linewidth=2)
    axes[2].plot(x_segments_3[n1 + n2:n1 + n2 + n3], l_t[2],
                 label=r"$B^{\prime}$ --> $B_{S2}^{\prime}$", linestyle="--", color="tab:red", linewidth=2)
    axes[2].plot(x_segments_3[n1 + n2 + n3:], l_t[3],
                 label=r"$B_{S2}^{\prime}$ --> C", linestyle="-.", color="tomato", linewidth=2)
    axes[2].set_xlabel("Primary interpolation coefficient ($\\alpha$)", fontsize=14)
    axes[2].set_ylabel("Loss", fontsize=14)
    axes[2].set_ylim(0.0004, 0.001)
    axes[2].set_title("Tertiary Loss Curves", fontsize=14)
    axes[2].legend(fontsize=9)

    # Horizontal dashed line at max tertiary loss
    max_tertiary = max(np.max(l) for l in l_t)
    for ax in axes:
        ax.axhline(y=max_tertiary, color="black", linestyle="--", linewidth=0.8)
    axes[0].plot([], [], "k--", label="max tertiary loss", linewidth=0.8)
    axes[0].legend(fontsize=10)
    axes[1].plot([], [], "k--", label="max tertiary loss", linewidth=0.8)
    axes[1].legend(fontsize=10)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_images(tensors, labels=None, losses=None,
                mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Show a row of images with optional labels and loss values."""
    n = len(tensors)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), dpi=150)
    if n == 1:
        axes = [axes]
    for i, (ax, t) in enumerate(zip(axes, tensors)):
        ax.imshow(unnormalize(t, mean, std), interpolation="none")
        title = labels[i] if labels else ""
        if losses is not None and losses[i] is not None:
            title += f"\nCE: {losses[i]:.4f}"
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    plt.tight_layout()
    return fig
