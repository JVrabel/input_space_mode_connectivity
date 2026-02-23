"""
Section 4.1 — Input space mode connectivity between real images.
Reproduces Figures 1–3 from the paper.

Two low-loss ImageNet images of the same class are linearly interpolated.
The barrier point B is optimized to B', giving a low-loss path A→B'→C.

Usage:
    python sec4_1_input_space_connectivity.py
    python sec4_1_input_space_connectivity.py --model resnet18 --class_id 574
    python sec4_1_input_space_connectivity.py --model vit --class_id 954
"""

import argparse
import torch
import matplotlib.pyplot as plt
from utils import (
    load_googlenet, load_resnet18_imagenet, load_vit,
    load_saved_tensors,
    linear_interpolation, compute_losses, find_barrier,
    bypass_barrier,
    plot_loss_curve, plot_barrier_comparison, plot_images, unnormalize,
)

MODEL_LOADERS = {
    "googlenet": load_googlenet,
    "resnet18":  load_resnet18_imagenet,
    "vit":       load_vit,
}

IMAGENET_CLASSES = {
    955: "jackfruit", 954: "banana", 574: "golf ball",
    409: "analog clock", 1: "goldfish", 71: "scorpion",
}


def run(model_name="googlenet", class_id=955, sel=(0, 1),
        tensor_dir="./data_sel/saved_tensors", steps=1000,
        opt_iters=1024, lr=0.005, lambda_mse=0.1, lambda_hf=1e-8):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print(f"Loading {model_name}...")
    model = MODEL_LOADERS[model_name](device)

    # Load two low-loss images
    x_a, x_c = load_saved_tensors(tensor_dir, class_id, sel, device)
    class_name = IMAGENET_CLASSES.get(class_id, f"class {class_id}")
    print(f"Class: {class_name} ({class_id})")

    # Show the two images
    plot_images([x_a, x_c], labels=["A", "C"])
    plt.suptitle(f"Input pair — {class_name}", fontsize=14)

    # Primary interpolation A → C
    print(f"Interpolating ({steps} steps)...")
    alphas, points = linear_interpolation(x_a, x_c, steps=steps)
    losses = compute_losses(model, points, class_id)
    idx, alpha_max, barrier = find_barrier(alphas, points, losses)
    print(f"Barrier at α={alpha_max:.3f}, loss={losses[idx]:.4f}")

    plot_loss_curve(alphas, losses, title=f"Primary loss — {class_name} ({model_name})")

    # Show A, B (barrier), C
    plot_images(
        [x_a, barrier, x_c],
        labels=["A (minimizer)", "B (barrier)", "C (minimizer)"],
        losses=[losses[0], losses[idx], losses[-1]],
    )

    # Optimize barrier B → B'
    print("Optimizing barrier...")
    b_prime, _ = bypass_barrier(
        model, x_a, x_c, barrier, class_id,
        lr=lr, iterations=opt_iters,
        lambda_mse=lambda_mse, lambda_hf=lambda_hf, verbose=True,
    )

    # Secondary interpolation through B'
    _, pts_ab = linear_interpolation(x_a, b_prime, steps=steps // 2)
    losses_ab = compute_losses(model, pts_ab, class_id)
    _, pts_bc = linear_interpolation(b_prime, x_c, steps=steps // 2)
    losses_bc = compute_losses(model, pts_bc, class_id)

    print(f"Secondary max A→B': {losses_ab.max():.6f}")
    print(f"Secondary max B'→C: {losses_bc.max():.6f}")

    plot_barrier_comparison(
        alphas, losses, losses_ab, losses_bc, alpha_max,
        title=f"{class_name} — {model_name}",
    )
    plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="googlenet", choices=list(MODEL_LOADERS))
    p.add_argument("--class_id", type=int, default=955)
    p.add_argument("--sel", nargs=2, type=int, default=[0, 1])
    p.add_argument("--tensor_dir", default="./data_sel/saved_tensors")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--opt_iters", type=int, default=1024)
    args = p.parse_args()
    run(model_name=args.model, class_id=args.class_id, sel=tuple(args.sel),
        tensor_dir=args.tensor_dir, steps=args.steps, opt_iters=args.opt_iters)
