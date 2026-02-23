#!/usr/bin/env python3
"""
Prepare ImageNet tensors for input space mode connectivity experiments.

Loads ImageNet validation images, applies standard preprocessing (Resize 256,
CenterCrop 224, ImageNet normalize), computes GoogLeNet cross-entropy loss per
image, sorts by loss (lowest first), and saves the top N as class_{id}_sel_{i}.pt.

Requires ImageNet validation data. Supports:
  - Directory of images with class in filename: ILSVRC2012_val_*_nXXXXXXXX.JPEG
  - Tar archive: val_images.tar.gz with same naming

Usage:
  python prepare_imagenet_tensors.py --images_dir ../data/rebuttal_val_images
  python prepare_imagenet_tensors.py --images_dir ../data/rebuttal_val_images --classes 954 955 574 763
  python prepare_imagenet_tensors.py --images_tar ../data/val_images.tar.gz --output ./data_sel/saved_tensors
"""

import argparse
import io
import os
import re
import tarfile
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from utils import IMAGENET_TRANSFORM, load_googlenet


def _load_imagenet_classes(path=None):
    if path:
        import importlib.util
        spec = importlib.util.spec_from_file_location("imagenet", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return {k: i for i, k in enumerate(mod.IMAGENET2012_CLASSES.keys())}
    from imagenet_classes import CLASS_TO_IDX
    return CLASS_TO_IDX


def _parse_class_from_filename(name):
    """Extract WordNet ID from filename, e.g. ILSVRC2012_val_00000002_n09193705.JPEG -> n09193705."""
    m = re.search(r"_n(\d{8})\.(?:JPEG|jpg|jpeg|png)$", name, re.I)
    if m:
        return "n" + m.group(1)
    return None


def _collect_images_from_dir(images_dir, class_to_idx):
    """Collect (path, numeric_label) for images in directory."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    collected = {}
    for ext in ("*.JPEG", "*.jpeg", "*.jpg", "*.png"):
        for fp in images_dir.glob(ext):
            wnid = _parse_class_from_filename(fp.name)
            if wnid and wnid in class_to_idx:
                numeric = class_to_idx[wnid]
                if numeric not in collected:
                    collected[numeric] = []
                collected[numeric].append(str(fp))
    return collected


def _collect_images_from_tar(tar_path, class_to_idx):
    """Collect (member, numeric_label) for images in tar archive."""
    tar_path = Path(tar_path)
    if not tar_path.exists():
        raise FileNotFoundError(f"Tar archive not found: {tar_path}")

    collected = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name:
                name = os.path.basename(m.name)
                wnid = _parse_class_from_filename(name)
                if wnid and wnid in class_to_idx:
                    numeric = class_to_idx[wnid]
                    if numeric not in collected:
                        collected[numeric] = []
                    collected[numeric].append(m)
    return collected


def _process_class_from_dir(images_dir, class_to_idx, numeric_label, model, transform, device, num_save, output_dir):
    """Process one class from directory, sort by loss, save top tensors."""
    collected = _collect_images_from_dir(images_dir, class_to_idx)
    if numeric_label not in collected:
        print(f"  Class {numeric_label}: no images found")
        return 0

    paths = collected[numeric_label]
    loss_tensors = []

    for path in paths[:num_save * 3]:  # process more to have enough for selection
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  Skip {path}: {e}")
            continue
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x)
            loss = F.cross_entropy(out, torch.tensor([numeric_label], device=device)).item()
        loss_tensors.append((loss, x.cpu()))

    if not loss_tensors:
        print(f"  Class {numeric_label}: no valid images")
        return 0

    loss_tensors.sort(key=lambda t: t[0])
    selected = loss_tensors[:num_save]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (loss, tensor) in enumerate(selected):
        out_path = output_dir / f"class_{numeric_label}_sel_{i}.pt"
        torch.save(tensor.squeeze(0), out_path)
        print(f"  class_{numeric_label}_sel_{i}.pt  loss={loss:.6f}")

    return len(selected)


def _process_class_from_tar(tar_path, class_to_idx, numeric_label, model, transform, device, num_save, output_dir):
    """Process one class from tar, sort by loss, save top tensors."""
    collected = _collect_images_from_tar(tar_path, class_to_idx)
    if numeric_label not in collected:
        print(f"  Class {numeric_label}: no images found")
        return 0

    members = collected[numeric_label]
    loss_tensors = []

    with tarfile.open(tar_path, "r:gz") as tar:
        for m in members[:num_save * 3]:
            try:
                f = tar.extractfile(m)
                if f is None:
                    continue
                img = Image.open(io.BytesIO(f.read())).convert("RGB")
            except Exception as e:
                print(f"  Skip {m.name}: {e}")
                continue
            x = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(x)
                loss = F.cross_entropy(out, torch.tensor([numeric_label], device=device)).item()
            loss_tensors.append((loss, x.cpu()))

    if not loss_tensors:
        print(f"  Class {numeric_label}: no valid images")
        return 0

    loss_tensors.sort(key=lambda t: t[0])
    selected = loss_tensors[:num_save]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, (loss, tensor) in enumerate(selected):
        out_path = output_dir / f"class_{numeric_label}_sel_{i}.pt"
        torch.save(tensor.squeeze(0), out_path)
        print(f"  class_{numeric_label}_sel_{i}.pt  loss={loss:.6f}")

    return len(selected)


def main():
    p = argparse.ArgumentParser(description="Prepare ImageNet tensors for walkthrough")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--images_dir", help="Directory of validation images (ILSVRC2012_val_*_nXXXXXXXX.JPEG)")
    g.add_argument("--images_tar", help="Path to val_images.tar.gz")
    p.add_argument("--output", default="./data_sel/saved_tensors", help="Output directory for .pt files")
    p.add_argument("--classes", type=int, nargs="+", default=[954, 955, 574, 763],
                   help="Numeric class IDs to process (default: 954 955 574 763)")
    p.add_argument("--num_per_class", type=int, default=3, help="Number of tensors to save per class (sel_0, sel_1, ...)")
    p.add_argument("--imagenet_classes", help="Path to imagenet_classes.py if not in parent data/")
    args = p.parse_args()

    class_to_idx = _load_imagenet_classes(args.imagenet_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_googlenet(device)
    transform = IMAGENET_TRANSFORM

    total = 0
    if args.images_dir:
        for c in args.classes:
            print(f"Processing class {c}...")
            n = _process_class_from_dir(
                args.images_dir, class_to_idx, c,
                model, transform, device, args.num_per_class, args.output
            )
            total += n
    else:
        for c in args.classes:
            print(f"Processing class {c}...")
            n = _process_class_from_tar(
                args.images_tar, class_to_idx, c,
                model, transform, device, args.num_per_class, args.output
            )
            total += n

    print(f"\nSaved {total} tensors to {args.output}")


if __name__ == "__main__":
    main()
