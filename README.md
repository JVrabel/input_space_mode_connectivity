# Input Space Mode Connectivity in Deep Neural Networks

Companion code for:

> **Input Space Mode Connectivity in Deep Neural Networks**
> Jakub Vrabel, Ori Shem-Ur, Yaron Oz, David Krueger
> ICLR 2025 — [OpenReview](https://openreview.net/forum?id=3qeOy7HwUT) · [PDF](https://openreview.net/pdf?id=3qeOy7HwUT)

## Setup

```bash
cd input_space_mode_connectivity
uv sync
```

## Usage

Open **`walkthrough.ipynb`** for an interactive guide, or run scripts directly:

```bash
# Sec 4.1 — real image connectivity (GoogLeNet/ImageNet)
python sec4_1_input_space_connectivity.py --model googlenet --class_id 955

# Sec 4.2 — adversarial attack barrier + statistics across 1000 classes
python sec4_2_adversarial_attacks.py --target_class 574 --source_class 763
python sec4_2_adversarial_attacks.py --compute --stat_tensors ./data_sel/stat_tensors  # recompute stats

# Sec 4.3 — synthetic inputs in an untrained model
python sec4_3_untrained_model.py --target_class 1
```

## Files

| File | Description |
|---|---|
| `utils.py` | Shared code: model loading, interpolation, optimization, plotting |
| `models.py` | Custom model definitions (ResNet-18 for CIFAR-10) |
| `walkthrough.ipynb` | Interactive notebook walking through all experiments |
| `sec4_1_input_space_connectivity.py` | Sec 4.1 — barrier between real images, bypass optimization |
| `sec4_2_adversarial_attacks.py` | Sec 4.2 — adversarial attack barrier + statistics boxplots |
| `sec4_3_untrained_model.py` | Sec 4.3 — FVO inputs in untrained models |
| `prepare_imagenet_tensors.py` | Generate saved_tensors from ImageNet validation |

## Data

**ImageNet tensors** (Sec 4.1, 4.2): Preprocessed validation images in `./data_sel/saved_tensors/` — a minimal set (classes 954, 955, 574, 763) is included in the repo.

To generate tensors from ImageNet validation (avoids redistributing ImageNet data):

```bash
# From a directory of images (filenames: ILSVRC2012_val_*_nXXXXXXXX.JPEG)
python prepare_imagenet_tensors.py --images_dir ../data/rebuttal_val_images

# From a tar archive
python prepare_imagenet_tensors.py --images_tar ../data/val_images.tar.gz
```

Requires `../data/imagenet_classes.py` (or `--imagenet_classes path`). Download ImageNet validation per [image-net.org](https://image-net.org/download.php).

**CIFAR-10** (Sec 4.3): Downloaded automatically to `./data/` on first run.

**Barrier statistics** (Sec 4.2, Figure 6): Pre-computed CSVs in `./data_sel/`:
- `interpolation_processed.csv` — within-class barriers
- `cross_class_interpolation_processed.csv` — adversarial barriers

To recompute from scratch: `python sec4_2_adversarial_attacks.py --compute --stat_tensors ./data_sel/stat_tensors` (requires `stat_tensors/` with `class_{id}_tensors_with_names.pt` per class).

## Repo layout

```
input_space_mode_connectivity/
├── data_sel/                   # bundled data (tensors + CSVs)
│   ├── saved_tensors/          # class_955_sel_0.pt, etc.
│   ├── interpolation_processed.csv
│   └── cross_class_interpolation_processed.csv
├── data/                       # CIFAR-10 (auto-downloaded for Sec 4.3)
└── ...
```

Run scripts from `input_space_mode_connectivity/`. Override paths with `--tensor_dir`, `--data_dir` if needed.

Sec 4.3 (untrained model) needs no ImageNet data — CIFAR-10 downloads automatically.

## Citation

```bibtex
@inproceedings{vrabel2025input,
  title={Input Space Mode Connectivity in Deep Neural Networks},
  author={Vrabel, Jakub and Shem-Ur, Ori and Oz, Yaron and Krueger, David},
  booktitle={International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=3qeOy7HwUT}
}
```
