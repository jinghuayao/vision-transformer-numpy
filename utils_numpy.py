"""NumPy experiment helpers — port of ``utils.py`` (no torch).

Covers checkpointing (``.npz``), metrics JSON, and matplotlib visualizations
for dataset samples and CLS-token attention maps.
"""

import json
import math
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from vit_numpy import ViTForClassification
from data_numpy import load_cifar10, CIFAR10_CLASSES


# ---------------------------------------------------------------------------
# Persistence (mirrors save_experiment / save_checkpoint / load_experiment)
# ---------------------------------------------------------------------------

def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    """Save model params to ``<base_dir>/<exp>/model_<epoch>.npz``.

    Args:
        experiment_name: experiment folder name.
        model: :class:`ViTForClassification`.
        epoch: epoch tag used in the filename.
    """
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"model_{epoch}.npz")
    np.savez(path, **model.state_dict())
    return path


def save_experiment(experiment_name, config, model, train_losses, test_losses,
                    accuracies, base_dir="experiments"):
    """Save ``config.json`` + ``metrics.json`` + final ``model_final.npz``.

    Args:
        config: model config dict (must be JSON-serializable).
        train_losses/test_losses/accuracies: per-epoch float lists.
    """
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f, sort_keys=True, indent=4)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump({"train_losses": list(map(float, train_losses)),
                   "test_losses": list(map(float, test_losses)),
                   "accuracies": list(map(float, accuracies))},
                  f, sort_keys=True, indent=4)
    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def load_experiment(experiment_name, checkpoint_name="model_final.npz",
                    base_dir="experiments"):
    """Load config, metrics and model weights from an experiment folder.

    Returns:
        config, model, train_losses, test_losses, accuracies.
    """
    outdir = os.path.join(base_dir, experiment_name)
    with open(os.path.join(outdir, "config.json")) as f:
        config = json.load(f)
    with open(os.path.join(outdir, "metrics.json")) as f:
        data = json.load(f)
    model = ViTForClassification(config)
    z = np.load(os.path.join(outdir, checkpoint_name), allow_pickle=False)
    model.load_state_dict({k: z[k] for k in z.files})
    return config, model, data["train_losses"], data["test_losses"], data["accuracies"]


def plot_metrics(train_losses, test_losses, accuracies, output=None):
    """Plot loss/accuracy curves (mirrors ``inspect.ipynb`` cell 3).

    Args:
        output: optional path to save the PNG figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, label="Train loss")
    ax1.plot(test_losses, label="Test loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.plot(accuracies)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    fig.tight_layout()
    if output is not None:
        fig.savefig(output)
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_images(data_root="./data", num_images=30, seed=0, output=None):
    """Plot a grid of random CIFAR-10 train images (port of ``utils.visualize_images``)."""
    rng = np.random.default_rng(seed)
    images, labels = load_cifar10(data_root, train=True)  # (N,32,32,3)
    idx = rng.permutation(len(images))[:num_images]
    fig = plt.figure(figsize=(10, 10))
    for i, j in enumerate(idx):
        ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(images[j])
        ax.set_title(CIFAR10_CLASSES[int(labels[j])])
    fig.tight_layout()
    if output is not None:
        fig.savefig(output)
    plt.close(fig)
    return fig


def _bilinear_upsample(maps, size=(32, 32)):
    """Upsample ``(N, h, w)`` float maps to ``(N, H, W)`` via PIL bilinear."""
    out = []
    for m in maps:
        m = m - m.min()
        if m.max() > 0:
            m = m / m.max()
        pil = Image.fromarray((m * 255).astype(np.uint8)).resize(
            (size[1], size[0]), Image.BILINEAR)
        out.append(np.array(pil).astype(np.float32) / 255.0)
    return np.stack(out)  # (N,H,W)


def visualize_attention(model, data_root="./data", num_images=30, seed=0,
                        output=None):
    """Overlay mean CLS-token attention (all layers+heads) on test images.

    Mirrors ``utils.visualize_attention``: forward with
    ``output_attentions=True`` -> concat layer maps ``(L,B,H,S,S)`` ->
    CLS row ``[..., 0, 1:]`` -> mean over layers+heads -> ``(B,h,w)`` ->
    upsample to ``32x32`` -> side-by-side plot with predictions.

    Returns:
        (fig, predicted_labels ``(num_images,)``).
    """
    rng = np.random.default_rng(seed)
    images, labels = load_cifar10(data_root, train=False)  # (N,32,32,3) uint8
    idx = rng.permutation(len(images))[:num_images]
    raw = [images[j] for j in idx]  # list of (32,32,3)
    gt = np.array([labels[j] for j in idx])
    # preprocess like test_transform: [0,1] -> normalize to [-1,1], NCHW
    batch = np.stack([(r.astype(np.float32) / 255.0 - 0.5) / 0.5 for r in raw])
    batch = batch.transpose(0, 3, 1, 2).astype(np.float32)  # (B,3,32,32)
    logits, attns = model.forward(batch, output_attentions=True, training=False)
    preds = logits.argmax(axis=1)  # (B,)
    # attns: list of L x (B,H,S,S) -> (L,B,H,S,S) -> (B,L*H,S,S)
    A = np.concatenate(attns, axis=1)  # (B, L*H, S, S)
    cls_maps = A[:, :, 0, 1:]  # (B, L*H, N_patches) CLS -> patches
    avg = cls_maps.mean(axis=1)  # (B, N)
    n = int(math.sqrt(avg.shape[-1]))  # patches per side
    maps = avg.reshape(-1, n, n)  # (B,h,w)
    maps = _bilinear_upsample(maps, (32, 32))  # (B,32,32)
    fig = plt.figure(figsize=(20, 10))
    mask = np.concatenate([np.ones((32, 32)), np.zeros((32, 32))], axis=1)
    for i in range(num_images):
        ax = fig.add_subplot(6, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(np.concatenate((raw[i], raw[i]), axis=1))
        ext = np.concatenate((np.zeros((32, 32)), maps[i]), axis=1)
        ax.imshow(np.ma.masked_where(mask == 1, ext), alpha=0.5, cmap="jet")
        g, p = CIFAR10_CLASSES[int(gt[i])], CIFAR10_CLASSES[int(preds[i])]
        ax.set_title(f"gt: {g} / pred: {p}", color=("green" if g == p else "red"))
    fig.tight_layout()
    if output is not None:
        fig.savefig(output)
    plt.close(fig)
    return fig, preds
