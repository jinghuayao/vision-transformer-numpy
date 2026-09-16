"""Pure-NumPy CIFAR-10 data loading + augmentation — port of ``data.py``.

Reads the raw ``cifar-10-batches-py`` pickles (no torchvision), applies
train/test transforms with NumPy/PIL, and yields NCHW float32 batches:

* ``ToTensor``: HWC uint8 ``(H, W, C)`` in ``[0,255]`` -> CHW float32 ``(C,H,W)``
  in ``[0,1]``.
* ``Resize``/``RandomResizedCrop``: PIL-based; shapes stay ``(C,32,32)``.
* ``RandomHorizontalFlip``: mirror width with prob ``p``.
* ``Normalize(mean, std)``: ``(x - mean) / std`` per channel -> ``[-1,1]``
  for mean/std 0.5.
"""

import os
import pickle
import numpy as np
from PIL import Image

CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')


# ---------------------------------------------------------------------------
# Low-level CIFAR-10 reading
# ---------------------------------------------------------------------------

def _unpickle(path):
    """Load a CIFAR-10 batch pickle; returns dict with ``data (N,3072)``."""
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='bytes')


def load_cifar10(root='./data', train=True):
    """Load CIFAR-10 images/labels as NumPy arrays.

    Args:
        root: dir containing ``cifar-10-batches-py`` (or that dir itself).
        train: True -> 50000 train samples; False -> 10000 test samples.

    Returns:
        images: ``(N, 32, 32, 3)`` uint8 HWC.
        labels: ``(N,)`` int64.
    """
    base = root if os.path.basename(root) == 'cifar-10-batches-py' \
        else os.path.join(root, 'cifar-10-batches-py')
    if train:
        xs, ys = [], []
        for i in range(1, 6):  # data_batch_1..5, each (10000, 3072)
            d = _unpickle(os.path.join(base, f'data_batch_{i}'))
            xs.append(d[b'data'])
            ys += d[b'labels']
        X = np.concatenate(xs, axis=0)  # (50000, 3072)
    else:
        d = _unpickle(os.path.join(base, 'test_batch'))
        X = d[b'data']  # (10000, 3072)
        ys = d[b'labels']
    images = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # (N,32,32,3) HWC
    return images, np.array(ys, dtype=np.int64)  # (N,), (N,)


# ---------------------------------------------------------------------------
# Transforms (PIL + NumPy, torchvision-equivalent)
# ---------------------------------------------------------------------------

def to_tensor(img_hwc):
    """HWC uint8 ``(H,W,C)`` -> CHW float32 ``(C,H,W)`` in ``[0,1]``."""
    return (img_hwc.transpose(2, 0, 1).astype(np.float32) / 255.0)


def normalize(img_chw, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """Per-channel normalize; ``(C,H,W)`` -> ``(C,H,W)`` (e.g. ``[0,1]``->``[-1,1]``)."""
    out = img_chw.copy()
    for c in range(out.shape[0]):
        out[c] = (out[c] - mean[c]) / std[c]
    return out


def resize_chw(img_chw, size=(32, 32)):
    """Resize CHW float ``(C,H,W)`` to ``(C,size[0],size[1])`` via PIL bilinear."""
    hwc = (np.clip(img_chw, 0, 1).transpose(1, 2, 0) * 255).astype(np.uint8)
    pil = Image.fromarray(hwc).resize((size[1], size[0]), Image.BILINEAR)
    return to_tensor(np.array(pil))


def random_horizontal_flip(img_chw, rng, p=0.5):
    """Flip width with prob ``p``; ``(C,H,W)`` -> ``(C,H,W)``."""
    if rng.random() < p:
        return img_chw[:, :, ::-1].copy()
    return img_chw


def random_resized_crop(img_chw, rng, size=(32, 32), scale=(0.8, 1.0),
                        ratio=(0.75, 4.0 / 3.0)):
    """Random crop with area/aspect sampling, resized back to ``size``.

    Args:
        img_chw: ``(C,H,W)`` float in ``[0,1]``.

    Returns:
        cropped: ``(C,size[0],size[1])`` float in ``[0,1]``.
    """
    C, H, W = img_chw.shape
    area = H * W
    for _ in range(10):  # up to 10 tries, else center crop
        target = rng.uniform(scale[0], scale[1]) * area
        log_ratio = (np.log(ratio[0]), np.log(ratio[1]))
        aspect = np.exp(rng.uniform(*log_ratio))
        w = int(round(np.sqrt(target * aspect)))
        h = int(round(np.sqrt(target / aspect)))
        if 0 < w <= W and 0 < h <= H:
            i = rng.integers(0, H - h + 1)
            j = rng.integers(0, W - w + 1)
            crop = img_chw[:, i:i + h, j:j + w]  # (C,h,w)
            return resize_chw(crop, size)
    # fallback: center crop at scale[0]
    h = w = int(np.sqrt(scale[0]) * min(H, W))
    i, j = (H - h) // 2, (W - w) // 2
    return resize_chw(img_chw[:, i:i + h, j:j + w], size)


def train_transform(img_hwc, rng):
    """Train pipeline: HWC uint8 ``(32,32,3)`` -> CHW float ``(3,32,32)`` in ``[-1,1]``."""
    x = to_tensor(img_hwc)  # (3,32,32) [0,1]
    x = resize_chw(x, (32, 32))  # (3,32,32) identity-ish
    x = random_horizontal_flip(x, rng, 0.5)  # (3,32,32)
    x = random_resized_crop(x, rng, (32, 32))  # (3,32,32)
    return normalize(x)  # (3,32,32) [-1,1]


def test_transform(img_hwc):
    """Test pipeline: HWC uint8 ``(32,32,3)`` -> CHW float ``(3,32,32)`` in ``[-1,1]``."""
    x = to_tensor(img_hwc)  # (3,32,32)
    x = resize_chw(x, (32, 32))  # (3,32,32)
    return normalize(x)  # (3,32,32)


# ---------------------------------------------------------------------------
# Dataset + DataLoader + prepare_data (mirrors data.py API)
# ---------------------------------------------------------------------------

class CIFAR10Dataset:
    """In-memory CIFAR-10 with on-the-fly transform.

    Args:
        images: ``(N,32,32,3)`` uint8; labels: ``(N,)`` int.
        train: select train vs test pipeline.
        seed: rng seed for stochastic train augmentation.
    """

    def __init__(self, images, labels, train=True, seed=0):
        self.images = images
        self.labels = labels
        self.train = train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        """Returns ``(img (3,32,32) float32, label int)``."""
        img = self.images[idx]  # (32,32,3)
        if self.train:
            out = train_transform(img, self.rng)
        else:
            out = test_transform(img)
        return out.astype(np.float32), int(self.labels[idx])


class DataLoader:
    """Minimal batched iterator: shuffles per epoch, drops nothing.

    Args:
        dataset: ``CIFAR10Dataset`` or subset thereof.
        batch_size: rows per batch.
        shuffle: reshuffle indices each iteration.
        seed: rng seed for ordering.
    """

    def __init__(self, dataset, batch_size=4, shuffle=True, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(idx)
        for s in range(0, len(idx), self.batch_size):
            b = idx[s:s + self.batch_size]
            imgs = np.stack([self.dataset[i][0] for i in b])  # (B,3,32,32)
            labels = np.array([self.dataset[i][1] for i in b], dtype=np.int64)  # (B,)
            yield imgs, labels

    @property
    def dataset_size(self):
        """Total number of samples (for loss normalization)."""
        return len(self.dataset)


class Subset:
    """Index subset view of a dataset (mirrors ``torch.utils.data.Subset``)."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.dataset[int(self.indices[i])]


def prepare_data(batch_size=4, train_sample_size=None, test_sample_size=None,
                 data_root='./data', seed=0):
    """Build train/test loaders (same signature/semantics as ``data.py``).

    Args:
        batch_size: batch size.
        train_sample_size: optionally subsample N train rows.
        test_sample_size: optionally subsample N test rows.

    Returns:
        trainloader, testloader: :class:`DataLoader` yielding
            ``(images (B,3,32,32) float32, labels (B,) int64)``.
        classes: 10-tuple of CIFAR-10 names.
    """
    rng = np.random.default_rng(seed)
    train_images, train_labels = load_cifar10(data_root, train=True)
    test_images, test_labels = load_cifar10(data_root, train=False)
    trainset = CIFAR10Dataset(train_images, train_labels, train=True, seed=seed)
    testset = CIFAR10Dataset(test_images, test_labels, train=False, seed=seed + 1)
    if train_sample_size is not None:
        idx = rng.permutation(len(trainset))[:train_sample_size]
        trainset = Subset(trainset, idx)
    if test_sample_size is not None:
        idx = rng.permutation(len(testset))[:test_sample_size]
        testset = Subset(testset, idx)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, seed=seed)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, seed=seed)
    return trainloader, testloader, CIFAR10_CLASSES
