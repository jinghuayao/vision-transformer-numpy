# Pure-NumPy Vision Transformer (port of vision-transformer-from-scratch)

Pure NumPy (+Pillow/matplotlib for data & plots) re-implementation of the
PyTorch ViT in this repo. No `torch`/`torchvision` required.

## Files (mirrors the original repo)

| This folder | Original | Purpose |
|---|---|---|
| `vit_numpy.py` | `vit.py` | GELU, Patch/Token embeddings, per-head + fused MHA, MLP, Block, Encoder, `ViTForClassification`, `CrossEntropyLoss`, `AdamW` — forward **and** manual backward, with input/output/intermediate shape annotations in every docstring |
| `data_numpy.py` | `data.py` | CIFAR-10 loader reading raw `cifar-10-batches-py` pickles; `ToTensor/Resize/RandomHorizontalFlip/RandomResizedCrop/Normalize` in NumPy/PIL; `DataLoader`, `prepare_data` |
| `train_numpy.py` | `train.py` | `Trainer` (train/train_epoch/evaluate), same `config` + asserts, argparse CLI |
| `utils_numpy.py` | `utils.py` | `save_experiment/save_checkpoint/load_experiment` (`.npz`), `plot_metrics`, `visualize_images`, `visualize_attention` |
| `test_vit_numpy.py` | — | 21 pytest tests: shapes, math vs closed forms, gradient checks, MHA equivalence, optimizer, save/load, data, end-to-end training |
| `vision_transformers_numpy.ipynb` | `vision_transformers.ipynb` | Same 8 sections (implement → data → utils → train → visualize dataset / plot / attention), torch-free; imports the tested modules above |
| `inspect_numpy.ipynb` | `inspect.ipynb` | Same 5 cells (imports → samples → load experiment → curves → attention) for `experiments/vit-numpy-demo/` |

## Notebooks

```bash
jupyter notebook vision_transformers_numpy.ipynb  # trains experiments/vit-numpy-demo/
jupyter notebook inspect_numpy.ipynb              # inspects it (run the first one first)
```

The two notebooks are ported cell-for-cell (same sections, torch-free);
the same flow is also available as scripts via `train_numpy.py` + the
`utils_numpy` visualizers.

## Setup

```bash
pip install -r requirements.txt
```

## Train (NumPy, CPU)

```bash
# needs the original CIFAR-10 pickles; point at the source repo's data dir
export VIT_DATA_ROOT=/Users/jinghuayao/Downloads/vision-transformer-from-scratch/data
python train_numpy.py --exp-name vit-numpy-test --epochs 2 --batch-size 32 \
    --train-samples 512 --test-samples 128
```

## Test

```bash
python -m pytest test_vit_numpy.py -v
```

## Notes vs the PyTorch version

* NCHW images / `(B,S,D)` sequences; `Linear` weight is `(in,out)`.
* Patch projection is stride-`P` conv implemented as patch-matrix multiply
  (exact for non-overlapping patches), with full backward.
* `MultiHeadAttention` (per-head linears) and `FasterMultiHeadAttention`
  (fused QKV) are both implemented and numerically identical given the same
  weights (tested).
* Dropout is inverted dropout with `training` flag; all original dropout
  probs are `0.0`, so eval/train agree by default.
* Checkpoints are `.npz` (NumPy) instead of `.pt`.
