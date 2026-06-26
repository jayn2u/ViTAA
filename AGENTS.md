# ViTAA Agent Notes

Use `uv run python` to execute Python code.
Use `uv` for dependency management (`pyproject.toml`, Python 3.11).

## Project role

This repository is the official [ViTAA](https://arxiv.org/abs/2005.07327) release (ECCV 2020): visual-textual attribute alignment for text-based person search. It is **not** integrated with `lab_clip` evaluation wrappers.

Supported dataset in code: **CUHK-PEDES only** (`paths_catalog.py`). ICFG-PEDES and RSTPReid are not implemented.

There is **no publicly released ViTAA checkpoint**. Paper numbers must be reproduced by completing preprocessing and training locally. ResNet-50 ImageNet weights download automatically when `MODEL.WEIGHT: imagenet`.

## Dataset location

Lab CUHK-PEDES raw files live at one of:

- `/mnt/data/lab_datasets/CUHK-PEDES`
- `/data/jayn2u/lab_datasets/CUHK-PEDES`

These paths refer to the same storage. Use whichever exists on the current machine.

Raw layout used by ViTAA preprocessing:

```
{lab_datasets_root}/CUHK-PEDES/
├── imgs/
└── reid_raw.json
```

Do **not** use `reid_raw_diverse_color.json`, `reid_raw_negative_*.json`, or other extended variants for ViTAA. They are not paired with the official `text_attribute_graph` release.

## Prerequisites (readiness checklist)

Before training or evaluation, verify every item below.

| # | Artifact | Required | Lab status (typical) | Notes |
|---|----------|----------|----------------------|-------|
| 1 | `datasets/cuhkpedes/imgs/` | Yes | Link from lab_datasets | Symlink to `{lab_datasets}/CUHK-PEDES/imgs` |
| 2 | `datasets/cuhkpedes/annotations/reid_raw.json` | Yes | Link from lab_datasets | `convert_to_json.py` reads `datadir/annotations/reid_raw.json`, not the dataset root |
| 3 | `datasets/cuhkpedes/text_attribute_graph/` | Yes | **Missing — download** | Official release only; one JSON per image caption |
| 4 | `datasets/cuhkpedes/segs/` | Yes | **Missing — generate** | Human Parsing Network output; same relative paths as `imgs/` |
| 5 | `datasets/cuhkpedes/annotations/train.json` | Yes | Created by convert | Plus `val.json`, `test.json` |
| 6 | ViTAA checkpoint (`epoch_*.pth`) | Eval only | **Missing — train** | Saved under `output/cuhkpedes/` |
| 7 | `uv sync` | Yes | Run once per machine | Optional: `uv sync --group tensorboard` for `--use-tensorboard` |

Training or testing without `text_attribute_graph`, `segs`, or converted JSON will fail at data load time.

## Target directory layout

All ViTAA-specific artifacts live under the project tree:

```
ViTAA/
├── datasets/cuhkpedes/
│   ├── annotations/
│   │   ├── reid_raw.json      # symlink to lab_datasets
│   │   ├── train.json         # convert_to_json output
│   │   ├── val.json
│   │   └── test.json
│   ├── imgs/                  # symlink to lab_datasets
│   ├── segs/                  # Human Parsing Network PNG masks
│   └── text_attribute_graph/  # downloaded attribute phrase JSONs
├── configs/cuhkpedes/bilstm_r50_seg.yaml
├── output/cuhkpedes/          # checkpoints (created by training)
└── tools/
```

### Symlink example

```bash
cd /data/jayn2u/ViTAA
mkdir -p datasets/cuhkpedes/annotations

ln -sfn /data/jayn2u/lab_datasets/CUHK-PEDES/imgs datasets/cuhkpedes/imgs
ln -sfn /data/jayn2u/lab_datasets/CUHK-PEDES/reid_raw.json \
  datasets/cuhkpedes/annotations/reid_raw.json
```

Use `/mnt/data/lab_datasets` when that path exists instead of `/data/jayn2u/lab_datasets`.

## text_attribute_graph

Download from the official ViTAA README links:

- [Google Drive](https://drive.google.com/file/d/1Sqm3V97hbqK9GxIwshZejJWLARfu5o1s/view?usp=sharing)
- [Baidu Yun (code: vbss)](https://pan.baidu.com/s/1TIX4lbvZmGwbBNHcRyA1ng)

Extract into `datasets/cuhkpedes/text_attribute_graph/`.

Naming convention (`tools/cuhkpedes/convert_to_json.py`):

- Image `file_path` `CUHK01/0363004.png` with caption index `0` → `CUHK01-0363004-0.json`
- Caption index `1` → `CUHK01-0363004-1.json`

Each file holds parsed attribute phrases mapped to five body regions: `head`, `upperbody`, `lowerbody`, `shoe`, `backpack`.

## segs (Human Parsing Network)

Segmentation maps are required at load time (`vitaa/data/datasets/cuhkpedes.py`):

- Image: `datasets/cuhkpedes/imgs/{file_path}`
- Seg: `datasets/cuhkpedes/segs/{file_path without extension}.png`

Example: `imgs/CUHK01/0363004.png` → `segs/CUHK01/0363004.png`

Generate with [Jarr0d/Human-Parsing-Network](https://github.com/Jarr0d/Human-Parsing-Network):

1. Clone the parsing repo (separate from ViTAA).
2. Download the pretrained parsing weight into `pretrained_models/`:
   - [Google Drive](https://drive.google.com/file/d/1CYhS5AXMnMtcv9MVq5luHLrZciAwhfqn/view?usp=sharing)
3. Run inference on CUHK-PEDES images via configs under `experiments/`.
4. Copy or symlink PNG outputs into `ViTAA/datasets/cuhkpedes/segs/` preserving the same relative paths as `imgs/`.

Every annotated image needs a matching seg file. Missing segs cause `FileNotFoundError` during training or evaluation.

## Annotation conversion

After `imgs/`, `annotations/reid_raw.json`, and `text_attribute_graph/` are in place:

```bash
cd /data/jayn2u/ViTAA
uv sync

uv run python tools/cuhkpedes/convert_to_json.py \
  --datadir datasets/cuhkpedes \
  --outdir datasets/cuhkpedes/annotations
```

The script tokenizes captions, builds vocabulary (default `word_count_threshold=2`), writes `onehot` caption IDs and `att_onehot` per-region word IDs, and emits `train.json`, `val.json`, `test.json`.

Config vocabulary sizes (`NUM_CLASSES: 12003`, `VOCABULARY_SIZE: 12000` in `configs/cuhkpedes/bilstm_r50_seg.yaml`) assume this conversion output.

## Training

```bash
cd /data/jayn2u/ViTAA
uv run python tools/train_net.py \
  --config-file configs/cuhkpedes/bilstm_r50_seg.yaml
```

| Setting | Value |
|---------|-------|
| Backbone | ResNet-50 (ImageNet) |
| Text encoder | BiLSTM |
| Input size | 384 × 128 |
| Batch size | 64 (paper used one V100; reduce batch and scale LR if needed) |
| Epochs | 80 |
| Train splits | `cuhkpedes_train`, `cuhkpedes_val` |
| Checkpoints | `output/cuhkpedes/epoch_{5,10,...,80}.pth` every 5 epochs |

Optional tensorboard logging requires `uv sync --group tensorboard` and `--use-tensorboard`.

## Evaluation

Entry script is `tools/test_net.py` (README mentions `tools/test.py`, which does not exist).

```bash
uv run python tools/test_net.py \
  --config-file configs/cuhkpedes/bilstm_r50_seg.yaml \
  --checkpoint-file output/cuhkpedes/epoch_80.pth
```

Metrics: text-to-image Rank@1, Rank@5, Rank@10 on `cuhkpedes_test`.

### CUHK-PEDES paper benchmark (test split)

| Method | R@1 | R@5 | R@10 |
|--------|-----|-----|------|
| ViTAA | 55.97 | 75.84 | 83.52 |

Use these as the reproduction target after full preprocessing and 80-epoch training.

## Benchmark reference in lab_clip

`lab_clip/AGENTS.md` cites the same ViTAA paper scores for CUHK-PEDES text-to-image plots. Those values are **not** from a local ViTAA checkpoint in this lab.

## Agent cautions

1. Complete the full prerequisite chain (`text_attribute_graph` → `convert_to_json` → `segs` → train) before expecting eval numbers.
2. Do not assume a downloadable ViTAA `best.pt` or Hugging Face release exists.
3. Human Parsing Network is a **separate** repository and environment; only its PNG outputs are consumed here.
4. Multi-GPU training code exists but upstream notes it was not tested.
5. Extended CUHK-PEDES JSON variants under `lab_datasets` are incompatible with the official `text_attribute_graph` naming scheme.
