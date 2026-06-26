# ViTAA Agent Notes

Use `uv run python` to execute Python code.
Use `uv` for dependency management (`pyproject.toml`, Python 3.11).
Download Google Drive artifacts with `uv run gdown` (`gdown` is listed in `pyproject.toml`; run `uv sync` first).

## Path policy

- **Do not use symlinks** to wire lab datasets into this repository. Other machines must run the same commands without link setup.
- **Do not hardcode machine-specific absolute paths** (for example `/data/jayn2u/...` or `/mnt/data/...`) in scripts, configs committed to git, or agent instructions. Resolve paths from environment variables at runtime.
- Read shared dataset location from `env/.env` or the process environment, following the same `DATASET_ROOT` convention as `lab_clip`, `negative-pedestrians`, and `clip-with-vectordb`.

## Environment file (`env/.env`)

Create `env/.env` before each run on a new machine. Example:

```bash
# env/.env
DATASET_ROOT="/mnt/data/lab_datasets"
VITAA_DATA_ROOT="/mnt/data/lab_datasets/CUHK-PEDES/vitaa"
```

| Variable | Role |
|----------|------|
| `DATASET_ROOT` | Root directory for lab datasets (`CUHK-PEDES`, …). Raw images and `reid_raw.json` are read from here. |
| `VITAA_DATA_ROOT` | ViTAA-only derived artifacts (`text_attribute_graph/`, `segs/`, processed `annotations/`). Default when unset: `${DATASET_ROOT}/CUHK-PEDES/vitaa`. |

Load before preprocessing, training, or evaluation:

```bash
set -a
source env/.env
set +a
```

### Resolving `DATASET_ROOT`

Prefer `DATASET_ROOT` from `env/.env` or the process environment. When unset, code may probe known lab roots only as a last resort:

```python
from pathlib import Path
import os

def resolve_dataset_root() -> Path:
    if root := os.environ.get("DATASET_ROOT"):
        return Path(root)
    for candidate in ("/mnt/data/lab_datasets", "/data/jayn2u/lab_datasets"):
        if Path(candidate).is_dir():
            return Path(candidate)
    raise FileNotFoundError("Set DATASET_ROOT in env/.env or the process environment.")
```

Agents adding or changing loaders and preprocessing must use this pattern instead of baking in a single host path.

### Resolved paths (CUHK-PEDES)

After loading `env/.env`:

| Purpose | Resolved path |
|---------|---------------|
| Images | `${DATASET_ROOT}/CUHK-PEDES/imgs/` |
| Raw annotation | `${DATASET_ROOT}/CUHK-PEDES/reid_raw.json` |
| Attribute phrases | `${VITAA_DATA_ROOT}/text_attribute_graph/` |
| Segmentation masks | `${VITAA_DATA_ROOT}/segs/` |
| Processed splits | `${VITAA_DATA_ROOT}/annotations/{train,val,test}.json` |

Do **not** use `reid_raw_diverse_color.json`, `reid_raw_negative_*.json`, or other extended variants. They are not paired with the official `text_attribute_graph` release.

## Project role

This repository is the official [ViTAA](https://arxiv.org/abs/2005.07327) release (ECCV 2020): visual-textual attribute alignment for text-based person search. It is **not** integrated with `lab_clip` evaluation wrappers.

Supported dataset in code: **CUHK-PEDES only** (`paths_catalog.py`). ICFG-PEDES and RSTPReid are not implemented.

There is **no publicly released ViTAA checkpoint**. Paper numbers must be reproduced by completing preprocessing and training locally. ResNet-50 ImageNet weights download automatically when `MODEL.WEIGHT: imagenet`.

## Prerequisites (readiness checklist)

Before training or evaluation, verify every item below.

| # | Artifact | Required | Resolved from |
|---|----------|----------|---------------|
| 1 | CUHK-PEDES images | Yes | `${DATASET_ROOT}/CUHK-PEDES/imgs/` |
| 2 | `reid_raw.json` | Yes | `${DATASET_ROOT}/CUHK-PEDES/reid_raw.json` |
| 3 | `text_attribute_graph/` | Yes | `${VITAA_DATA_ROOT}/text_attribute_graph/` (download) |
| 4 | `segs/` | Yes | `${VITAA_DATA_ROOT}/segs/` (Human Parsing Network) |
| 5 | `train.json`, `val.json`, `test.json` | Yes | `${VITAA_DATA_ROOT}/annotations/` (`convert_to_json`) |
| 6 | ViTAA checkpoint (`epoch_*.pth`) | Eval only | `output/cuhkpedes/` (training output) |
| 7 | `uv sync` | Yes | Once per machine |

Training or testing without `text_attribute_graph`, `segs`, or converted JSON will fail at data load time.

## Target directory layout

Raw CUHK-PEDES stays under `DATASET_ROOT`. ViTAA-specific files stay under `VITAA_DATA_ROOT` (default `${DATASET_ROOT}/CUHK-PEDES/vitaa`):

```
${DATASET_ROOT}/CUHK-PEDES/
├── imgs/
├── reid_raw.json
└── vitaa/
    ├── annotations/
    │   ├── train.json
    │   ├── val.json
    │   └── test.json
    ├── segs/
    └── text_attribute_graph/

ViTAA/
├── configs/cuhkpedes/bilstm_r50_seg.yaml
├── env/.env
├── output/cuhkpedes/
└── tools/
```

## One-shot prerequisite runner

From the project root, after `env/.env` is set:

```bash
uv sync
uv run python tools/cuhkpedes/prepare_prerequisites.py
```

This validates `${DATASET_ROOT}/CUHK-PEDES`, downloads and extracts `text_attribute_graph` with `uv run gdown`, and runs `convert_to_json.py`. It warns when `segs/` is still missing. Use `--require-segs` to fail instead of warn.

`segs/` must be generated separately with Human-Parsing-Network (see below).

## Downloads (gdown)

Official Google Drive releases used by ViTAA preprocessing. Baidu mirrors are listed in the upstream README when Drive is unavailable.

| Artifact | Google Drive ID | Local path |
|----------|-----------------|------------|
| `text_attribute_graph` archive | `1Sqm3V97hbqK9GxIwshZejJWLARfu5o1s` | `${VITAA_DATA_ROOT}/text_attribute_graph/` |
| Human Parsing Network weight | `1CYhS5AXMnMtcv9MVq5luHLrZciAwhfqn` | `{Human-Parsing-Network}/pretrained_models/` (separate repo) |

### text_attribute_graph

```bash
set -a
source env/.env
set +a

uv sync
mkdir -p "${VITAA_DATA_ROOT}"

uv run gdown 1Sqm3V97hbqK9GxIwshZejJWLARfu5o1s \
  -O "${VITAA_DATA_ROOT}/text_attribute_graph.zip"
unzip "${VITAA_DATA_ROOT}/text_attribute_graph.zip" -d "${VITAA_DATA_ROOT}/"
```

Or run the full prerequisite chain (download, extract, `convert_to_json`):

```bash
uv run python tools/cuhkpedes/prepare_prerequisites.py
```

After extraction, JSON files must sit directly under `${VITAA_DATA_ROOT}/text_attribute_graph/` (for example `CUHK01-0363004-0.json`). If the archive unpacks into a nested folder, move that folder to `text_attribute_graph/`. Do not replace this with a symlink.

Baidu fallback: [link (code: vbss)](https://pan.baidu.com/s/1TIX4lbvZmGwbBNHcRyA1ng).

### Human Parsing Network weight

Used in the separate [Human-Parsing-Network](https://github.com/Jarr0d/Human-Parsing-Network) repo to generate `segs/`:

```bash
mkdir -p "${HUMAN_PARSING_ROOT}/pretrained_models"

uv run gdown 1CYhS5AXMnMtcv9MVq5luHLrZciAwhfqn \
  -O "${HUMAN_PARSING_ROOT}/pretrained_models/"
```

Set `HUMAN_PARSING_ROOT` to the clone location of Human-Parsing-Network on the current machine.

Naming convention (`tools/cuhkpedes/convert_to_json.py`):

- Image `file_path` `CUHK01/0363004.png` with caption index `0` → `CUHK01-0363004-0.json`
- Caption index `1` → `CUHK01-0363004-1.json`

Each file holds parsed attribute phrases mapped to five body regions: `head`, `upperbody`, `lowerbody`, `shoe`, `backpack`.

## segs (Human Parsing Network)

Segmentation maps are required at load time (`vitaa/data/datasets/cuhkpedes.py`):

- Image: `${DATASET_ROOT}/CUHK-PEDES/imgs/{file_path}`
- Seg: `${VITAA_DATA_ROOT}/segs/{file_path without extension}.png`

Example: `imgs/CUHK01/0363004.png` → `segs/CUHK01/0363004.png`

Generate with [Jarr0d/Human-Parsing-Network](https://github.com/Jarr0d/Human-Parsing-Network):

1. Clone the parsing repo (separate from ViTAA); set `HUMAN_PARSING_ROOT` to its path.
2. Download the pretrained parsing weight (see **Downloads (gdown)** → Human Parsing Network weight).
3. Run inference on `${DATASET_ROOT}/CUHK-PEDES/imgs` via configs under `experiments/`.
4. Write PNG outputs under `${VITAA_DATA_ROOT}/segs/` with the same relative paths as under `imgs/`.

Every annotated image needs a matching seg file. Missing segs cause `FileNotFoundError` during training or evaluation.

## Annotation conversion

After `text_attribute_graph/` is in place and `reid_raw.json` is available under `DATASET_ROOT`:

```bash
set -a
source env/.env
set +a

uv sync

uv run python tools/cuhkpedes/convert_to_json.py \
  --datadir "${VITAA_DATA_ROOT}" \
  --outdir "${VITAA_DATA_ROOT}/annotations" \
  --reid-raw "${DATASET_ROOT}/CUHK-PEDES/reid_raw.json"
```

The script tokenizes captions, builds vocabulary (default `word_count_threshold=2`), writes `onehot` caption IDs and `att_onehot` per-region word IDs, and emits `train.json`, `val.json`, `test.json`.

`--reid-raw` points at the shared lab annotation file; do not copy or symlink it into `VITAA_DATA_ROOT`. When wiring `paths_catalog.py` and the dataset loader, read images from `${DATASET_ROOT}/CUHK-PEDES/imgs` and segs from `${VITAA_DATA_ROOT}/segs`.

Config vocabulary sizes (`NUM_CLASSES: 12003`, `VOCABULARY_SIZE: 12000` in `configs/cuhkpedes/bilstm_r50_seg.yaml`) assume this conversion output.

## Training

```bash
set -a
source env/.env
set +a

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
set -a
source env/.env
set +a

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
6. **Never symlink** `${DATASET_ROOT}` assets into `${VITAA_DATA_ROOT}` or this repository; resolve paths from `env/.env` in code and shell commands instead.
7. When updating `paths_catalog.py`, `convert_to_json.py`, or dataset loaders, wire `DATASET_ROOT` and `VITAA_DATA_ROOT` explicitly; upstream relative `datasets/cuhkpedes/imgs` layout alone is not portable across lab machines.
