# Visuomotor BC / Flow Matching Baseline

This repository is a compact imitation-learning baseline for robot manipulation from HDF5 (`.h5`) demonstrations.

The main use case is:
- demonstrations stored as one H5 file per episode
- RGB images + robot observations inside each file
- behavior cloning (BC) training first
- optional flow-matching and simulator integration later

It now supports:
- multiple camera streams
- multiple proprio/action keys concatenated together
- local H5 files
- a parallel Hugging Face H5 loader path

For our current setup, that means we can train a **bimanual arm policy** by concatenating:
- left arm proprio + right arm proprio
- left arm action + right arm action

## Repo Overview

```text
BC_3DV/
  configs/        YAML experiment and dataset configs
  data/           local split files and optional processed metadata
  datasets/       H5 inspection, preprocessing, and dataset loaders
  evaluation/     offline metrics and rollout entry points
  models/         image encoders and policy definitions
  sim/            Isaac Sim interface stubs
  training/       BC / flow training scripts and losses
  utils/          config loading, logging, seeding
```

## Folder Guide

### `configs/`

- [configs/dataset.yaml](/Users/maxence/Desktop/3dv/BC_3DV/configs/dataset.yaml)
  Central dataset contract. Defines:
  - where local H5 files live
  - which split file to use
  - which image keys to read
  - which proprio keys to concatenate
  - which action keys to concatenate
  - frame stack / action chunk / image resize settings

- [configs/bc.yaml](/Users/maxence/Desktop/3dv/BC_3DV/configs/bc.yaml)
  Training config for the main BC baseline.

- [configs/flow.yaml](/Users/maxence/Desktop/3dv/BC_3DV/configs/flow.yaml)
  Training config for the flow-matching model.

- [configs/eval.yaml](/Users/maxence/Desktop/3dv/BC_3DV/configs/eval.yaml)
  Offline evaluation config and checkpoint paths.

### `data/`

- `data/raw/`
  Expected location for local H5 episodes. Not checked in.

- [data/processed/splits.json](/Users/maxence/Desktop/3dv/BC_3DV/data/processed/splits.json)
  Train/val/test episode lists. Right now this is set up as a one-file smoke test.

- `data/processed/metadata.json`
  Optional metadata written by preprocessing.

- `data/processed/index_cache.json`
  Optional precomputed sample index written by preprocessing.

### `datasets/`

- [datasets/h5_dataset.py](/Users/maxence/Desktop/3dv/BC_3DV/datasets/h5_dataset.py)
  Main local dataset loader.

  It:
  - reads one H5 episode per file
  - loads one or more image streams
  - concatenates one or more proprio keys
  - concatenates one or more action keys
  - produces action chunks for supervised learning

  This is the most important loader for current BC training.

- [datasets/hf_h5_dataset.py](/Users/maxence/Desktop/3dv/BC_3DV/datasets/hf_h5_dataset.py)
  Hugging Face-backed dataset loader with the same sample structure as the local loader.

  Important:
  - this path is still less mature than the local loader
  - it is useful for experiments where some episodes live on HF
  - the local BC path should be considered the primary path for now

- [datasets/inspect_h5.py](/Users/maxence/Desktop/3dv/BC_3DV/datasets/inspect_h5.py)
  Scans raw H5 files and reports likely image/state/action keys, shapes, and dataset inventory.

- [datasets/inspect_action_semantics.py](/Users/maxence/Desktop/3dv/BC_3DV/datasets/inspect_action_semantics.py)
  Heuristic tool to inspect what `actions_*` likely mean.

  It helps answer questions like:
  - do these 7D actions look like joint commands?
  - do they look quaternion-like in the last 4 dimensions?
  - are they numerically closer to next-step observations or simple observation deltas?

- [datasets/preprocess_dataset.py](/Users/maxence/Desktop/3dv/BC_3DV/datasets/preprocess_dataset.py)
  Builds split files and optional metadata/index cache from a directory of H5 files.

- [datasets/utils.py](/Users/maxence/Desktop/3dv/BC_3DV/datasets/utils.py)
  Shared helpers for H5 traversal and file discovery.

### `models/`

- [models/encoders.py](/Users/maxence/Desktop/3dv/BC_3DV/models/encoders.py)
  Visual encoder definitions:
  - `small_cnn`
  - `resnet18`

- [models/bc_policy.py](/Users/maxence/Desktop/3dv/BC_3DV/models/bc_policy.py)
  BC policy.

  Inputs:
  - concatenated image tensor
  - concatenated proprio vector

  Output:
  - flattened future action chunk

- [models/flow_policy.py](/Users/maxence/Desktop/3dv/BC_3DV/models/flow_policy.py)
  Flow-matching policy over action chunks.

### `training/`

- [training/train_bc.py](/Users/maxence/Desktop/3dv/BC_3DV/training/train_bc.py)
  Main training entry point for local H5 BC.

- [training/train_bc_hf.py](/Users/maxence/Desktop/3dv/BC_3DV/training/train_bc_hf.py)
  BC training entry point for Hugging Face-hosted H5 episodes.

- [training/train_flow.py](/Users/maxence/Desktop/3dv/BC_3DV/training/train_flow.py)
  Flow-matching training script.

- [training/losses.py](/Users/maxence/Desktop/3dv/BC_3DV/training/losses.py)
  BC and flow losses.

- [training/engine.py](/Users/maxence/Desktop/3dv/BC_3DV/training/engine.py)
  Shared batch-to-device helper.

### `evaluation/`

- [evaluation/metrics.py](/Users/maxence/Desktop/3dv/BC_3DV/evaluation/metrics.py)
  Offline metrics like action MSE and endpoint error.

- [evaluation/evaluate_offline.py](/Users/maxence/Desktop/3dv/BC_3DV/evaluation/evaluate_offline.py)
  Offline checkpoint comparison script.

- [evaluation/evaluate_rollout.py](/Users/maxence/Desktop/3dv/BC_3DV/evaluation/evaluate_rollout.py)
  Rollout entry point for simulator evaluation.

  Important:
  - this is still a stubbed integration path
  - observation preprocessing is not implemented yet

### `sim/`

- [sim/isaac_wrapper.py](/Users/maxence/Desktop/3dv/BC_3DV/sim/isaac_wrapper.py)
  Boundary layer for Isaac Sim integration.

- [sim/replay_episode.py](/Users/maxence/Desktop/3dv/BC_3DV/sim/replay_episode.py)
  Demo replay harness for simulator wiring.

### `utils/`

- [utils/config_utils.py](/Users/maxence/Desktop/3dv/BC_3DV/utils/config_utils.py)
  YAML loading helper.

- [utils/logging_utils.py](/Users/maxence/Desktop/3dv/BC_3DV/utils/logging_utils.py)
  TensorBoard / no-op logging wrapper.

- [utils/seed.py](/Users/maxence/Desktop/3dv/BC_3DV/utils/seed.py)
  Random seed setup.

## Data Contract

The repo assumes:
- one H5 file = one episode
- first dimension = timesteps

Typical H5 structure:

```text
episode.h5
  actions_arm_left
  actions_arm_right
  observations/
    images/
      aria_rgb_cam/color
      oakd_front_view/color
    ...
```

The current `dataset.yaml` is configured for **both arms together**:

```yaml
keys:
  image_keys:
    - observations/images/aria_rgb_cam/color
    - observations/images/oakd_front_view/color
  proprio_keys:
    - observations/qpos_arm_left
    - observations/qpos_arm_right
  action_keys:
    - actions_arm_left
    - actions_arm_right
```

This means a single sample contains:
- image: both camera streams concatenated channel-wise
- proprio: left + right proprio concatenated
- action: future left + right action chunk concatenated

If your observation keys are not `qpos_*` but end-effector poses, replace `proprio_keys` with the real pose keys.

## What BC Produces

The BC model predicts a flattened future action chunk.

Example with:
- left arm action dim = 7
- right arm action dim = 7
- total per-step action dim = 14
- `action_chunk = 8`

Then the model output is:
- `8 x 14 = 112` values per sample

Conceptually:

```text
[
  left_t0(7), right_t0(7),
  left_t1(7), right_t1(7),
  ...
  left_t7(7), right_t7(7)
]
```

Checkpoints are written to:
- `outputs/bc/checkpoint_best.pt`
- `outputs/bc/checkpoint_last.pt`
- `outputs/bc/checkpoint_XXXX.pt`

TensorBoard logs go to:
- `outputs/bc/tb/`

## Recommended Workflow For New Data

### 1. Inspect raw H5 structure

```bash
python -m datasets.inspect_h5 \
  --input_dir data/raw \
  --glob "*.h5" \
  --output_json data/processed/h5_inspection.json \
  --output_csv data/processed/h5_dataset_summary.csv
```

Use this first when the key structure is still uncertain.

### 2. Inspect action semantics

```bash
python -m datasets.inspect_action_semantics \
  --input_h5 data/raw/20250827_151212.h5 \
  --action_keys actions_arm_left actions_arm_right \
  --observation_keys observations/qpos_arm_left observations/qpos_arm_right \
  --output_json data/processed/action_semantics.json
```

If your observations are end-effector poses, replace the observation keys with those pose keys.

This script is heuristic, not definitive, but it is useful for checking whether `actions_*` look more like:
- absolute targets
- deltas
- quaternion-style pose commands

### 3. Create splits

For a multi-file dataset:

```bash
python -m datasets.preprocess_dataset \
  --raw_dir data/raw \
  --glob "*.h5" \
  --proprio_key observations/qpos_arm_left \
  --processed_dir data/processed \
  --write_index_cache
```

For a one-file smoke test, using the checked-in `data/processed/splits.json` is fine.

### 4. Train BC locally

```bash
python -m training.train_bc --config configs/bc.yaml
```

### 5. Train BC from HF-hosted H5 files

```bash
python -m training.train_bc_hf \
  --config configs/bc.yaml \
  --hf_repo_id your-name/your-dataset
```

## Current Status

### Most stable path

- local H5 BC training via [training/train_bc.py](/Users/maxence/Desktop/3dv/BC_3DV/training/train_bc.py)

### Usable but less mature

- Hugging Face H5 loading via [datasets/hf_h5_dataset.py](/Users/maxence/Desktop/3dv/BC_3DV/datasets/hf_h5_dataset.py)
- HF BC entry point via [training/train_bc_hf.py](/Users/maxence/Desktop/3dv/BC_3DV/training/train_bc_hf.py)

### Still scaffold / TODO territory

- full simulator rollout evaluation
- Isaac Sim integration
- polished mixed local + HF dataset orchestration
- confirming action semantics from robot/controller source code

## Notes For Teammates

- This repo is intentionally small. Most logic is in the dataset loaders and BC training script.
- The dataset contract matters more than the model code. If keys or action semantics are wrong, training can run but still learn the wrong behavior.
- `actions_*` should be treated as logged control targets unless proven otherwise.
- If actions are missing and we derive them from observations, that is only a fallback baseline.
- The safest first milestone on a new machine is always:
  1. inspect one H5
  2. confirm dataset keys
  3. run one-file BC smoke test
  4. only then scale to more episodes or remote training

## Minimal Smoke Test

Put one episode at:

```text
data/raw/20250827_151212.h5
```

Then run:

```bash
python -m training.train_bc --config configs/bc.yaml
```

If the dataset keys in `configs/dataset.yaml` match the file, this should:
- load both cameras
- concatenate both arm streams
- train BC
- write checkpoints to `outputs/bc/`
