# Visuomotor Imitation Learning Pipeline (H5 -> BC + Flow Matching + Isaac Sim)

This repository provides an end-to-end baseline for object manipulation imitation learning from `.h5` demonstrations.

## Repository Tree

```text
project/
  configs/
    dataset.yaml
    bc.yaml
    flow.yaml
    eval.yaml
  data/
    raw/
    processed/
    splits.json
  models/
    encoders.py
    bc_policy.py
    flow_policy.py
  datasets/
    inspect_h5.py
    h5_dataset.py
    preprocess_dataset.py
    utils.py
  training/
    train_bc.py
    train_flow.py
    losses.py
    engine.py
  evaluation/
    evaluate_offline.py
    evaluate_rollout.py
    metrics.py
  sim/
    isaac_wrapper.py
    replay_episode.py
  utils/
    logging_utils.py
    config_utils.py
    seed.py
  README.md
```

## Expected Dataset Format

- One `.h5` file = one episode (default assumption).
- Typical keys (override in `configs/dataset.yaml`):
  - image key: `observations/images/front`
  - proprio key: `observations/qpos`
  - action key: `actions`

If `actions` is missing, the dataset loader can derive surrogate targets from next-step proprio delta.

## 1) Inspect your H5 files first

```bash
python -m datasets.inspect_h5 \
  --input_dir data/raw \
  --glob "*.h5" \
  --output_json data/processed/h5_inspection.json \
  --output_csv data/processed/h5_dataset_summary.csv
```

Outputs:
- nested key listing
- shape + dtype inventory
- likely image/state/action keys
- per-file timestep estimate
- aggregate defaults recommendation

Then set `configs/dataset.yaml` keys accordingly.

## 2) Preprocess metadata and splits

```bash
python -m datasets.preprocess_dataset \
  --raw_dir data/raw \
  --glob "*.h5" \
  --proprio_key observations/qpos \
  --processed_dir data/processed \
  --write_index_cache
```

This writes:
- `data/processed/metadata.json`
- `data/processed/splits.json`
- optionally `data/processed/index_cache.json`

## 3) Train Behavior Cloning baseline

```bash
python -m training.train_bc --config configs/bc.yaml
```

BC model:
- image encoder (small CNN or ResNet18)
- proprio MLP
- fusion MLP
- action chunk regression with MSE (+ optional smoothness)

## 4) Train Flow Matching policy

```bash
python -m training.train_flow --config configs/flow.yaml
```

Flow model:
- same visual/proprio conditioning
- sinusoidal time embedding
- vector field over action chunk
- standard flow-matching objective on linear interpolation path from Gaussian base

## 5) Offline Evaluation

```bash
python -m evaluation.evaluate_offline --config configs/eval.yaml
```

Metrics:
- action MSE
- endpoint error (last step in chunk)
- per-dimension MSE

Outputs JSON comparison for BC vs Flow.

## 6) Isaac Sim Integration

The integration layer is intentionally isolated in `sim/isaac_wrapper.py`.

Implemented as concrete interface methods with explicit TODO markers:
- `reset_episode`
- `set_state_from_demo`
- `apply_action`
- `get_observation`
- `step`
- `run_closed_loop_rollout`

Use:
- `python -m sim.replay_episode --episode_h5 <path>` for demo replay wiring
- `python -m evaluation.evaluate_rollout --policy bc --config configs/eval.yaml` for rollout eval wiring

## Reproducibility Notes

- Seed control: `utils/seed.py`
- Config-driven runs via YAML files in `configs/`
- Deterministic split generation in preprocessing
- Optional TensorBoard logging

## Known TODO(project-specific)

- Isaac Sim environment API bindings.
- Observation preprocessing in rollout evaluation.
- Choosing final action target derivation when real `actions` key is absent.
- Multi-camera support expansion (current default is one stream key).
