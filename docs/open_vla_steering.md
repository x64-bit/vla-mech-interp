# OpenVLA Steering & LIBERO Evaluation Guide

Documentation for the changes we made inside `openvla/analysis/steering` and `experiments/robot/libero`, plus the steps required to re-run the full mechanistic interpretability → activation-steering pipeline.

---

## 1. Repository Additions

- **FFN extraction & analysis**
  - `extract_ffn_vectors.py`: single-image probe with hook registration.
  - `bulk_extract_ffn_vectors.py`: TFDS-based iterator over LIBERO RLDS (chunked `.npz` per layer).
  - `project_ffn_vectors.py`: projection onto the OpenVLA token embedding basis.
  - `cluster_projected_vectors.py`: KMeans clustering + JSON summaries.
  - `visualize_tsne.py`: 2-D scatter plots for cluster sanity checks.
  - `build_steering_vector.py`: averages a cluster into a steering vector.
  - `steering_vectors/` & `steering_configs/`: versioned vectors plus JSON configs (e.g., `fast_layer13.npy/json`).

- **Activation steering + evaluation**
  - `apply_activation_steering.py`: smoke-test script for steering hooks.
  - `experiments/robot/libero/run_libero_eval.py`: now accepts `--steer-config`, `--metrics-output-dir`, `--dump-actions` and logs `episodes.jsonl` + `metrics_summary.json`.
  - `analysis/steering/run_libero_eval_with_configs.py`: automation driver that iterates suites/configs and appends to `libero_results.csv`.
  - All steering utilities add `Path(__file__).resolve().parents[3]` to `sys.path`, so you must launch from `/home/ubuntu/eecs182proj/src/openvla`.

- **Config tweak**
  - `~/.libero/config.yaml` now points `datasets:` to `/home/ubuntu/eecs182proj/src/openvla/modified_libero_rlds`, matching the TFDS builders we created.

---

## 2. Environment Requirements

```bash
cd /home/ubuntu/eecs182proj/src
. /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate openvla
export TFDS_DATA_DIR=/home/ubuntu/eecs182proj/src/openvla/modified_libero_rlds
export HF_HOME=/home/ubuntu/.cache/huggingface
```

Make sure MuJoCo runs headless:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
```

---

## 3. Data Locations

- **Modified LIBERO RLDS** (TFDS layout expected by extraction scripts):  
  `/home/ubuntu/eecs182proj/src/openvla/modified_libero_rlds`
- **Projected FFN chunks & metadata**:  
  `/home/ubuntu/eecs182proj/src/openvla/analysis/steering/spatial_ffn_chunks/`  
  `/home/ubuntu/eecs182proj/src/openvla/analysis/steering/spatial_projections/`
- **Cluster and steering assets**:  
  `/home/ubuntu/eecs182proj/src/openvla/analysis/steering/spatial_clusters*/`, `steering_vectors/`, `steering_configs/`
- **Evaluation logs and rollouts**:  
  `/home/ubuntu/eecs182proj/src/openvla/analysis/steering/libero_runs/`  
  `/home/ubuntu/eecs182proj/src/openvla/rollouts/YYYY_MM_DD/`

---

## 4. Pipeline Steps

### 4.1 Extract FFN value vectors at scale

```bash
cd /home/ubuntu/eecs182proj/src/openvla
python -m openvla.analysis.steering.bulk_extract_ffn_vectors \
  --model-id openvla/openvla-7b-finetuned-libero-spatial \
  --dataset libero_spatial_no_noops \
  --data-dir /home/ubuntu/eecs182proj/src/openvla/modified_libero_rlds \
  --output-root analysis/steering/spatial_ffn_chunks \
  --device cuda:0 \
  --dtype bfloat16 \
  --max-examples 5000        # optional throttle while testing
```

Artifacts: chunked `.npz` files per transformer layer plus metadata JSON for example indices.

### 4.2 Project FFN vectors into token space

```bash
python -m openvla.analysis.steering.project_ffn_vectors \
  --model-id openvla/openvla-7b-finetuned-libero-spatial \
  --ffn-root analysis/steering/spatial_ffn_chunks \
  --output-root analysis/steering/spatial_projections \
  --top-k 32 \
  --dtype bfloat16
```

This stores `(token_id, score)` pairs per FFN vector so we can reason about semantics.

### 4.3 Cluster projected vectors

```bash
python -m openvla.analysis.steering.cluster_projected_vectors \
  --projections-root analysis/steering/spatial_projections \
  --output-root analysis/steering/spatial_clusters_k10 \
  --k 10 \
  --layers 13 14 15
```

Outputs: `clusters_layerXX.json` with cluster centroids, member counts, and top tokens.

### 4.4 Visual sanity checks (optional)

```bash
python -m openvla.analysis.steering.visualize_tsne \
  --projections-root analysis/steering/spatial_projections \
  --clusters-root analysis/steering/spatial_clusters_k10 \
  --layer 13 \
  --output-path analysis/steering/plots/layer_13_tsne_fastslow.png
```

### 4.5 Build steering vectors

```bash
python -m openvla.analysis.steering.build_steering_vector \
  --ffn-root analysis/steering/spatial_ffn_chunks \
  --clusters-path analysis/steering/spatial_clusters_k10/layer_13_clusters.json \
  --cluster-id 2 \
  --output-path analysis/steering/steering_vectors/fast_layer13.npy
```

Update/create a config (example already committed):

```json
{
  "entries": [
    {
      "layer": 13,
      "vector_path": "/home/ubuntu/eecs182proj/src/openvla/analysis/steering/steering_vectors/fast_layer13.npy",
      "scale": 10.0
    }
  ]
}
```

### 4.6 Smoke-test steering in isolation

```bash
python -m openvla.analysis.steering.apply_activation_steering \
  --model-id openvla/openvla-7b-finetuned-libero-spatial \
  --image-path /home/ubuntu/eecs182proj/data/libero_lerobot/.../frame_000000.png \
  --instruction "place seal on blue plate" \
  --steer-config analysis/steering/steering_configs/fast_layer13.json \
  --max-new-tokens 64 \
  --device cuda:0
```

The script prints generated tokens and saves them to `analysis/steering/runs/*.json`.

### 4.7 Run LIBERO evaluation (single config)

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
python -m openvla.experiments.robot.libero.run_libero_eval \
  --suite libero_spatial \
  --policy-id openvla/openvla-7b-finetuned-libero-spatial \
  --episodes-per-task 50 \
  --steer-config-path analysis/steering/steering_configs/fast_layer13.json \
  --metrics-output-dir analysis/steering/libero_runs/libero_spatial/fast \
  --dump-actions
```

Products:
- `episodes.jsonl`: per-episode status, optional action/token dumps.
- `metrics_summary.json`: success counts, average episode length, etc.
- `rollouts/YYYY_MM_DD/*.mp4`: optional video (requires `--record` flag in config).

### 4.8 Batch multiple configs/suites

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
python -m openvla.analysis.steering.run_libero_eval_with_configs \
  --suite libero_spatial \
  --steer-config analysis/steering/steering_configs/fast_layer13.json \
  --label fast \
  --episodes-per-task 50 \
  --log-root analysis/steering/libero_runs \
  --results-csv analysis/steering/libero_results.csv
```

`run_libero_eval_with_configs.py` wraps `run_libero_eval.py`, persists stdout/stderr per run, and appends a single CSV row summarizing the metrics.

---

## 5. Tips & Gotchas

- **Mixed dtypes**: models default to `bfloat16`. Every script casts `pixel_values` to the model dtype, but if you override `--dtype float32`, ensure enough GPU memory or pass `--device cpu`.
- **TFDS builder names**: pass only the builder ID (e.g., `libero_spatial_no_noops`). The script relies on `--data-dir` or `TFDS_DATA_DIR`.
- **Path resolution**: run scripts from `/home/ubuntu/eecs182proj/src/openvla` or ensure `PYTHONPATH` includes the repo root.
- **MuJoCo GL backend**: headless boxes without EGL need `osmesa`. Confirm via `echo $MUJOCO_GL` before launching evaluations.
- **Long sweeps**: 10 tasks × 50 episodes × multiple configs takes hours. Use `tmux` or the automation driver, and watch GPU memory with `nvidia-smi`.
- **Results aggregation**: `analysis/steering/libero_results.csv` is the canonical table. Downstream plotting scripts can read this file directly.

---

## 6. Working for full data

Since we did not have the compute to run all of our experiments with full data converage. Here are some steps that may be helpful for larger testing.

1. Finish remaining LIBERO suites (Object, Goal, Long) with baseline + steering configs.
2. Expand clustering beyond layer 13 to capture “slow” and “precision” directions.
3. Mirror the same pipeline for OpenPI once the `envs/openpi` virtualenv is ready.
4. Produce figures/tables: t‑SNE plots, rollout frame comparisons, and the Table 1 replica.

With this document and the committed scripts, future users can recreate the entire pipeline by following Sections 2–4 end to end.

