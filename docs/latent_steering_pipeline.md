## Latent Steering Pipeline

This document explains how to reproduce the concept-specific latent steering vectors for π0.5 LIBERO tasks. The workflow has three phases:

1. **Prepare contrastive datasets and concept ranges**
2. **Capture & slice instruction-token residuals**
3. **Project through the SAE and compute steering vectors**


### 1. Prepare contrastive datasets

1. **Filter demos (optional):**
   ```bash
   mkdir -p data/libero_filtered/open_close
   cp data/libero_100_original/libero_90/KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_demo.hdf5 data/libero_filtered/open_close/
   # …repeat for the other demos of interest…
   ```

2. **Convert to LeRobot format:**
   ```bash
   source src/openpi/.venv/bin/activate
   uv run src/openpi/examples/libero/convert_libero90_data_to_lerobot.py \
     --args.input-dir data/libero_filtered/open_close \
     --args.repo-id libero_open_close_lerobot \
     --args.output-dir data/libero_lerobot/libero_open_close_lerobot
   ```

3. **Build contrast manifests for each concept pair:**
   ```bash
   python data/utils/build_contrasting_pairs.py \
     --backend lerobot \
     --lerobot-root data/libero_lerobot/libero_open_close_lerobot \
     --task-a-text "open the top drawer of the cabinet" \
     --task-b-text "close the top drawer of the cabinet" \
     --output data/manifests/open_vs_close.json
   ```

4. **Map manifest segments to dataset indices:**
   ```bash
   python src/openpi/scripts/map_concept_samples.py \
     --manifest-path data/manifests/open_vs_close.json \
     --output-path data/manifests/open_vs_close_concept_map.json \
     --lerobot-root data/libero_lerobot/libero_open_close_lerobot \
     --repo_id libero_open_close_lerobot
   ```

Repeat steps 3–4 for each task pair (e.g., “turn on/off the stove”).


### 2. Capture and slice instruction residuals

1. **Add a train config pointing at the filtered repo** (done once by extending `training/config.py` with `pi05_libero_open_close`).

2. **Capture instruction-token residuals at layer 12:**
   ```bash
   python -m openpi.analysis.save_residuals \
     --args.checkpoint-path data/openpi-assets/checkpoints/pi05_libero_pytorch/model.safetensors \
     --args.config-name pi05_libero_open_close \
     --args.output-dir data/residuals/open_close_run \
     --args.layers 12 \
     --args.batch-size 128 \
     --args.capture-instructions \
     --args.skip-norm-stats
   ```

3. **Slice residuals by concept:**
   ```bash
   python src/openpi/scripts/slice_residuals_by_concept.py \
     --run-dir data/residuals/open_close_run \
     --concept-map-path data/manifests/open_vs_close_concept_map.json \
     --output-dir data/residuals/open_vs_close_slices \
     --token-type instructions \
     --layers 12 \
     --device cpu
   ```

   Outputs live at `data/residuals/<pair>_slices/layer12/<concept>.pt`.


### 3. Compute latent steering vectors

1. **Describe the concept pairs for the SAE script:**
   ```json
   // data/latent_steering/layer12_pairs.json
   {
     "pairs": [
       {
         "name": "open_vs_close",
         "positive_label": "open the top drawer of the cabinet",
         "positive_path": "data/residuals/open_vs_close_slices/layer12/open the top drawer of the cabinet.pt",
         "negative_label": "close the top drawer of the cabinet",
         "negative_path": "data/residuals/open_vs_close_slices/layer12/close the top drawer of the cabinet.pt"
       }
     ]
   }
   ```

2. **Run the SAE utility:**
   ```bash
   python src/openpi/scripts/compute_latent_steering.py \
     --args.pairs-config data/latent_steering/layer12_pairs.json \
     --args.output-root data/latent_steering \
     --args.layer 12 \
     --args.device cuda \
     --args.batch-size 512
   ```

   The script:
   - Loads the Gemma-2B SAE (`gemma-2b-it-res-jb / blocks.12.hook_resid_post`)
   - Encodes each concept tensor, saving per-concept latent means
   - Computes `positive − negative` in latent space and decodes it back to residual space
   - Writes outputs to `data/latent_steering/layer12/<pair>/` and optionally cosine stats under `pairwise_stats.pt`


### Outputs

- `data/residuals/...` – concept-aligned instruction residual tensors
- `data/latent_steering/layer12/<pair>/` – `*_latent_mean.pt`, `latent_direction.pt`, `residual_direction.pt`
- `data/latent_steering/layer12/pairwise_stats.pt` – cosine similarity and magnitude comparisons between pairs

These artifacts can be fed into SAE probing notebooks, steering experiments, or downstream evaluations.

