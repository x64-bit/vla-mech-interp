# Custom Libero Contrast Suites

We currently curate four binary “concept suites” inside `openpi` to study π0/π0.5 behavior under tightly controlled instruction pairs. Each suite consists of:

- A filtered LeRobot repo (`data/libero_lerobot/*`) derived from the LIBERO demonstrations.
- A contrast manifest (`data/manifests/*.json`) describing per-demo interaction windows along the axis that most cleanly separates the two tasks.
- A concept map (`*_concept_map.json`) that anchors every segment back to dataset indices so we can slice model activations.
- Residual capture runs plus pre-sliced tensors under `data/residuals/`.

Use `docs/latent_steering_pipeline.md` if you need the end-to-end recipe (conversion → manifest → concept slicing → SAE projections). This document focuses on what already exists and how to reuse or extend it.

## Build Workflow (shared by every suite)

1. **Filter/convert demos**  
   `uv run src/openpi/examples/libero/convert_libero90_data_to_lerobot.py --args.input-dir <filtered_root> --args.repo-id <repo> --args.output-dir data/libero_lerobot/<repo>`
2. **Generate contrast windows**  
   `python data/utils/build_contrasting_pairs.py --backend lerobot --lerobot-root data/libero_lerobot/<repo> --task-a-text "<positive>" --task-b-text "<negative>" --output data/manifests/<suite>.json`
3. **Map to concept indices**  
   `python src/openpi/scripts/map_concept_samples.py --manifest-path data/manifests/<suite>.json --lerobot-root data/libero_lerobot/<repo> --repo_id <repo> --output-path data/manifests/<suite>_concept_map.json`
4. **Capture + slice residuals**  
   - Capture: `python -m openpi.analysis.save_residuals --args.config-name <train_config> --args.checkpoint-path <weights> --args.output-dir data/residuals/<run_dir> --args.layers 12 --args.capture-instructions`
   - Slice: `python src/openpi/scripts/slice_residuals_by_concept.py --run-dir data/residuals/<run_dir> --concept-map-path data/manifests/<suite>_concept_map.json --output-dir data/residuals/<suite>_slices --layers 12 --token-type instructions`

Once a suite is built, add it to `data/latent_steering/layer12_pairs.json` so SAE utilities can compute directions.

## Suite Catalog

| Suite | Positive / Negative task | `repo_id` | Manifest → Concept map | Segments per task | Axis idx / effect | Residual slices |
| --- | --- | --- | --- | --- | --- | --- |
| `open_vs_close` | “open the top drawer of the cabinet” / “close the top drawer of the cabinet” | `libero_open_close_lerobot` | `data/manifests/open_vs_close.json` → `data/manifests/open_vs_close_concept_map.json` | 50 / 50 | 5 / 1.60 | `data/residuals/open_vs_close_slices/layer12/*.pt` |
| `on_vs_off` | “turn on the stove” / “turn off the stove” | `libero_open_close_lerobot` | `data/manifests/on_vs_off.json` → `data/manifests/on_vs_off_concept_map.json` | 50 / 50 | 4 / 8.96 | `data/residuals/on_vs_off_slices/layer12/*.pt` |
| `open_vs_close_front_back` | Same drawer pair as above but restricted to the front/back composite repo | `libero_open_close_front_back_lerobot` | `data/manifests/open_vs_close_front_back.json` → `data/manifests/open_vs_close_front_back_concept_map.json` | 50 / 50 | 5 / 1.60 | `data/residuals/open_vs_close_front_back_slices/layer12/*.pt` |
| `back_vs_front` | “put the black bowl at the back on the plate” / “…at the front…” | `libero_open_close_front_back_lerobot` | `data/manifests/back_vs_front.json` → `data/manifests/back_vs_front_concept_map.json` | 50 / 50 | 1 / -3.08 | `data/residuals/back_vs_front_slices/layer12/*.pt` |

`axis idx` refers to the Libero state dimension chosen by `build_contrasting_pairs.py`; higher absolute effect sizes mean a cleaner separation. All manifests currently hold 100 segments (50 per concept).

## Suite Details

### Open vs Close
- **Purpose:** isolates drawer opening/closing to probe actuated yaw/pitch behaviors without camera ambiguity.
- **Data roots:** `data/libero_lerobot/libero_open_close_lerobot` (repo `libero_open_close_lerobot` on Hugging Face).
- **Stats:** `axis_index=5`, `effect_size=1.60`, mean window length ≈64 steps, concept sample count 22,259 frames.
- **Training configs:** `pi0_libero_open_close` and `pi05_libero_open_close` inside `src/openpi/src/openpi/training/config.py`.
- **Residual assets:** raw capture at `data/residuals/open_close_run/`, sliced tensors in `data/residuals/open_vs_close_slices/`.
- **Latent steering entry:** `data/latent_steering/layer12_pairs.json` → `name: "open_vs_close"`.

### On vs Off
- **Purpose:** contrasts stove power toggling, giving a language-only shift without major arm motion changes.
- **Data roots:** shares `libero_open_close_lerobot` with the drawer pair; only task text differs.
- **Stats:** `axis_index=4`, `effect_size=8.96`, mean window length ≈94 steps with a short median (56) because motion bursts are brief, concept sample count 22,259.
- **Residual assets:** `data/residuals/on_vs_off_slices/` plus precomputed latent means (`layer12_on_off_latents.pt`).
- **Latent steering entry:** `name: "on_vs_off"` in `data/latent_steering/layer12_pairs.json`.

### Open vs Close (Front/Back repo)
- **Purpose:** same drawer verbs but evaluated inside the harder composite workspace (`libero_open_close_front_back_lerobot`) so the visual backdrop changes from front/back placements.
- **Stats:** matches the vanilla drawer axis (`axis_index=5`, `effect_size=1.60`) but draws from 20,041 frames due to fewer demos. Window lengths mirror the original suite (mean ≈64 steps).
- **Residual assets:** capture run `data/residuals/open_close_front_back_run/`, slices at `data/residuals/open_vs_close_front_back_slices/`.
- **Training config:** `pi05_libero_open_close_front_back`.
- **Latent steering entry:** `name: "open_vs_close_front_back_dataset"`.

### Back vs Front
- **Purpose:** probes spatial disambiguation (putting a bowl on the front vs back of a plate) to test whether instruction tokens encode egocentric references.
- **Data roots:** `libero_open_close_front_back_lerobot`.
- **Stats:** `axis_index=1`, `effect_size=-3.08` (large magnitude), mean window length ≈114 steps (long approach motions), concept sample count 20,041.
- **Residual assets:** `data/residuals/back_vs_front_slices/`.
- **Latent steering entry:** `name: "back_vs_front"`.

## Using the Suites

- **Training / fine-tuning:** point `TrainConfig.data.repo_id` at the relevant LeRobot repo (`libero_open_close_lerobot` or `libero_open_close_front_back_lerobot`) and reuse the provided configs.
- **Residual collection:** feed the same config names to `openpi.analysis.save_residuals`; the metadata under each run (`metadata.json`) captures checkpoints, layers, and token types for provenance.
- **Latent steering:** keep `data/latent_steering/layer12_pairs.json` in sync whenever you add/remove suites so downstream scripts automatically discover them.
- **Visualization:** cosine plots already live under `data/latent_steering(_finetuned)/layer12/*.png` for quick sanity checks of each direction.

## Extending with New Suites

1. Duplicate the manifest + concept-map commands with your new task texts.
2. Capture fresh residuals (change `--config-name`, `--checkpoint-path`, and `--output-dir`).
3. Add the resulting slices to `data/latent_steering/layer12_pairs.json`.
4. Update this document’s catalog so future experiments know where to find the assets.

Because every artifact lives under `data/` with consistent naming, scripts such as `src/openpi/scripts/compute_latent_steering.py` and `src/openpi/scripts/slice_residuals_by_concept.py` can be reused without modification for any suite that follows the pattern above.


