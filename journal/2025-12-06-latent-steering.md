## 2025-12-06 – Latent Steering Pipeline Build

We finalized an end-to-end workflow for extracting concept-specific instruction residuals from π0.5 LIBERO demos, projecting them through the Gemma-2B SAE, and deriving contrastive “steering” vectors. Key milestones:

1. **Dataset filtering + conversion** – Copied four LIBERO-90 demos (open/close drawer, turn on/off stove) into `data/libero_filtered/open_close` and converted them into a dedicated LeRobot repo at `data/libero_lerobot/libero_open_close_lerobot`.
2. **Contrast manifest + mapping** – Used `build_contrasting_pairs.py --backend lerobot` to create manifests for each pair, then `map_concept_samples.py` to record the dataset row spans per concept.
3. **Residual capture** – Added `pi05_libero_open_close` config, captured layer-12 instruction residuals with `save_residuals.py` (pi05 weights), and stored them in `data/residuals/open_close_run/instructions/layer12_chunk*.pt`.
4. **Concept slicing** – Ran `slice_residuals_by_concept.py` twice (open/close and on/off) to produce per-concept tensors under `data/residuals/*_slices/layer12/`.
5. **SAE latent directions** – Wrote `compute_latent_steering.py`, plus `data/latent_steering/layer12_pairs.json`, to automate SAE encoding, latent means, contrast directions, decoded residuals, and cosine metrics. The script now reproduces the earlier manual calculations.




