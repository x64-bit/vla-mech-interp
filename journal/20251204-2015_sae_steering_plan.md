# SAE Finetuning & Steering Plan

## Phase 1 – Prep Residual Dataset

1. **dataset-prep** – Enumerate the new LIBERO-90 residual shards under [`data/residuals/pi0_libero/libero90`](data/residuals/pi0_libero/libero90), split into training/validation, and record metadata (layer, token type, sample counts).
2. **conversion-script** – Add a small loader (e.g., in [`src/openpi/src/openpi/analysis/save_residuals.py`](src/openpi/src/openpi/analysis/save_residuals.py) or a new helper) that streams residual tensors into numpy/torch arrays compatible with `sae_lens.ActivationsStore` (float32, contiguous batches, optional downsampling).

## Phase 2 – Finetune Gemma Layer-14 SAE

3. **sae-load** – Use `sae_lens.SAE.from_pretrained("gemma-2b-it-res-jb", "blocks.12.hook_resid_post")` as initialization (per the Hugging Face card) and adapt hidden size to pi0.5 layer-14 (2048). Save config overrides for dimension or sparsity changes.
4. **sae-train** – Implement a training loop (new script, e.g., [`src/openpi/scripts/finetune_sae.py`](src/openpi/scripts/finetune_sae.py)) that:
   - streams residual mini-batches from Phase 1 loader,
   - applies SAE forward/reconstruction loss, L1 sparsity, dead-feature refresh,
   - checkpoints weights to `data/sae/pi05_layer14/` and logs metrics (loss, L0, dead features).
5. **sae-validate** – Evaluate reconstruction on held-out shards plus basic interpretability probes (activation histograms, cosine similarity vs. original HF SAE) to ensure the finetuned latent space is stable.

## Phase 3 – Contrastive Steering Experiments

6. **label-events** – Reuse LIBERO-90 annotations to tag timesteps as push/pull or open/grasp and map them onto decision-token residuals (instructions vs. actions). Persist paired residual IDs so Haon clusters and SAE latents consume identical contrastive splits.
7. **latent-vectors** – Build a contrastive pipeline patterned after *Controlling VLA Policies through Sparse Latent Directions*: (a) encode each class subset with the finetuned SAE to collect sparse activations; (b) compute class prototypes (mean activation per latent) and optionally fit a logistic head in latent space; (c) form steering vectors as normalized prototype differences, optionally masking to the top-k features by activation magnitude; (d) decode each latent vector back into residual space and cache both latent and decoded forms for later injections.
8. **steer-test** – Extend the inference harness so that during `model.sample_actions` we can (a) hook layer-14 residuals, (b) inject the decoded steering vector at the desired timestep with a sweep over scaling coefficients, (c) log whether the resulting action flips push↔pull or open↔grasp, (d) compare against Haon’s clustering-based edits under identical conditions, and (e) record qualitative traces (logits, token projections, video snippets) for downstream analysis.

## Phase 4 – Token-Space Analysis

9. **latent-to-token** – For each steering vector, decode to residual space via SAE decoder, run through the appropriate RMSNorm + LM head (Paligemma for instructions, Gemma expert + AdaRMS for decisions) to obtain token logits.
10. **cosine-metrics** – Compute cosine similarity matrices between SAE-derived directions, Haon’s cluster centroids, and Khan et al.’s steering vectors in token space; visualize overlaps and disentanglement (heatmaps, top-token tables) and summarize in the journal.
