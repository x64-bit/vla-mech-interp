## CS182

## Residual Capture and Export

This guide explains how to tap the Gemma residual stream inside pi0, persist the activations under `data/residuals/`, and use them later for SAE steering experiments. The workflow is motivated by recent activation-steering studies in VLAs ([Häon et al. 2025](file://Mechanistic interpretability for steering vision-language-action models.pdf); [Khan et al. 2025](file://SAE on VLA.pdf)) which compare token-space clustering against SAE latents.

### 1. Hooking Gemma layers

`openpi.analysis.residual_capture.ResidualRecorder` attaches PyTorch forward hooks directly to `model.paligemma_with_expert.gemma_expert.model.layers`. Each hook:

1. Detaches the decoder output tensor (the residual stream after `_gated_residual`).
2. Optionally slices specific tokens via a `TokenSelector`, for example:
   - `select_action_token(action_horizon, offset=-1)` to keep the final decision token.
   - `select_last_tokens(k)` to keep the tail of the sequence for Haon-style comparisons.
3. Casts/CPU-moves tensors for safe serialization.

Use the context manager to scope the hooks:

```python
from openpi.analysis import residual_capture

cfg = residual_capture.ResidualCaptureConfig(
    layers=[12, 14, 17],
    token_selector=residual_capture.select_action_token(model.config.action_horizon, offset=-1),
)

with residual_capture.ResidualRecorder(model, cfg) as recorder:
    model.sample_actions(device, observation)
    activations = recorder.flush()   # {layer_idx: torch.Tensor[num_tokens, hidden_dim]}
```

`flush()` returns the latest batch of activations and clears the buffers, which keeps memory bounded when iterating over large datasets.

### 2. Streaming activations to `data/residuals`

`openpi.analysis.save_residuals` is a CLI utility that reproduces the Libero preprocessing pipeline, runs a pi0 PyTorch checkpoint over the dataset, and writes chunked activation tensors to disk. Key responsibilities:

- Loads a train config (default `pi0_libero`) to reuse the Libero transforms and normalization stats.
- Instruments the specified Gemma layers via `ResidualRecorder`.
- Streams batches, respecting optional limits (max batches, max samples per layer).
- For each layer, splits and stores two corpora by default:
  - `decision/`: the final action (“decision”) token per time step.
  - `instructions/`: every language token in the prompt (useful for semantics-focused probes).
- When the checkpoint is pi05 (AdaRMS enabled), also logs the per-step conditioning vectors under `adarms/` so logits can be reconstructed later without re-running the diffusion loop.
- Writes `.pt` shards plus a `metadata.json` manifest describing sample counts, file lists, and collection settings.

Example invocation:

```bash
uv run -m openpi.analysis.save_residuals \
  --checkpoint-path data/openpi-assets/checkpoints/pi05_libero_pytorch/model.safetensors \
  --output-dir data/residuals/pi0_libero/run01 \
  --config-name pi0_libero \
  --lerobot-home data/libero_lerobot \
  --assets-dir data/openpi-assets/checkpoints/pi05_libero/assets \
  --layers 12 14 \
  --max-samples-per-layer 20000 \
  --samples-per-file 8192 \
  --capture-instructions true
```

Flags of interest:

| Flag | Description |
| --- | --- |
| `--layers` | Gemma decoder layers (0‑indexed). |
| `--capture-instructions/--no-capture-instructions` | Toggle whether prompt tokens are exported (defaults to on). |
| `--max-samples-per-layer` | Stops recording once the layer reaches this many tokens. Set `None` to stream the entire dataset. |
| `--samples-per-file` | Controls shard size; useful when training SAEs on multiple GPUs. |
| `--skip-norm-stats` | Bypass Libero normalization assets (only if the dataset is already normalized). |

Output layout:

```
data/residuals/pi0_libero/run01/
  decision/
    layer12_chunk0000.pt   # decision tokens, shape [N, hidden_dim]
    layer14_chunk0000.pt
  instructions/
    layer12_chunk0000.pt   # prompt tokens, concatenated across the batch
    layer14_chunk0000.pt
  adarms/
    chunk0000.pt           # AdaRMS conditioning rows aligned with decision samples
  metadata.json           # manifest with sample counts per split, etc.
```

The metadata includes the config name, checkpoint path, layers, action horizon, per-split sample counts, and every file written (including AdaRMS files when present). This makes SAE training reproducible without re-running the expensive pi0 forward pass and preserves everything needed to decode residuals back into logits.

### 3. Training downstream SAEs or steering vectors

Once residual shards exist:

```python
import torch

tensor = torch.load("data/residuals/pi0_libero/run01/layer14_chunk0001.pt")
# tensor.shape == [num_samples, hidden_dim]
```

Use the tensors as your SAE training data (e.g., finetune the Gemma-2B SAE with an orthogonality penalty) or to compare residual-space directions against Haon et al.’s token projections. The chunked format makes it easy to stream samples while keeping RAM usage small.

### 4. Tips & troubleshooting

- **Normalization assets:** if the script can’t find Libero norm stats automatically, pass `--assets-dir` pointing to a directory that contains `<asset_id>/norm_stats.json`.
- **HF_LEROBOT_HOME:** set `--lerobot-home` so the LeRobot dataset loader resolves the right cache (defaults to whatever is configured in your environment).
- **Device selection:** `--device cuda:0` overrides automatic detection; gradients are disabled so 8–12 GB GPUs are usually sufficient.
- **Token sanity check:** inspect a small chunk by decoding tokens with `ffn_probe.py` or comparing activations for obvious contrastive behaviors (push/pull drawer, slow/fast, etc.) before launching SAE fine-tuning.

With these utilities, you can capture a single set of residuals from pi0-LIBERO, store them under `data/residuals`, and iterate on SAE or token-space steering experiments without re-running Libero rollouts. 

