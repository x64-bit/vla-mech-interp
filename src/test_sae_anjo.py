from openpi.training import config as _config
from openpi.policies import policy_config
from pathlib import Path
import math
import torch
from torch.utils.data import Dataset, DataLoader
from sae_lens import SAE
import numpy as np
import os

torch.set_grad_enabled(False)

################################################################################
# 1. Load π0.5 Policy + Gemma Language Model
################################################################################

config = _config.get_config("pi05_libero")
checkpoint_dir = "data/openpi-assets/checkpoints/pi05_libero_pytorch"

policy = policy_config.create_trained_policy(config, checkpoint_dir)
model = policy._model
model.eval()

vlm = model.paligemma_with_expert.paligemma.language_model
device = policy._pytorch_device

print(f"Model device: {device}")
print("Gemma layers:", len(vlm.layers))


################################################################################
# 2. Load Pretrained SAE for Layer L
################################################################################

LAYER = 12
SAE_RELEASE = "gemma-2b-it-res-jb"
SAE_ID = f"blocks.{LAYER}.hook_resid_post"

print(f"\nLoading SAE: {SAE_RELEASE} / {SAE_ID}")

sae = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device=device,
)

sae = sae.to(device).eval()

# d_sae is number of features → W_dec is [d_sae, d_model]
feature_dim = sae.W_dec.shape[0]
d_model = sae.W_dec.shape[1]
sqrt_d = math.sqrt(d_model)
print("Loaded SAE. Num features:", feature_dim)
print(f"d_model: {d_model} | sqrt(d) scale: {sqrt_d:.4f}")

APPLY_SQRT_D_SCALE = True
residual_scale = sqrt_d if APPLY_SQRT_D_SCALE else None
# residual_scale = .00001 if APPLY_SQRT_D_SCALE else None
if residual_scale is not None:
    print(f"Applying sqrt(d) scaling factor {residual_scale:.4f} to residual batches")


################################################################################
# 3. Dataset that lazily loads residual shards from disk
################################################################################

class ResidualDataset(Dataset):
    def __init__(
        self,
        residuals_dir: Path,
        layer: int,
        dtype=torch.bfloat16,
        scale_factor: float | None = None,
    ):
        self.dir = residuals_dir / "instructions"
        self.layer = layer
        self.dtype = dtype
        self.scale_factor = (
            None if scale_factor is None else torch.tensor(scale_factor, dtype=dtype)
        )

        pattern = f"layer{layer:02d}_chunk*.pt"
        self.files = sorted(self.dir.glob(pattern))
        if not self.files:
            raise ValueError(f"No residuals found for: {pattern}")

        print("\nFound chunks:")
        for x in self.files:
            print(" -", x)

        self.cumulative = [0]
        for f in self.files:
            chunk = torch.load(f, map_location="cpu")
            self.cumulative.append(self.cumulative[-1] + chunk.shape[0])

        self.total = self.cumulative[-1]
        print(f"Total residual vectors: {self.total}")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.cumulative, idx, side="right") - 1
        local_idx = idx - self.cumulative[file_idx]

        chunk = torch.load(self.files[file_idx], map_location="cpu")
        vec = chunk[local_idx].to(self.dtype)

        if self.scale_factor is not None:
            vec = vec * self.scale_factor

        return vec


def create_loader(residuals_dir, layer, bs=32, scale_factor=None):
    ds = ResidualDataset(residuals_dir, layer, scale_factor=scale_factor)
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    return loader, ds


################################################################################
# 4. Create loader
################################################################################

residuals_dir = Path("data/residuals/pi0_libero/libero90/20251204-232330")

loader, ds = create_loader(residuals_dir, LAYER, bs=32, scale_factor=residual_scale)
print(f"Loader ready. Dataset size: {len(ds)}")


################################################################################
# 5. Run SAE on dataset and collect diagnostics
################################################################################

all_feats = []
all_mse = []

print("\nRunning SAE on ALL residuals...")

with torch.no_grad():
    for i, batch in enumerate(loader):
        batch = batch.to(device)

        # SAE v6 API
        feats = sae.encode(batch)    # [B, d_sae]
        recon = sae.decode(feats)    # [B, d_model]

        # mse = torch.mean((batch - recon)**2, dim=-1)  # [B]

        # UN-SCALE for valid metrics
        # We divide by the scale factor to get back to "Real World" units
        if residual_scale is not None:
            real_batch = batch / residual_scale
            real_recon = recon / residual_scale
        else:
            real_batch = batch
            real_recon = recon

        # Now calculate MSE in the original unit space
        mse = torch.mean((real_batch - real_recon)**2, dim=-1)

        # Move to CPU (float32 for safe concatenation)
        all_feats.append(feats.float().cpu())
        all_mse.append(mse.float().cpu())

        if i % 50 == 0:
            print(f"Batch {i}/{len(loader)} | feats={feats.shape} mse={mse.mean().item():.6f}")
            # Inside your loop
            # 1. Calculate the variance of the original data (the "scale" of the signal)
            #    We sum variance across the feature dimension (d_model)
            total_variance = torch.var(real_batch, dim=0).sum()

            # 2. Calculate the Squared Error (MSE * d_model essentially)
            #    We sum the squared error across the feature dimension
            squared_error = torch.sum((real_batch - real_recon)**2, dim=-1).mean()

            # 3. Calculate FVU (Metric independent of scale)
            fvu = squared_error / total_variance

            print(f"MSE (Raw): {mse.mean():.4f} | Data Variance: {total_variance:.4f} | FVU: {fvu:.4f}")

print("Done running SAE.")


################################################################################
# 6. Save outputs
################################################################################

all_feats = torch.cat(all_feats, dim=0)
all_mse = torch.cat(all_mse, dim=0)

os.makedirs("sae_outputs_anjo", exist_ok=True)
torch.save(all_feats, f"sae_outputs_anjo/layer{LAYER}_latent_acts.pt")
torch.save(all_mse, f"sae_outputs_anjo/layer{LAYER}_recon_mse.pt")

print(f"\nSaved SAE outputs for layer {LAYER}:"
      f"\n - Latents: sae_outputs_anjo/layer{LAYER}_latent_acts.pt"
      f"\n - MSEs:   sae_outputs_anjo/layer{LAYER}_recon_mse.pt")
