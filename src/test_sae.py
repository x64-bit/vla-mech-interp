from openpi.training import config as _config
from openpi.policies import policy_config
from pathlib import Path
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
print("Loaded SAE. Num features:", feature_dim)


################################################################################
# 3. Dataset that lazily loads residual shards from disk
################################################################################

class ResidualDataset(Dataset):
    def __init__(self, residuals_dir: Path, layer: int, dtype=torch.bfloat16):
        self.dir = residuals_dir / "instructions"
        self.layer = layer
        self.dtype = dtype

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
        chunk = chunk.to(self.dtype)

        return chunk[local_idx]


def create_loader(residuals_dir, layer, bs=32):
    ds = ResidualDataset(residuals_dir, layer)
    loader = DataLoader(ds, batch_size=bs, shuffle=False)
    return loader, ds


################################################################################
# 4. Create loader
################################################################################

residuals_dir = Path("data/residuals/pi0_libero/libero90/20251204-232330")

loader, ds = create_loader(residuals_dir, LAYER, bs=32)
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

        mse = torch.mean((batch - recon)**2, dim=-1)  # [B]

        # Move to CPU (float32 for safe concatenation)
        all_feats.append(feats.float().cpu())
        all_mse.append(mse.float().cpu())

        if i % 50 == 0:
            print(f"Batch {i}/{len(loader)} | feats={feats.shape} mse={mse.mean().item():.6f}")

print("Done running SAE.")


################################################################################
# 6. Save outputs
################################################################################

all_feats = torch.cat(all_feats, dim=0)
all_mse = torch.cat(all_mse, dim=0)

os.makedirs("sae_outputs", exist_ok=True)
torch.save(all_feats, f"sae_outputs/layer{LAYER}_latent_acts.pt")
torch.save(all_mse, f"sae_outputs/layer{LAYER}_recon_mse.pt")

print(f"\nSaved SAE outputs for layer {LAYER}:"
      f"\n - Latents: sae_outputs/layer{LAYER}_latent_acts.pt"
      f"\n - MSEs:   sae_outputs/layer{LAYER}_recon_mse.pt")
