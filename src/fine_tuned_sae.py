import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sae_lens import SAE
from pathlib import Path
import numpy as np
import random
import os


###############################################################################
# 0. Normalization Helpers
###############################################################################

def normalize_batch(batch, mode, d_model=2048, eps=1e-6):
    if mode == "none":
        return batch

    if mode == "expected_avg_only_in":
        # Anthropic April Update style normalization
        mean_abs = batch.abs().mean(dim=-1, keepdim=True).clamp(min=eps)
        scale = (d_model ** 0.5) / mean_abs
        return batch * scale

    if mode == "layernorm_in":
        mean = batch.mean(dim=-1, keepdim=True)
        var = batch.var(dim=-1, unbiased=False, keepdim=True)
        return (batch - mean) / torch.sqrt(var + eps)

    raise ValueError(f"Unknown normalization mode: {mode}")


###############################################################################
# 1. Dataset Loader
###############################################################################

class ResidualDataset(Dataset):
    def __init__(self, residual_dir, layer, max_samples=20000, dtype=torch.float32):
        files = sorted((residual_dir/"instructions").glob(f"layer{layer:02d}_chunk*.pt"))
        if not files:
            raise ValueError("No residuals found.")

        self.data = []
        for f in files:
            x = torch.load(f, map_location="cpu")
            for row in x:
                self.data.append(row)

        random.shuffle(self.data)
        self.data = self.data[:max_samples]

        self.data = torch.stack(self.data).to(dtype)
        print(f"Loaded {len(self.data)} residuals")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


###############################################################################
# 2. Train/Eval Loader Factory
###############################################################################

def create_loaders(residual_dir, layer, batch_size, train_frac=0.8):
    ds = ResidualDataset(residual_dir, layer)

    N = len(ds)
    split = int(N * train_frac)

    train_ds = torch.utils.data.Subset(ds, list(range(split)))
    eval_ds  = torch.utils.data.Subset(ds, list(range(split, N)))

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    )


###############################################################################
# 3. Training Step
###############################################################################

def train_one_epoch(sae, loader, optimizer, lambda_l1, norm_mode, device):
    sae.train()
    total_mse = 0.0
    total_l1  = 0.0
    count = 0

    for batch in loader:
        batch = batch.to(device)

        # apply chosen normalization
        batch = normalize_batch(batch, norm_mode)

        optimizer.zero_grad()
        latent = sae.encode(batch)
        recon = sae.decode(latent)

        mse_loss = ((batch - recon)**2).mean()
        l1_loss = lambda_l1 * latent.abs().mean()

        loss = mse_loss + l1_loss
        loss.backward()
        optimizer.step()

        total_mse += mse_loss.item() * len(batch)
        total_l1  += l1_loss.item() * len(batch)
        count += len(batch)

    return total_mse/count, total_l1/count


###############################################################################
# 4. Evaluation
###############################################################################

def eval_metrics(sae, loader, norm_mode, device):
    sae.eval()
    mses = []
    L0s = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch = normalize_batch(batch, norm_mode)

            latent = sae.encode(batch)
            recon  = sae.decode(latent)

            mse = ((batch - recon)**2).mean(dim=-1)
            mses.extend(mse.cpu().tolist())

            L0 = (latent.abs() > 1e-6).sum(dim=-1)
            L0s.extend(L0.cpu().tolist())

    mses = np.array(mses)
    L0s  = np.array(L0s)

    return {
        "eval_mse_mean": float(mses.mean()),
        "eval_mse_std":  float(mses.std()),
        "L0_mean":       float(L0s.mean())
    }


###############################################################################
# 5. Full Hyperparameter Sweep
###############################################################################

def full_hparam_sweep(residual_dir, layer, device="cuda"):
    lambdas = [1e-5, 3e-5, 1e-4]
    lrs     = [3e-5, 1e-4, 3e-4]
    batches = [128, 256, 512]
    norms   = ["none", "expected_avg_only_in", "layernorm_in"]

    results = {}

    for norm_mode in norms:
        for bs in batches:
            train_loader, eval_loader = create_loaders(residual_dir, layer, batch_size=bs)

            for lr in lrs:
                for lam in lambdas:

                    print(f"\n=== norm={norm_mode}, bs={bs}, lr={lr}, λ={lam} ===")

                    # load pretrained SAE
                    sae = SAE.from_pretrained(
                        release="gemma-2b-it-res-jb",
                        sae_id=f"blocks.{layer}.hook_resid_post",
                        device=device,
                    )

                    optimizer = optim.Adam(sae.parameters(), lr=lr)

                    # train a few epochs
                    for epoch in range(2):
                        mse, l1 = train_one_epoch(sae, train_loader, optimizer, lam, norm_mode, device)
                        print(f"Epoch {epoch}: mse={mse:.5f}, l1={l1:.5f}")

                    metrics = eval_metrics(sae, eval_loader, norm_mode, device)

                    key = (norm_mode, bs, lr, lam)
                    results[key] = metrics

                    # persist
                    os.makedirs("sae_hparam_logs", exist_ok=True)
                    torch.save(sae.state_dict(), f"sae_hparam_logs/sae_{norm_mode}_bs{bs}_lr{lr}_lam{lam}.pt")

                    with open(f"sae_hparam_logs/metrics_{norm_mode}_bs{bs}_lr{lr}_lam{lam}.txt", "w") as f:
                        f.write(str(metrics))

    return results


###############################################################################
# 6. Run Sweep
###############################################################################

if __name__ == "__main__":
    residual_dir = Path("data/residuals/pi0_libero/libero90/20251204-232330")
    layer = 12
    device = "cuda" if torch.cuda.is_available() else "cpu"

    res = full_hparam_sweep(residual_dir, layer, device)
    print("\n=== FINAL RESULTS ===")
    for k, v in res.items():
        print(k, v)
