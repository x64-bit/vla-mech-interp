import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sae_lens import SAE


###############################################################################
# 1. Chunked Residual Dataset (no 10M-length Python list)
###############################################################################

class ResidualDataset(Dataset):
    """
    Loads all residual chunks into RAM as tensors and indexes them via
    a cumulative length array. You get O(log N) indexing and avoid
    a 10M-element Python list.
    """

    def __init__(self, residual_dir: Path, layer: int, dtype=torch.float32):
        files = sorted((residual_dir / "instructions").glob(f"layer{layer:02d}_chunk*.pt"))
        if not files:
            raise ValueError(f"No residuals found for layer {layer} in {residual_dir}")

        self.chunks = []
        self.cum_lengths = []
        total = 0

        for f in files:
            x = torch.load(f, map_location="cpu").to(dtype)   # [chunk_size, d_model]
            self.chunks.append(x)
            total += x.shape[0]
            self.cum_lengths.append(total)

        self.total = total
        print(f"[INFO] Loaded {len(self.chunks)} chunks, total {self.total} residuals.")

    def __len__(self):
        return self.total

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Binary search over cumulative lengths to find which chunk this index falls in
        lo, hi = 0, len(self.cum_lengths) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if idx < self.cum_lengths[mid]:
                hi = mid
            else:
                lo = mid + 1

        chunk_idx = lo
        prev_cum = 0 if chunk_idx == 0 else self.cum_lengths[chunk_idx - 1]
        row_idx = idx - prev_cum
        return self.chunks[chunk_idx][row_idx]


###############################################################################
# 2. Training Loop (encoder-only fine-tuning, AMP + big batch)
###############################################################################

def train_one_epoch(
    sae: SAE,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    lambda_l1: float,
    device: str,
    scaler: torch.amp.GradScaler,
    max_steps: int | None = None,
    grad_accum: int = 1,
):
    sae.train()
    total_mse = 0.0
    total_l1 = 0.0
    count = 0

    step = 0
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break

        batch = batch.to(device, non_blocking=True)

        # AMP with new API
        with torch.amp.autocast("cuda"):
            # SAE v6 API: use encode/decode, no return_latent arg
            latent = sae.encode(batch)
            recon = sae.decode(latent)

            mse_loss = ((batch - recon) ** 2).mean()
            l1_loss = lambda_l1 * latent.abs().mean()
            loss = (mse_loss + l1_loss) / grad_accum

        scaler.scale(loss).backward()

        # Gradient accumulation to simulate larger batches
        if (batch_idx + 1) % grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = batch.size(0)
        total_mse += mse_loss.item() * bs
        total_l1 += l1_loss.item() * bs
        count += bs
        step += 1

    mean_mse = total_mse / max(count, 1)
    mean_l1 = total_l1 / max(count, 1)
    return mean_mse, mean_l1


###############################################################################
# 3. Evaluation (on small subset so it doesn't dominate runtime)
###############################################################################

def eval_sae(
    sae: SAE,
    loader: DataLoader,
    device: str,
    max_batches: int = 200,
):
    sae.eval()
    mses = []
    L0s = []

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break

            batch = batch.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                latent = sae.encode(batch)
                recon = sae.decode(latent)
                mse = ((batch - recon) ** 2).mean(dim=-1)

            mses.extend(mse.cpu().tolist())
            L0s.extend((latent.abs() > 1e-6).sum(dim=-1).cpu().tolist())

    if not mses:
        return {"mse_mean": float("nan"), "mse_std": float("nan"), "L0_mean": float("nan")}

    mses = np.array(mses)
    L0s = np.array(L0s)
    return {
        "mse_mean": float(mses.mean()),
        "mse_std": float(mses.std()),
        "L0_mean": float(L0s.mean()),
    }


###############################################################################
# 4. Fine-Tune Driver
###############################################################################

def fine_tune_sae(residual_dir: Path, layer: int, device: str = "cuda"):
    print("DEVICE =", device, "| CUDA available =", torch.cuda.is_available())

    # From your sweep:
    LR = 1e-4
    LAMBDA = 3e-5

    # Aggressive but safe for A100-40GB; if you OOM, drop to 2048
    BATCH_SIZE = 4096

    # How many "logical epochs"
    EPOCHS = 12

    # Optional cap on steps per epoch; None = full pass over all 10M residuals
    MAX_STEPS_PER_EPOCH = None  # e.g. 5000 if you want to cap work per epoch

    # Gradient accumulation (effective batch = BATCH_SIZE * GRAD_ACCUM)
    GRAD_ACCUM = 1

    print("[INFO] Loading residual dataset (chunked)...")
    dataset = ResidualDataset(residual_dir, layer)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=12,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # Separate eval loader, no shuffle
    eval_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    print("[INFO] Loading pretrained Gemma SAE (from sae_lens)...")
    sae: SAE = SAE.from_pretrained(
        release="gemma-2b-it-res-jb",
        sae_id=f"blocks.{layer}.hook_resid_post",
        device="cpu",
    )
    sae = sae.to(device)
    print("[INFO] SAE moved to device:", next(sae.parameters()).device)

    # Freeze decoder + bias for stability and feature identity
    for name, param in sae.named_parameters():
        if "W_dec" in name or "b_dec" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    print("[INFO] Encoder-only fine-tuning (W_dec, b_dec frozen).")

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, sae.parameters()),
        lr=LR,
    )

    scaler = torch.amp.GradScaler("cuda")

    for epoch in range(EPOCHS):
        mse, l1 = train_one_epoch(
            sae=sae,
            loader=loader,
            optimizer=optimizer,
            lambda_l1=LAMBDA,
            device=device,
            scaler=scaler,
            max_steps=MAX_STEPS_PER_EPOCH,
            grad_accum=GRAD_ACCUM,
        )
        print(f"[Epoch {epoch}] train MSE={mse:.6f} | L1={l1:.6f}")

        # Light eval every few epochs (or every epoch if you want)
        if epoch % 3 == 0:
            metrics = eval_sae(sae, eval_loader, device=device, max_batches=200)
            print(
                f"  Eval → MSE={metrics['mse_mean']:.4f} "
                f"(±{metrics['mse_std']:.4f}) | L0={metrics['L0_mean']:.1f}"
            )

    # Save final fine-tuned SAE
    os.makedirs("sae_finetuned", exist_ok=True)
    save_path = f"sae_finetuned/sae_pi0_layer{layer:02d}_finetuned.pt"
    torch.save(sae.state_dict(), save_path)
    print(f"[DONE] Saved fine-tuned SAE to {save_path}")

    return sae


###############################################################################
# 5. Entry Point
###############################################################################

if __name__ == "__main__":
    residual_dir = Path("data/residuals/pi0_libero/libero90/20251204-232330")
    layer = 12
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fine_tune_sae(residual_dir, layer, device)
