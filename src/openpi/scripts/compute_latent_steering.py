#!/usr/bin/env python3
"""Compute SAE-based latent steering vectors for concept pairs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import tyro  # type: ignore[import-not-found]
from sae_lens import SAE  # type: ignore[import-not-found]


@dataclass
class Args:
    """CLI arguments."""

    pairs_config: Path
    output_root: Path = Path("data/latent_steering")
    layer: int = 12
    sae_release: str = "gemma-2b-it-res-jb"
    sae_id: str | None = None
    finetuned_sae_path: Path | None = None
    device: str = "cuda"
    batch_size: int = 512
    save_pairwise_stats: bool = True


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError(f"{path} must contain a non-empty 'pairs' list")
    required = {"name", "positive_label", "positive_path", "negative_label", "negative_path"}
    for entry in pairs:
        missing = required - entry.keys()
        if missing:
            raise ValueError(f"Pair entry missing fields {missing}: {entry}")
    return pairs


def _encode_residuals(
    sae: SAE,
    tensor: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if tensor.dim() == 3:
        tensor = tensor.squeeze(1)
    tensor = tensor.to(torch.float32)
    feats: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, tensor.shape[0], batch_size):
            chunk = tensor[start : start + batch_size].to(device)
            feats.append(sae.encode(chunk).cpu())
    return torch.cat(feats, dim=0)


def _decode_direction(sae: SAE, direction: torch.Tensor, *, device: torch.device) -> torch.Tensor:
    with torch.no_grad():
        resid = sae.decode(direction.to(device))
    return resid.cpu()


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0), dim=-1
    ).item()


def main(args: Args) -> None:
    sae_id = args.sae_id or f"blocks.{args.layer}.hook_resid_post"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    sae = SAE.from_pretrained(release=args.sae_release, sae_id=sae_id, device="cpu")
    if args.finetuned_sae_path is not None:
        state = torch.load(args.finetuned_sae_path, map_location="cpu")
        if isinstance(state, dict) and {"W_dec", "W_enc", "b_dec", "b_enc"} <= state.keys():
            sae.load_state_dict(state, strict=False)
        elif isinstance(state, dict) and "state_dict" in state:
            sae.load_state_dict(state["state_dict"], strict=False)
        else:
            raise ValueError(f"Unrecognized SAE checkpoint format at {args.finetuned_sae_path}")
    sae = sae.to(device).eval()

    pairs = _load_pairs(args.pairs_config)
    layer_dir = args.output_root / f"layer{args.layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)

    latent_vectors: dict[str, torch.Tensor] = {}
    residual_vectors: dict[str, torch.Tensor] = {}

    for entry in pairs:
        pair_name = entry["name"]
        pair_dir = layer_dir / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        concept_means: dict[str, torch.Tensor] = {}
        concept_counts: dict[str, int] = {}

        for label_key in ("positive", "negative"):
            label = entry[f"{label_key}_label"]
            path = Path(entry[f"{label_key}_path"])
            tensor = torch.load(path, map_location="cpu")
            feats = _encode_residuals(sae, tensor, device=device, batch_size=args.batch_size)
            mean = feats.mean(dim=0)
            concept_means[label_key] = mean
            concept_counts[label_key] = feats.shape[0]
            torch.save(
                {"concept": label, "count": feats.shape[0], "mean": mean},
                pair_dir / f"{label_key}_latent_mean.pt",
            )

        direction_latent = concept_means["positive"] - concept_means["negative"]
        direction_resid = _decode_direction(sae, direction_latent, device=device)

        latent_vectors[pair_name] = direction_latent.flatten()
        residual_vectors[pair_name] = direction_resid.flatten()

        torch.save(
            {
                "pair": pair_name,
                "positive_concept": entry["positive_label"],
                "negative_concept": entry["negative_label"],
                "latent_direction": direction_latent,
                "latent_norm": direction_latent.norm().item(),
                "counts": concept_counts,
            },
            pair_dir / "latent_direction.pt",
        )
        torch.save(
            {
                "pair": pair_name,
                "positive_concept": entry["positive_label"],
                "negative_concept": entry["negative_label"],
                "residual_direction": direction_resid,
                "residual_norm": direction_resid.norm().item(),
                "counts": concept_counts,
            },
            pair_dir / "residual_direction.pt",
        )

    if args.save_pairwise_stats and len(latent_vectors) >= 2:
        stats = {}
        names = list(latent_vectors.keys())
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                stats[f"{a}__{b}"] = {
                    "latent_cosine": _cosine(latent_vectors[a], latent_vectors[b]),
                    "residual_cosine": _cosine(residual_vectors[a], residual_vectors[b]),
                    "latent_norm_a": latent_vectors[a].norm().item(),
                    "latent_norm_b": latent_vectors[b].norm().item(),
                    "residual_norm_a": residual_vectors[a].norm().item(),
                    "residual_norm_b": residual_vectors[b].norm().item(),
                }
        torch.save(stats, layer_dir / "pairwise_stats.pt")


if __name__ == "__main__":
    tyro.cli(main)

