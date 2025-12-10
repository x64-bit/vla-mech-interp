#!/usr/bin/env python3
"""Plot cosine similarity between two steering vectors (latent + residual)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import tyro  # type: ignore[import-not-found]


@dataclass
class SteeringVectorSpec:
    """Descriptor for a saved steering direction."""

    label: str
    latent_path: Path
    residual_path: Path | None = None


@dataclass
class Args:
    """CLI arguments for steering cosine plots."""

    vector_a: SteeringVectorSpec
    vector_b: SteeringVectorSpec
    output_path: Path = Path("data/analysis/steering_cosine.png")
    include_flipped: bool = False
    plot_residual: bool = True


def _load_vector(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu")
    direction = payload.get("latent_direction")
    if direction is None:
        direction = payload.get("residual_direction")
    if direction is None:
        raise ValueError(f"{path} missing 'latent_direction'/'residual_direction'")
    return direction.flatten().to(torch.float32)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    value = torch.nn.functional.cosine_similarity(
        a.unsqueeze(0), b.unsqueeze(0), dim=-1
    ).item()
    return abs(value)


def _norm(vec: torch.Tensor) -> float:
    return torch.linalg.vector_norm(vec).item()


def main(args: Args) -> None:
    lat_a = _load_vector(args.vector_a.latent_path)
    lat_b = _load_vector(args.vector_b.latent_path)
    lat_cos = _cosine(lat_a, lat_b)
    latent_values = [("latent", lat_cos)]

    resid_values: list[tuple[str, float]] = []
    if args.plot_residual and args.vector_a.residual_path and args.vector_b.residual_path:
        res_a = _load_vector(args.vector_a.residual_path)
        res_b = _load_vector(args.vector_b.residual_path)
        resid_values.append(("residual", _cosine(res_a, res_b)))

    bars = [("latent", latent_values[0][1])]
    if resid_values:
        bars.append(("residual", resid_values[0][1]))

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.bar(
        [label for label, _ in bars],
        [value for _, value in bars],
        color=["#1f77b4", "#2ca02c"][: len(bars)],
        alpha=0.9,
    )
    ax.set_ylim(0, 1)
    ax.set_ylabel("absolute cosine similarity")
    ax.set_title(f"{args.vector_a.label} vs {args.vector_b.label}")
    fig.tight_layout()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=200)
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    tyro.cli(main)

