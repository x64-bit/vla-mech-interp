#!/usr/bin/env python3
"""Compute cosine overlap between concept directions in residual and token space."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import tyro

from openpi.analysis import concept_similarity as cs


@dataclass
class Args:
    """CLI arguments for the concept similarity suite."""

    concept_config: Path
    haon_residual_dir: Path
    sae_decoded_dir: Path
    haon_clusters_path: Path
    checkpoint_path: Path
    output_path: Path = Path("analysis/concept_similarity.json")
    layer: int = 14
    config_name: str = "pi0_libero"
    residual_max_samples: int | None = None
    latent_max_samples: int | None = None
    reducer: str = "mean"
    normalize_vectors: bool = True
    vector_dump_dir: Path | None = None
    tensor_device: str = "cpu"
    projection_device: str = "cpu"
    model_device: str = "cpu"
    top_k_tokens: int = 10


def main(args: Args) -> None:
    tensor_device = torch.device(args.tensor_device)

    concept_specs = cs.load_concept_specs(args.concept_config)
    if len(concept_specs) < 2:
        raise ValueError("Provide at least two concepts to compare.")

    haon_vectors = _build_vectors(
        concept_specs,
        root=args.haon_residual_dir,
        layer=args.layer,
        file_getter=lambda spec: spec.residual_file,
        max_samples=args.residual_max_samples,
        reducer=args.reducer,
        normalize=args.normalize_vectors,
        device=tensor_device,
    )

    sae_vectors = _build_vectors(
        concept_specs,
        root=args.sae_decoded_dir,
        layer=args.layer,
        file_getter=lambda spec: spec.latent_file,
        max_samples=args.latent_max_samples,
        reducer=args.reducer,
        normalize=args.normalize_vectors,
        device=tensor_device,
    )

    residual_cos = {
        "haon": cs.compute_pairwise_cosines(haon_vectors),
        "sae": cs.compute_pairwise_cosines(sae_vectors),
    }

    clusters = cs.load_clusters(args.haon_clusters_path)
    haon_cluster_vectors = {
        spec.name: _maybe_normalize(
            torch.from_numpy(cs.select_cluster_centroid(clusters, spec)),
            args.normalize_vectors,
        )
        for spec in concept_specs
    }

    lm_head, token_embeddings = cs.load_model_components(
        args.checkpoint_path,
        config_name=args.config_name,
        device=args.model_device,
    )

    haon_token_vectors = cs.project_vectors_to_token_space(
        haon_cluster_vectors,
        lm_head=lm_head,
        token_embeddings=token_embeddings,
        top_k=args.top_k_tokens,
        device=args.projection_device,
    )
    sae_token_vectors = cs.project_vectors_to_token_space(
        sae_vectors,
        lm_head=lm_head,
        token_embeddings=token_embeddings,
        top_k=args.top_k_tokens,
        device=args.projection_device,
    )

    token_cos = {
        "haon": cs.compute_pairwise_cosines(haon_token_vectors),
        "sae": cs.compute_pairwise_cosines(sae_token_vectors),
    }

    per_concept = _summarize_concepts(
        concept_specs,
        haon_vectors,
        sae_vectors,
        haon_token_vectors,
        sae_token_vectors,
    )

    if args.vector_dump_dir is not None:
        _dump_vectors(
            args.vector_dump_dir,
            {
                "haon_residual": haon_vectors,
                "sae_residual": sae_vectors,
                "haon_token": haon_token_vectors,
                "sae_token": sae_token_vectors,
            },
        )

    payload: dict[str, Any] = {
        "layer": args.layer,
        "concepts": [spec.name for spec in concept_specs],
        "residual_space": residual_cos,
        "token_space": token_cos,
        "per_concept": per_concept,
        "metadata": {
            "concept_config": str(args.concept_config),
            "haon_residual_dir": str(args.haon_residual_dir),
            "sae_decoded_dir": str(args.sae_decoded_dir),
            "haon_clusters_path": str(args.haon_clusters_path),
            "checkpoint_path": str(args.checkpoint_path),
            "config_name": args.config_name,
            "reducer": args.reducer,
            "normalize_vectors": args.normalize_vectors,
            "top_k_tokens": args.top_k_tokens,
        },
    }

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(payload, indent=2))

    print(f"[concept-sim] wrote {args.output_path}")  # noqa: T201


def _build_vectors(
    specs: list[cs.ConceptSpec],
    *,
    root: Path,
    layer: int,
    file_getter: Callable[[cs.ConceptSpec], str],
    max_samples: int | None,
    reducer: str,
    normalize: bool,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    vectors: dict[str, torch.Tensor] = {}
    for spec in specs:
        matrix = cs.load_concept_matrix(
            root=root,
            layer=layer,
            file_stem=file_getter(spec),
            max_samples=max_samples,
            device=device,
        )
        vector = cs.reduce_samples(matrix, reducer=reducer, normalize=normalize)
        vectors[spec.name] = vector.detach().cpu()
    return vectors


def _maybe_normalize(vector: torch.Tensor, normalize: bool) -> torch.Tensor:
    if not normalize:
        return vector
    return torch.nn.functional.normalize(vector, dim=0)


def _summarize_concepts(
    specs: list[cs.ConceptSpec],
    haon_vectors: dict[str, torch.Tensor],
    sae_vectors: dict[str, torch.Tensor],
    haon_token_vectors: dict[str, torch.Tensor],
    sae_token_vectors: dict[str, torch.Tensor],
) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for spec in specs:
        name = spec.name
        summary[name] = {
            "haon_residual_norm": float(torch.linalg.vector_norm(haon_vectors[name])),
            "sae_residual_norm": float(torch.linalg.vector_norm(sae_vectors[name])),
            "haon_token_norm": float(torch.linalg.vector_norm(haon_token_vectors[name])),
            "sae_token_norm": float(torch.linalg.vector_norm(sae_token_vectors[name])),
        }
    return summary


def _dump_vectors(base_dir: Path, collections: dict[str, dict[str, torch.Tensor]]) -> None:
    for label, vectors in collections.items():
        target_dir = base_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        for concept, vector in vectors.items():
            torch.save(vector.cpu(), target_dir / f"{concept}.pt")


if __name__ == "__main__":
    main(tyro.cli(Args))


