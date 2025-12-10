#!/usr/bin/env python3
"""Simulate Haon-style cluster activation selection constrained to a specific layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tyro

from openpi.analysis import (
    ClusterResult,
    SteeringDirection,
    average_value_vectors,
    index_projections,
    load_clusters,
    load_projections,
    rank_clusters_by_keywords,
    save_steering_config,
    select_cluster_records,
    summarize_tokens,
)


@dataclass
class Args:
    """CLI arguments for simulating activation selection from cluster centroids."""

    projections_path: Path
    clusters_path: Path
    layer_index: int = 12
    cluster_id: int | None = None
    keywords: Sequence[str] | None = None
    top_members: int = 6
    normalize_vector: bool = True
    vector_output: Path | None = None
    steering_output: Path | None = None
    steering_gain: float = 1.0


def main(args: Args) -> None:
    if args.cluster_id is None and not args.keywords:
        raise ValueError("Provide either --cluster-id or --keywords to pick a cluster.")

    records = load_projections(args.projections_path)
    projection_lookup = index_projections(records)
    clusters = load_clusters(args.clusters_path)
    cluster = _select_cluster(clusters, cluster_id=args.cluster_id, keywords=args.keywords)

    layer_records = select_cluster_records(
        cluster,
        projection_lookup=projection_lookup,
        layer_index=args.layer_index,
    )
    if not layer_records:
        raise ValueError(
            f"Cluster {cluster.cluster_id} has no members at layer {args.layer_index}. "
            "Re-run clustering with layer-specific inputs or choose a different cluster."
        )

    if args.top_members and args.top_members > 0:
        selected_records = layer_records[: args.top_members]
    else:
        selected_records = layer_records

    vector = average_value_vectors(selected_records, normalize=args.normalize_vector)
    token_summary = summarize_tokens(selected_records, top_k=10)

    print(f"[simulate] cluster={cluster.cluster_id} layer={args.layer_index}")  # noqa: T201
    print(f"[simulate] members_considered={len(layer_records)} selected={len(selected_records)}")  # noqa: T201
    print("[simulate] aggregated_tokens=" + ", ".join(f"{tok}:{score:.2f}" for tok, score in token_summary))  # noqa: T201

    for record in selected_records:
        token_preview = ", ".join(token for token, _ in record.top_tokens[:5])
        print(  # noqa: T201
            f"  - neuron={record.neuron_index:04d} max_act={record.max_activation:.3f} tokens=[{token_preview}]"
        )

    if args.vector_output is not None:
        args.vector_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.vector_output, vector)
        print(f"[simulate] wrote averaged vector to {args.vector_output}")  # noqa: T201

    if args.steering_output is not None:
        direction = SteeringDirection(layer_index=args.layer_index, vector=vector.tolist(), gain=args.steering_gain)
        save_steering_config([direction], args.steering_output)
        print(f"[simulate] wrote steering config to {args.steering_output}")  # noqa: T201


def _select_cluster(
    clusters: Sequence[ClusterResult],
    *,
    cluster_id: int | None,
    keywords: Sequence[str] | None,
) -> ClusterResult:
    if cluster_id is not None:
        for cluster in clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        raise ValueError(f"Cluster id {cluster_id} not found in {len(clusters)} clusters.")

    assert keywords is not None  # checked by caller
    ranked = rank_clusters_by_keywords(clusters, keywords)
    if not ranked or ranked[0][1] <= 0:
        raise ValueError(f"No clusters matched keywords={keywords}")
    best_cluster, score = ranked[0]
    print(f"[simulate] selected cluster {best_cluster.cluster_id} via keywords (score={score:.3f})")  # noqa: T201
    return best_cluster


if __name__ == "__main__":
    main(tyro.cli(Args))


