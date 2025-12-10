"""Cluster FFN value vectors and summarize semantic directions."""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Sequence

import tyro

try:
    import torch  # pyright: ignore[reportMissingImports]
    import safetensors.torch  # pyright: ignore[reportMissingImports]
    from openpi.config import TrainConfig
    from openpi.models.pi0_config import Pi0Config

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from openpi.analysis import (
    ClusterConfig,
    cluster_projections,
    extract_model_components_for_clustering,
    load_projections,
    rank_clusters_by_keywords,
    save_clusters,
)


@dataclass
class Args:
    projections_path: pathlib.Path
    output_path: pathlib.Path
    num_clusters: int = 10
    max_iterations: int = 100
    normalize: bool = True
    random_seed: int = 0
    layers: Sequence[int] | None = None
    keywords: Sequence[str] | None = None
    use_semantic_space: bool = False
    model_checkpoint: pathlib.Path | None = None
    model_config: pathlib.Path | None = None
    top_k_tokens: int = 10
    device: str = "cpu"


def main(args: Args) -> None:
    records = load_projections(args.projections_path)
    if args.layers:
        layer_set = set(args.layers)
        records = [rec for rec in records if rec.layer_index in layer_set]
    if not records:
        raise ValueError("No projections available after filtering. Did you forget to store value vectors?")

    # Prepare semantic clustering components if requested
    lm_head_weight = None
    token_embeddings = None
    
    if args.use_semantic_space:
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for semantic space clustering. "
                "Install it with: pip install torch safetensors"
            )
        
        if args.model_checkpoint is None:
            raise ValueError(
                "model_checkpoint must be provided when use_semantic_space=True. "
                "Provide path to model checkpoint directory or model.safetensors file."
            )
        
        print(f"Loading model components from {args.model_checkpoint}...")  # noqa: T201
        
        # Determine checkpoint path
        checkpoint_path = pathlib.Path(args.model_checkpoint)
        if checkpoint_path.is_dir():
            safetensors_path = checkpoint_path / "model.safetensors"
            if not safetensors_path.exists():
                raise FileNotFoundError(f"No model.safetensors found in {checkpoint_path}")
            checkpoint_path = safetensors_path
        elif not checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
        
        # Load model config
        if args.model_config is None:
            # Try to find config in checkpoint directory
            config_path = checkpoint_path.parent / "config.json"
            if not config_path.exists():
                raise ValueError(
                    f"Could not find config.json. Please provide --model-config path or "
                    f"ensure config.json exists in {checkpoint_path.parent}"
                )
        else:
            config_path = args.model_config
        
        # Load train config
        train_config = TrainConfig.from_json(config_path)
        
        # Load model
        model = train_config.model.load_pytorch(train_config, str(checkpoint_path))
        model.eval()
        
        # Extract components
        lm_head_weight, token_embeddings = extract_model_components_for_clustering(model)
        print(f"Extracted LM head: {lm_head_weight.shape}, Token embeddings: {token_embeddings.shape}")  # noqa: T201
        
        # Clean up model from memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    config = ClusterConfig(
        num_clusters=args.num_clusters,
        max_iterations=args.max_iterations,
        normalize=args.normalize,
        random_seed=args.random_seed,
        use_semantic_space=args.use_semantic_space,
        top_k_tokens=args.top_k_tokens,
    )
    
    clustering_mode = "semantic token space" if args.use_semantic_space else "raw weight space"
    print(f"Clustering {len(records)} projections in {clustering_mode}...")  # noqa: T201
    
    clusters = cluster_projections(
        records,
        config,
        lm_head_weight=lm_head_weight,
        token_embeddings=token_embeddings,
        device=args.device,
    )
    save_clusters(clusters, args.output_path)
    print(f"Saved {len(clusters)} clusters to {args.output_path}")  # noqa: T201

    if args.keywords:
        scored = rank_clusters_by_keywords(clusters, args.keywords)
        print("Top clusters by keyword score:")  # noqa: T201
        for cluster, score in scored[:5]:
            tokens = ", ".join(token for token, _ in cluster.token_summary[:5])
            print(f"  cluster={cluster.cluster_id} score={score:.3f} tokens=[{tokens}]")  # noqa: T201


if __name__ == "__main__":
    main(tyro.cli(Args))

