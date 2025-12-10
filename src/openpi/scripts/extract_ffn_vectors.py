"""Run FFN activation probing on a trained PyTorch pi0 policy."""

from __future__ import annotations

import glob
import json
import pathlib
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import torch  # pyright: ignore[reportMissingImports]
import tyro

from openpi.analysis.ffn_probe import FFNProbeConfig, run_probe_over_batches
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.policies import policy_config
from openpi.training import config as training_config


def _load_npz(path: pathlib.Path) -> dict:
    with np.load(path, allow_pickle=True) as data:
        if "arr_0" in data:
            return data["arr_0"].item()
        return {key: data[key].item() if data[key].dtype == object else data[key] for key in data.files}


def _iter_observations(paths: Iterable[pathlib.Path]) -> Iterable[dict]:
    for path in paths:
        yield _load_npz(path)


@dataclass
class Args:
    """CLI arguments for FFN vector extraction."""

    config_name: str
    checkpoint_dir: pathlib.Path
    observations_glob: str
    output_path: pathlib.Path
    top_k_per_layer: int = 10
    max_batches: int | None = 64
    device: str = "cuda"
    store_value_vectors: bool = False
    min_activation: float = 0.0
    activation_target: Literal["gemma_expert", "paligemma_language"] = "gemma_expert"


def main(args: Args) -> None:
    observation_paths = sorted(pathlib.Path(p) for p in glob.glob(args.observations_glob))
    if not observation_paths:
        raise ValueError(f"No observation files matched pattern: {args.observations_glob}")

    train_config = training_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        pytorch_device=args.device,
    )
    if not getattr(policy, "_is_pytorch_model", False):
        raise ValueError("FFN probe currently supports PyTorch pi0 policies only.")

    model = policy._model  # pyright: ignore[reportPrivateUsage]
    tokenizer = PaligemmaTokenizer()
    batches = _iter_observations(observation_paths)

    def forward_fn(batch: dict) -> None:
        with torch.no_grad():
            policy.infer(batch)

    probe_config = FFNProbeConfig(
        top_k_per_layer=args.top_k_per_layer,
        store_value_vectors=args.store_value_vectors,
        min_activation=args.min_activation,
    )
    limited_batches = batches if args.max_batches is None else (obs for idx, obs in enumerate(batches) if idx < args.max_batches)

    projections = run_probe_over_batches(
        model=model,
        tokenizer=tokenizer,
        device=args.device,
        batches=limited_batches,
        forward_fn=forward_fn,
        probe_config=probe_config,
        target=args.activation_target,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as f:
        json.dump([proj.to_dict() for proj in projections], f, indent=2)
    print(f"Wrote {len(projections)} neuron projections to {args.output_path}")  # noqa: T201


if __name__ == "__main__":
    main(tyro.cli(Args))

