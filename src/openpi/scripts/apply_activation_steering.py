"""Run policy inference with optional activation steering."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import pathlib
from typing import Iterable

import contextlib
import numpy as np
import tyro

from openpi.analysis import ActivationSteerer, load_steering_config
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
    config_name: str
    checkpoint_dir: pathlib.Path
    steering_config: pathlib.Path
    observations_glob: str
    device: str = "cuda"
    max_batches: int | None = 8


def main(args: Args) -> None:
    paths = sorted(pathlib.Path(p) for p in glob.glob(args.observations_glob))
    if not paths:
        raise ValueError(f"No observation files matched: {args.observations_glob}")

    train_config = training_config.get_config(args.config_name)
    policy = policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        pytorch_device=args.device,
    )
    if not getattr(policy, "_is_pytorch_model", False):
        raise ValueError("Activation steering currently supports PyTorch policies only.")

    model = policy._model  # pyright: ignore[reportPrivateUsage]
    directions = load_steering_config(args.steering_config)

    def run_inference(tag: str, context, batch_iter: Iterable[dict]):
        print(f"[{tag}] running inference")  # noqa: T201
        with context:
            for idx, batch in enumerate(batch_iter):
                if args.max_batches is not None and idx >= args.max_batches:
                    break
                result = policy.infer(batch)
                actions = result["actions"]
                mean_action = actions.mean()
                std_action = actions.std()
                print(f"  batch={idx} mean={mean_action:.4f} std={std_action:.4f}")  # noqa: T201

    run_inference("baseline", contextlib.nullcontext(), _iter_observations(paths))
    steerer = ActivationSteerer(model, directions)
    run_inference("steered", steerer, _iter_observations(paths))


if __name__ == "__main__":
    main(tyro.cli(Args))

