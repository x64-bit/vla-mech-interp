"""Catalog residual shards for LIBERO runs and emit train/val splits."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import torch
import tyro


@dataclass
class Args:
    """CLI arguments for cataloging a residual run."""

    run_dir: Path
    output_name: str = "catalog.json"
    val_ratio: float = 0.1
    seed: int = 0
    include_decision: bool = True
    include_instructions: bool = True
    include_adarms: bool = True
    verbose: bool = True


def main(args: Args) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"{run_dir} does not exist")

    catalog: dict = {
        "run_dir": str(run_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "token_types": {},
    }

    if args.include_decision:
        decision_summary = _catalog_token_type(
            run_dir / "decision",
            val_ratio=args.val_ratio,
            seed=args.seed,
            verbose=args.verbose,
        )
        if decision_summary is not None:
            catalog["token_types"]["decision"] = decision_summary

    if args.include_instructions:
        instruction_summary = _catalog_token_type(
            run_dir / "instructions",
            val_ratio=args.val_ratio,
            seed=args.seed,
            verbose=args.verbose,
        )
        if instruction_summary is not None:
            catalog["token_types"]["instructions"] = instruction_summary

    if args.include_adarms:
        adarms_summary = _catalog_adarms(
            run_dir / "adarms",
            val_ratio=args.val_ratio,
            seed=args.seed,
            verbose=args.verbose,
        )
        if adarms_summary is not None:
            catalog["token_types"]["adarms"] = adarms_summary

    if not catalog["token_types"]:
        raise RuntimeError(f"No residual subdirectories found under {run_dir}")

    output_path = run_dir / args.output_name
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(catalog, fp, indent=2)

    if args.verbose:
        print(f"[catalog] Wrote {output_path}")  # noqa: T201


def _catalog_token_type(
    token_dir: Path,
    *,
    val_ratio: float,
    seed: int,
    verbose: bool,
) -> dict | None:
    if not token_dir.exists():
        if verbose:
            print(f"[catalog] Skipping missing directory {token_dir}")  # noqa: T201
        return None

    layer_chunks = _gather_layer_chunks(token_dir)
    if not layer_chunks:
        if verbose:
            print(f"[catalog] No layer chunks detected in {token_dir}")  # noqa: T201
        return None

    token_summary: dict[str, dict] = {
        "layers": {},
        "total_samples": 0,
        "chunk_count": 0,
    }

    for layer_id, chunks in sorted(layer_chunks.items()):
        train_chunks, val_chunks = _split_chunks(chunks, val_ratio=val_ratio, seed=seed)
        total_samples = sum(chunk["samples"] for chunk in chunks)
        token_summary["layers"][str(layer_id)] = {
            "total_samples": total_samples,
            "chunk_count": len(chunks),
            "chunks": chunks,
            "splits": {
                "train": [chunk["file"] for chunk in train_chunks],
                "val": [chunk["file"] for chunk in val_chunks],
            },
        }
        token_summary["total_samples"] += total_samples
        token_summary["chunk_count"] += len(chunks)

        if verbose:
            train_count = sum(chunk["samples"] for chunk in train_chunks)
            val_count = sum(chunk["samples"] for chunk in val_chunks)
            print(  # noqa: T201
                f"[catalog] {token_dir.name} layer {layer_id:02d}: "
                f"{len(chunks)} chunks, total={total_samples}, "
                f"train={train_count}, val={val_count}"
            )

    return token_summary


def _catalog_adarms(
    adarms_dir: Path,
    *,
    val_ratio: float,
    seed: int,
    verbose: bool,
) -> dict | None:
    if not adarms_dir.exists():
        if verbose:
            print(f"[catalog] Skipping missing directory {adarms_dir}")  # noqa: T201
        return None

    chunks = _gather_chunks(adarms_dir.glob("chunk*.pt"))
    if not chunks:
        if verbose:
            print(f"[catalog] No AdaRMS chunks detected in {adarms_dir}")  # noqa: T201
        return None

    train_chunks, val_chunks = _split_chunks(chunks, val_ratio=val_ratio, seed=seed)
    total_samples = sum(chunk["samples"] for chunk in chunks)
    if verbose:
        train_count = sum(chunk["samples"] for chunk in train_chunks)
        val_count = sum(chunk["samples"] for chunk in val_chunks)
        print(  # noqa: T201
            f"[catalog] adarms: {len(chunks)} chunks, "
            f"total={total_samples}, train={train_count}, val={val_count}"
        )

    return {
        "total_samples": total_samples,
        "chunk_count": len(chunks),
        "chunks": chunks,
        "splits": {
            "train": [chunk["file"] for chunk in train_chunks],
            "val": [chunk["file"] for chunk in val_chunks],
        },
    }


def _gather_layer_chunks(token_dir: Path) -> dict[int, list[dict]]:
    """Collect chunk metadata keyed by layer id."""
    layer_map: dict[int, list[dict]] = defaultdict(list)
    for path in sorted(token_dir.glob("layer*_chunk*.pt")):
        layer_id = _parse_layer_id(path)
        chunk_entry = _inspect_chunk(path)
        layer_map[layer_id].append(chunk_entry)
    return layer_map


def _gather_chunks(paths: Iterable[Path]) -> list[dict]:
    return [_inspect_chunk(path) for path in sorted(paths)]


def _parse_layer_id(path: Path) -> int:
    prefix = path.stem.split("_", 1)[0]
    if not prefix.startswith("layer"):
        raise ValueError(f"Unable to parse layer id from {path.name}")
    return int(prefix.replace("layer", ""))


def _inspect_chunk(path: Path) -> dict:
    tensor = torch.load(path, map_location="cpu")
    if not torch.is_tensor(tensor):
        raise TypeError(f"{path} did not contain a tensor")

    samples = int(tensor.shape[0])
    shape = tuple(int(dim) for dim in tensor.shape)
    dtype = str(tensor.dtype)
    entry = {
        "file": path.name,
        "samples": samples,
        "shape": list(shape),
        "dtype": dtype,
        "size_bytes": path.stat().st_size,
    }
    del tensor
    return entry


def _split_chunks(
    chunks: Sequence[dict],
    *,
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    chunk_count = len(chunks)
    if chunk_count == 0 or val_ratio <= 0:
        return list(chunks), []
    if chunk_count == 1:
        return list(chunks), []

    rng = random.Random(seed)
    indices = list(range(chunk_count))
    rng.shuffle(indices)

    proposed = max(1, int(round(chunk_count * val_ratio)))
    val_count = min(proposed, chunk_count - 1)
    val_indices = set(indices[:val_count])

    train_chunks = [chunks[i] for i in range(chunk_count) if i not in val_indices]
    val_chunks = [chunks[i] for i in range(chunk_count) if i in val_indices]
    return train_chunks, val_chunks


if __name__ == "__main__":
    main(tyro.cli(Args))









