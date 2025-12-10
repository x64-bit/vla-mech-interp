#!/usr/bin/env python3
"""Slice residual shards into concept-specific tensors using a concept sample map."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import tyro  # type: ignore[import-not-found]


@dataclass
class ConceptRange:
    segment_id: int
    episode_index: int
    frame_start: int
    frame_end: int
    start_index: int
    end_index: int
    length: int


@dataclass
class Args:
    """CLI arguments for slicing residuals by concept."""

    run_dir: Path
    concept_map_path: Path
    output_dir: Path
    token_type: str = "instructions"
    layers: List[int] | None = None
    chunk_pattern: str = "layer{layer:02d}_chunk*.pt"
    catalog_path: Path | None = None
    max_samples_per_concept: int | None = None
    device: str = "cpu"
    verbose: bool = True


def _load_concept_map(path: Path) -> dict[str, List[ConceptRange]]:
    payload = json.loads(path.read_text())
    concepts = payload.get("concepts")
    if not isinstance(concepts, dict):
        raise ValueError(f"{path} missing 'concepts' field")
    parsed: dict[str, List[ConceptRange]] = {}
    for concept, ranges in concepts.items():
        bucket: list[ConceptRange] = []
        for entry in ranges:
            bucket.append(
                ConceptRange(
                    segment_id=int(entry["segment_id"]),
                    episode_index=int(entry["episode_index"]),
                    frame_start=int(entry["frame_start"]),
                    frame_end=int(entry["frame_end"]),
                    start_index=int(entry["start_index"]),
                    end_index=int(entry["end_index"]),
                    length=int(entry["length"]),
                )
            )
        parsed[concept] = sorted(bucket, key=lambda r: r.start_index)
    return parsed


def _iter_chunks(run_dir: Path, token_type: str, layer: int, pattern: str):
    token_dir = run_dir / token_type
    for path in sorted(token_dir.glob(pattern.format(layer=layer))):
        yield path


def slice_residuals(args: Args) -> None:
    concept_ranges = _load_concept_map(args.concept_map_path)
    if not concept_ranges:
        raise ValueError("Concept map is empty.")

    layers = args.layers
    if layers is None:
        layers = sorted({int(path.name.split("_")[0].replace("layer", "")) for path in (args.run_dir / args.token_type).glob("layer*_chunk*.pt")})

    device = torch.device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    concept_buffers: dict[str, List[torch.Tensor]] = {name: [] for name in concept_ranges}
    concept_counts: dict[str, int] = {name: 0 for name in concept_ranges}

    for layer in layers:
        if args.verbose:
            print(f"[slice] processing layer {layer}")  # noqa: T201
        layer_output_dir = output_dir / f"layer{layer:02d}"
        layer_output_dir.mkdir(parents=True, exist_ok=True)
        concept_buffers = {name: [] for name in concept_ranges}
        concept_counts = {name: 0 for name in concept_ranges}

        ranges_by_concept = {
            concept: list(ranges)
            for concept, ranges in concept_ranges.items()
        }

        global_index = 0
        for chunk_path in _iter_chunks(args.run_dir, args.token_type, layer, args.chunk_pattern):
            tensor = torch.load(chunk_path, map_location=device)
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(1)
            if tensor.dim() != 3:
                raise ValueError(f"{chunk_path} expected rank-3 tensor, got shape {tensor.shape}")
            chunk_rows = tensor.shape[0]
            chunk_start = global_index
            chunk_end = global_index + chunk_rows - 1

            for concept, ranges in ranges_by_concept.items():
                while ranges and ranges[0].end_index < chunk_start:
                    ranges.pop(0)
                if not ranges:
                    continue
                for rng in ranges:
                    if rng.start_index > chunk_end:
                        break
                    overlap_start = max(rng.start_index, chunk_start)
                    overlap_end = min(rng.end_index, chunk_end)
                    if overlap_end < overlap_start:
                        continue
                    row_start = overlap_start - chunk_start
                    row_end = overlap_end - chunk_start + 1
                    rows = tensor[row_start:row_end].cpu()
                    concept_buffers[concept].append(rows)
                    concept_counts[concept] += rows.shape[0]
                    if args.max_samples_per_concept is not None and concept_counts[concept] >= args.max_samples_per_concept:
                        break
            global_index += chunk_rows

        for concept, buffers in concept_buffers.items():
            if not buffers:
                continue
            combined = torch.cat(buffers, dim=0)
            if args.max_samples_per_concept is not None:
                combined = combined[: args.max_samples_per_concept]
            out_path = layer_output_dir / f"{concept}.pt"
            torch.save(combined, out_path)
            if args.verbose:
                print(f"[slice] wrote {combined.shape[0]} samples for concept '{concept}' to {out_path}")  # noqa: T201


def main() -> None:
    args = tyro.cli(Args)
    slice_residuals(args)


if __name__ == "__main__":
    main()

