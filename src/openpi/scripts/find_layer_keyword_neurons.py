#!/usr/bin/env python3
"""Rank Paligemma/Gemma FFN neurons by token keywords without running inference."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Sequence

import torch
import tyro

from openpi.training import config as training_config
from openpi.models.tokenizer import PaligemmaTokenizer


@dataclass
class ConceptConfig:
    name: str
    keywords: tuple[str, ...]


@dataclass
class Args:
    checkpoint_path: Path
    config_name: str = "pi05_libero"
    layer_index: int = 12
    target: str = "paligemma_language"
    top_tokens: int = 20
    top_neurons: int = 8
    concepts: Sequence[str] = ("open", "close", "front", "back")
    concepts_json: Path | None = None
    output_json: Path | None = None
    device: str = "cpu"


def load_model(checkpoint_path: Path, config_name: str, device: str):
    if checkpoint_path.is_dir():
        model_path = checkpoint_path / "model.safetensors"
        if not model_path.exists():
            raise FileNotFoundError(f"model.safetensors not found under {checkpoint_path}")
        checkpoint_path = model_path
    train_config = training_config.get_config(config_name)
    model = train_config.model.load_pytorch(train_config, str(checkpoint_path))
    model.to(device)
    model.eval()
    return model


def resolve_layers_and_head(model, target: str):
    if target == "paligemma_language":
        paligemma = model.paligemma_with_expert.paligemma
        language_core = paligemma.language_model
        layers_src = getattr(language_core, "model", language_core)
        lm_head = getattr(paligemma, "lm_head", None)
        if lm_head is None:
            lm_head = getattr(language_core, "lm_head", None)
    elif target == "gemma_expert":
        layers_src = model.paligemma_with_expert.gemma_expert.model
        lm_head = model.paligemma_with_expert.gemma_expert.lm_head
    else:
        raise ValueError(f"Unknown target '{target}'.")
    if lm_head is None:
        lm_head = model.paligemma_with_expert.gemma_expert.lm_head
    layers = list(layers_src.layers)
    return layers, lm_head


def concept_specs_from_args(args: Args) -> list[ConceptConfig]:
    if args.concepts_json is not None:
        payload = json.loads(args.concepts_json.read_text())
        if not isinstance(payload, dict) or not payload:
            raise ValueError("--concepts-json must map concept names to keyword lists.")
        specs = []
        for name, keywords in payload.items():
            if isinstance(keywords, str):
                keywords = [keywords]
            specs.append(ConceptConfig(name=name, keywords=tuple(str(kw).lower() for kw in keywords if kw)))
        return specs
    return [
        ConceptConfig(name=concept, keywords=(concept.lower(),))
        for concept in args.concepts
    ]


def keyword_score(tokens: list[tuple[str, float]], keywords: tuple[str, ...]) -> float:
    score = 0.0
    for token, value in tokens:
        lt = token.lower()
        if any(keyword in lt for keyword in keywords):
            score += float(value)
    return score


def find_neurons(args: Args) -> dict[str, list[dict]]:
    device = torch.device(args.device)
    model = load_model(args.checkpoint_path, args.config_name, device=args.device)
    layers, lm_head = resolve_layers_and_head(model, args.target)

    if args.layer_index < 0 or args.layer_index >= len(layers):
        raise IndexError(f"Layer {args.layer_index} is out of bounds (model has {len(layers)} layers).")
    layer = layers[int(args.layer_index)]
    down_proj: torch.nn.Linear = layer.mlp.down_proj

    tokenizer = PaligemmaTokenizer()
    value_matrix = down_proj.weight.detach().to(device=device, dtype=torch.float32)
    lm_w = lm_head.weight.detach().to(device=device, dtype=torch.float32)
    logits = torch.matmul(lm_w, value_matrix)
    if lm_head.bias is not None:
        logits += lm_head.bias.detach().to(device=device, dtype=torch.float32).unsqueeze(1)

    top_vals, top_indices = torch.topk(logits, k=min(args.top_tokens, logits.shape[0]), dim=0)
    specs = concept_specs_from_args(args)

    concept_hits: dict[str, list[dict]] = {spec.name: [] for spec in specs}
    for neuron_idx in range(value_matrix.shape[1]):
        tokens = [
            (
                tokenizer._tokenizer.id_to_piece(int(tok_id)),
                float(score),
            )
            for tok_id, score in zip(top_indices[:, neuron_idx].tolist(), top_vals[:, neuron_idx].tolist(), strict=True)
        ]
        for spec in specs:
            score = keyword_score(tokens, spec.keywords)
            if score <= 0:
                continue
            concept_hits[spec.name].append(
                {
                    "layer": args.layer_index,
                    "neuron_index": neuron_idx,
                    "score": score,
                    "top_tokens": tokens[:5],
                }
            )

    for spec in specs:
        hits = sorted(concept_hits[spec.name], key=lambda item: item["score"], reverse=True)
        concept_hits[spec.name] = hits[: args.top_neurons]
    return concept_hits


def main(args: Args) -> None:
    concept_hits = find_neurons(args)
    for concept, neurons in concept_hits.items():
        print(f"\n[concept] {concept}")  # noqa: T201
        if not neurons:
            print("  (no matching neurons)")  # noqa: T201
            continue
        for entry in neurons:
            tokens = ", ".join(f"{tok}:{score:.2f}" for tok, score in entry["top_tokens"])
            print(  # noqa: T201
                f"  neuron={entry['neuron_index']:04d} score={entry['score']:.3f} tokens=[{tokens}]"
            )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(concept_hits, indent=2))
        print(f"\n[wrote] {args.output_json}")  # noqa: T201


if __name__ == "__main__":
    main(tyro.cli(Args))


