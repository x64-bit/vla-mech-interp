#!/usr/bin/env python3
"""Build concept-to-sample mappings from a contrast manifest and LeRobot dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import tyro
from lerobot.common.datasets import lerobot_dataset


@dataclass
class Args:
    """CLI arguments for mapping concept segments to dataset sample indices."""

    manifest_path: Path
    output_path: Path
    lerobot_root: Path = Path("data/libero_lerobot/libero_90_lerobot")
    repo_id: str = "libero_90_lerobot"
    concept_map: Path | None = None
    verbose: bool = True


@dataclass(frozen=True)
class Segment:
    id: int
    episode_index: int
    start_frame: int
    end_frame: int
    concept: str


def _load_concept_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    mapping = json.loads(path.read_text())
    if not isinstance(mapping, dict):
        raise ValueError("concept_map must contain a JSON object of task->concept mappings.")
    return {str(k): str(v) for k, v in mapping.items()}


def _prepare_segments(manifest_path: Path, concept_map: dict[str, str]) -> dict[int, list[Segment]]:
    payload = json.loads(manifest_path.read_text())
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        raise ValueError(f"Manifest {manifest_path} did not contain a 'segments' list.")

    grouped: dict[int, list[Segment]] = {}
    for seg_id, entry in enumerate(raw_segments):
        episode_index = entry.get("episode_index")
        start = entry.get("start_step")
        end = entry.get("end_step")
        task = entry.get("task")
        if episode_index is None:
            raise ValueError(
                "Manifest segments must include 'episode_index'. "
                "Re-run build_contrasting_pairs with the LeRobot backend."
            )
        if start is None or end is None:
            raise ValueError("Manifest segments must include 'start_step' and 'end_step'.")
        if task is None:
            raise ValueError("Manifest segments must include 'task'.")
        start_frame = int(start)
        end_frame = int(end)
        if end_frame < start_frame:
            raise ValueError(f"Segment {seg_id} has end < start ({end_frame} < {start_frame}).")
        concept = concept_map.get(task, task)
        grouped.setdefault(int(episode_index), []).append(
            Segment(
                id=seg_id,
                episode_index=int(episode_index),
                start_frame=start_frame,
                end_frame=end_frame,
                concept=concept,
            )
        )
    for segments in grouped.values():
        segments.sort(key=lambda seg: seg.start_frame)
    return grouped


def _to_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError("Expected scalar tensor for metadata fields.")
        return int(value.item())
    if isinstance(value, np.generic):
        return int(value.item())
    return int(value)


def _build_mapping(
    dataset: lerobot_dataset.LeRobotDataset,
    segments_by_episode: dict[int, list[Segment]],
    verbose: bool,
) -> dict[str, Any]:
    if not segments_by_episode:
        raise ValueError("No segments found in manifest for mapping.")

    hf_dataset = dataset.hf_dataset
    if hf_dataset is None:
        hf_dataset = dataset.load_hf_dataset()

    episode_states: dict[int, dict[str, Any]] = {}
    segment_states: dict[int, dict[str, Any]] = {}
    for episode_index, segments in segments_by_episode.items():
        episode_states[episode_index] = {"segments": segments, "cursor": 0}
        for seg in segments:
            segment_states[seg.id] = {
                "segment": seg,
                "active": False,
                "start_index": None,
                "last_index": None,
                "start_frame_obs": None,
                "last_frame_obs": None,
                "completed": False,
            }

    concept_ranges: dict[str, list[dict[str, int]]] = {}
    global_index = 0

    def finalize_segment(seg_id: int) -> None:
        state = segment_states[seg_id]
        seg = state["segment"]
        if state["completed"]:
            return
        if not state["active"] or state["start_index"] is None or state["last_index"] is None:
            raise RuntimeError(
                f"Segment {seg_id} ({seg.concept}, episode {seg.episode_index}) "
                "ended without capturing any samples. Check manifest frame ranges."
            )
        state["active"] = False
        state["completed"] = True
        concept_ranges.setdefault(seg.concept, []).append(
            {
                "segment_id": seg.id,
                "episode_index": seg.episode_index,
                "frame_start": seg.start_frame,
                "frame_end": seg.end_frame,
                "observed_frame_start": int(state["start_frame_obs"]),
                "observed_frame_end": int(state["last_frame_obs"]),
                "start_index": int(state["start_index"]),
                "end_index": int(state["last_index"]),
                "length": int(state["last_index"] - state["start_index"] + 1),
            }
        )

    for row in hf_dataset:
        episode_index = _to_int(row["episode_index"])
        frame_index = _to_int(row["frame_index"])

        state = episode_states.get(episode_index)
        if state is not None:
            cursor = state["cursor"]
            segments = state["segments"]
            while cursor < len(segments) and frame_index > segments[cursor].end_frame:
                finalize_segment(segments[cursor].id)
                cursor += 1
            state["cursor"] = cursor
            if cursor < len(segments):
                seg = segments[cursor]
                seg_state = segment_states[seg.id]
                if frame_index < seg.start_frame:
                    pass
                elif frame_index <= seg.end_frame:
                    if not seg_state["active"]:
                        seg_state["active"] = True
                        seg_state["start_index"] = global_index
                        seg_state["start_frame_obs"] = frame_index
                    seg_state["last_index"] = global_index
                    seg_state["last_frame_obs"] = frame_index
                    if frame_index == seg.end_frame:
                        finalize_segment(seg.id)
                        state["cursor"] = cursor + 1
        global_index += 1

    # Finalize any trailing segments that ended exactly at dataset end.
    for episode_state in episode_states.values():
        cursor = episode_state["cursor"]
        segments = episode_state["segments"]
        while cursor < len(segments):
            finalize_segment(segments[cursor].id)
            cursor += 1

    unfinished = [seg_id for seg_id, st in segment_states.items() if not st["completed"]]
    if unfinished:
        raise RuntimeError(
            f"Failed to cover {len(unfinished)} segments from manifest. "
            "Ensure the dataset and manifest refer to the same conversion."
        )

    if verbose:
        for concept, entries in concept_ranges.items():
            total = sum(entry["length"] for entry in entries)
            print(f"[concept-map] {concept}: {len(entries)} segments, {total} samples")  # noqa: T201

    return {
        "sample_count": global_index,
        "concepts": concept_ranges,
    }


def main(args: Args) -> None:
    concept_map = _load_concept_map(args.concept_map)
    segments_by_episode = _prepare_segments(args.manifest_path, concept_map)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id=args.repo_id,
        root=str(args.lerobot_root),
        force_cache_sync=False,
        download_videos=False,
    )
    mapping = _build_mapping(dataset, segments_by_episode, verbose=args.verbose)
    output = {
        "manifest": str(args.manifest_path),
        "lerobot_root": str(args.lerobot_root),
        "repo_id": args.repo_id,
        "concept_count": {concept: len(entries) for concept, entries in mapping["concepts"].items()},
        "sample_count": mapping["sample_count"],
        "concepts": mapping["concepts"],
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(output, indent=2))
    if args.verbose:
        print(f"[concept-map] Wrote {args.output_path}")  # noqa: T201


if __name__ == "__main__":
    main(tyro.cli(Args))






