#!/usr/bin/env python3
"""Contrastive segment extractor for paired LIBERO manipulation demos."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import h5py  # type: ignore
import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-not-found]


DEFAULT_DATA_ROOTS: tuple[Path, ...] = (
    Path("data/libero_100_original/libero_10"),
    Path("data/libero_100_original/libero_90"),
    Path("data/libero_goal_original/libero_goal"),
    Path("data/libero_object_original/libero_object"),
    Path("data/libero_spatial_original/libero_spatial"),
)

ALLOWED_DATASETS: frozenset[str] = frozenset(
    {
        "KITCHEN_SCENE8_turn_off_the_stove_demo.hdf5",
        "KITCHEN_SCENE9_turn_on_the_stove_demo.hdf5",
        "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet_demo.hdf5",
        "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5",
    }
)


@dataclass
class DemoSegment:
    task: str
    file: str
    demo: str
    start_step: int
    end_step: int
    axis_delta: float
    axis_values: List[float]
    episode_index: int | None = None


@dataclass
class AxisSummary:
    index: int
    effect_size: float
    mean_delta_per_task: Dict[str, float]


@dataclass
class ContrastManifest:
    axis: AxisSummary
    segments: List[DemoSegment]


@dataclass
class TaskDataset:
    """Container for demos plus per-demo metadata."""

    demos: Dict[str, np.ndarray]
    default_source: str
    file_overrides: Dict[str, str]
    episode_indices: Dict[str, int | None]


TaskName = str


class LeRobotDataset:
    """Lightweight reader for converted LeRobot datasets."""

    def __init__(self, root: Path):
        self.root = root
        self.meta_dir = root / "meta"
        if not self.meta_dir.is_dir():
            raise FileNotFoundError(f"LeRobot metadata directory not found: {self.meta_dir}")
        info_path = self.meta_dir / "info.json"
        self.info = json.loads(info_path.read_text())
        self.chunk_size = self.info.get("chunks_size", 1_000)
        self.data_path_template = self.info["data_path"]
        self._tasks_by_index: Dict[int, str] = {}
        self._task_index_by_text: Dict[str, int] = {}
        tasks_path = self.meta_dir / "tasks.jsonl"
        with tasks_path.open() as handle:
            for line in handle:
                entry = json.loads(line)
                task = entry["task"]
                task_index = entry["task_index"]
                self._tasks_by_index[task_index] = task
                self._task_index_by_text[task.lower()] = task_index
        self._episodes: Dict[int, dict[str, Any]] = {}
        self._episodes_by_task: Dict[str, list[int]] = defaultdict(list)
        episodes_path = self.meta_dir / "episodes.jsonl"
        with episodes_path.open() as handle:
            for line in handle:
                entry = json.loads(line)
                episode_index = entry["episode_index"]
                self._episodes[episode_index] = entry
                for task in entry.get("tasks", []):
                    self._episodes_by_task[task].append(episode_index)

    def task_text_from_index(self, task_index: int) -> str:
        if task_index not in self._tasks_by_index:
            raise ValueError(f"Unknown task index {task_index}")
        return self._tasks_by_index[task_index]

    def resolve_task_index(self, *, task_index: int | None, task_text: str | None) -> int:
        if task_index is not None:
            return task_index
        if not task_text:
            raise ValueError("Provide either task index or task text for LeRobot datasets")
        normalized = task_text.strip().lower()
        if normalized not in self._task_index_by_text:
            raise ValueError(f"Task text '{task_text}' not found in metadata")
        return self._task_index_by_text[normalized]

    def episodes_for_task(self, task_text: str) -> list[int]:
        return sorted(self._episodes_by_task.get(task_text, []))

    def load_episode_states(self, episode_index: int) -> tuple[str, np.ndarray]:
        if episode_index not in self._episodes:
            raise ValueError(f"Episode {episode_index} not found in metadata")
        relative = self.data_path_template.format(
            episode_chunk=episode_index // self.chunk_size,
            episode_index=episode_index,
        )
        rel_path = Path(relative)
        abs_path = self.root / rel_path
        if not abs_path.exists():
            raise FileNotFoundError(f"Episode parquet not found: {abs_path}")
        table = pq.read_table(abs_path)
        states = np.asarray(table["state"].to_pylist(), dtype=np.float64)
        return rel_path.as_posix(), states

def resolve_dataset_file(value: str, extra_roots: Sequence[Path]) -> Path:
    """Resolve a dataset argument to an existing .hdf5 path."""

    raw_path = Path(value)
    candidates = [raw_path]
    if not raw_path.suffix:
        candidates.append(raw_path.with_suffix(".hdf5"))
    # Search user-provided and default roots when only a filename is given.
    search_roots = list(extra_roots) + list(DEFAULT_DATA_ROOTS)
    for candidate in candidates:
        if candidate.exists():
            return candidate
        if candidate.is_absolute():
            continue
        for root in search_roots:
            root_candidate = root / candidate.name
            if root_candidate.exists():
                return root_candidate
    raise FileNotFoundError(f"Could not locate dataset '{value}'. Specify an absolute or relative path.")


def ensure_allowed_dataset(path: Path) -> None:
    """Validate that the dataset file corresponds to a supported single-action task."""

    if path.name not in ALLOWED_DATASETS:
        allowed_str = ", ".join(sorted(ALLOWED_DATASETS))
        raise ValueError(
            "Unsupported dataset '{}' provided. This tool currently supports only the following single-action tasks: {}".format(
                path.name, allowed_str
            )
        )


def load_demo_states(h5_path: Path) -> Dict[str, np.ndarray]:
    """Load every demo's state trajectory from a LIBERO HDF5 file."""

    demos: Dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as handle:
        data_grp = handle["data"]
        for demo_key in sorted(data_grp.keys()):
            states = data_grp[f"{demo_key}"]["states"][...]
            demos[demo_key] = states.astype(np.float64)
    return demos


def task_dataset_from_hdf5(path: Path) -> TaskDataset:
    demos = load_demo_states(path)
    return TaskDataset(
        demos=demos,
        default_source=path.name,
        file_overrides={},
        episode_indices={demo_id: None for demo_id in demos},
    )


def task_dataset_from_lerobot(
    dataset: LeRobotDataset,
    task_index: int,
    *,
    episode_indices: Sequence[int] | None = None,
    max_demos: int | None = None,
) -> tuple[TaskDataset, list[int]]:
    task_text = dataset.task_text_from_index(task_index)
    available = dataset.episodes_for_task(task_text)
    if not available:
        raise ValueError(f"No episodes found for task '{task_text}'")
    if episode_indices is not None:
        missing = [idx for idx in episode_indices if idx not in available]
        if missing:
            raise ValueError(f"Episodes {missing} do not belong to task '{task_text}'")
        selected = list(episode_indices)
    else:
        selected = list(available)
    selected.sort()
    if max_demos is not None:
        selected = selected[:max_demos]
    demos: Dict[str, np.ndarray] = {}
    file_overrides: Dict[str, str] = {}
    episode_map: Dict[str, int | None] = {}
    for episode_index in selected:
        rel_path, states = dataset.load_episode_states(episode_index)
        demo_id = f"episode_{episode_index:06d}"
        demos[demo_id] = states
        file_overrides[demo_id] = rel_path
        episode_map[demo_id] = episode_index
    if not demos:
        raise ValueError(f"No demos loaded for task '{task_text}'")
    return (
        TaskDataset(
            demos=demos,
            default_source=str(dataset.root),
            file_overrides=file_overrides,
            episode_indices=episode_map,
        ),
        selected,
    )


def compute_delta_stats(demos: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per-dimension mean/std of terminal displacement for each demo set."""

    if not demos:
        raise ValueError("No demos found in dataset")
    deltas = np.stack([states[-1] - states[0] for states in demos.values()])
    return deltas.mean(axis=0), deltas.std(axis=0)


def select_opposite_axis(
    mean_by_task: Dict[TaskName, np.ndarray],
    std_by_task: Dict[TaskName, np.ndarray],
    min_abs_delta: float,
) -> Tuple[int, float]:
    """Pick the state dimension with large, opposing motion between two tasks."""

    tasks = list(mean_by_task.keys())
    if len(tasks) != 2:
        raise ValueError("Axis selection currently supports exactly two tasks")
    a, b = tasks
    diff = mean_by_task[a] - mean_by_task[b]
    pooled_std = np.sqrt(0.5 * (std_by_task[a] ** 2 + std_by_task[b] ** 2))
    effect = diff / (pooled_std + 1e-9)
    candidate_order = np.argsort(np.abs(diff))[::-1]
    for idx in candidate_order:
        sign_a = np.sign(mean_by_task[a][idx])
        sign_b = np.sign(mean_by_task[b][idx])
        if sign_a == 0 or sign_b == 0:
            continue
        if sign_a != sign_b:
            if abs(mean_by_task[a][idx]) < min_abs_delta or abs(mean_by_task[b][idx]) < min_abs_delta:
                continue
            return int(idx), float(effect[idx])
    raise RuntimeError("Failed to find an axis with opposite signed motion")


def locate_interaction_window(
    states: np.ndarray,
    axis_idx: int,
    progress_frac: float,
    pad: int,
) -> Tuple[int, int]:
    """Find the time span where the chosen axis makes most of its progress."""

    axis_series = states[:, axis_idx]
    displacement = axis_series - axis_series[0]
    total_delta = axis_series[-1] - axis_series[0]
    threshold = max(abs(total_delta) * progress_frac, 1e-6)
    active = np.flatnonzero(np.abs(displacement) >= threshold)
    if active.size == 0:
        gradients = np.abs(np.diff(axis_series, prepend=axis_series[0]))
        if not np.any(gradients):
            return 0, len(axis_series) - 1
        top_idx = np.argsort(gradients)[-max(5, len(axis_series) // 10) :]
        top_idx = np.sort(top_idx)
        start, end = int(top_idx[0]), int(top_idx[-1])
    else:
        start, end = int(active[0]), int(active[-1])
    start = max(start - pad, 0)
    end = min(end + pad, len(axis_series) - 1)
    if end <= start:
        end = min(start + 1, len(axis_series) - 1)
    return start, end


def demo_segments(
    demos: Dict[str, np.ndarray],
    task: str,
    file_name: str,
    axis_idx: int,
    progress_frac: float,
    pad: int,
    file_overrides: Dict[str, str] | None = None,
    episode_indices: Dict[str, int | None] | None = None,
) -> Iterable[DemoSegment]:
    """Yield interaction windows for every demo belonging to one task label."""

    for demo_id, states in demos.items():
        start, end = locate_interaction_window(states, axis_idx, progress_frac, pad)
        axis_values = states[start : end + 1, axis_idx]
        source_name = file_overrides.get(demo_id, file_name) if file_overrides else file_name
        episode_index = episode_indices.get(demo_id) if episode_indices else None
        yield DemoSegment(
            task=task,
            file=source_name,
            demo=demo_id,
            start_step=start,
            end_step=end,
            axis_delta=float(axis_values[-1] - axis_values[0]),
            axis_values=axis_values.tolist(),
            episode_index=episode_index,
        )


def build_manifest(
    task_datasets: Dict[str, TaskDataset],
    progress_frac: float,
    pad: int,
    min_abs_delta: float,
) -> ContrastManifest:
    """Construct the full contrastive manifest for two labeled task datasets."""

    if len(task_datasets) != 2:
        raise ValueError("Manifest builder expects exactly two task paths")
    mean_stats: Dict[str, np.ndarray] = {}
    std_stats: Dict[str, np.ndarray] = {}
    for task, dataset in task_datasets.items():
        mean_stats[task], std_stats[task] = compute_delta_stats(dataset.demos)
    axis_idx, effect = select_opposite_axis(mean_stats, std_stats, min_abs_delta)
    axis_summary = AxisSummary(
        index=axis_idx,
        effect_size=effect,
        mean_delta_per_task={task: float(mean[axis_idx]) for task, mean in mean_stats.items()},
    )
    segments: List[DemoSegment] = []
    for task, dataset in task_datasets.items():
        segments.extend(
            demo_segments(
                demos=dataset.demos,
                task=task,
                file_name=dataset.default_source,
                axis_idx=axis_idx,
                progress_frac=progress_frac,
                pad=pad,
                file_overrides=dataset.file_overrides,
                episode_indices=dataset.episode_indices,
            )
        )
    return ContrastManifest(axis=axis_summary, segments=segments)


def save_manifest(manifest: ContrastManifest, output_path: Path) -> None:
    payload = {
        "axis": asdict(manifest.axis),
        "segments": [asdict(segment) for segment in manifest.segments],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build contrastive pairs for two labeled tasks.")
    parser.add_argument(
        "--backend",
        choices=("hdf5", "lerobot"),
        default="hdf5",
        help="Source backend for demos. Use 'lerobot' for converted LIBERO datasets.",
    )
    parser.add_argument(
        "--task-a-file",
        type=str,
        help="Path or filename for the first task HDF5 file (hdf5 backend only).",
    )
    parser.add_argument(
        "--task-b-file",
        type=str,
        help="Path or filename for the second task HDF5 file (hdf5 backend only).",
    )
    parser.add_argument("--task-a-label", type=str, help="Optional label for the first task.")
    parser.add_argument("--task-b-label", type=str, help="Optional label for the second task.")
    parser.add_argument("--output", type=Path, default=Path("data/utils/manifests/contrast_pairs.json"), help="Destination JSON manifest.")
    parser.add_argument("--progress-frac", type=float, default=0.35, help="Fraction of total axis progress used to bracket the interaction window.")
    parser.add_argument("--pad", type=int, default=8, help="Extra steps appended to both ends of each interaction window.")
    parser.add_argument("--min-abs-delta", type=float, default=0.05, help="Minimum absolute displacement each task must show on the selected axis.")
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=None,
        help="Additional directory to search when dataset arguments are given as filenames.",
    )
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=Path("data/libero_lerobot/libero_90_lerobot"),
        help="Root directory of the converted LeRobot dataset.",
    )
    parser.add_argument("--task-a-index", type=int, help="Task index (from tasks.jsonl) for the first concept (lerobot backend).")
    parser.add_argument("--task-b-index", type=int, help="Task index (from tasks.jsonl) for the second concept (lerobot backend).")
    parser.add_argument("--task-a-text", type=str, help="Exact task text for the first concept (lerobot backend alternative).")
    parser.add_argument("--task-b-text", type=str, help="Exact task text for the second concept (lerobot backend alternative).")
    parser.add_argument(
        "--task-a-episodes",
        type=int,
        nargs="+",
        help="Explicit episode indices to use for the first task (lerobot backend).",
    )
    parser.add_argument(
        "--task-b-episodes",
        type=int,
        nargs="+",
        help="Explicit episode indices to use for the second task (lerobot backend).",
    )
    parser.add_argument(
        "--max-demos-per-task",
        type=int,
        default=None,
        help="Maximum demos to load per task (lerobot backend).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_datasets: Dict[str, TaskDataset] = {}
    if args.backend == "hdf5":
        if not args.task_a_file or not args.task_b_file:
            raise ValueError("task-a-file and task-b-file must be provided for the hdf5 backend")
        extra_roots = tuple(args.search_root or [])
        task_a_path = resolve_dataset_file(args.task_a_file, extra_roots)
        task_b_path = resolve_dataset_file(args.task_b_file, extra_roots)
        ensure_allowed_dataset(task_a_path)
        ensure_allowed_dataset(task_b_path)
        label_a = args.task_a_label or Path(args.task_a_file).stem
        label_b = args.task_b_label or Path(args.task_b_file).stem
        if label_a == label_b:
            raise ValueError("Task labels must be distinct")
        task_datasets[label_a] = task_dataset_from_hdf5(task_a_path)
        task_datasets[label_b] = task_dataset_from_hdf5(task_b_path)
    else:
        dataset = LeRobotDataset(args.lerobot_root)
        task_a_index = dataset.resolve_task_index(task_index=args.task_a_index, task_text=args.task_a_text)
        task_b_index = dataset.resolve_task_index(task_index=args.task_b_index, task_text=args.task_b_text)
        label_a = args.task_a_label or dataset.task_text_from_index(task_a_index)
        label_b = args.task_b_label or dataset.task_text_from_index(task_b_index)
        if label_a == label_b:
            raise ValueError("Task labels must be distinct")
        dataset_a, episodes_a = task_dataset_from_lerobot(
            dataset,
            task_a_index,
            episode_indices=args.task_a_episodes,
            max_demos=args.max_demos_per_task,
        )
        dataset_b, episodes_b = task_dataset_from_lerobot(
            dataset,
            task_b_index,
            episode_indices=args.task_b_episodes,
            max_demos=args.max_demos_per_task,
        )
        task_datasets[label_a] = dataset_a
        task_datasets[label_b] = dataset_b
        print(f"[lerobot] {label_a}: episodes {episodes_a}")
        print(f"[lerobot] {label_b}: episodes {episodes_b}")

    manifest = build_manifest(
        task_datasets=task_datasets,
        progress_frac=args.progress_frac,
        pad=args.pad,
        min_abs_delta=args.min_abs_delta,
    )
    save_manifest(manifest, args.output)

    print(f"Contrastive manifest saved to {args.output}")
    task_labels = list(manifest.axis.mean_delta_per_task.keys())
    print(
        "Axis index {} with effect size {:.3f} ({} mean delta={:.3f}, {} mean delta={:.3f})".format(
            manifest.axis.index,
            manifest.axis.effect_size,
            task_labels[0],
            manifest.axis.mean_delta_per_task[task_labels[0]],
            task_labels[1],
            manifest.axis.mean_delta_per_task[task_labels[1]],
        )
    )
    segment_counts = {task: sum(1 for seg in manifest.segments if seg.task == task) for task in task_labels}
    print("Segments: " + ", ".join(f"{task}={count}" for task, count in segment_counts.items()))


if __name__ == "__main__":
    main()
