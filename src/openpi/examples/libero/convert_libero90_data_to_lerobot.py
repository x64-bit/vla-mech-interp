"""
Utility for converting raw LIBERO-90 (short-horizon) demos stored as HDF5 files
into the LeRobot dataset format used by OpenPi/LeRobot training.

The script mirrors the feature schema published alongside the HF dataset
`physical-intelligence/libero`, so the resulting dataset can be consumed via
`LeRobotDataset` just like the official Libero suites.

Usage example (writes into the source directory by default):
    uv run examples/libero/convert_libero90_data_to_lerobot.py \
        --input_dir data/libero_100_original/libero_90 \
        --repo_id libero_90_lerobot

To place the dataset elsewhere (e.g., HF cache), pass --output_dir.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from tqdm import tqdm
import tyro


def _resize_image(image: np.ndarray, size: int) -> np.ndarray:
    """Resize an RGB image to (size, size) using bicubic filtering."""
    pil_img = Image.fromarray(image)
    resized = pil_img.resize((size, size), resample=Image.BICUBIC)
    return np.asarray(resized, dtype=np.uint8)


def _format_task_name(file_path: Path) -> str:
    """
    Convert the filename (plus demo id) into a readable natural-language task prompt.

    Example:
        KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet_demo.hdf5
        -> "kitchen scene10 put the black bowl in the top drawer of the cabinet"
    """
    stem = file_path.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    # Drop scene prefixes like "KITCHEN_SCENE10" to keep the prompt purely task-focused.
    parts = stem.split("_", maxsplit=2)
    if len(parts) >= 3 and parts[1].startswith("SCENE"):
        stem = stem.split("_", 2)[-1]
    task = stem.replace("_", " ").strip()
    return task.lower()


def _load_state_components(obs_group, index: int) -> np.ndarray:
    """Compose the 8-D state vector expected by OpenPi."""
    ee_pos = np.asarray(obs_group["ee_pos"][index], dtype=np.float32)
    ee_ori = np.asarray(obs_group["ee_ori"][index], dtype=np.float32)
    gripper = np.asarray(obs_group["gripper_states"][index], dtype=np.float32)
    return np.concatenate([ee_pos, ee_ori, gripper], dtype=np.float32)


@dataclass
class Args:
    input_dir: Path = Path("data/libero_100_original/libero_90")
    repo_id: str = "libero_90_lerobot"
    fps: int = 10
    image_size: int = 256
    output_dir: Path | None = None


def main(args: Args) -> None:
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    demo_paths = sorted(input_dir.glob("*.hdf5"))
    if not demo_paths:
        raise ValueError(f"No .hdf5 demos found under '{input_dir}'.")

    if args.output_dir is not None:
        output_root = args.output_dir.expanduser().resolve()
    else:
        default_name = args.repo_id.replace("/", "_")
        output_root = (args.input_dir / default_name).resolve()

    output_root.parent.mkdir(parents=True, exist_ok=True)

    if output_root.exists():
        shutil.rmtree(output_root)

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        robot_type="panda",
        fps=args.fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (args.image_size, args.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (args.image_size, args.image_size, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=4,
    )

    for demo_path in tqdm(demo_paths, desc="Converting LIBERO-90 demos"):
        with h5py.File(demo_path, "r") as handle:
            for demo_key in sorted(handle["data"].keys()):
                demo = handle["data"][demo_key]
                obs = demo["obs"]
                actions = demo["actions"]
                num_steps = actions.shape[0]

                if num_steps == 0:
                    continue

                task_prompt = _format_task_name(demo_path)

                for idx in range(num_steps):
                    dataset.add_frame(
                        {
                            "image": _resize_image(obs["agentview_rgb"][idx], args.image_size),
                            "wrist_image": _resize_image(obs["eye_in_hand_rgb"][idx], args.image_size),
                            "state": _load_state_components(obs, idx),
                            "actions": np.asarray(actions[idx], dtype=np.float32),
                            "task": task_prompt,
                        }
                    )
                dataset.save_episode()

if __name__ == "__main__":
    tyro.cli(main)

