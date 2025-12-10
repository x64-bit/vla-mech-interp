"""
Test extract FFN vectors using the existing extract_ffn_vectors.py script.
First converts JAX checkpoint to PyTorch if needed.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import numpy as np
from pathlib import Path
import torch
import random
from PIL import Image
import sys
import subprocess

# Add the scripts directory to the path so we can import extract_ffn_vectors
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

# Import the extract_ffn_vectors script
from extract_ffn_vectors import Args, main
from openpi.shared import download

# Configuration
CONFIG_NAME = "pi05_libero"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_OUTPUT_DIR = Path("./test_outputs")
TEST_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def convert_checkpoint_if_needed(jax_checkpoint_dir: Path, pytorch_output_dir: Path):
    """Convert JAX checkpoint to PyTorch if PyTorch version doesn't exist."""
    if pytorch_output_dir.exists() and (pytorch_output_dir / "model.safetensors").exists():
        print(f"[✓] PyTorch checkpoint already exists: {pytorch_output_dir}")
        # Still need to ensure assets are copied
        if not (pytorch_output_dir / "assets").exists() and (jax_checkpoint_dir / "assets").exists():
            print(f"[*] Copying assets directory...")
            import shutil
            shutil.copytree(jax_checkpoint_dir / "assets", pytorch_output_dir / "assets")
        return pytorch_output_dir
    
    print(f"[*] Converting JAX checkpoint to PyTorch...")
    print(f"    JAX checkpoint: {jax_checkpoint_dir}")
    print(f"    PyTorch output: {pytorch_output_dir}")
    
    # Run the conversion script
    conversion_script = Path(__file__).parent / "examples" / "convert_jax_model_to_pytorch.py"
    
    if not conversion_script.exists():
        raise FileNotFoundError(
            f"Conversion script not found: {conversion_script}\n"
            f"Expected at: {conversion_script.absolute()}"
        )
    
    cmd = [
        sys.executable,
        str(conversion_script),
        "--checkpoint_dir", str(jax_checkpoint_dir),
        "--config_name", CONFIG_NAME,
        "--output_path", str(pytorch_output_dir),
        "--precision", "bfloat16",
    ]
    
    print(f"[*] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(
            f"Conversion failed:\n{result.stderr}\n{result.stdout}"
        )
    
    print(f"[✓] Conversion complete!")
    
    # Manually copy assets directory if conversion script didn't (it has a bug)
    if (jax_checkpoint_dir / "assets").exists() and not (pytorch_output_dir / "assets").exists():
        print(f"[*] Copying assets directory from JAX checkpoint...")
        import shutil
        shutil.copytree(jax_checkpoint_dir / "assets", pytorch_output_dir / "assets")
        print(f"[✓] Assets copied!")
    return pytorch_output_dir


def test_extract_ffn_vectors():
    """Test extracting FFN vectors using the existing extract_ffn_vectors.py script."""
    print("\n" + "="*80)
    print("TEST: Extract FFN Vectors (using extract_ffn_vectors.py)")
    print("="*80)
    
    # Download JAX checkpoint
    checkpoint_path = f"gs://openpi-assets/checkpoints/{CONFIG_NAME}"
    print(f"[*] Downloading checkpoint: {checkpoint_path}")
    checkpoint_dir = download.maybe_download(checkpoint_path)
    checkpoint_dir = Path(checkpoint_dir)
    
    print(f"[*] Checkpoint directory: {checkpoint_dir}")
    
    # Check if it's JAX (has params directory)
    if (checkpoint_dir / "params").exists():
        # Convert to PyTorch
        pytorch_checkpoint_dir = checkpoint_dir.parent / f"{CONFIG_NAME}_pytorch"
        pytorch_checkpoint_dir = convert_checkpoint_if_needed(checkpoint_dir, pytorch_checkpoint_dir)
        checkpoint_dir = pytorch_checkpoint_dir
    
    # Verify PyTorch checkpoint exists
    model_safetensors = checkpoint_dir / "model.safetensors"
    if not model_safetensors.exists():
        raise FileNotFoundError(
            f"PyTorch model file not found: {model_safetensors}\n"
            f"Checkpoint directory: {checkpoint_dir}\n"
            f"Contents: {list(checkpoint_dir.iterdir())}"
        )
    print(f"[✓] Found PyTorch model file: {model_safetensors}")
    
    # Load actual image files
    print("[*] Loading test images...")
    image_path = Path("data/libero_lerobot/physical-intelligence/libero/images/image/episode_000194/frame_000000.png")
    wrist_image_path = Path("data/libero_lerobot/physical-intelligence/libero/images/wrist_image/episode_000194/frame_000000.png")
    
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not wrist_image_path.exists():
        raise FileNotFoundError(f"Wrist image not found: {wrist_image_path}")
    
    # Load and convert images to numpy arrays
    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    wrist_image = np.array(Image.open(wrist_image_path).convert("RGB"), dtype=np.uint8)
    
    print(f"[*] Loaded image: {image.shape}, wrist_image: {wrist_image.shape}")
    
    # Create observation dictionary
    dummy_state = np.random.randn(7).astype(np.float32)
    
    # Random prompts for testing
    random_prompts = [
        "pick up the object",
        "place the item on the table",
        "move the cup to the left",
        "grasp the bowl",
        "put down the plate",
        "lift the container",
        "transfer the object",
        "pick and place the item",
    ]
    random_prompt = random.choice(random_prompts)
    
    obs = {
        "observation/image": image,
        "observation/state": dummy_state,
        "prompt": random_prompt,
        "observation/wrist_image": wrist_image,
    }
    
    print(f"[*] Using prompt: '{random_prompt}'")
    
    # Save observation to npz file (this is what extract_ffn_vectors.py expects)
    obs_npz_path = TEST_OUTPUT_DIR / "test_observation.npz"
    np.savez(obs_npz_path, **obs)
    print(f"[*] Saved observation to {obs_npz_path}")
    
    # Create Args object for extract_ffn_vectors.py
    args = Args(
        config_name=CONFIG_NAME,
        checkpoint_dir=checkpoint_dir,
        observations_glob=str(obs_npz_path),
        output_path=TEST_OUTPUT_DIR / "test_ffn_projections.json",
        top_k_per_layer=10,
        max_batches=1,
        device=DEVICE,
        store_value_vectors=True,
        min_activation=0.0,
    )
    
    # Run the extract_ffn_vectors.py main function
    print("[*] Running extract_ffn_vectors.py...")
    main(args)
    
    # Verify output
    output_path = args.output_path
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[✓] Output file contains {len(data)} neuron projections")
        if data:
            first_proj = data[0]
            print(f"  Example projection:")
            print(f"    layer_index: {first_proj['layer_index']}")
            print(f"    neuron_index: {first_proj['neuron_index']}")
            print(f"    max_activation: {first_proj['max_activation']:.4f}")
            print(f"    top_tokens: {first_proj['top_tokens'][:3]}")
        print("[✓] FFN extraction test PASSED")
    else:
        print("[✗] Output file not found")
    
    return output_path


if __name__ == "__main__":
    test_extract_ffn_vectors()