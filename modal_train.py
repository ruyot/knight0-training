"""
Modal entrypoint for knight0 training.

This script sets up a Modal app that runs training on a GPU worker.

Usage:
    modal run modal_train.py                    # Run training with default settings
    modal run modal_train.py --config large     # Use large model
    modal run modal_train.py --epochs 100       # Train for 100 epochs
"""

import modal

# Create Modal app
app = modal.App("knight0-training")

# Define the Docker image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "torchaudio>=2.0.0",
        "python-chess>=1.999",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "datasets>=2.14.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "onnxscript>=0.1.0",  # Required for PyTorch 2.x ONNX export
        "requests>=2.31.0",
    )
    .apt_install("stockfish")
    .add_local_dir("knight0", remote_path="/root/knight0_pkg/knight0")
    .add_local_dir("data", remote_path="/root/knight0_pkg/data")
)

# Create a persistent volume for storing data, checkpoints, and models
volume = modal.Volume.from_name("knight0-volume", create_if_missing=True)

# Volume mount path
VOLUME_PATH = "/root/knight0"


@app.function(
    image=image,
    gpu="A10G",  # Use A10G GPU (or "A100", "T4", etc.)
    timeout=3600 * 4,  # 4 hour timeout
    volumes={VOLUME_PATH: volume},
)
def train_on_modal(
    model_config: str = "medium",
    batch_size: int = 256,
    num_epochs: int = 50,
    learning_rate: float = 1e-3,
    use_test_data: bool = False,
    export_onnx: bool = True,
):
    """
    Run training on Modal GPU worker.
    
    Args:
        model_config: Model size ("small", "medium", "large")
        batch_size: Training batch size
        num_epochs: Number of epochs
        learning_rate: Initial learning rate
        use_test_data: If True, use small test dataset for quick debugging
        export_onnx: If True, export ONNX model after training
    """
    import sys
    sys.path.insert(0, "/root/knight0_pkg")
    
    from knight0.train_loop import train_main
    
    print("Starting knight0 training on Modal...")
    print(f"Volume path: {VOLUME_PATH}")
    
    # Run training
    train_main(
        root_dir=VOLUME_PATH,
        model_config=model_config,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        use_test_data=use_test_data,
        export_onnx_after=export_onnx,
    )
    
    # Commit volume changes
    volume.commit()
    
    print("\nTraining completed! Model saved to volume.")
    print(f"  - Checkpoints: {VOLUME_PATH}/checkpoints/")
    print(f"  - ONNX model: {VOLUME_PATH}/knight0_model.onnx")


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def download_onnx_model():
    """
    Download the trained ONNX model from the volume.
    
    Returns:
        bytes: The ONNX model file contents
    """
    from pathlib import Path
    
    source_path = Path(VOLUME_PATH) / "knight0_model.onnx"
    
    if not source_path.exists():
        print(f"Error: ONNX model not found at {source_path}")
        print("Please run training first.")
        return None
    
    # Read the file from the volume
    with open(source_path, 'rb') as f:
        model_data = f.read()
    
    print(f"Read ONNX model from volume ({len(model_data) / 1e6:.2f} MB)")
    return model_data


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def list_volume_contents():
    """
    List the contents of the Modal volume.
    """
    import os
    from pathlib import Path
    
    root = Path(VOLUME_PATH)
    
    print(f"\nContents of {VOLUME_PATH}:")
    print("="*80)
    
    if not root.exists():
        print("Volume is empty or doesn't exist yet.")
        return
    
    for item in sorted(root.rglob("*")):
        if item.is_file():
            size_mb = item.stat().st_size / 1e6
            rel_path = item.relative_to(root)
            print(f"  {rel_path} ({size_mb:.2f} MB)")
    
    print("="*80)


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def clear_volume():
    """
    Clear all contents from the volume (use with caution!).
    """
    import shutil
    from pathlib import Path
    
    root = Path(VOLUME_PATH)
    
    if root.exists():
        for item in root.iterdir():
            if item.is_file():
                item.unlink()
                print(f"Deleted file: {item.name}")
            elif item.is_dir():
                shutil.rmtree(item)
                print(f"Deleted directory: {item.name}")
        
        volume.commit()
        print("\nVolume cleared successfully.")
    else:
        print("Volume is already empty.")


@app.local_entrypoint()
def main(
    config: str = "medium",
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    test: bool = False,
    action: str = "train",
):
    """
    Local entrypoint for the Modal app.
    
    Args:
        config: Model configuration ("small", "medium", "large")
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        test: Use test data for quick debugging
        action: Action to perform ("train", "download", "list", "clear")
    """
    if action == "train":
        print("Starting training job on Modal...")
        train_on_modal.remote(
            model_config=config,
            batch_size=batch_size,
            num_epochs=epochs,
            learning_rate=lr,
            use_test_data=test,
            export_onnx=True,
        )
    
    elif action == "download":
        print("Downloading ONNX model from Modal volume...")
        model_data = download_onnx_model.remote()
        if model_data:
            output_path = "knight0_model.onnx"
            with open(output_path, 'wb') as f:
                f.write(model_data)
            print(f"✓ Saved ONNX model to {output_path} ({len(model_data) / 1e6:.2f} MB)")
    
    elif action == "list":
        print("Listing Modal volume contents...")
        list_volume_contents.remote()
    
    elif action == "clear":
        response = input("Are you sure you want to clear the volume? (yes/no): ")
        if response.lower() == "yes":
            clear_volume.remote()
        else:
            print("Cancelled.")
    
    else:
        print(f"Unknown action: {action}")
        print("Available actions: train, download, list, clear")


# For interactive debugging
if __name__ == "__main__":
    print("knight0 Modal Training Script")
    print("\nUsage examples:")
    print("  modal run modal_train.py")
    print("  modal run modal_train.py --config large --epochs 100")
    print("  modal run modal_train.py --test true  # Quick test with small dataset")
    print("  modal run modal_train.py --action download")
    print("  modal run modal_train.py --action list")

