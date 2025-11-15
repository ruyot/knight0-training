"""
Modal parallel training script for knight0.

This version processes PGN files in PARALLEL for 10x faster data extraction!

Usage:
    modal run modal_train_parallel.py
"""

import modal
from pathlib import Path

# Create Modal app
app = modal.App("knight0-parallel-training")

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
        "onnxscript>=0.1.0",
        "requests>=2.31.0",
    )
    .apt_install("stockfish")
    .add_local_dir("knight0", remote_path="/root/knight0_pkg/knight0")
    .add_local_dir("data", remote_path="/root/knight0_pkg/data")
)

# Create a persistent volume
volume = modal.Volume.from_name("knight0-volume", create_if_missing=True)
VOLUME_PATH = "/root/knight0"


@app.function(
    image=image,
    cpu=2,  # Each worker gets 2 CPUs for Stockfish
    timeout=3600,  # 1 hour per PGN
    volumes={VOLUME_PATH: volume},
)
def process_single_pgn(pgn_filename: str, max_games: int = None):
    """
    Process a single PGN file on one worker.
    
    Args:
        pgn_filename: Name of the PGN file (not full path)
        max_games: Optional limit on games to process
    
    Returns:
        Tuple of (shard_path, num_positions)
    """
    import sys
    sys.path.insert(0, "/root/knight0_pkg")
    
    from pathlib import Path
    from knight0.extract_positions import extract_single_pgn_shard
    
    # Paths
    pgn_path = Path("/root/knight0_pkg/data") / pgn_filename
    output_dir = Path(VOLUME_PATH) / "processed"
    
    # Extract positions for this PGN
    shard_path = extract_single_pgn_shard(
        pgn_path=pgn_path,
        output_dir=output_dir,
        stockfish_path="/usr/games/stockfish",
        max_games=max_games
    )
    
    # Load to get count
    import pickle
    with open(shard_path, 'rb') as f:
        positions = pickle.load(f)
    
    # Commit volume after each shard
    volume.commit()
    
    return (str(shard_path), len(positions))


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600 * 4,  # 4 hours
    volumes={VOLUME_PATH: volume},
)
def train_with_parallel_extraction(
    model_config: str = "medium",
    batch_size: int = 512,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
):
    """
    Extract data in PARALLEL, then train.
    
    Args:
        model_config: Model size
        batch_size: Training batch size
        num_epochs: Number of epochs
        learning_rate: Initial learning rate
    """
    import sys
    sys.path.insert(0, "/root/knight0_pkg")
    
    import pickle
    from pathlib import Path
    from knight0.train_loop import train_main
    
    print("=" * 80)
    print("PARALLEL EXTRACTION + TRAINING PIPELINE")
    print("=" * 80)
    
    # Check if training data already exists
    training_data_path = Path(VOLUME_PATH) / "training_data.pkl"
    
    if not training_data_path.exists():
        print("\n🚀 PHASE 1: PARALLEL DATA EXTRACTION")
        print("-" * 80)
        
        # Find all PGN files
        data_dir = Path("/root/knight0_pkg/data")
        pgn_files = sorted([f.name for f in data_dir.glob("*.pgn")])
        
        print(f"Found {len(pgn_files)} PGN files:")
        for pgn in pgn_files:
            print(f"  - {pgn}")
        
        print(f"\n🔥 Processing {len(pgn_files)} PGNs in PARALLEL...")
        print("(Each PGN gets its own worker + CPU for Stockfish)")
        
        # Process all PGNs in parallel using Modal's .map()
        # .map() passes each item as separate arguments to the function
        results = list(process_single_pgn.starmap(
            [(pgn, None) for pgn in pgn_files]
        ))
        
        print("\n✓ All workers completed!")
        print("\nResults:")
        total_positions = 0
        for shard_path, num_positions in results:
            shard_name = Path(shard_path).name
            print(f"  {shard_name}: {num_positions:,} positions")
            total_positions += num_positions
        
        print(f"\nTotal positions: {total_positions:,}")
        
        # Merge all shards
        print("\n📦 Merging shards into training_data.pkl...")
        processed_dir = Path(VOLUME_PATH) / "processed"
        all_positions = []
        
        for shard_file in sorted(processed_dir.glob("shard_*.pkl")):
            with open(shard_file, 'rb') as f:
                shard_positions = pickle.load(f)
                all_positions.extend(shard_positions)
                print(f"  Loaded {len(shard_positions):,} from {shard_file.name}")
        
        # Save merged dataset
        with open(training_data_path, 'wb') as f:
            pickle.dump(all_positions, f)
        
        volume.commit()
        
        print(f"\n✓ Saved {len(all_positions):,} positions to training_data.pkl")
    else:
        print("\n✓ Training data already exists, skipping extraction")
    
    # Now train
    print("\n" + "=" * 80)
    print("🎯 PHASE 2: TRAINING")
    print("=" * 80 + "\n")
    
    train_main(
        root_dir=VOLUME_PATH,
        model_config=model_config,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        use_test_data=False,
        export_onnx_after=True,
    )
    
    print("\n" + "=" * 80)
    print("✓ PIPELINE COMPLETE!")
    print("=" * 80)


@app.local_entrypoint()
def main(
    config: str = "medium",
    epochs: int = 100,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
):
    """
    Main entrypoint for parallel training.
    
    Args:
        config: Model configuration (small/medium/large)
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
    """
    print("Starting PARALLEL knight0 training on Modal...")
    train_with_parallel_extraction.remote(
        model_config=config,
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=learning_rate,
    )

