"""
Download processed Stockfish-labeled data from Modal volume.
Use the shards that were already created!
"""

import modal
import pickle
from pathlib import Path

app = modal.App("download-shards")
volume = modal.Volume.from_name("knight0-volume")

@app.function(volumes={"/data": volume}, timeout=600)
def get_processed_data():
    """Download all processed shards from Modal."""
    import os
    
    processed_dir = Path("/data/processed")
    
    if not processed_dir.exists():
        print("No processed directory found!")
        return []
    
    # Get all shard files
    shard_files = list(processed_dir.glob("shard_*.pkl"))
    print(f"Found {len(shard_files)} shards")
    
    all_positions = []
    for shard_file in sorted(shard_files):
        with open(shard_file, 'rb') as f:
            positions = pickle.load(f)
            all_positions.extend(positions)
            print(f"  {shard_file.name}: {len(positions):,} positions")
    
    print(f"\nTotal: {len(all_positions):,} positions")
    return all_positions

@app.local_entrypoint()
def main():
    print("="*80)
    print("DOWNLOADING PROCESSED DATA FROM MODAL")
    print("="*80)
    print()
    
    positions = get_processed_data.remote()
    
    if not positions:
        print("No data found! Need to run extraction first.")
        return
    
    # Sample 10k for tiny model training
    sample_size = 10000
    if len(positions) > sample_size:
        print(f"\nSampling {sample_size:,} positions for tiny model...")
        import random
        random.shuffle(positions)
        positions = positions[:sample_size]
    
    # Save
    output_file = Path("training_data.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(positions, f)
    
    print(f"\n✓ Saved {len(positions):,} positions to {output_file}")
    print(f"  File size: {output_file.stat().st_size / (1024*1024):.1f} MB")
    print()
    print("Ready to train! Run:")
    print("  python3 train_tiny_simple.py")

if __name__ == "__main__":
    main()

