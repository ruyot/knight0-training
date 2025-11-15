"""
Extraction Worker 1: Process first 5 PGNs
"""
import modal
from pathlib import Path

app = modal.App("knight0-extract-worker-1")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("python-chess>=1.999", "numpy>=1.24.0", "tqdm>=4.65.0")
    .apt_install("stockfish")
    .add_local_dir("knight0", remote_path="/root/knight0_pkg/knight0")
    .add_local_dir("data", remote_path="/root/knight0_pkg/data")
)

volume = modal.Volume.from_name("knight0-volume", create_if_missing=True)
VOLUME_PATH = "/root/knight0"


@app.function(
    image=image,
    cpu=4,  # 4 CPUs for faster Stockfish
    timeout=7200,  # 2 hours
    volumes={VOLUME_PATH: volume},
)
def extract_pgns_batch_1():
    """Extract positions from PGNs 1-5"""
    import sys
    sys.path.insert(0, "/root/knight0_pkg")
    
    from pathlib import Path
    from knight0.extract_positions import extract_single_pgn_shard
    
    data_dir = Path("/root/knight0_pkg/data")
    output_dir = Path(VOLUME_PATH) / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all PGNs and take first 5
    all_pgns = sorted([f for f in data_dir.glob("*.pgn")])
    my_pgns = all_pgns[0:5]
    
    print(f"Worker 1 processing {len(my_pgns)} PGNs:")
    for pgn in my_pgns:
        print(f"  - {pgn.name}")
    
    results = []
    for pgn_path in my_pgns:
        print(f"\n{'='*80}")
        print(f"Processing: {pgn_path.name}")
        print('='*80)
        
        shard_path = extract_single_pgn_shard(
            pgn_path=pgn_path,
            output_dir=output_dir,
            stockfish_path="/usr/games/stockfish",
            max_games=None
        )
        
        # Count positions
        import pickle
        with open(shard_path, 'rb') as f:
            positions = pickle.load(f)
        
        results.append((pgn_path.name, len(positions)))
        print(f"✓ Saved {len(positions):,} positions")
        
        # Commit after each PGN
        volume.commit()
    
    print(f"\n{'='*80}")
    print("WORKER 1 COMPLETE!")
    print('='*80)
    for name, count in results:
        print(f"  {name}: {count:,} positions")
    
    total = sum(count for _, count in results)
    print(f"\nTotal: {total:,} positions")
    
    return results


@app.local_entrypoint()
def main():
    print("Starting Extraction Worker 1 (PGNs 1-5)...")
    results = extract_pgns_batch_1.remote()
    print("\n✓ Worker 1 finished!")
    return results

