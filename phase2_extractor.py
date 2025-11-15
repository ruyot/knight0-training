"""
Phase 2 Deep Extraction Worker

Processes chess games with:
- Stockfish depth 20-30
- MultiPV 3-5
- Position filtering for tactical/strategic moments
- Higher Elo requirements (2400+)
"""

import modal
from pathlib import Path

app = modal.App("knight0-phase2-extraction")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "python-chess>=1.999",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",  # For parquet support
    )
    .apt_install("stockfish", "git-lfs")
    .add_local_dir("knight0", remote_path="/root/knight0_pkg/knight0")
)

volume = modal.Volume.from_name("knight0-volume", create_if_missing=True)
VOLUME_PATH = "/root/knight0"


@app.function(
    image=image,
    cpu=4,  # 4 CPUs for deep Stockfish analysis
    timeout=7200,  # 2 hours per chunk
    volumes={VOLUME_PATH: volume},
)
def extract_phase2_chunk(chunk_path_str: str, depth: int = 25, multipv: int = 3):
    """
    Extract Phase 2 positions from a single PGN chunk.
    
    Args:
        chunk_path_str: Path to PGN chunk file
        depth: Stockfish analysis depth (20-30)
        multipv: Number of principal variations
    
    Returns:
        Tuple of (shard_path, num_positions, avg_quality)
    """
    import sys
    sys.path.insert(0, "/root/knight0_pkg")
    
    from pathlib import Path
    from knight0.deep_extract import extract_deep_shard
    import pickle
    
    chunk_path = Path(chunk_path_str)
    output_dir = Path(VOLUME_PATH) / "phase2_processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"=" * 80)
    print(f"PHASE 2 EXTRACTION: {chunk_path.name}")
    print(f"Settings: depth={depth}, multiPV={multipv}")
    print("=" * 80)
    
    # Extract with deep analysis
    shard_path = extract_deep_shard(
        pgn_path=chunk_path,
        output_dir=output_dir,
        depth=depth,
        multipv=multipv,
        max_games=None,  # Process all games in chunk
        stockfish_path="/usr/games/stockfish"
    )
    
    # Load to get stats
    with open(shard_path, 'rb') as f:
        positions = pickle.load(f)
    
    # Compute average quality
    avg_quality = sum(p.get('quality', 0.5) for p in positions) / len(positions) if positions else 0.0
    
    print(f"\n✓ Completed: {len(positions)} positions (avg quality: {avg_quality:.3f})")
    
    # Commit volume
    volume.commit()
    
    return (str(shard_path), len(positions), avg_quality)


@app.local_entrypoint()
def main(
    data_source: str = "lichess",  # "lichess" or "hf" (huggingface)
    depth: int = 25,
    multipv: int = 3,
    max_chunks: int = 5,  # Limit chunks for testing
):
    """
    Run Phase 2 extraction on specified data source.
    
    Args:
        data_source: "lichess" (current PGNs) or "hf" (HuggingFace dataset)
        depth: Stockfish depth (20-30)
        multipv: MultiPV count (3-5)
        max_chunks: Maximum chunks to process (for testing)
    """
    print("=" * 80)
    print("PHASE 2: DEEP FINE-TUNING EXTRACTION")
    print("=" * 80)
    print(f"Data source: {data_source}")
    print(f"Depth: {depth}")
    print(f"MultiPV: {multipv}")
    print(f"Max chunks: {max_chunks}")
    print("")
    
    if data_source == "hf":
        # Use HuggingFace dataset chunks
        chunk_dir = Path("data/hf_chunks")
        if not chunk_dir.exists():
            print("ERROR: HuggingFace chunks not found!")
            print("Run: python download_hf_dataset.py")
            return
        
        chunk_files = sorted(chunk_dir.glob("hf_chunk_*.pgn"))[:max_chunks]
    else:
        # Use existing Lichess/TCEC PGNs
        data_dir = Path("data")
        all_pgns = sorted(data_dir.glob("*.pgn"))
        
        # Take highest-quality PGNs for Phase 2
        chunk_files = [
            p for p in all_pgns
            if "elite" in p.name.lower() or "TCEC" in p.name
        ][:max_chunks]
    
    if not chunk_files:
        print("ERROR: No data files found!")
        return
    
    print(f"Processing {len(chunk_files)} chunks:")
    for chunk in chunk_files:
        print(f"  - {chunk.name}")
    print("")
    
    # Process in parallel (each chunk on separate worker)
    print("🔥 Launching parallel deep extraction workers...")
    
    results = []
    for chunk in chunk_files:
        result = extract_phase2_chunk.remote(str(chunk), depth, multipv)
        results.append(result)
    
    # Summary
    print("\n" + "=" * 80)
    print("PHASE 2 EXTRACTION COMPLETE!")
    print("=" * 80)
    
    total_positions = 0
    for shard_path, num_positions, avg_quality in results:
        shard_name = Path(shard_path).name
        print(f"  {shard_name}: {num_positions:,} positions (quality: {avg_quality:.3f})")
        total_positions += num_positions
    
    print(f"\nTotal Phase 2 positions: {total_positions:,}")
    print(f"Average depth: {depth}")
    print(f"\n✓ Ready for Phase 2 fine-tuning!")

