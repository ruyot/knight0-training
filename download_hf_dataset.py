"""
Download and prepare HuggingFace chess dataset for Phase 2 fine-tuning.

Dataset: angeluriot/chess_games (7.31 GB)
URL: https://huggingface.co/datasets/angeluriot/chess_games
"""

from pathlib import Path
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_hf_dataset():
    """
    Download the HuggingFace chess dataset using git-lfs.
    """
    output_dir = Path("data/hf_chess")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Downloading HuggingFace chess dataset")
    logger.info("=" * 80)
    logger.info(f"Dataset: angeluriot/chess_games (7.31 GB)")
    logger.info(f"Output: {output_dir}")
    logger.info("")
    
    # Clone the dataset repo
    logger.info("Cloning dataset repository...")
    subprocess.run([
        "git", "clone",
        "https://huggingface.co/datasets/angeluriot/chess_games",
        str(output_dir)
    ], check=True)
    
    # Pull large files with git-lfs
    logger.info("\nDownloading dataset file (7.31 GB)...")
    subprocess.run([
        "git", "-C", str(output_dir),
        "lfs", "pull"
    ], check=True)
    
    logger.info("\n✓ Download complete!")
    logger.info(f"Dataset location: {output_dir}/dataset.parquet")
    
    return output_dir / "dataset.parquet"


def convert_parquet_to_pgn_chunks(
    parquet_path: Path,
    output_dir: Path,
    chunk_size: int = 50000
):
    """
    Convert parquet dataset to PGN chunks for parallel processing.
    
    Args:
        parquet_path: Path to dataset.parquet
        output_dir: Output directory for PGN chunks
        chunk_size: Games per chunk
    """
    import pandas as pd
    import io
    import chess.pgn
    
    logger.info(f"\nConverting parquet to PGN chunks...")
    logger.info(f"Chunk size: {chunk_size} games")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load parquet
    logger.info("Loading parquet file...")
    df = pd.read_parquet(parquet_path)
    total_games = len(df)
    
    logger.info(f"Total games: {total_games:,}")
    
    # Split into chunks
    num_chunks = (total_games + chunk_size - 1) // chunk_size
    logger.info(f"Creating {num_chunks} PGN chunks...")
    
    chunk_files = []
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_games)
        
        chunk_df = df.iloc[start_idx:end_idx]
        chunk_path = output_dir / f"hf_chunk_{chunk_idx:03d}.pgn"
        
        # Convert chunk to PGN
        with open(chunk_path, 'w') as f:
            for _, row in chunk_df.iterrows():
                # Create PGN game from row
                game = chess.pgn.Game()
                
                # Set headers if available
                if 'white' in row:
                    game.headers["White"] = str(row['white'])
                if 'black' in row:
                    game.headers["Black"] = str(row['black'])
                if 'white_elo' in row:
                    game.headers["WhiteElo"] = str(row['white_elo'])
                if 'black_elo' in row:
                    game.headers["BlackElo"] = str(row['black_elo'])
                if 'result' in row:
                    game.headers["Result"] = str(row['result'])
                
                # Add moves (assuming 'moves' column has PGN movetext)
                if 'moves' in row and pd.notna(row['moves']):
                    try:
                        moves_str = str(row['moves'])
                        pgn_io = io.StringIO(f"[Event \"?\"]\n\n{moves_str}")
                        temp_game = chess.pgn.read_game(pgn_io)
                        if temp_game:
                            game = temp_game
                    except Exception as e:
                        logger.debug(f"Skipping malformed game: {e}")
                        continue
                
                # Write game
                print(game, file=f, end="\n\n")
        
        chunk_files.append(chunk_path)
        logger.info(f"  Created: {chunk_path.name} ({end_idx - start_idx} games)")
    
    logger.info(f"\n✓ Created {len(chunk_files)} PGN chunks")
    return chunk_files


if __name__ == "__main__":
    # Download dataset
    parquet_path = download_hf_dataset()
    
    # Convert to PGN chunks
    chunk_dir = Path("data/hf_chunks")
    chunks = convert_parquet_to_pgn_chunks(
        parquet_path=parquet_path,
        output_dir=chunk_dir,
        chunk_size=50000
    )
    
    logger.info(f"\n✓ Ready for Phase 2 extraction!")
    logger.info(f"  Chunk directory: {chunk_dir}")
    logger.info(f"  Total chunks: {len(chunks)}")

