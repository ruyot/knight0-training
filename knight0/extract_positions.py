"""
Position extraction and Stockfish labeling.

This module:
1. Parses PGN files
2. Extracts positions from games
3. Labels positions using Stockfish analysis
4. Saves labeled positions to disk
"""

import chess
import chess.pgn
import chess.engine
import pickle
import gzip
import bz2
import signal
from pathlib import Path
from typing import List, Dict, Optional, Iterator
from tqdm import tqdm
import logging

from .config import STOCKFISH_CONFIG, DATA_FILTERS
from .utils import normalize_score

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    """Raised when a timeout occurs"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Operation timed out")


class PositionExtractor:
    """
    Extracts and labels positions from PGN files using Stockfish.
    """
    
    def __init__(
        self,
        stockfish_path: str = "stockfish",
        depth: int = STOCKFISH_CONFIG["depth"],
        time_limit: float = STOCKFISH_CONFIG["time_limit"],
        sample_rate: int = STOCKFISH_CONFIG["sample_rate"],
        min_move: int = STOCKFISH_CONFIG["min_move"],
        max_move: int = STOCKFISH_CONFIG["max_move"],
    ):
        """
        Args:
            stockfish_path: Path to Stockfish binary
            depth: Analysis depth
            time_limit: Time limit per analysis (seconds)
            sample_rate: Label every Nth move
            min_move: Start labeling from this move number
            max_move: Stop labeling after this move number
        """
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.time_limit = time_limit
        self.sample_rate = sample_rate
        self.min_move = min_move
        self.max_move = max_move
        self.engine = None
    
    def __enter__(self):
        """Context manager entry: open Stockfish engine."""
        print(f"DEBUG: __enter__ called, starting Stockfish at {self.stockfish_path}...", flush=True)
        logger.info(f"Starting Stockfish engine: {self.stockfish_path}")
        print("DEBUG: About to call popen_uci...", flush=True)
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        print("DEBUG: Stockfish started successfully!", flush=True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: close Stockfish engine."""
        if self.engine:
            self.engine.quit()
            logger.info("Closed Stockfish engine")
    
    def open_pgn_file(self, pgn_path: Path):
        """
        Open a PGN file, handling compression automatically.
        
        Args:
            pgn_path: Path to PGN file (may be .gz or .bz2)
            
        Returns:
            File handle
        """
        if pgn_path.suffix == '.gz':
            return gzip.open(pgn_path, 'rt', encoding='utf-8', errors='ignore')
        elif pgn_path.suffix == '.bz2':
            return bz2.open(pgn_path, 'rt', encoding='utf-8', errors='ignore')
        else:
            return open(pgn_path, 'r', encoding='utf-8', errors='ignore')
    
    def should_process_game(self, game: chess.pgn.Game) -> bool:
        """
        Check if a game meets quality criteria.
        
        Args:
            game: chess.pgn.Game object
            
        Returns:
            True if game should be processed
        """
        headers = game.headers
        
        # Check if game has result
        result = headers.get("Result", "*")
        if result == "*":
            return False
        
        # Check minimum Elo if available
        try:
            white_elo = int(headers.get("WhiteElo", 0))
            black_elo = int(headers.get("BlackElo", 0))
            min_elo = DATA_FILTERS["min_elo"]
            
            if white_elo > 0 and black_elo > 0:
                if white_elo < min_elo or black_elo < min_elo:
                    return False
        except ValueError:
            pass  # Elo not available or invalid
        
        # Check game length
        mainline = list(game.mainline_moves())
        num_moves = len(mainline)
        
        if num_moves < DATA_FILTERS["min_moves"]:
            return False
        if num_moves > DATA_FILTERS["max_moves"]:
            return False
        
        return True
    
    def analyze_position(self, board: chess.Board) -> Optional[Dict]:
        """
        Analyze a position with Stockfish.
        
        Args:
            board: chess.Board object
            
        Returns:
            Dict with 'move' (UCI string) and 'score' (centipawns), or None if analysis fails
        """
        if not self.engine:
            raise RuntimeError("Engine not initialized. Use context manager.")
        
        try:
            # Run analysis
            limit = chess.engine.Limit(depth=self.depth, time=self.time_limit)
            result = self.engine.analyse(board, limit)
            
            # Extract best move
            if "pv" not in result or len(result["pv"]) == 0:
                return None
            
            best_move = result["pv"][0]
            
            # Extract score
            score = result.get("score")
            if score is None:
                return None
            
            # Convert score to centipawns (from white's perspective)
            score_cp = score.white().score(mate_score=10000)
            if score_cp is None:
                return None
            
            return {
                "move": best_move.uci(),
                "score": score_cp
            }
        
        except Exception as e:
            logger.warning(f"Analysis failed: {e}")
            return None
    
    def extract_from_game(self, game: chess.pgn.Game) -> List[Dict]:
        """
        Extract labeled positions from a single game.
        
        Args:
            game: chess.pgn.Game object
            
        Returns:
            List of dicts with 'fen', 'move', 'value'
        """
        positions = []
        board = game.board()
        
        for move_num, move in enumerate(game.mainline_moves(), start=1):
            # Check if we should sample this position
            if move_num < self.min_move or move_num > self.max_move:
                board.push(move)
                continue
            
            if move_num % self.sample_rate != 0:
                board.push(move)
                continue
            
            # Analyze position
            analysis = self.analyze_position(board)
            
            if analysis is not None:
                positions.append({
                    "fen": board.fen(),
                    "move": analysis["move"],
                    "value": normalize_score(analysis["score"])
                })
            
            board.push(move)
        
        return positions
    
    def process_pgn_file(
        self,
        pgn_path: Path,
        max_games: Optional[int] = None
    ) -> List[Dict]:
        """
        Process a single PGN file and extract labeled positions.
        
        Args:
            pgn_path: Path to PGN file
            max_games: Optional limit on number of games to process
            
        Returns:
            List of labeled positions
        """
        print(f"DEBUG: process_pgn_file called for {pgn_path}", flush=True)
        logger.info(f"Processing PGN file: {pgn_path}")
        print(f"DEBUG: Logger done, opening PGN file...", flush=True)
        
        all_positions = []
        games_processed = 0
        games_skipped = 0
        
        print(f"DEBUG: About to call open_pgn_file...", flush=True)
        with self.open_pgn_file(pgn_path) as pgn_file:
            print(f"DEBUG: PGN file opened, entering game loop...", flush=True)
            while True:
                if max_games and games_processed >= max_games:
                    break
                
                try:
                    if games_processed == 1:
                        print(f"DEBUG: Top of loop, about to read game 2...", flush=True)
                    if games_processed == 0:
                        print(f"DEBUG: Reading first game...", flush=True)
                    
                    # Try to read game with 10-second timeout
                    game = None
                    try:
                        signal.signal(signal.SIGALRM, timeout_handler)
                        signal.alarm(10)  # 10-second timeout
                        game = chess.pgn.read_game(pgn_file)
                        signal.alarm(0)  # Cancel alarm
                    except TimeoutError:
                        signal.alarm(0)  # Cancel alarm
                        logger.warning(f"Timeout reading game after {games_processed} games - skipping")
                        print(f"DEBUG: Timeout on game read, skipping...", flush=True)
                        # Try to recover by reading rest of game
                        continue
                    except Exception as parse_error:
                        signal.alarm(0)  # Cancel alarm
                        logger.warning(f"Error parsing game (skipping): {parse_error}")
                        print(f"DEBUG: Parse error, continuing...", flush=True)
                        continue
                    
                    if games_processed == 0:
                        print(f"DEBUG: First game read: {game is not None}", flush=True)
                    if games_processed == 1:
                        print(f"DEBUG: Second game read: {game is not None}", flush=True)
                    if game is None:
                        break
                    
                    if games_processed == 1:
                        print(f"DEBUG: Checking if second game should be processed...", flush=True)
                    if not self.should_process_game(game):
                        games_skipped += 1
                        if games_processed == 0 and games_skipped < 5:
                            print(f"DEBUG: Game {games_skipped} skipped (filters)", flush=True)
                        if games_processed == 1:
                            print(f"DEBUG: Second game skipped by filters", flush=True)
                        continue
                    
                    if games_processed == 0:
                        print(f"DEBUG: Extracting from first game...", flush=True)
                    if games_processed == 1:
                        print(f"DEBUG: Extracting from second game...", flush=True)
                    positions = self.extract_from_game(game)
                    if games_processed == 1:
                        print(f"DEBUG: Extracted {len(positions)} from second game", flush=True)
                    if games_processed == 0:
                        print(f"DEBUG: Extracted {len(positions)} positions from first game", flush=True)
                    all_positions.extend(positions)
                    games_processed += 1
                    if games_processed == 1:
                        print(f"DEBUG: First game COMPLETE! Total positions so far: {len(all_positions)}", flush=True)
                        print(f"DEBUG: Looping back to read game 2...", flush=True)
                    
                    # More frequent logging!
                    if games_processed % 50 == 0:
                        avg_per_game = len(all_positions) / games_processed if games_processed > 0 else 0
                        logger.info(f"  ✓ Processed {games_processed} games, "
                                  f"extracted {len(all_positions)} positions "
                                  f"(avg: {avg_per_game:.1f} pos/game)")
                
                except Exception as e:
                    logger.warning(f"Error processing game: {e}")
                    continue
        
        avg_per_game = len(all_positions) / games_processed if games_processed > 0 else 0
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"✓ PGN COMPLETE: {pgn_path.name}")
        logger.info(f"  Games processed: {games_processed:,}")
        logger.info(f"  Games skipped:   {games_skipped:,}")
        logger.info(f"  Positions:       {len(all_positions):,}")
        logger.info(f"  Avg pos/game:    {avg_per_game:.1f}")
        logger.info(f"{'='*80}")
        
        return all_positions
    
    def process_multiple_pgns(
        self,
        pgn_paths: List[Path],
        output_path: Path,
        max_games_per_file: Optional[int] = None
    ):
        """
        Process multiple PGN files with incremental saving (resumable).
        
        Each PGN is saved as a separate shard. If a shard exists, skip that PGN.
        At the end, merge all shards into the final output file.
        
        Args:
            pgn_paths: List of PGN file paths
            output_path: Path to save final merged pickle file
            max_games_per_file: Optional limit on games per file
        """
        logger.info(f"Processing {len(pgn_paths)} PGN files with incremental saving")
        
        # Create processed/ directory for shards
        shards_dir = output_path.parent / "processed"
        shards_dir.mkdir(exist_ok=True)
        logger.info(f"Shards directory: {shards_dir}")
        
        # Process each PGN (or skip if shard exists)
        processed_shards = []
        
        for pgn_path in tqdm(pgn_paths, desc="Processing PGN files"):
            # Generate shard name from PGN filename
            shard_name = f"shard_{pgn_path.stem}.pkl"
            shard_path = shards_dir / shard_name
            
            if shard_path.exists():
                logger.info(f"✓ Shard exists for {pgn_path.name}, skipping extraction")
                processed_shards.append(shard_path)
                continue
            
            # Extract positions for this PGN
            logger.info(f"Processing {pgn_path.name}...")
            positions = self.process_pgn_file(pgn_path, max_games_per_file)
            
            # Save shard immediately
            logger.info(f"Saving shard to {shard_path} ({len(positions)} positions)")
            with open(shard_path, 'wb') as f:
                pickle.dump(positions, f)
            
            processed_shards.append(shard_path)
            logger.info(f"✓ Saved shard for {pgn_path.name}")
        
        # Merge all shards into final output
        logger.info(f"\nMerging {len(processed_shards)} shards into {output_path}")
        all_positions = []
        
        for shard_path in processed_shards:
            with open(shard_path, 'rb') as f:
                shard_positions = pickle.load(f)
                all_positions.extend(shard_positions)
                logger.info(f"  Loaded {len(shard_positions)} positions from {shard_path.name}")
        
        logger.info(f"Total positions: {len(all_positions)}")
        
        # Save merged dataset
        logger.info(f"Saving merged dataset to {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(all_positions, f)
        
        logger.info(f"✓ Saved {len(all_positions)} positions to {output_path}")


def create_training_data(
    root_dir: str,
    pgn_paths: Optional[List[Path]] = None,
    output_filename: str = "training_data.pkl",
    stockfish_path: str = "/usr/games/stockfish",  # Default Debian path
    max_games_per_file: Optional[int] = None,
    use_test_data: bool = False,
) -> Path:
    """
    High-level function to create training data from PGN files.
    
    Args:
        root_dir: Root directory (e.g., /root/knight0)
        pgn_paths: List of PGN files to process (if None, auto-discover)
        output_filename: Name of output pickle file
        stockfish_path: Path to Stockfish binary
        max_games_per_file: Optional limit on games per file
        use_test_data: If True, create small test dataset
        
    Returns:
        Path to output pickle file
    """
    from .data_sources import DataSourceManager
    
    root_path = Path(root_dir)
    output_path = root_path / output_filename
    
    # If output already exists, return it
    if output_path.exists():
        logger.info(f"Training data already exists: {output_path}")
        return output_path
    
    # Get PGN files
    if pgn_paths is None:
        manager = DataSourceManager(root_dir)
        
        if use_test_data:
            # Create and use test data
            test_pgn = manager.setup_quick_test_data()
            pgn_paths = [test_pgn]
        else:
            # Auto-discover PGN files
            pgn_paths = manager.list_available_pgns()
            
            if len(pgn_paths) == 0:
                logger.warning("No PGN files found. Creating test data.")
                test_pgn = manager.setup_quick_test_data()
                pgn_paths = [test_pgn]
    
    # Process PGN files with Stockfish
    with PositionExtractor(stockfish_path=stockfish_path) as extractor:
        extractor.process_multiple_pgns(
            pgn_paths=pgn_paths,
            output_path=output_path,
            max_games_per_file=max_games_per_file
        )
    
    return output_path


def extract_single_pgn_shard(
    pgn_path: Path,
    output_dir: Path,
    stockfish_path: str = "/usr/games/stockfish",
    max_games: Optional[int] = None
) -> Path:
    """
    Extract positions from a single PGN file and save as a shard.
    Designed for parallel execution across multiple workers.
    
    Args:
        pgn_path: Path to the PGN file
        output_dir: Directory to save the shard
        stockfish_path: Path to Stockfish binary
        max_games: Optional limit on games to proc
    Returns:
        Path to the saved shard file
    """
    import sys
    print(f"\n{'='*80}", flush=True)
    print(f"EXTRACT_SINGLE_PGN_SHARD CALLED", flush=True)
    print(f"PGN: {pgn_path}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"{'='*80}\n", flush=True)
    sys.stdout.flush()
    
    print("DEBUG: About to call logger.info...", flush=True)
    logger.info(f"Worker processing: {pgn_path.name}")
    print("DEBUG: Logger.info done", flush=True)
    
    # Create output directory
    print("DEBUG: Creating output directory...", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"DEBUG: Output dir created: {output_dir}", flush=True)
    
    # Generate shard name
    shard_name = f"shard_{pgn_path.stem}.pkl"
    shard_path = output_dir / shard_name
    print(f"DEBUG: Shard path: {shard_path}", flush=True)
    
    # Check if already processed
    if shard_path.exists():
        print(f"DEBUG: Shard exists, skipping", flush=True)
        logger.info(f"✓ Shard exists for {pgn_path.name}, skipping")
        return shard_path
    
    print(f"DEBUG: Creating PositionExtractor with stockfish={stockfish_path}...", flush=True)
    # Extract positions
    with PositionExtractor(stockfish_path=stockfish_path) as extractor:
        print("DEBUG: PositionExtractor created, calling process_pgn_file...", flush=True)
        positions = extractor.process_pgn_file(pgn_path, max_games)
    
    # Save shard
    logger.info(f"Saving shard to {shard_path} ({len(positions)} positions)")
    with open(shard_path, 'wb') as f:
        pickle.dump(positions, f)
    
    logger.info(f"✓ Completed {pgn_path.name}: {len(positions)} positions")
    return shard_path


if __name__ == "__main__":
    import tempfile
    import logging
    
    logging.basicConfig(level=logging.INFO)
    
    # Test with a small PGN
    with tempfile.TemporaryDirectory() as tmpdir:
        from .data_sources import DataSourceManager
        
        # Create test data
        manager = DataSourceManager(tmpdir)
        test_pgn = manager.setup_quick_test_data()
        
        print(f"\nProcessing test PGN: {test_pgn}")
        
        # Extract positions
        output_path = create_training_data(
            root_dir=tmpdir,
            pgn_paths=[test_pgn],
            stockfish_path="stockfish",  # Assumes stockfish is in PATH
            max_games_per_file=2
        )
        
        # Load and inspect
        print(f"\nLoading positions from {output_path}")
        with open(output_path, 'rb') as f:
            positions = pickle.load(f)
        
        print(f"Loaded {len(positions)} positions")
        if len(positions) > 0:
            print("\nFirst position:")
            print(f"  FEN: {positions[0]['fen']}")
            print(f"  Move: {positions[0]['move']}")
            print(f"  Value: {positions[0]['value']:.3f}")

