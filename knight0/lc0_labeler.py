"""
Fast position labeling using Leela Chess Zero (LC0).

LC0 uses a neural network (like AlphaZero) for instant evaluation.
This is 10-100x faster than Stockfish for our use case!
"""

import chess
import chess.pgn
import subprocess
import json
import tempfile
from pathlib import Path
from typing import Dict, Optional, List
import logging

from .config import LC0_CONFIG, DATA_FILTERS
from .utils import normalize_score

logger = logging.getLogger(__name__)


class LC0Labeler:
    """
    Fast position labeler using Leela Chess Zero.
    
    LC0 outputs:
    - Policy: move probabilities (perfect for our policy head!)
    - Value: win probability (convert to our value target)
    """
    
    def __init__(
        self,
        lc0_binary: str = "lc0",
        weights_file: Optional[str] = None,
        visits: int = 50,
        sample_rate: int = 8,
        min_move: int = 15,
        max_move: int = 50
    ):
        self.lc0_binary = lc0_binary
        self.weights_file = weights_file  # Will auto-download if None
        self.visits = visits
        self.sample_rate = sample_rate
        self.min_move = min_move
        self.max_move = max_move
        self.process = None
    
    def __enter__(self):
        """Start LC0 process."""
        logger.info(f"Starting LC0 engine: {self.lc0_binary}")
        
        cmd = [self.lc0_binary]
        if self.weights_file:
            cmd.extend(["--weights", self.weights_file])
        
        # Run in UCI mode
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Initialize UCI
        self.send_command("uci")
        self.wait_for("uciok")
        self.send_command("ucinewgame")
        
        logger.info("LC0 engine ready!")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop LC0 process."""
        if self.process:
            self.send_command("quit")
            self.process.wait(timeout=5)
            logger.info("Closed LC0 engine")
    
    def send_command(self, cmd: str):
        """Send command to LC0."""
        if self.process and self.process.stdin:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()
    
    def wait_for(self, expected: str) -> List[str]:
        """Wait for specific output from LC0."""
        lines = []
        while True:
            line = self.process.stdout.readline().strip()
            lines.append(line)
            if expected in line:
                break
        return lines
    
    def analyze_position(self, board: chess.Board) -> Optional[Dict]:
        """
        Analyze position with LC0.
        
        Returns:
            Dict with 'move' (best move UCI) and 'score' (win probability)
        """
        if not self.process:
            raise RuntimeError("LC0 not started. Use context manager.")
        
        try:
            # Set position
            fen = board.fen()
            self.send_command(f"position fen {fen}")
            
            # Analyze with limited visits (fast!)
            self.send_command(f"go nodes {self.visits}")
            
            # Parse output
            best_move = None
            value = 0.0
            
            for line in self.wait_for("bestmove"):
                if line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2:
                        best_move = parts[1]
                elif "score cp" in line:
                    # Extract centipawn score
                    parts = line.split()
                    if "cp" in parts:
                        idx = parts.index("cp")
                        if idx + 1 < len(parts):
                            value = float(parts[idx + 1])
                elif "score wdl" in line:
                    # Win/Draw/Loss probability (better!)
                    parts = line.split()
                    if "wdl" in parts:
                        idx = parts.index("wdl")
                        if idx + 3 < len(parts):
                            # Convert W-D-L to value
                            w = float(parts[idx + 1])
                            d = float(parts[idx + 2])
                            l = float(parts[idx + 3])
                            total = w + d + l
                            if total > 0:
                                # Win rate (0 to 1)
                                value = (w + 0.5 * d) / total
                                # Convert to tanh-like range (-1 to 1)
                                value = 2 * value - 1
            
            if best_move and best_move != "(none)":
                return {
                    "move": best_move,
                    "score": value * 100  # Scale to centipawn-ish range
                }
            return None
            
        except Exception as e:
            logger.warning(f"LC0 analysis error: {e}")
            return None
    
    def open_pgn_file(self, pgn_path: Path):
        """Open PGN file (handles compression)."""
        import gzip
        import bz2
        
        if pgn_path.suffix == '.gz':
            return gzip.open(pgn_path, 'rt', encoding='utf-8', errors='ignore')
        elif pgn_path.suffix == '.bz2':
            return bz2.open(pgn_path, 'rt', encoding='utf-8', errors='ignore')
        else:
            return open(pgn_path, 'r', encoding='utf-8', errors='ignore')
    
    def should_process_game(self, game: chess.pgn.Game) -> bool:
        """Check if game meets quality criteria."""
        headers = game.headers
        
        # Check game length
        mainline = list(game.mainline_moves())
        num_moves = len(mainline)
        
        if num_moves < DATA_FILTERS["min_moves"]:
            return False
        if num_moves > DATA_FILTERS["max_moves"]:
            return False
        
        return True
    
    def extract_from_game(self, game: chess.pgn.Game) -> List[Dict]:
        """Extract and label positions from a game."""
        positions = []
        board = game.board()
        
        move_num = 0
        for move in game.mainline_moves():
            move_num += 1
            
            # Sample positions
            if (move_num >= self.min_move and 
                move_num <= self.max_move and 
                move_num % self.sample_rate == 0):
                
                # Analyze position
                analysis = self.analyze_position(board)
                if analysis:
                    positions.append({
                        "board": board.fen(),
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
        """Process a PGN file and extract labeled positions."""
        logger.info(f"Processing PGN with LC0: {pgn_path}")
        
        all_positions = []
        games_processed = 0
        games_skipped = 0
        
        with self.open_pgn_file(pgn_path) as pgn_file:
            while True:
                if max_games and games_processed >= max_games:
                    break
                
                try:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break
                    
                    if not self.should_process_game(game):
                        games_skipped += 1
                        continue
                    
                    positions = self.extract_from_game(game)
                    all_positions.extend(positions)
                    games_processed += 1
                    
                    # Progress logging
                    if games_processed % 100 == 0:
                        avg_per_game = len(all_positions) / games_processed
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


def extract_single_pgn_shard_lc0(
    pgn_path: Path,
    output_dir: Path,
    lc0_binary: str = "lc0",
    max_games: Optional[int] = None
) -> Path:
    """
    Extract positions from a single PGN using LC0 (FAST!).
    """
    import pickle
    
    logger.info(f"LC0 Worker processing: {pgn_path.name}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate shard name
    shard_name = f"shard_{pgn_path.stem}.pkl"
    shard_path = output_dir / shard_name
    
    # Check if already processed
    if shard_path.exists():
        logger.info(f"✓ Shard exists for {pgn_path.name}, skipping")
        return shard_path
    
    # Extract positions with LC0
    with LC0Labeler(lc0_binary=lc0_binary) as labeler:
        positions = labeler.process_pgn_file(pgn_path, max_games)
    
    # Save shard
    logger.info(f"Saving shard to {shard_path} ({len(positions)} positions)")
    with open(shard_path, 'wb') as f:
        pickle.dump(positions, f)
    
    logger.info(f"✓ Completed {pgn_path.name}: {len(positions)} positions")
    return shard_path

