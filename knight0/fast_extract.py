"""
ULTRA-FAST position extraction using Stockfish with nodes limit.

This is 10-50x faster than depth-based analysis because:
- Uses "nodes" limit (e.g., 1000 nodes) instead of depth
- Much faster per position
- Still provides decent move quality
"""

import chess
import chess.pgn
import chess.engine
import pickle
import gzip
import bz2
from pathlib import Path
from typing import List, Dict, Optional
import logging

from .config import DATA_FILTERS
from .utils import normalize_score

logger = logging.getLogger(__name__)


class FastExtractor:
    """Ultra-fast Stockfish-based position labeler."""
    
    def __init__(
        self,
        stockfish_path: str = "/usr/games/stockfish",
        nodes: int = 500,  # Very low for speed!
        sample_rate: int = 12,  # Sample less frequently
        min_move: int = 15,
        max_move: int = 45
    ):
        self.stockfish_path = stockfish_path
        self.nodes = nodes
        self.sample_rate = sample_rate
        self.min_move = min_move
        self.max_move = max_move
        self.engine = None
    
    def __enter__(self):
        """Start Stockfish."""
        logger.info(f"Starting FAST Stockfish (nodes={self.nodes})")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop Stockfish."""
        if self.engine:
            self.engine.quit()
    
    def analyze_position(self, board: chess.Board) -> Optional[Dict]:
        """Fast analysis with nodes limit."""
        if not self.engine:
            raise RuntimeError("Engine not started")
        
        try:
            # Use NODES limit (much faster than depth!)
            limit = chess.engine.Limit(nodes=self.nodes)
            result = self.engine.analyse(board, limit)
            
            if "score" in result and "pv" in result:
                pv = result["pv"]
                if pv:
                    score = result["score"].relative
                    if score.is_mate():
                        cp_score = 10000 if score.mate() > 0 else -10000
                    else:
                        cp_score = score.score()
                    
                    return {
                        "move": pv[0].uci(),
                        "score": cp_score
                    }
            return None
        except Exception as e:
            logger.warning(f"Analysis error: {e}")
            return None
    
    def open_pgn_file(self, pgn_path: Path):
        """Open PGN file."""
        if pgn_path.suffix == '.gz':
            return gzip.open(pgn_path, 'rt', encoding='utf-8', errors='ignore')
        elif pgn_path.suffix == '.bz2':
            return bz2.open(pgn_path, 'rt', encoding='utf-8', errors='ignore')
        else:
            return open(pgn_path, 'r', encoding='utf-8', errors='ignore')
    
    def should_process_game(self, game: chess.pgn.Game) -> bool:
        """Check game quality."""
        mainline = list(game.mainline_moves())
        num_moves = len(mainline)
        
        if num_moves < DATA_FILTERS["min_moves"]:
            return False
        if num_moves > DATA_FILTERS["max_moves"]:
            return False
        
        return True
    
    def extract_from_game(self, game: chess.pgn.Game) -> List[Dict]:
        """Extract positions from game."""
        positions = []
        board = game.board()
        
        move_num = 0
        for move in game.mainline_moves():
            move_num += 1
            
            # Sample positions
            if (move_num >= self.min_move and 
                move_num <= self.max_move and 
                move_num % self.sample_rate == 0):
                
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
        """Process PGN file."""
        logger.info(f"FAST extraction from: {pgn_path.name}")
        
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
                    
                    # Progress logging every 200 games
                    if games_processed % 200 == 0:
                        avg = len(all_positions) / games_processed
                        logger.info(f"  ✓ {games_processed} games, "
                                  f"{len(all_positions)} positions "
                                  f"(avg: {avg:.1f}/game)")
                
                except Exception as e:
                    logger.warning(f"Game error: {e}")
                    continue
        
        avg = len(all_positions) / games_processed if games_processed > 0 else 0
        logger.info(f"")
        logger.info(f"{'='*80}")
        logger.info(f"✓ DONE: {pgn_path.name}")
        logger.info(f"  Games: {games_processed:,}")
        logger.info(f"  Positions: {len(all_positions):,}")
        logger.info(f"  Avg: {avg:.1f}/game")
        logger.info(f"{'='*80}")
        
        return all_positions


def extract_fast_shard(
    pgn_path: Path,
    output_dir: Path,
    stockfish_path: str = "/usr/games/stockfish",
    max_games: Optional[int] = None
) -> Path:
    """
    Fast extraction using nodes-based Stockfish analysis.
    """
    logger.info(f"Fast worker: {pgn_path.name}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    shard_name = f"shard_{pgn_path.stem}.pkl"
    shard_path = output_dir / shard_name
    
    if shard_path.exists():
        logger.info(f"✓ Shard exists, skipping")
        return shard_path
    
    # Fast extraction with nodes limit
    with FastExtractor(stockfish_path=stockfish_path, nodes=500) as extractor:
        positions = extractor.process_pgn_file(pgn_path, max_games)
    
    logger.info(f"Saving {len(positions)} positions...")
    with open(shard_path, 'wb') as f:
        pickle.dump(positions, f)
    
    logger.info(f"✓ Done: {len(positions)} positions")
    return shard_path

