"""
Deep position extraction for Phase 2 fine-tuning.

Uses:
- Stockfish depth 20-30
- MultiPV 3-5 for detecting tactical pivots
- Position filtering for quality
"""

import chess
import chess.engine
import chess.pgn
from pathlib import Path
from typing import List, Dict, Optional
import pickle
import logging
from tqdm import tqdm

from .position_filter import PositionFilter
from .encoding import board_to_tensor, move_to_index

logger = logging.getLogger(__name__)


class DeepExtractor:
    """
    Extracts high-quality positions with deep Stockfish analysis.
    """
    
    def __init__(
        self,
        stockfish_path: str = "/usr/games/stockfish",
        depth: int = 25,
        multipv: int = 3,
        time_limit: float = 0.5,
    ):
        """
        Args:
            stockfish_path: Path to Stockfish binary
            depth: Analysis depth (20-30 recommended)
            multipv: Number of best moves to analyze
            time_limit: Max time per position (seconds)
        """
        self.stockfish_path = stockfish_path
        self.depth = depth
        self.multipv = multipv
        self.time_limit = time_limit
        self.engine = None
        self.filter = PositionFilter()
    
    def __enter__(self):
        logger.info(f"Starting deep Stockfish analysis: depth={self.depth}, multiPV={self.multipv}")
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        self.engine.configure({"Threads": 2})  # Use 2 threads for speed
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.engine:
            self.engine.quit()
    
    def analyze_position_deep(
        self,
        board: chess.Board,
        prev_eval: Optional[float] = None,
        eval_history: Optional[List[float]] = None
    ) -> Optional[Dict]:
        """
        Deeply analyze a position with multiPV.
        
        Returns:
            Dict with position data if it passes filter, None otherwise
        """
        if board.is_game_over():
            return None
        
        # Analyze with MultiPV
        info = self.engine.analyse(
            board,
            chess.engine.Limit(depth=self.depth, time=self.time_limit),
            multipv=self.multipv
        )
        
        if not info or len(info) == 0:
            return None
        
        # Get best move and eval
        best_info = info[0]
        best_move = best_info.get("pv", [None])[0]
        best_score = best_info.get("score")
        
        if not best_move or not best_score:
            return None
        
        # Normalize score
        if best_score.is_mate():
            mate_in = best_score.mate()
            value = 1.0 if mate_in > 0 else -1.0
        else:
            cp = best_score.relative.score()
            value = max(min(cp / 1000.0, 1.0), -1.0)
        
        # Get second-best eval for tactical pivot detection
        second_best_eval = None
        if len(info) >= 2:
            second_score = info[1].get("score")
            if second_score and not second_score.is_mate():
                cp2 = second_score.relative.score()
                second_best_eval = max(min(cp2 / 1000.0, 1.0), -1.0)
        
        # Apply position filter
        if not self.filter.should_keep_position(
            board=board,
            eval_score=value,
            prev_eval=prev_eval,
            eval_history=eval_history,
            best_move=best_move,
            second_best_eval=second_best_eval
        ):
            return None
        
        # Compute quality score for weighting during training
        quality_score = self.filter.compute_position_quality_score(
            board=board,
            eval_score=value,
            prev_eval=prev_eval
        )
        
        return {
            "fen": board.fen(),
            "board": board.copy(),
            "move": move_to_index(best_move),
            "value": value,
            "quality": quality_score,
            "depth": self.depth,
            "multipv_gap": abs(value - second_best_eval) if second_best_eval else 0.0,
        }
    
    def extract_from_game(
        self,
        game: chess.pgn.Game,
        sample_rate: int = 2,  # Sample more frequently for deep analysis
        min_move: int = 12,
        max_move: int = 50,
    ) -> List[Dict]:
        """
        Extract high-quality positions from a single game.
        
        Args:
            game: PGN game
            sample_rate: Sample every Nth move
            min_move: Start sampling from this move
            max_move: Stop sampling at this move
        
        Returns:
            List of position dictionaries
        """
        positions = []
        board = game.board()
        
        eval_history = []
        prev_eval = None
        
        for move_num, move in enumerate(game.mainline_moves(), start=1):
            board.push(move)
            
            # Check sampling criteria
            if move_num < min_move or move_num > max_move:
                continue
            
            if move_num % sample_rate != 0:
                continue
            
            # Deep analysis
            position_data = self.analyze_position_deep(
                board=board,
                prev_eval=prev_eval,
                eval_history=eval_history
            )
            
            if position_data:
                positions.append(position_data)
                
                # Update history
                current_eval = position_data["value"]
                eval_history.append(current_eval)
                prev_eval = current_eval
                
                # Keep history small
                if len(eval_history) > 5:
                    eval_history.pop(0)
        
        return positions
    
    def process_pgn_deep(
        self,
        pgn_path: Path,
        max_games: Optional[int] = None,
        min_elo: int = 2400,  # Higher Elo for fine-tuning
    ) -> List[Dict]:
        """
        Process a PGN file with deep analysis.
        
        Args:
            pgn_path: Path to PGN file
            max_games: Optional game limit
            min_elo: Minimum player Elo
        
        Returns:
            List of high-quality positions
        """
        logger.info(f"Deep extraction from: {pgn_path.name}")
        logger.info(f"Settings: depth={self.depth}, multiPV={self.multipv}, min_elo={min_elo}")
        
        all_positions = []
        games_processed = 0
        
        with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
            while True:
                if max_games and games_processed >= max_games:
                    break
                
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                # Filter by Elo
                white_elo = game.headers.get("WhiteElo", "0")
                black_elo = game.headers.get("BlackElo", "0")
                
                try:
                    if int(white_elo) < min_elo or int(black_elo) < min_elo:
                        continue
                except (ValueError, TypeError):
                    continue
                
                # Extract positions
                positions = self.extract_from_game(game)
                all_positions.extend(positions)
                games_processed += 1
                
                if games_processed % 50 == 0:
                    logger.info(f"  Processed {games_processed} games, extracted {len(all_positions)} positions")
        
        logger.info(f"Completed: {games_processed} games → {len(all_positions)} high-quality positions")
        return all_positions


def extract_deep_shard(
    pgn_path: Path,
    output_dir: Path,
    depth: int = 25,
    multipv: int = 3,
    max_games: Optional[int] = None,
    stockfish_path: str = "/usr/games/stockfish"
) -> Path:
    """
    Extract a single shard with deep analysis.
    
    Args:
        pgn_path: Path to PGN file
        output_dir: Output directory for shard
        depth: Stockfish depth
        multipv: MultiPV count
        max_games: Optional game limit
        stockfish_path: Path to Stockfish
    
    Returns:
        Path to saved shard
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    shard_name = f"deep_shard_{pgn_path.stem}_d{depth}.pkl"
    shard_path = output_dir / shard_name
    
    if shard_path.exists():
        logger.info(f"✓ Deep shard exists: {shard_name}")
        return shard_path
    
    with DeepExtractor(
        stockfish_path=stockfish_path,
        depth=depth,
        multipv=multipv
    ) as extractor:
        positions = extractor.process_pgn_deep(pgn_path, max_games)
    
    # Save shard
    logger.info(f"Saving deep shard: {shard_name} ({len(positions)} positions)")
    with open(shard_path, 'wb') as f:
        pickle.dump(positions, f)
    
    return shard_path

