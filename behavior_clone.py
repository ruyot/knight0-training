"""
FAST behavior cloning from raw PGN games.

NO STOCKFISH NEEDED! Just copy moves from winning players.
This gives us a basic NN to start curriculum training immediately!
"""

import chess
import chess.pgn
import pickle
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_winner_moves(pgn_path: Path, max_games: int = 5000) -> List[Dict]:
    """
    Extract moves from WINNING players only.
    Fast! No engine analysis needed!
    """
    positions = []
    games_processed = 0
    
    logger.info(f"Behavior cloning from: {pgn_path.name}")
    
    with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
        while games_processed < max_games:
            try:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                # Only use games with clear winner
                result = game.headers.get("Result", "*")
                if result not in ["1-0", "0-1"]:
                    continue
                
                # Determine winner
                white_won = (result == "1-0")
                
                board = game.board()
                move_num = 0
                
                for move in game.mainline_moves():
                    move_num += 1
                    
                    # Only sample moves from winning side
                    is_white_move = (move_num % 2 == 1)
                    
                    if (white_won and is_white_move) or (not white_won and not is_white_move):
                        # Sample moves from opening/middlegame
                        if 10 <= move_num <= 50 and move_num % 3 == 0:
                            positions.append({
                                "board": board.fen(),
                                "move": move.uci(),
                                "value": 1.0 if white_won else -1.0  # Winner = good
                            })
                    
                    board.push(move)
                
                games_processed += 1
                
                if games_processed % 500 == 0:
                    logger.info(f"  Processed {games_processed} games, {len(positions)} positions")
            
            except Exception as e:
                continue
    
    logger.info(f"✓ Extracted {len(positions)} positions from {games_processed} games")
    return positions


if __name__ == "__main__":
    data_dir = Path("data")
    
    all_positions = []
    
    # Quick extraction from a few PGNs
    pgns_to_use = [
        "2025-01.bare.[28156].pgn",
        "2025-02.bare.[27510].pgn",
        "2025-03.bare.[30865].pgn"
    ]
    
    logger.info("🚀 FAST BEHAVIOR CLONING (NO STOCKFISH!)")
    logger.info("="*80)
    
    for pgn_name in pgns_to_use:
        pgn_path = data_dir / pgn_name
        if pgn_path.exists():
            positions = extract_winner_moves(pgn_path, max_games=5000)
            all_positions.extend(positions)
    
    # Save
    output_path = Path("behavior_clone_data.pkl")
    logger.info(f"\n💾 Saving {len(all_positions)} positions to {output_path}")
    
    with open(output_path, 'wb') as f:
        pickle.dump(all_positions, f)
    
    logger.info("✓ Behavior cloning data ready!")
    logger.info(f"  Total positions: {len(all_positions):,}")
    logger.info(f"  ~{len(all_positions) / 15000:.1f} positions per game")

