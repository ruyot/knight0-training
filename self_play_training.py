"""
Self-play training to handle novel positions.

After Phase 1 & 2 (supervised learning), use self-play to:
1. Explore positions Stockfish never saw
2. Learn to handle human creativity
3. Develop unique strategies
4. Improve long-term planning

This is how AlphaZero surpassed Stockfish!
"""

import chess
import torch
from pathlib import Path
from typing import List, Dict, Tuple
import random
from dataclasses import dataclass

from search_engine import Knight0Search


@dataclass
class SelfPlayGame:
    """Result of a self-play game."""
    positions: List[chess.Board]
    moves: List[chess.Move]
    values: List[float]  # Backpropagated from final result
    result: str  # "1-0", "0-1", or "1/2-1/2"


class SelfPlayGenerator:
    """
    Generate training data through self-play.
    """
    
    def __init__(self, model_path: str, temperature: float = 1.0):
        """
        Args:
            model_path: Path to ONNX model
            temperature: Exploration temperature (higher = more random)
        """
        self.search = Knight0Search(model_path)
        self.temperature = temperature
    
    def play_game(
        self,
        max_moves: int = 200,
        search_depth: int = 6,
        add_noise: bool = True
    ) -> SelfPlayGame:
        """
        Play one self-play game.
        
        Args:
            max_moves: Maximum game length
            search_depth: Search depth per move
            add_noise: Add exploration noise
        
        Returns:
            SelfPlayGame with positions and moves
        """
        board = chess.Board()
        positions = []
        moves = []
        
        for move_num in range(max_moves):
            if board.is_game_over():
                break
            
            positions.append(board.copy())
            
            # Search for best move
            result = self.search.iterative_deepening(
                board,
                max_depth=search_depth,
                time_limit=0.5
            )
            
            # Add exploration noise (for training diversity)
            if add_noise and random.random() < 0.1:  # 10% random moves
                move = random.choice(list(board.legal_moves))
            else:
                move = result.best_move
            
            moves.append(move)
            board.push(move)
        
        # Determine result
        if board.is_checkmate():
            result = "0-1" if board.turn == chess.WHITE else "1-0"
        elif board.is_stalemate() or board.is_insufficient_material():
            result = "1/2-1/2"
        else:
            result = "1/2-1/2"  # Draw by length
        
        # Backpropagate result to all positions
        game_value = {
            "1-0": 1.0,
            "0-1": -1.0,
            "1/2-1/2": 0.0
        }[result]
        
        # Alternate value based on whose turn it was
        values = []
        for pos in positions:
            if pos.turn == chess.WHITE:
                values.append(game_value)
            else:
                values.append(-game_value)
        
        return SelfPlayGame(
            positions=positions,
            moves=moves,
            values=values,
            result=result
        )
    
    def generate_training_data(
        self,
        num_games: int = 100,
        output_path: Path = None
    ) -> List[Dict]:
        """
        Generate self-play training data.
        
        Args:
            num_games: Number of games to play
            output_path: Optional path to save data
        
        Returns:
            List of position dictionaries
        """
        print(f"Generating {num_games} self-play games...")
        
        all_positions = []
        
        for game_num in range(num_games):
            game = self.play_game()
            
            # Convert to training format
            for pos, move, value in zip(game.positions, game.moves, game.values):
                from knight0.encoding import move_to_index
                
                all_positions.append({
                    'fen': pos.fen(),
                    'board': pos.copy(),
                    'move': move_to_index(move),
                    'value': value,
                    'source': 'selfplay'
                })
            
            if (game_num + 1) % 10 == 0:
                print(f"  Completed {game_num + 1}/{num_games} games "
                      f"({len(all_positions)} positions)")
        
        # Save if requested
        if output_path:
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(all_positions, f)
            print(f"\nSaved {len(all_positions)} positions to {output_path}")
        
        return all_positions


def mix_supervised_and_selfplay(
    supervised_data: List[Dict],
    selfplay_data: List[Dict],
    selfplay_ratio: float = 0.2
) -> List[Dict]:
    """
    Mix supervised and self-play data for training.
    
    Args:
        supervised_data: Positions from Stockfish labeling
        selfplay_data: Positions from self-play
        selfplay_ratio: Fraction of self-play data (0.2 = 20%)
    
    Returns:
        Mixed dataset
    """
    n_selfplay = int(len(supervised_data) * selfplay_ratio / (1 - selfplay_ratio))
    n_selfplay = min(n_selfplay, len(selfplay_data))
    
    # Sample selfplay data
    sampled_selfplay = random.sample(selfplay_data, n_selfplay)
    
    # Combine
    mixed = supervised_data + sampled_selfplay
    random.shuffle(mixed)
    
    print(f"Mixed dataset: {len(supervised_data)} supervised + "
          f"{n_selfplay} self-play = {len(mixed)} total")
    
    return mixed


if __name__ == "__main__":
    model_path = "knight0_model.onnx"
    
    if not Path(model_path).exists():
        print("ERROR: Model not found! Train Phase 1 first.")
        exit(1)
    
    # Generate self-play data
    generator = SelfPlayGenerator(model_path, temperature=1.0)
    
    selfplay_data = generator.generate_training_data(
        num_games=10,  # Start small
        output_path=Path("selfplay_data.pkl")
    )
    
    print(f"\n✓ Generated {len(selfplay_data)} self-play positions!")
    print("\nNext steps:")
    print("1. Mix with supervised data (80/20 split)")
    print("2. Fine-tune model on mixed dataset")
    print("3. Repeat self-play with stronger model")
    print("4. This creates a continuous improvement cycle!")

