"""
Curriculum Learning: Train knight0 by progressively facing harder opponents.

Training progression:
1. Stockfish Level 1 (Elo ~1000) → Win 90%+
2. Stockfish Level 2 (Elo ~1100) → Win 90%+
3. Stockfish Level 3 (Elo ~1200) → Win 90%+
... continue until target strength

This teaches:
- Exploiting blunders (low levels make mistakes!)
- Tactical awareness (must capitalize on errors)
- Consistent play (90% win rate = reliable)
- Progressive improvement (each level harder)

After Phase 1 training, this refines the model against realistic opponents!
"""

import chess
import chess.engine
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pickle
import time
from dataclasses import dataclass
from collections import defaultdict

from search_engine import Knight0Search
from knight0.encoding import board_to_tensor, move_to_index


@dataclass
class GameResult:
    """Result of a single game."""
    result: str  # "1-0", "0-1", "1/2-1/2"
    moves: int
    time_ms: int
    positions: List[chess.Board]
    knight0_moves: List[chess.Move]
    knight0_values: List[float]


@dataclass
class TournamentResult:
    """Result of a tournament (N games)."""
    wins: int
    losses: int
    draws: int
    games: List[GameResult]
    
    @property
    def total_games(self) -> int:
        return self.wins + self.losses + self.draws
    
    @property
    def win_rate(self) -> float:
        return self.wins / self.total_games if self.total_games > 0 else 0.0
    
    @property
    def score(self) -> float:
        """Chess score (1 for win, 0.5 for draw, 0 for loss)."""
        return (self.wins + 0.5 * self.draws) / self.total_games if self.total_games > 0 else 0.0


class StockfishOpponent:
    """
    Stockfish opponent with configurable skill level.
    """
    
    # Approximate Elo for each Stockfish skill level
    SKILL_ELO = {
        0: 800,
        1: 1000,
        2: 1100,
        3: 1200,
        4: 1300,
        5: 1400,
        6: 1500,
        7: 1600,
        8: 1700,
        9: 1800,
        10: 1900,
        11: 2000,
        12: 2100,
        13: 2200,
        14: 2300,
        15: 2400,
        16: 2500,
        17: 2600,
        18: 2700,
        19: 2800,
        20: 3200,
    }
    
    def __init__(self, stockfish_path: str = "/usr/games/stockfish"):
        self.stockfish_path = stockfish_path
        self.engine = None
    
    def start(self, skill_level: int = 1):
        """
        Start Stockfish at specified skill level.
        
        Args:
            skill_level: 0-20 (0=weakest, 20=strongest)
        """
        self.engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        
        # Configure skill level
        self.engine.configure({
            "Skill Level": skill_level,
            "UCI_LimitStrength": True,
            "UCI_Elo": self.SKILL_ELO.get(skill_level, 1000)
        })
        
        self.skill_level = skill_level
        print(f"Stockfish started at skill level {skill_level} (≈{self.SKILL_ELO[skill_level]} Elo)")
    
    def get_move(self, board: chess.Board, time_limit: float = 0.1) -> chess.Move:
        """Get Stockfish's move."""
        result = self.engine.play(board, chess.engine.Limit(time=time_limit))
        return result.move
    
    def quit(self):
        """Stop engine."""
        if self.engine:
            self.engine.quit()
            self.engine = None


class CurriculumTrainer:
    """
    Train knight0 through curriculum learning.
    """
    
    def __init__(
        self,
        model_path: str,
        stockfish_path: str = "/usr/games/stockfish",
        search_depth: int = 6
    ):
        """
        Args:
            model_path: Path to knight0 ONNX model
            stockfish_path: Path to Stockfish binary
            search_depth: Search depth for knight0
        """
        self.knight0 = Knight0Search(model_path)
        self.stockfish = StockfishOpponent(stockfish_path)
        self.search_depth = search_depth
        
        # Track progress
        self.current_level = 1
        self.level_history = defaultdict(list)  # skill_level -> [win_rates]
    
    def play_game(
        self,
        knight0_white: bool = True,
        max_moves: int = 200,
        time_per_move: float = 1.0
    ) -> GameResult:
        """
        Play one game: knight0 vs Stockfish.
        
        Args:
            knight0_white: If True, knight0 plays White
            max_moves: Maximum game length
            time_per_move: Time per move (seconds)
        
        Returns:
            GameResult
        """
        board = chess.Board()
        positions = []
        knight0_moves = []
        knight0_values = []
        move_count = 0
        start_time = time.time()
        
        while not board.is_game_over() and move_count < max_moves:
            is_knight0_turn = (board.turn == chess.WHITE) == knight0_white
            
            if is_knight0_turn:
                # knight0's turn
                positions.append(board.copy())
                
                result = self.knight0.iterative_deepening(
                    board,
                    max_depth=self.search_depth,
                    time_limit=time_per_move
                )
                
                move = result.best_move
                knight0_moves.append(move)
                knight0_values.append(result.eval)
            else:
                # Stockfish's turn
                move = self.stockfish.get_move(board, time_per_move)
            
            board.push(move)
            move_count += 1
        
        # Determine result
        if board.is_checkmate():
            result = "0-1" if board.turn == chess.WHITE else "1-0"
        elif board.is_stalemate() or board.is_insufficient_material():
            result = "1/2-1/2"
        elif board.can_claim_draw():
            result = "1/2-1/2"
        else:
            result = "1/2-1/2"  # Draw by length
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return GameResult(
            result=result,
            moves=move_count,
            time_ms=elapsed_ms,
            positions=positions,
            knight0_moves=knight0_moves,
            knight0_values=knight0_values
        )
    
    def run_tournament(
        self,
        skill_level: int,
        num_games: int = 50,
        alternate_colors: bool = True
    ) -> TournamentResult:
        """
        Run a tournament against Stockfish at specified level.
        
        Args:
            skill_level: Stockfish skill level (0-20)
            num_games: Number of games to play
            alternate_colors: If True, alternate colors each game
        
        Returns:
            TournamentResult
        """
        print("\n" + "=" * 70)
        print(f"TOURNAMENT: knight0 vs Stockfish Level {skill_level}")
        print(f"Games: {num_games}")
        print("=" * 70)
        
        # Start Stockfish at this level
        self.stockfish.start(skill_level)
        
        wins = 0
        losses = 0
        draws = 0
        games = []
        
        for game_num in range(num_games):
            knight0_white = (game_num % 2 == 0) if alternate_colors else True
            
            color_str = "White" if knight0_white else "Black"
            print(f"\nGame {game_num + 1}/{num_games} (knight0 = {color_str})...", end=" ")
            
            game = self.play_game(knight0_white)
            games.append(game)
            
            # Count result
            if game.result == "1-0":
                if knight0_white:
                    wins += 1
                    print("✓ WIN")
                else:
                    losses += 1
                    print("✗ LOSS")
            elif game.result == "0-1":
                if knight0_white:
                    losses += 1
                    print("✗ LOSS")
                else:
                    wins += 1
                    print("✓ WIN")
            else:
                draws += 1
                print("= DRAW")
            
            # Progress update every 10 games
            if (game_num + 1) % 10 == 0:
                current_win_rate = wins / (game_num + 1)
                print(f"  Progress: W:{wins} L:{losses} D:{draws} "
                      f"(Win rate: {current_win_rate:.1%})")
        
        self.stockfish.quit()
        
        result = TournamentResult(
            wins=wins,
            losses=losses,
            draws=draws,
            games=games
        )
        
        print("\n" + "-" * 70)
        print("TOURNAMENT RESULTS:")
        print(f"  Wins:     {wins}")
        print(f"  Losses:   {losses}")
        print(f"  Draws:    {draws}")
        print(f"  Win Rate: {result.win_rate:.1%}")
        print(f"  Score:    {result.score:.1%}")
        print("=" * 70)
        
        return result
    
    def train_curriculum(
        self,
        start_level: int = 1,
        end_level: int = 10,
        win_rate_threshold: float = 0.90,
        games_per_level: int = 50,
        save_training_data: bool = True
    ) -> Dict[int, TournamentResult]:
        """
        Train through curriculum: progressively harder opponents.
        
        Only advances to next level if win_rate >= threshold.
        
        Args:
            start_level: Starting Stockfish level
            end_level: Target Stockfish level
            win_rate_threshold: Required win rate to advance (0.90 = 90%)
            games_per_level: Games to play at each level
            save_training_data: Save positions for retraining
        
        Returns:
            Dict of level -> TournamentResult
        """
        print("\n" + "=" * 70)
        print("CURRICULUM LEARNING TRAINING")
        print("=" * 70)
        print(f"Start Level: {start_level} (≈{StockfishOpponent.SKILL_ELO[start_level]} Elo)")
        print(f"End Level:   {end_level} (≈{StockfishOpponent.SKILL_ELO[end_level]} Elo)")
        print(f"Win Rate Required: {win_rate_threshold:.0%}")
        print(f"Games per Level: {games_per_level}")
        print("=" * 70)
        
        results = {}
        training_positions = []
        
        current_level = start_level
        
        while current_level <= end_level:
            print(f"\n{'*' * 70}")
            print(f"LEVEL {current_level} CHALLENGE")
            print(f"Target: Win {win_rate_threshold:.0%} of {games_per_level} games")
            print(f"{'*' * 70}")
            
            # Run tournament at this level
            result = self.run_tournament(
                skill_level=current_level,
                num_games=games_per_level
            )
            
            results[current_level] = result
            self.level_history[current_level].append(result.win_rate)
            
            # Collect training data from losses/draws (positions to improve on)
            if save_training_data:
                for game in result.games:
                    if game.result != "1-0":  # Lost or drew
                        # These are positions where we made mistakes
                        for pos, move, value in zip(
                            game.positions,
                            game.knight0_moves,
                            game.knight0_values
                        ):
                            training_positions.append({
                                'board': pos,
                                'move': move_to_index(move),
                                'value': value,
                                'skill_level': current_level,
                                'source': 'curriculum'
                            })
            
            # Check if we can advance
            if result.win_rate >= win_rate_threshold:
                print(f"\n✓ LEVEL {current_level} PASSED!")
                print(f"  Win rate: {result.win_rate:.1%} >= {win_rate_threshold:.0%}")
                
                if current_level < end_level:
                    print(f"  → Advancing to Level {current_level + 1}")
                    current_level += 1
                else:
                    print(f"\n🎉 CURRICULUM COMPLETE!")
                    print(f"  Successfully beat Level {end_level}!")
                    break
            else:
                print(f"\n⚠ LEVEL {current_level} NOT PASSED")
                print(f"  Win rate: {result.win_rate:.1%} < {win_rate_threshold:.0%}")
                print(f"  Need to retrain before advancing!")
                print(f"\n  Options:")
                print(f"    1. Fine-tune model on failed positions")
                print(f"    2. Increase search depth")
                print(f"    3. Continue training at this level")
                
                # For now, we'll stop and let user retrain
                break
        
        # Save training data
        if save_training_data and training_positions:
            output_path = Path(f"curriculum_training_data_level{current_level}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(training_positions, f)
            print(f"\n💾 Saved {len(training_positions)} training positions to {output_path}")
            print("   Use these to fine-tune the model on weak positions!")
        
        return results
    
    def evaluate_current_strength(self) -> int:
        """
        Estimate current strength by finding highest beatable level.
        
        Returns:
            Approximate Elo rating
        """
        print("\nEvaluating current strength...")
        
        # Binary search for skill level
        low, high = 0, 20
        best_level = 0
        
        while low <= high:
            mid = (low + high) // 2
            
            # Quick tournament (20 games)
            result = self.run_tournament(mid, num_games=20)
            
            if result.score >= 0.75:  # 75%+ score
                best_level = mid
                low = mid + 1
            else:
                high = mid - 1
        
        estimated_elo = StockfishOpponent.SKILL_ELO[best_level]
        print(f"\n📊 Estimated Strength: Level {best_level} (≈{estimated_elo} Elo)")
        
        return estimated_elo


if __name__ == "__main__":
    model_path = "knight0_model.onnx"
    
    if not Path(model_path).exists():
        print("ERROR: knight0_model.onnx not found!")
        print("Train Phase 1 first, then run curriculum training.")
        exit(1)
    
    print("\n" + "=" * 70)
    print("KNIGHT0 CURRICULUM TRAINING")
    print("=" * 70)
    print("\nThis will train your model by progressively facing harder opponents.")
    print("Starting from Stockfish Level 1, advancing only after 90%+ win rate.\n")
    
    # Create trainer
    trainer = CurriculumTrainer(
        model_path=model_path,
        search_depth=6  # Adjust based on time constraints
    )
    
    # Run curriculum
    results = trainer.train_curriculum(
        start_level=1,
        end_level=10,  # Target Level 10 (≈1900 Elo)
        win_rate_threshold=0.90,
        games_per_level=50
    )
    
    print("\n" + "=" * 70)
    print("CURRICULUM TRAINING SUMMARY")
    print("=" * 70)
    
    for level, result in results.items():
        elo = StockfishOpponent.SKILL_ELO[level]
        print(f"Level {level:2d} ({elo} Elo): {result.win_rate:.1%} win rate")
    
    # Estimate final strength
    final_elo = trainer.evaluate_current_strength()
    
    print(f"\n🎯 Final Estimated Strength: {final_elo} Elo")

