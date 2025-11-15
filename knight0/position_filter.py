"""
Position filtering for high-quality fine-tuning data.

Filters for:
- Sharp tactical moments
- Evaluation changes (critical positions)
- Non-terminal positions
- Strategic pivots
"""

import chess
import chess.engine
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class PositionFilter:
    """
    Filters positions for deep fine-tuning based on complexity and criticality.
    """
    
    @staticmethod
    def is_sharp_position(board: chess.Board, eval_score: float, prev_eval: Optional[float] = None) -> bool:
        """
        Detect sharp tactical moments.
        
        Criteria:
        - Large evaluation swing from previous position (if available)
        - Multiple pieces attacked
        - King safety issues
        - Material imbalance
        
        Args:
            board: Current position
            eval_score: Current evaluation (normalized -1 to 1)
            prev_eval: Previous evaluation (optional)
        
        Returns:
            True if position is tactically sharp
        """
        # Check for eval swing
        if prev_eval is not None:
            eval_change = abs(eval_score - prev_eval)
            if eval_change > 0.3:  # Significant swing (30 centipawns normalized)
                return True
        
        # Count attacked pieces
        attacked_pieces = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and board.is_attacked_by(not piece.color, square):
                attacked_pieces += 1
        
        if attacked_pieces >= 3:  # Multiple pieces under attack
            return True
        
        # Check king safety
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square:
                attackers = board.attackers(not color, king_square)
                if len(attackers) >= 2:  # King under multi-piece attack
                    return True
        
        return False
    
    @staticmethod
    def is_stable_eval(eval_history: list, threshold: float = 0.2) -> bool:
        """
        Check if evaluation is stable (not fluctuating wildly).
        
        Args:
            eval_history: List of recent evaluations
            threshold: Maximum allowed fluctuation
        
        Returns:
            True if evaluation is stable
        """
        if len(eval_history) < 3:
            return True
        
        # Check variance in recent evals
        recent = eval_history[-3:]
        max_val = max(recent)
        min_val = min(recent)
        
        return (max_val - min_val) <= threshold
    
    @staticmethod
    def is_non_terminal(board: chess.Board, eval_score: float) -> bool:
        """
        Check if position is non-terminal (not clearly winning/losing).
        
        Args:
            board: Current position
            eval_score: Evaluation score (-1 to 1)
        
        Returns:
            True if position is balanced enough for learning
        """
        # Skip positions that are clearly winning/losing
        if abs(eval_score) > 0.8:  # More than 800 centipawns advantage
            return False
        
        # Skip positions with very few pieces (simple endgames)
        piece_count = len(board.piece_map())
        if piece_count < 6:  # Less than 6 pieces total
            return False
        
        # Skip checkmate/stalemate
        if board.is_checkmate() or board.is_stalemate():
            return False
        
        return True
    
    @staticmethod
    def is_tactical_pivot(
        board: chess.Board,
        best_move: chess.Move,
        eval_score: float,
        second_best_eval: Optional[float] = None
    ) -> bool:
        """
        Detect tactical pivot points (critical decision moments).
        
        Criteria:
        - Multiple reasonable candidate moves
        - Best move significantly better than alternatives
        - Position complexity (checks, captures, threats)
        
        Args:
            board: Current position
            best_move: Stockfish's best move
            eval_score: Evaluation after best move
            second_best_eval: Eval of second-best move (if using multiPV)
        
        Returns:
            True if position is a tactical pivot
        """
        # Check if best move is clearly superior (using multiPV)
        if second_best_eval is not None:
            advantage = abs(eval_score - second_best_eval)
            if advantage > 0.15:  # Best move is 150+ centipawns better
                return True
        
        # Check for forcing moves
        if board.is_check():
            return True
        
        if best_move and board.is_capture(best_move):
            return True
        
        # Count legal moves (complexity)
        num_legal_moves = board.legal_moves.count()
        if 15 <= num_legal_moves <= 40:  # Sweet spot for complex positions
            return True
        
        return False
    
    @staticmethod
    def should_keep_position(
        board: chess.Board,
        eval_score: float,
        prev_eval: Optional[float] = None,
        eval_history: Optional[list] = None,
        best_move: Optional[chess.Move] = None,
        second_best_eval: Optional[float] = None,
    ) -> bool:
        """
        Main filter: decide if position should be kept for fine-tuning.
        
        Args:
            board: Current position
            eval_score: Current evaluation
            prev_eval: Previous evaluation
            eval_history: Recent evaluation history
            best_move: Stockfish's best move
            second_best_eval: Second-best move evaluation (multiPV)
        
        Returns:
            True if position meets quality criteria
        """
        # Must be non-terminal
        if not PositionFilter.is_non_terminal(board, eval_score):
            return False
        
        # Must have stable evaluation (not random noise)
        if eval_history and not PositionFilter.is_stable_eval(eval_history):
            return False
        
        # Keep if ANY of these conditions are met:
        criteria_met = 0
        
        if PositionFilter.is_sharp_position(board, eval_score, prev_eval):
            criteria_met += 1
        
        if best_move and PositionFilter.is_tactical_pivot(
            board, best_move, eval_score, second_best_eval
        ):
            criteria_met += 1
        
        # Keep position if at least one criterion is met
        return criteria_met >= 1
    
    @staticmethod
    def compute_position_quality_score(
        board: chess.Board,
        eval_score: float,
        prev_eval: Optional[float] = None,
    ) -> float:
        """
        Compute a quality score for prioritizing positions.
        
        Higher scores = more valuable for training.
        
        Returns:
            Quality score (0.0 to 1.0)
        """
        score = 0.0
        
        # Reward sharp positions
        if prev_eval is not None:
            eval_change = abs(eval_score - prev_eval)
            score += min(eval_change, 0.5) * 0.4  # Up to 0.2
        
        # Reward complex positions (moderate piece count)
        piece_count = len(board.piece_map())
        if 10 <= piece_count <= 24:
            score += 0.2
        
        # Reward balanced positions
        eval_balance = 1.0 - abs(eval_score)
        score += eval_balance * 0.3
        
        # Reward positions with king pressure
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square:
                attackers = board.attackers(not color, king_square)
                if len(attackers) >= 1:
                    score += 0.15
                    break
        
        # Reward checks and captures
        if board.is_check():
            score += 0.15
        
        return min(score, 1.0)

