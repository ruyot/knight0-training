"""
Data augmentation for chess positions.

Increases training data diversity and prevents memorization by:
1. Board flipping (left-right symmetry)
2. Color inversion (play as Black)
3. Small evaluation noise
4. Move probability smoothing

This forces model to learn CONCEPTS not POSITIONS.
"""

import chess
import torch
import numpy as np
from typing import Dict, List, Tuple
import random


class ChessAugmentation:
    """
    Augment chess positions to improve generalization.
    """
    
    @staticmethod
    def flip_board_horizontal(board: chess.Board) -> chess.Board:
        """
        Flip board left-right (a-file ↔ h-file).
        
        This creates a symmetric position that's strategically identical
        but looks different to the NN.
        """
        flipped = chess.Board(None)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Map square to flipped square
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                flipped_file = 7 - file
                flipped_square = chess.square(flipped_file, rank)
                flipped.set_piece_at(flipped_square, piece)
        
        flipped.turn = board.turn
        
        # Adjust castling rights
        if board.has_kingside_castling_rights(chess.WHITE):
            flipped.set_castling_fen('Q' if board.has_queenside_castling_rights(chess.WHITE) else '')
        if board.has_queenside_castling_rights(chess.WHITE):
            flipped.set_castling_fen('K' if board.has_kingside_castling_rights(chess.WHITE) else '')
        
        return flipped
    
    @staticmethod
    def invert_colors(board: chess.Board) -> chess.Board:
        """
        Swap colors: White ↔ Black.
        
        Forces model to learn from both perspectives.
        """
        inverted = chess.Board(None)
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Flip piece color
                inverted_piece = chess.Piece(piece.piece_type, not piece.color)
                # Flip rank (mirror vertically)
                file = chess.square_file(square)
                rank = 7 - chess.square_rank(square)
                inverted_square = chess.square(file, rank)
                inverted.set_piece_at(inverted_square, inverted_piece)
        
        inverted.turn = not board.turn
        return inverted
    
    @staticmethod
    def add_eval_noise(value: float, noise_std: float = 0.02) -> float:
        """
        Add small noise to evaluation to prevent overfitting to exact values.
        
        Args:
            value: Original evaluation
            noise_std: Standard deviation of noise
        
        Returns:
            Noisy evaluation
        """
        noise = np.random.normal(0, noise_std)
        return np.clip(value + noise, -1.0, 1.0)
    
    @staticmethod
    def smooth_policy(
        move_probs: Dict[chess.Move, float],
        legal_moves: List[chess.Move],
        smoothing: float = 0.05
    ) -> Dict[chess.Move, float]:
        """
        Smooth policy distribution to prevent overconfidence.
        
        Adds small probability to all legal moves, preventing
        model from being 100% certain (which causes brittleness).
        
        Args:
            move_probs: Original move probabilities
            legal_moves: All legal moves
            smoothing: Amount of smoothing (0.0 = none, 1.0 = uniform)
        
        Returns:
            Smoothed probabilities
        """
        n_moves = len(legal_moves)
        uniform_prob = 1.0 / n_moves
        
        smoothed = {}
        for move in legal_moves:
            original_prob = move_probs.get(move, 0.0)
            smoothed[move] = (1 - smoothing) * original_prob + smoothing * uniform_prob
        
        # Renormalize
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {k: v / total for k, v in smoothed.items()}
        
        return smoothed
    
    @staticmethod
    def augment_position(
        board: chess.Board,
        value: float,
        best_move: chess.Move,
        augmentation_prob: float = 0.5
    ) -> List[Tuple[chess.Board, float, chess.Move]]:
        """
        Generate augmented versions of a position.
        
        Args:
            board: Original position
            value: Evaluation
            best_move: Best move
            augmentation_prob: Probability of applying each augmentation
        
        Returns:
            List of (board, value, move) tuples
        """
        augmented = [(board, value, best_move)]  # Original
        
        # Horizontal flip (50% chance)
        if random.random() < augmentation_prob:
            flipped_board = ChessAugmentation.flip_board_horizontal(board)
            # Need to flip the move too
            flipped_move = chess.Move(
                chess.square(7 - chess.square_file(best_move.from_square),
                           chess.square_rank(best_move.from_square)),
                chess.square(7 - chess.square_file(best_move.to_square),
                           chess.square_rank(best_move.to_square)),
                promotion=best_move.promotion
            )
            augmented.append((flipped_board, value, flipped_move))
        
        # Color inversion (50% chance)
        if random.random() < augmentation_prob:
            inverted_board = ChessAugmentation.invert_colors(board)
            # Invert evaluation
            inverted_value = -value
            # Transform move
            inverted_move = chess.Move(
                chess.square(chess.square_file(best_move.from_square),
                           7 - chess.square_rank(best_move.from_square)),
                chess.square(chess.square_file(best_move.to_square),
                           7 - chess.square_rank(best_move.to_square)),
                promotion=best_move.promotion
            )
            augmented.append((inverted_board, inverted_value, inverted_move))
        
        return augmented


# Example usage in training loop
def augment_batch(positions: List[Dict], augment_prob: float = 0.3) -> List[Dict]:
    """
    Augment a batch of training positions.
    
    Args:
        positions: List of position dicts
        augment_prob: Probability of augmenting each position
    
    Returns:
        Augmented position list (potentially larger)
    """
    augmenter = ChessAugmentation()
    augmented_positions = []
    
    for pos in positions:
        board = pos['board']
        value = pos['value']
        move_idx = pos['move']
        
        # Get move from index (need to implement)
        # For now, skip augmentation during extraction
        # Apply during DataLoader instead
        
        # Add original
        augmented_positions.append(pos)
        
        # Maybe add augmented version
        if random.random() < augment_prob:
            # Add eval noise
            noisy_value = augmenter.add_eval_noise(value)
            aug_pos = pos.copy()
            aug_pos['value'] = noisy_value
            augmented_positions.append(aug_pos)
    
    return augmented_positions

