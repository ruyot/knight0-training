"""
Board and move encoding for the knight0 chess engine.

This module provides consistent encoding between training and inference:
- board_to_tensor: Converts chess.Board -> numpy array [C, 8, 8]
- move_to_index: Converts chess.Move -> integer index [0, N_MOVES)
- index_to_move: Converts integer index -> chess.Move
"""

import numpy as np
import chess
from typing import Optional


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert a chess.Board to a tensor representation [C, 8, 8].
    
    Planes (21 total):
    - 0-5: White pieces (pawn, knight, bishop, rook, queen, king)
    - 6-11: Black pieces (pawn, knight, bishop, rook, queen, king)
    - 12: Side to move (1.0 if white, 0.0 if black)
    - 13: White kingside castling
    - 14: White queenside castling
    - 15: Black kingside castling
    - 16: Black queenside castling
    - 17: En passant file (one-hot encoding of file, or all zeros)
    - 18: Fifty-move rule counter (normalized)
    - 19: Move number (normalized)
    - 20: Game phase indicator (opening/middlegame/endgame heuristic)
    
    Args:
        board: python-chess Board object
        
    Returns:
        numpy array of shape [21, 8, 8], dtype float32
    """
    tensor = np.zeros((21, 8, 8), dtype=np.float32)
    
    # Planes 0-11: Piece positions
    piece_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            rank = chess.square_rank(square)
            file = chess.square_file(square)
            plane_offset = 0 if piece.color == chess.WHITE else 6
            plane_idx = plane_offset + piece_to_plane[piece.piece_type]
            tensor[plane_idx, rank, file] = 1.0
    
    # Plane 12: Side to move
    tensor[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    
    # Planes 13-16: Castling rights
    tensor[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    tensor[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    tensor[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    tensor[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    
    # Plane 17: En passant
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        tensor[17, :, ep_file] = 1.0
    
    # Plane 18: Fifty-move rule counter (normalized to [0, 1])
    tensor[18, :, :] = board.halfmove_clock / 100.0
    
    # Plane 19: Move number (normalized, capped at 100 for normalization)
    move_number = board.fullmove_number
    tensor[19, :, :] = min(move_number / 100.0, 1.0)
    
    # Plane 20: Game phase indicator (heuristic based on material)
    # Simple heuristic: count total material (excluding kings)
    material_count = 0
    for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        material_count += len(board.pieces(piece_type, chess.WHITE))
        material_count += len(board.pieces(piece_type, chess.BLACK))
    # Normalize: opening ~32 pieces, endgame ~4-8 pieces
    game_phase = max(0.0, min(1.0, (material_count - 4) / 28.0))
    tensor[20, :, :] = game_phase
    
    return tensor


def move_to_index(move: chess.Move, board: Optional[chess.Board] = None) -> int:
    """
    Convert a chess.Move to an integer index.
    
    Encoding scheme:
    - Standard moves: from_square * 64 + to_square
    - Promotions: We handle them by mapping to the same index as the base move,
      since the neural network will learn the most common promotion (queen).
      For simplicity in this hackathon version, we treat all promotions
      as the same move index (the base from->to).
    
    This gives us a 4096-dimensional policy output (64 * 64).
    
    Args:
        move: python-chess Move object
        board: Optional Board object (for future use if needed)
        
    Returns:
        Integer index in [0, 4096)
    """
    from_square = move.from_square
    to_square = move.to_square
    
    # For promotions, we use the base move index
    # The model will implicitly learn to prefer queen promotions
    index = from_square * 64 + to_square
    
    assert 0 <= index < 4096, f"Invalid move index: {index}"
    return index


def index_to_move(idx: int, board: chess.Board) -> Optional[chess.Move]:
    """
    Convert an integer index to a chess.Move.
    
    This function validates that the resulting move is legal on the given board.
    For promotions, it defaults to queen promotion.
    
    Args:
        idx: Integer index in [0, 4096)
        board: Board object to validate legality
        
    Returns:
        chess.Move object if the move is legal, None otherwise
    """
    if not (0 <= idx < 4096):
        return None
    
    from_square = idx // 64
    to_square = idx % 64
    
    # Check if this is a pawn promotion move
    piece = board.piece_at(from_square)
    if piece and piece.piece_type == chess.PAWN:
        to_rank = chess.square_rank(to_square)
        if (piece.color == chess.WHITE and to_rank == 7) or \
           (piece.color == chess.BLACK and to_rank == 0):
            # This is a promotion move, default to queen
            move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
            if move in board.legal_moves:
                return move
    
    # Standard move
    move = chess.Move(from_square, to_square)
    if move in board.legal_moves:
        return move
    
    return None


def get_legal_move_indices(board: chess.Board) -> list[int]:
    """
    Get the list of legal move indices for the current board position.
    
    This is useful for masking the policy output during inference.
    
    Args:
        board: Board object
        
    Returns:
        List of integer indices corresponding to legal moves
    """
    legal_indices = []
    for move in board.legal_moves:
        idx = move_to_index(move, board)
        legal_indices.append(idx)
    return legal_indices

