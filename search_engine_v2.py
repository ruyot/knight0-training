"""
ENHANCED Search Engine for knight0 - Professional strength!

New features over v1:
1. Null Move Pruning (+50-100 Elo)
2. Late Move Reductions/LMR (+100-200 Elo)
3. Killer Moves (+30-50 Elo)
4. MVV-LVA Capture Ordering (+20-40 Elo)
5. Check Extensions (+30-50 Elo)
6. Aspiration Windows (+20-30 Elo)
7. History Heuristic (+30-50 Elo)
8. Better Transposition Table with aging
"""

import chess
import onnxruntime as ort
import torch
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict

from knight0.encoding import board_to_tensor, move_to_index


@dataclass
class SearchResult:
    """Result of a search."""
    best_move: chess.Move
    eval: float
    depth: int
    nodes: int
    time_ms: int
    pv: List[chess.Move]  # Principal variation


class TranspositionTable:
    """
    Enhanced TT with aging and better replacement scheme.
    """
    def __init__(self, max_size: int = 10_000_000):  # 10M entries
        self.table = {}
        self.max_size = max_size
        self.generation = 0  # Age entries
    
    def new_search(self):
        """Start new search - increment generation for aging."""
        self.generation += 1
    
    def get(self, board_hash: int) -> Optional[Dict]:
        """Get cached evaluation."""
        return self.table.get(board_hash)
    
    def store(self, board_hash: int, depth: int, eval_score: float, 
              best_move: Optional[chess.Move], flag: str):
        """
        Store with depth-preferred replacement.
        """
        existing = self.table.get(board_hash)
        
        # Replace if: new entry, deeper search, or same gen
        if (not existing or 
            depth >= existing['depth'] or 
            existing['generation'] < self.generation):
            
            self.table[board_hash] = {
                'depth': depth,
                'eval': eval_score,
                'move': best_move,
                'flag': flag,
                'generation': self.generation
            }
        
        # Limit size (simple random eviction)
        if len(self.table) > self.max_size:
            # Remove 10% oldest entries
            to_remove = sorted(self.table.items(), 
                             key=lambda x: x[1]['generation'])[:self.max_size // 10]
            for key, _ in to_remove:
                del self.table[key]


class MoveOrdering:
    """
    Advanced move ordering for better alpha-beta cutoffs.
    Uses killer moves, history heuristic, and MVV-LVA.
    """
    def __init__(self):
        # Killer moves: quiet moves that caused cutoffs [ply][killer_slot]
        self.killer_moves = [[None, None] for _ in range(100)]
        
        # History heuristic: [piece][to_square]
        self.history = [[0] * 64 for _ in range(12)]
        
        # Piece values for MVV-LVA
        self.piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 300,
            chess.BISHOP: 300,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
    
    def clear_killers(self):
        """Clear killer moves for new search."""
        self.killer_moves = [[None, None] for _ in range(100)]
    
    def add_killer(self, move: chess.Move, ply: int):
        """Add killer move at this ply."""
        if move != self.killer_moves[ply][0]:
            self.killer_moves[ply][1] = self.killer_moves[ply][0]
            self.killer_moves[ply][0] = move
    
    def add_history(self, piece: chess.Piece, to_square: int, depth: int):
        """Update history score for this move."""
        piece_idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
        self.history[piece_idx][to_square] += depth * depth
    
    def mvv_lva_score(self, board: chess.Board, move: chess.Move) -> int:
        """
        Most Valuable Victim - Least Valuable Attacker.
        Higher score = better capture.
        """
        if not board.is_capture(move):
            return 0
        
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        
        if not victim or not attacker:
            return 0
        
        victim_value = self.piece_values.get(victim.piece_type, 0)
        attacker_value = self.piece_values.get(attacker.piece_type, 0)
        
        # Higher victim value - lower attacker value = better
        return victim_value * 10 - attacker_value
    
    def score_move(self, board: chess.Board, move: chess.Move, 
                   ply: int, tt_move: Optional[chess.Move],
                   nn_score: float) -> float:
        """
        Score a move for ordering.
        Higher = search earlier.
        """
        score = 0.0
        
        # 1. TT move (highest priority)
        if tt_move and move == tt_move:
            return 1_000_000.0
        
        # 2. Promotions
        if move.promotion:
            return 900_000.0 + (move.promotion * 1000)
        
        # 3. Captures (MVV-LVA)
        if board.is_capture(move):
            score = 800_000.0 + self.mvv_lva_score(board, move)
            return score
        
        # 4. Killer moves
        if move == self.killer_moves[ply][0]:
            return 700_000.0
        if move == self.killer_moves[ply][1]:
            return 600_000.0
        
        # 5. History heuristic
        piece = board.piece_at(move.from_square)
        if piece:
            piece_idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
            history_score = self.history[piece_idx][move.to_square]
            score = 500_000.0 + min(history_score, 100_000)
        
        # 6. NN policy score
        score += nn_score * 1000
        
        return score
    
    def order_moves(self, board: chess.Board, legal_moves: List[chess.Move],
                    ply: int, tt_move: Optional[chess.Move],
                    move_probs: Dict[chess.Move, float]) -> List[chess.Move]:
        """
        Order moves for search.
        """
        scored_moves = []
        for move in legal_moves:
            nn_score = move_probs.get(move, 0.0)
            score = self.score_move(board, move, ply, tt_move, nn_score)
            scored_moves.append((score, move))
        
        # Sort by score (highest first)
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored_moves]


class Knight0SearchV2:
    """
    Enhanced search engine with professional techniques.
    """
    
    def __init__(self, model_path: str):
        """
        Args:
            model_path: Path to ONNX model
        """
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.tt = TranspositionTable()
        self.move_ordering = MoveOrdering()
        self.nodes_searched = 0
        
        # Search parameters
        self.NULL_MOVE_REDUCTION = 2  # R value for null move pruning
        self.LMR_FULL_DEPTH_MOVES = 4  # First N moves at full depth
        self.LMR_REDUCTION_LIMIT = 3  # Don't reduce more than this
    
    def evaluate_position(self, board: chess.Board) -> Tuple[float, Dict[chess.Move, float]]:
        """
        Get NN evaluation of position.
        
        Returns:
            (value, move_probs) - value in [-1, 1], move_probs as dict
        """
        board_tensor = board_to_tensor(board).unsqueeze(0)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: board_tensor.numpy()})
        
        policy_logits = torch.tensor(outputs[0])
        value = torch.tensor(outputs[1]).item()
        
        # Get legal move probabilities
        probs = torch.softmax(policy_logits, dim=-1).squeeze()
        
        move_probs = {}
        for move in board.legal_moves:
            move_idx = move_to_index(move)
            move_probs[move] = probs[move_idx].item()
        
        return value, move_probs
    
    def quiescence_search(
        self,
        board: chess.Board,
        alpha: float,
        beta: float,
        ply: int,
        max_depth: int = 8
    ) -> float:
        """
        Enhanced quiescence search with check extension.
        """
        self.nodes_searched += 1
        
        # Check for draw by repetition or 50-move rule
        if board.is_repetition() or board.halfmove_clock >= 100:
            return 0.0
        
        # Stand-pat
        if not board.is_check():
            stand_pat, _ = self.evaluate_position(board)
            
            if stand_pat >= beta:
                return beta
            if alpha < stand_pat:
                alpha = stand_pat
            
            if max_depth <= 0:
                return stand_pat
        else:
            # In check - must search all moves
            if max_depth <= 0:
                max_depth = 1  # Extend in check
            stand_pat = -999.0
        
        # Generate tactical moves
        if board.is_check():
            # All moves when in check
            moves = list(board.legal_moves)
        else:
            # Only captures and promotions
            moves = [m for m in board.legal_moves 
                    if board.is_capture(m) or m.promotion]
        
        if not moves:
            if board.is_check():
                return -10.0 + ply  # Checkmate (prefer later mates)
            return stand_pat
        
        # Simple ordering for qsearch
        moves.sort(key=lambda m: (
            m.promotion if m.promotion else 0,
            self.move_ordering.mvv_lva_score(board, m)
        ), reverse=True)
        
        for move in moves:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, ply + 1, max_depth - 1)
            board.pop()
            
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        
        return alpha
    
    def alpha_beta(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
        pv: List[chess.Move],
        do_null: bool = True
    ) -> float:
        """
        Enhanced alpha-beta with all modern techniques.
        """
        self.nodes_searched += 1
        
        # Check draw
        if ply > 0 and (board.is_repetition() or board.halfmove_clock >= 100):
            return 0.0
        
        # Mate distance pruning
        alpha = max(alpha, -10.0 + ply)
        beta = min(beta, 10.0 - ply)
        if alpha >= beta:
            return alpha
        
        # Check TT
        board_hash = chess.polyglot.zobrist_hash(board)
        tt_entry = self.tt.get(board_hash)
        tt_move = None
        
        if tt_entry and tt_entry['depth'] >= depth:
            tt_move = tt_entry['move']
            if tt_entry['flag'] == 'exact':
                return tt_entry['eval']
            elif tt_entry['flag'] == 'lowerbound':
                alpha = max(alpha, tt_entry['eval'])
            elif tt_entry['flag'] == 'upperbound':
                beta = min(beta, tt_entry['eval'])
            
            if alpha >= beta:
                return tt_entry['eval']
        else:
            if tt_entry:
                tt_move = tt_entry['move']
        
        # Terminal node
        if depth <= 0:
            return self.quiescence_search(board, alpha, beta, ply)
        
        if board.is_game_over():
            if board.is_checkmate():
                return -10.0 + ply
            return 0.0
        
        in_check = board.is_check()
        
        # NULL MOVE PRUNING
        # If we can pass and still maintain beta, position is too good
        if (do_null and not in_check and depth >= 3 and 
            ply > 0 and not self._is_endgame(board)):
            
            board.push(chess.Move.null())
            null_score = -self.alpha_beta(
                board, depth - 1 - self.NULL_MOVE_REDUCTION,
                -beta, -beta + 0.001, ply + 1, [], False
            )
            board.pop()
            
            if null_score >= beta:
                return beta  # Null move cutoff
        
        # Get moves and order them
        eval_score, move_probs = self.evaluate_position(board)
        legal_moves = list(board.legal_moves)
        
        if not legal_moves:
            if in_check:
                return -10.0 + ply
            return 0.0
        
        ordered_moves = self.move_ordering.order_moves(
            board, legal_moves, ply, tt_move, move_probs
        )
        
        # Check extension
        if in_check:
            depth += 1
        
        best_move = None
        best_score = -999.0
        flag = 'upperbound'
        local_pv = []
        moves_searched = 0
        
        for move_idx, move in enumerate(ordered_moves):
            board.push(move)
            
            # LATE MOVE REDUCTIONS (LMR)
            # Search later moves at reduced depth
            reduction = 0
            if (moves_searched >= self.LMR_FULL_DEPTH_MOVES and
                depth >= 3 and
                not in_check and
                not board.is_check() and
                not board.is_capture(move) and
                not move.promotion):
                
                # Reduce more for later moves
                reduction = 1 + (moves_searched // 6)
                reduction = min(reduction, self.LMR_REDUCTION_LIMIT)
                reduction = min(reduction, depth - 1)
            
            child_pv = []
            
            # Search with reduction first
            if reduction > 0:
                score = -self.alpha_beta(
                    board, depth - 1 - reduction, -alpha - 0.001, -alpha,
                    ply + 1, child_pv, True
                )
                # If it raised alpha, re-search at full depth
                if score > alpha:
                    child_pv = []
                    score = -self.alpha_beta(
                        board, depth - 1, -beta, -alpha,
                        ply + 1, child_pv, True
                    )
            else:
                # PVS (Principal Variation Search)
                if moves_searched == 0:
                    # First move - full window
                    score = -self.alpha_beta(
                        board, depth - 1, -beta, -alpha,
                        ply + 1, child_pv, True
                    )
                else:
                    # Later moves - null window search
                    score = -self.alpha_beta(
                        board, depth - 1, -alpha - 0.001, -alpha,
                        ply + 1, [], True
                    )
                    # Re-search if null window failed high
                    if score > alpha and score < beta:
                        child_pv = []
                        score = -self.alpha_beta(
                            board, depth - 1, -beta, -alpha,
                            ply + 1, child_pv, True
                        )
            
            board.pop()
            moves_searched += 1
            
            if score > best_score:
                best_score = score
                best_move = move
                
                if score > alpha:
                    alpha = score
                    flag = 'exact'
                    local_pv = [move] + child_pv
                    
                    # Update history for good quiet moves
                    if not board.is_capture(move) and not move.promotion:
                        piece = board.piece_at(move.from_square)
                        if piece:
                            self.move_ordering.add_history(piece, move.to_square, depth)
                
                if score >= beta:
                    # Beta cutoff
                    flag = 'lowerbound'
                    
                    # Add killer move for quiet moves
                    if not board.is_capture(move) and not move.promotion:
                        self.move_ordering.add_killer(move, ply)
                    
                    break
        
        # Store in TT
        self.tt.store(board_hash, depth, best_score, best_move, flag)
        
        if flag == 'exact':
            pv[:] = local_pv
        
        return best_score
    
    def _is_endgame(self, board: chess.Board) -> bool:
        """Check if position is endgame (for null move pruning)."""
        # Simple heuristic: no queens or limited material
        white_material = sum(1 for sq in chess.SQUARES 
                           if board.piece_at(sq) and 
                           board.piece_at(sq).color == chess.WHITE and
                           board.piece_at(sq).piece_type != chess.KING)
        black_material = sum(1 for sq in chess.SQUARES 
                           if board.piece_at(sq) and 
                           board.piece_at(sq).color == chess.BLACK and
                           board.piece_at(sq).piece_type != chess.KING)
        return white_material <= 5 or black_material <= 5
    
    def iterative_deepening(
        self,
        board: chess.Board,
        max_depth: int = 12,
        time_limit: float = 1.0
    ) -> SearchResult:
        """
        Iterative deepening with aspiration windows.
        """
        start_time = time.time()
        self.nodes_searched = 0
        self.tt.new_search()
        self.move_ordering.clear_killers()
        
        best_move = None
        best_eval = 0.0
        best_pv = []
        
        # Start with wide window
        alpha = -999.0
        beta = 999.0
        
        for depth in range(1, max_depth + 1):
            if time.time() - start_time > time_limit:
                break
            
            # ASPIRATION WINDOWS (after depth 4)
            if depth >= 5:
                # Narrow window around previous eval
                window = 0.3
                alpha = best_eval - window
                beta = best_eval + window
                
                # Try narrow window
                pv = []
                eval_score = self.alpha_beta(board, depth, alpha, beta, 0, pv, True)
                
                # If failed high or low, re-search with wider window
                if eval_score <= alpha or eval_score >= beta:
                    alpha = -999.0
                    beta = 999.0
                    pv = []
                    eval_score = self.alpha_beta(board, depth, alpha, beta, 0, pv, True)
            else:
                pv = []
                eval_score = self.alpha_beta(board, depth, alpha, beta, 0, pv, True)
            
            if pv:
                best_move = pv[0]
                best_eval = eval_score
                best_pv = pv
            
            elapsed = time.time() - start_time
            nps = int(self.nodes_searched / elapsed) if elapsed > 0 else 0
            
            pv_str = ' '.join(m.uci() for m in pv[:8])
            print(f"depth {depth:2d}: eval={eval_score:+.3f}, "
                  f"nodes={self.nodes_searched:,}, nps={nps:,}, "
                  f"pv={pv_str}")
        
        elapsed_ms = int((time.time() - start_time) * 1000)
        
        return SearchResult(
            best_move=best_move or list(board.legal_moves)[0],
            eval=best_eval,
            depth=depth - 1,
            nodes=self.nodes_searched,
            time_ms=elapsed_ms,
            pv=best_pv
        )
    
    def get_best_move(
        self,
        board: chess.Board,
        depth: int = 12,
        time_limit: float = 1.0
    ) -> chess.Move:
        """
        Get best move for position.
        
        Args:
            board: Current position
            depth: Search depth
            time_limit: Time limit in seconds
        
        Returns:
            Best move
        """
        result = self.iterative_deepening(board, depth, time_limit)
        print(f"\nBest: {result.best_move.uci()} "
              f"(eval={result.eval:+.3f}, depth={result.depth}, "
              f"nodes={result.nodes:,}, nps={result.nodes*1000//result.time_ms:,})")
        return result.best_move


if __name__ == "__main__":
    # Test enhanced search
    from pathlib import Path
    
    model_path = Path("knight0_model.onnx")
    if not model_path.exists():
        print("ERROR: knight0_model.onnx not found!")
        exit(1)
    
    search = Knight0SearchV2(str(model_path))
    
    # Test tactical position
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    
    print("Enhanced Search Engine V2 - Testing")
    print("=" * 60)
    print(board)
    print()
    
    best_move = search.get_best_move(board, depth=10, time_limit=5.0)
    print(f"\nRecommended: {best_move}")

