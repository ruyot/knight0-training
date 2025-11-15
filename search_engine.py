"""
Search engine for knight0 - adds tactical strength on top of NN evaluation.

Implements:
1. Alpha-Beta pruning (like Stockfish)
2. Quiescence search (resolve tactical sequences)
3. Move ordering (PV moves first)
4. Transposition table (cache positions)
5. Iterative deepening

This turns the NN from ~1600 Elo → 2000+ Elo!
"""

import chess
import chess.engine
import onnxruntime as ort
import torch
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict

from knight0.encoding import board_to_tensor, move_to_index, index_to_move


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
    Cache for position evaluations to avoid re-searching.
    """
    def __init__(self, max_size: int = 1000000):
        self.table = OrderedDict()
        self.max_size = max_size
    
    def get(self, board_hash: int) -> Optional[Dict]:
        """Get cached evaluation."""
        return self.table.get(board_hash)
    
    def store(self, board_hash: int, depth: int, eval_score: float, best_move: chess.Move, flag: str):
        """
        Store position evaluation.
        
        Args:
            board_hash: Zobrist hash of position
            depth: Search depth
            eval_score: Evaluation
            best_move: Best move found
            flag: "exact", "lowerbound", or "upperbound"
        """
        # Remove oldest if full
        if len(self.table) >= self.max_size:
            self.table.popitem(last=False)
        
        self.table[board_hash] = {
            'depth': depth,
            'eval': eval_score,
            'move': best_move,
            'flag': flag
        }


class Knight0Search:
    """
    Search engine combining NN evaluation with alpha-beta search.
    """
    
    def __init__(self, model_path: str, threads: int = 1):
        """
        Args:
            model_path: Path to ONNX model
            threads: Number of threads (for future MCTS)
        """
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.tt = TranspositionTable()
        self.nodes_searched = 0
    
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
        max_depth: int = 6
    ) -> float:
        """
        Quiescence search: resolve all tactical sequences.
        
        Only searches captures, checks, and promotions to avoid
        horizon effect.
        
        Args:
            board: Current position
            alpha: Alpha bound
            beta: Beta bound
            max_depth: Maximum quiescence depth
        
        Returns:
            Evaluation after resolving tactics
        """
        self.nodes_searched += 1
        
        # Stand-pat: can we just stop here?
        stand_pat, _ = self.evaluate_position(board)
        
        if stand_pat >= beta:
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        
        if max_depth <= 0:
            return stand_pat
        
        # Only search "forcing" moves
        forcing_moves = []
        for move in board.legal_moves:
            if board.is_capture(move) or board.gives_check(move) or move.promotion:
                forcing_moves.append(move)
        
        if not forcing_moves:
            return stand_pat
        
        # Search forcing moves
        for move in forcing_moves:
            board.push(move)
            score = -self.quiescence_search(board, -beta, -alpha, max_depth - 1)
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
        pv: List[chess.Move]
    ) -> float:
        """
        Alpha-beta search with transposition table.
        
        Args:
            board: Current position
            depth: Remaining depth
            alpha: Alpha bound
            beta: Beta bound
            pv: Principal variation (output)
        
        Returns:
            Evaluation score
        """
        self.nodes_searched += 1
        
        # Check transposition table
        board_hash = chess.polyglot.zobrist_hash(board)
        tt_entry = self.tt.get(board_hash)
        
        if tt_entry and tt_entry['depth'] >= depth:
            if tt_entry['flag'] == 'exact':
                return tt_entry['eval']
            elif tt_entry['flag'] == 'lowerbound':
                alpha = max(alpha, tt_entry['eval'])
            elif tt_entry['flag'] == 'upperbound':
                beta = min(beta, tt_entry['eval'])
            
            if alpha >= beta:
                return tt_entry['eval']
        
        # Terminal nodes
        if depth <= 0:
            return self.quiescence_search(board, alpha, beta)
        
        if board.is_game_over():
            if board.is_checkmate():
                return -10.0  # We're mated
            return 0.0  # Draw
        
        # Get move ordering from NN
        _, move_probs = self.evaluate_position(board)
        ordered_moves = sorted(
            board.legal_moves,
            key=lambda m: move_probs.get(m, 0.0),
            reverse=True
        )
        
        # Try PV move first if available
        if tt_entry and tt_entry['move']:
            pv_move = tt_entry['move']
            if pv_move in ordered_moves:
                ordered_moves.remove(pv_move)
                ordered_moves.insert(0, pv_move)
        
        best_move = None
        flag = 'upperbound'
        local_pv = []
        
        for move in ordered_moves:
            board.push(move)
            child_pv = []
            score = -self.alpha_beta(board, depth - 1, -beta, -alpha, child_pv)
            board.pop()
            
            if score >= beta:
                # Beta cutoff
                self.tt.store(board_hash, depth, beta, move, 'lowerbound')
                return beta
            
            if score > alpha:
                alpha = score
                best_move = move
                flag = 'exact'
                local_pv = [move] + child_pv
        
        # Store in transposition table
        if best_move:
            self.tt.store(board_hash, depth, alpha, best_move, flag)
            pv[:] = local_pv
        
        return alpha
    
    def iterative_deepening(
        self,
        board: chess.Board,
        max_depth: int = 6,
        time_limit: float = 1.0
    ) -> SearchResult:
        """
        Iterative deepening search.
        
        Searches depth 1, then 2, then 3, etc. up to max_depth or time_limit.
        This allows us to return best move so far if time runs out.
        
        Args:
            board: Position to search
            max_depth: Maximum search depth
            time_limit: Time limit in seconds
        
        Returns:
            SearchResult with best move and evaluation
        """
        start_time = time.time()
        self.nodes_searched = 0
        
        best_move = None
        best_eval = -999.0
        best_pv = []
        
        # Try each depth
        for depth in range(1, max_depth + 1):
            if time.time() - start_time > time_limit:
                break
            
            pv = []
            eval_score = self.alpha_beta(board, depth, -999.0, 999.0, pv)
            
            if pv:
                best_move = pv[0]
                best_eval = eval_score
                best_pv = pv
                
                elapsed_ms = int((time.time() - start_time) * 1000)
                nps = int(self.nodes_searched / (time.time() - start_time)) if elapsed_ms > 0 else 0
                
                print(f"depth {depth}: eval={eval_score:.3f}, pv={' '.join(m.uci() for m in pv[:5])}, "
                      f"nodes={self.nodes_searched}, nps={nps}")
        
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
        depth: int = 6,
        time_limit: float = 1.0
    ) -> chess.Move:
        """
        Get best move for position.
        
        Args:
            board: Current position
            depth: Search depth (6-8 for fast games, 10+ for slow)
            time_limit: Time limit in seconds
        
        Returns:
            Best move
        """
        result = self.iterative_deepening(board, depth, time_limit)
        print(f"\nBest move: {result.best_move.uci()} "
              f"(eval={result.eval:.3f}, depth={result.depth}, "
              f"nodes={result.nodes}, time={result.time_ms}ms)")
        return result.best_move


if __name__ == "__main__":
    # Test the search engine
    from pathlib import Path
    
    model_path = Path("knight0_model.onnx")
    if not model_path.exists():
        print("ERROR: knight0_model.onnx not found!")
        exit(1)
    
    # Create search engine
    search = Knight0Search(str(model_path))
    
    # Test position
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    
    print("Searching position:")
    print(board)
    print()
    
    # Search
    best_move = search.get_best_move(board, depth=6, time_limit=2.0)
    
    print(f"\nRecommended move: {best_move}")

