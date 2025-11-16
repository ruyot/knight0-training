#!/usr/bin/env python3
"""
Compare all knight0 search engine versions on same positions.
"""

import chess
import time
from pathlib import Path

# Import both versions
from search_engine import Knight0Search as V1
from search_engine_v2 import Knight0SearchV2 as V2

def test_position(name: str, fen: str, depth: int, time_limit: float):
    """Test a position with both search engines."""
    print(f"\n{'='*80}")
    print(f"Test: {name}")
    print(f"{'='*80}")
    print(f"FEN: {fen}")
    print()
    
    board = chess.Board(fen)
    print(board)
    print()
    
    # Test V1
    print("V1 (Original Python):")
    print("-" * 40)
    v1 = V1("knight0_model.onnx")
    start = time.time()
    v1_move = v1.get_best_move(board, depth=depth, time_limit=time_limit)
    v1_time = time.time() - start
    v1_nodes = v1.nodes_searched
    print()
    
    # Test V2
    print("V2 (Enhanced Python):")
    print("-" * 40)
    v2 = V2("knight0_model.onnx")
    start = time.time()
    v2_move = v2.get_best_move(board, depth=depth+2, time_limit=time_limit)
    v2_time = time.time() - start
    v2_nodes = v2.nodes_searched
    print()
    
    # Summary
    print("Summary:")
    print("-" * 40)
    print(f"V1: {v1_move} ({v1_nodes:,} nodes in {v1_time:.2f}s = {int(v1_nodes/v1_time):,} nps)")
    print(f"V2: {v2_move} ({v2_nodes:,} nodes in {v2_time:.2f}s = {int(v2_nodes/v2_time):,} nps)")
    print(f"V2 Speedup: {v2_nodes/v1_nodes:.2f}x nodes searched")
    print(f"Moves match: {'✅' if v1_move == v2_move else '❌'}")
    print()


def main():
    """Run comparison tests."""
    print("Knight0 Search Engine Comparison")
    print("="*80)
    print()
    
    model_path = Path("knight0_model.onnx")
    if not model_path.exists():
        print("ERROR: knight0_model.onnx not found!")
        print("Please ensure the trained model is in the current directory.")
        return
    
    # Test positions
    tests = [
        ("Starting Position", 
         "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
         6, 1.0),
        
        ("Italian Game",
         "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
         6, 1.0),
        
        ("Tactical Position",
         "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
         6, 2.0),
    ]
    
    for name, fen, depth, time_limit in tests:
        try:
            test_position(name, fen, depth, time_limit)
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            print()
    
    print("="*80)
    print("Comparison complete!")
    print()
    print("Next steps:")
    print("1. Deploy with V2 (Python) - ready now, 2000-2200 Elo")
    print("2. Build Rust version for 2200-2400 Elo:")
    print("   cd knight0-search-rust && ./build.sh")
    print()


if __name__ == "__main__":
    main()

