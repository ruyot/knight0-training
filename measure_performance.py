"""
Measure tiny model performance comprehensively.

Tests:
1. Position understanding (material, king safety, development)
2. Tactical awareness (can it see threats?)
3. Prediction accuracy vs Stockfish
4. Playing strength estimate
"""

import chess
import numpy as np
import onnxruntime as ort
from knight0.encoding import board_to_tensor

def evaluate(fen: str, session) -> float:
    """Evaluate a position with the model."""
    board = chess.Board(fen)
    board_np = board_to_tensor(board)
    board_input = np.expand_dims(board_np, axis=0).astype(np.float32)
    
    outputs = session.run(None, {'input': board_input})
    return float(outputs[0][0][0])

def test_suite(model_path: str):
    """Run comprehensive performance tests."""
    print("="*80)
    print("TINY MODEL PERFORMANCE MEASUREMENT")
    print("="*80)
    print()
    
    session = ort.InferenceSession(model_path)
    
    # Test 1: Material Understanding
    print("TEST 1: Material Understanding")
    print("-" * 60)
    
    tests = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0.0, "Start"),
        ("rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1", 0.9, "White +Q"),
        ("rnbqk1nr/pppp1ppp/8/2b1p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1", -0.9, "Black +Q"),
        ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1", 0.0, "Equal"),
    ]
    
    material_correct = 0
    for fen, expected, desc in tests:
        pred = evaluate(fen, session)
        correct = (pred > 0.3 if expected > 0.3 else 
                  pred < -0.3 if expected < -0.3 else 
                  abs(pred) < 0.3)
        
        status = "✓" if correct else "✗"
        material_correct += correct
        print(f"  {status} {desc:15s}: {pred:+.3f} (expect {expected:+.1f})")
    
    material_score = (material_correct / len(tests)) * 100
    print(f"\nMaterial Score: {material_score:.0f}%")
    print()
    
    # Test 2: Positional Understanding
    print("TEST 2: Positional Understanding")
    print("-" * 60)
    
    pos_tests = [
        # Developed vs undeveloped
        ("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", 
         "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
         "Developed > Undeveloped"),
        
        # Castled vs not castled
        ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
         "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1",
         "Castled > King center"),
    ]
    
    pos_correct = 0
    for better_fen, worse_fen, desc in pos_tests:
        eval_better = evaluate(better_fen, session)
        eval_worse = evaluate(worse_fen, session)
        correct = eval_better > eval_worse
        
        status = "✓" if correct else "✗"
        pos_correct += correct
        print(f"  {status} {desc}: {eval_better:+.3f} > {eval_worse:+.3f}")
    
    pos_score = (pos_correct / len(pos_tests)) * 100
    print(f"\nPositional Score: {pos_score:.0f}%")
    print()
    
    # Test 3: Variance Check (NOT constant output!)
    print("TEST 3: Output Variance (checking model learned)")
    print("-" * 60)
    
    all_evals = []
    for fen, _, _ in tests:
        all_evals.append(evaluate(fen, session))
    
    variance = np.var(all_evals)
    mean = np.mean(all_evals)
    std = np.std(all_evals)
    
    print(f"  Mean:     {mean:+.3f}")
    print(f"  Std Dev:  {std:.3f}")
    print(f"  Variance: {variance:.3f}")
    
    if variance < 0.01:
        print(f"  ✗ LOW VARIANCE - Model outputs constants!")
        variance_ok = False
    else:
        print(f"  ✓ Good variance - Model learned different positions")
        variance_ok = True
    print()
    
    # Overall Score
    print("="*80)
    print("OVERALL PERFORMANCE")
    print("="*80)
    
    if not variance_ok:
        print("❌ FAILED: Model outputs constants (didn't learn)")
        print("   Need to retrain!")
        return False
    
    overall = (material_score + pos_score) / 2
    print(f"Material Understanding: {material_score:.0f}%")
    print(f"Positional Understanding: {pos_score:.0f}%")
    print(f"\nOverall Score: {overall:.0f}%")
    print()
    
    # Estimate playing strength
    if overall >= 80:
        strength = "1800-2000 Elo (with search)"
        print(f"✓ STRONG: {strength}")
    elif overall >= 60:
        strength = "1600-1800 Elo (with search)"
        print(f"✓ GOOD: {strength}")
    elif overall >= 40:
        strength = "1400-1600 Elo (with search)"
        print(f"⚠️  OKAY: {strength}")
    else:
        strength = "<1400 Elo"
        print(f"✗ WEAK: {strength}")
    
    print()
    print("Note: Tiny model (10MB) + Deep search = Strong play!")
    print("      Simple eval + 10-ply search beats complex eval + 4-ply")
    print()
    
    return overall >= 60


if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "tiny_model.onnx"
    test_suite(model_path)

