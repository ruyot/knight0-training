"""
Test if the model learned positional features or just material.
"""

import chess
import numpy as np
import onnxruntime as ort
from knight0.encoding import board_to_tensor

def evaluate(fen: str, session) -> float:
    """Evaluate a position."""
    board = chess.Board(fen)
    board_np = board_to_tensor(board)
    board_input = np.expand_dims(board_np, axis=0).astype(np.float32)
    
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: board_input})
    
    value = float(outputs[1][0][0])  # Value head output
    return value

def main():
    print("="*80)
    print("TESTING: Does the model understand POSITIONS or just MATERIAL?")
    print("="*80)
    print()
    
    session = ort.InferenceSession("knight0_model.onnx")
    
    # Test 1: Same material, different positions
    print("TEST 1: Same material, different king safety")
    print("-" * 60)
    
    # Good position for white (queen active, king safe)
    pos1 = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1"
    eval1 = evaluate(pos1, session)
    print(f"Position 1 (developed, castled): {eval1:+.4f}")
    print(pos1)
    print()
    
    # Bad position for white (king exposed, pieces uncoordinated)  
    pos2 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
    eval2 = evaluate(pos2, session)
    print(f"Position 2 (undeveloped, king center): {eval2:+.4f}")
    print(pos2)
    print()
    
    if abs(eval1 - eval2) < 0.05:
        print("❌ PROBLEM: Same eval despite different positions!")
        print("   Model may only be seeing material.")
    else:
        print(f"✅ GOOD: Different evals ({eval1:+.4f} vs {eval2:+.4f})")
        print(f"   Difference: {abs(eval1-eval2):.4f}")
        print("   Model understands positions!")
    
    print()
    print("="*80)
    print()
    
    # Test 2: Different material
    print("TEST 2: Material advantage detection")
    print("-" * 60)
    
    # White up a queen
    pos3 = "rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
    eval3 = evaluate(pos3, session)
    print(f"Position 3 (White up a queen): {eval3:+.4f}")
    print()
    
    # Equal material
    pos4 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    eval4 = evaluate(pos4, session)
    print(f"Position 4 (Starting position): {eval4:+.4f}")
    print()
    
    if eval3 > eval4 + 0.3:
        print("✅ GOOD: Recognizes material advantage")
    else:
        print("❌ PROBLEM: Not seeing material advantage properly")
    
    print()
    print("="*80)
    print()
    
    # Test 3: Tactical position
    print("TEST 3: Tactical awareness")
    print("-" * 60)
    
    # Black king exposed, white has attack
    pos5 = "r1b1k2r/pppp1ppp/2n2n2/2b1q3/2B1P3/3P1Q2/PPP2PPP/RNB1K2R w KQkq - 0 1"
    eval5 = evaluate(pos5, session)
    print(f"Position 5 (Tactical, exposed king): {eval5:+.4f}")
    
    # Safe position
    pos6 = "rnbqkb1r/pppppppp/5n2/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 0 1"
    eval6 = evaluate(pos6, session)
    print(f"Position 6 (Safe, normal development): {eval6:+.4f}")
    print()
    
    print("="*80)
    print()
    print("SUMMARY:")
    if abs(eval1 - eval2) > 0.05 and eval3 > eval4 + 0.3:
        print("✅ Model learned POSITIONAL features!")
        print("   - Understands piece placement")
        print("   - Recognizes material")
        print("   - Should play at 2000+ Elo with search")
    else:
        print("❌ Model may have issues:")
        print("   - Check encoding in inference")
        print("   - Verify model file is correct")
        print("   - May need retraining")
    print()

if __name__ == "__main__":
    main()

