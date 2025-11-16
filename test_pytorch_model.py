"""
Test the PyTorch model directly (not ONNX) to see if the issue is in export.
"""

import torch
import chess
from knight0.model import ChessNet, CONFIGS
from knight0.encoding import board_to_tensor

def evaluate_pytorch(fen: str, model) -> float:
    """Evaluate using PyTorch model directly."""
    board = chess.Board(fen)
    board_np = board_to_tensor(board)
    board_tensor = torch.from_numpy(board_np).unsqueeze(0).float()
    
    with torch.no_grad():
        _, value = model(board_tensor)
    
    return float(value[0][0])

def main():
    print("="*80)
    print("TESTING PYTORCH MODEL (not ONNX) - Is it the export that's broken?")
    print("="*80)
    print()
    
    # Load PyTorch checkpoint
    print("Loading PyTorch checkpoint...")
    checkpoint = torch.load("knight0_best_model.pth", map_location='cpu')
    
    print(f"Checkpoint info:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Val Loss: {checkpoint['val_loss']:.6f}")
    print()
    
    # Create and load model
    config = CONFIGS["large"]
    model = ChessNet(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded!")
    print()
    
    # Test same positions as before
    print("TEST: Different positions")
    print("-" * 60)
    
    pos1 = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1"
    pos2 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"
    pos3 = "rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1"  # White up a queen
    pos4 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Starting
    
    eval1 = evaluate_pytorch(pos1, model)
    eval2 = evaluate_pytorch(pos2, model)
    eval3 = evaluate_pytorch(pos3, model)
    eval4 = evaluate_pytorch(pos4, model)
    
    print(f"Position 1 (developed):     {eval1:+.4f}")
    print(f"Position 2 (undeveloped):   {eval2:+.4f}")
    print(f"Position 3 (white +queen):  {eval3:+.4f}")
    print(f"Position 4 (starting pos):  {eval4:+.4f}")
    print()
    
    # Check if they're different
    all_same = (abs(eval1 - eval2) < 0.01 and 
                abs(eval1 - eval3) < 0.01 and 
                abs(eval1 - eval4) < 0.01)
    
    print("="*80)
    print()
    if all_same:
        print("❌ PyTorch model ALSO outputs constants!")
        print("   → Training failed (not an ONNX export issue)")
        print()
        print("Possible causes:")
        print("  1. Value head didn't get gradients (check dropout/architecture)")
        print("  2. Training data had issues")
        print("  3. Learning rate too high/low")
        print("  4. Batch norm frozen")
    else:
        print("✅ PyTorch model works!")
        print(f"   Position variance: {max(eval1,eval2,eval3,eval4) - min(eval1,eval2,eval3,eval4):.4f}")
        print()
        print("❌ ONNX export is BROKEN!")
        print("   → Need to fix export_onnx.py")
    print()

if __name__ == "__main__":
    main()

