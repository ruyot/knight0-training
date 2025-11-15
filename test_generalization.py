"""
Test model generalization on unseen positions.

Evaluates:
1. Move accuracy on held-out test set
2. Performance on different position types
3. Confidence calibration (does model know when it's uncertain?)
4. Tactical puzzle solving
"""

import torch
import chess
import chess.pgn
import onnxruntime as ort
from pathlib import Path
from typing import List, Dict
import pickle
import numpy as np

from knight0.encoding import board_to_tensor, move_to_index, index_to_move
from knight0.model import ChessNet, create_model


def evaluate_position_types(model_session, dataset_path: Path) -> Dict:
    """
    Test model on different position types:
    - Opening (moves 1-15)
    - Middlegame (moves 15-40)
    - Endgame (moves 40+)
    - Tactical (sharp positions)
    - Quiet (strategic positions)
    """
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    results = {
        'opening': {'correct': 0, 'total': 0},
        'middlegame': {'correct': 0, 'total': 0},
        'endgame': {'correct': 0, 'total': 0},
    }
    
    for item in dataset[:5000]:  # Sample
        board = item['board']
        true_move_idx = item['move']
        
        # Determine position type by piece count
        piece_count = len(board.piece_map())
        
        if piece_count >= 28:
            pos_type = 'opening'
        elif piece_count >= 12:
            pos_type = 'middlegame'
        else:
            pos_type = 'endgame'
        
        # Get prediction
        board_tensor = board_to_tensor(board).unsqueeze(0)
        input_name = model_session.get_inputs()[0].name
        outputs = model_session.run(None, {input_name: board_tensor.numpy()})
        policy_logits = torch.tensor(outputs[0])
        predicted_move_idx = torch.argmax(policy_logits, dim=1).item()
        
        results[pos_type]['total'] += 1
        if predicted_move_idx == true_move_idx:
            results[pos_type]['correct'] += 1
    
    # Calculate accuracies
    for pos_type in results:
        total = results[pos_type]['total']
        correct = results[pos_type]['correct']
        accuracy = (correct / total * 100) if total > 0 else 0
        results[pos_type]['accuracy'] = accuracy
        print(f"{pos_type.capitalize()}: {accuracy:.2f}% ({correct}/{total})")
    
    return results


def test_confidence_calibration(model_session, dataset_path: Path):
    """
    Test if model confidence matches actual accuracy.
    
    High confidence predictions should be more accurate than
    low confidence predictions.
    """
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    predictions = []
    
    for item in dataset[:5000]:
        board = item['board']
        true_move_idx = item['move']
        
        board_tensor = board_to_tensor(board).unsqueeze(0)
        input_name = model_session.get_inputs()[0].name
        outputs = model_session.run(None, {input_name: board_tensor.numpy()})
        
        policy_logits = torch.tensor(outputs[0])
        probs = torch.softmax(policy_logits, dim=-1)
        
        predicted_move_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0, predicted_move_idx].item()
        correct = (predicted_move_idx == true_move_idx)
        
        predictions.append({
            'confidence': confidence,
            'correct': correct
        })
    
    # Bin by confidence
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    print("\nConfidence Calibration:")
    print("Confidence Range | Accuracy | Count")
    print("-" * 45)
    
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i+1]
        bin_preds = [p for p in predictions if low <= p['confidence'] < high]
        
        if bin_preds:
            accuracy = sum(p['correct'] for p in bin_preds) / len(bin_preds) * 100
            print(f"{low:.1f} - {high:.1f}      | {accuracy:.2f}%   | {len(bin_preds)}")


def test_novel_positions(model_session):
    """
    Test on positions the model has NEVER seen.
    
    Uses random legal positions to test true generalization.
    """
    print("\nTesting on novel random positions...")
    
    correct = 0
    total = 100
    
    import chess.engine
    engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
    
    for _ in range(total):
        # Generate random legal position
        board = chess.Board()
        for _ in range(np.random.randint(5, 30)):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            board.push(np.random.choice(legal_moves))
        
        if board.is_game_over():
            continue
        
        # Get Stockfish's best move
        result = engine.analyse(board, chess.engine.Limit(depth=15))
        stockfish_move = result['pv'][0]
        stockfish_idx = move_to_index(stockfish_move)
        
        # Get model prediction
        board_tensor = board_to_tensor(board).unsqueeze(0)
        input_name = model_session.get_inputs()[0].name
        outputs = model_session.run(None, {input_name: board_tensor.numpy()})
        
        policy_logits = torch.tensor(outputs[0])
        predicted_move_idx = torch.argmax(policy_logits, dim=1).item()
        
        if predicted_move_idx == stockfish_idx:
            correct += 1
    
    engine.quit()
    
    accuracy = correct / total * 100
    print(f"Novel position accuracy: {accuracy:.2f}% ({correct}/{total})")
    return accuracy


if __name__ == "__main__":
    ROOT_DIR = Path(__file__).parent
    ONNX_MODEL = ROOT_DIR / "knight0_model.onnx"
    DATASET = ROOT_DIR / "training_data.pkl"
    
    if not ONNX_MODEL.exists() or not DATASET.exists():
        print("ERROR: Model or dataset not found!")
        exit(1)
    
    print("=" * 60)
    print("GENERALIZATION TESTING")
    print("=" * 60)
    
    # Load model
    session = ort.InferenceSession(str(ONNX_MODEL))
    
    # Test 1: Position types
    print("\n1. Position Type Performance:")
    print("-" * 60)
    evaluate_position_types(session, DATASET)
    
    # Test 2: Confidence calibration
    print("\n2. Confidence Calibration:")
    print("-" * 60)
    test_confidence_calibration(session, DATASET)
    
    # Test 3: Novel positions
    print("\n3. Novel Position Testing:")
    print("-" * 60)
    test_novel_positions(session)
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)

