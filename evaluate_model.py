"""
Evaluate knight0 model on validation data.

This script computes move accuracy metrics to quantify model quality.
"""

import torch
import pickle
from pathlib import Path
from tqdm import tqdm
import numpy as np

from knight0.model import ChessNet, MODEL_CONFIGS
from knight0.dataset import ChessDataset


def evaluate_model(
    model_path: str,
    data_path: str,
    model_config: str = "medium",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_samples: int = 10000,
):
    """
    Evaluate model on validation data.
    
    Metrics:
    - Top-1 accuracy: % where best move matches Stockfish
    - Top-3 accuracy: % where Stockfish move is in top 3
    - Top-5 accuracy: % where Stockfish move is in top 5
    - Average rank: Mean rank of Stockfish move in policy output
    
    Args:
        model_path: Path to model (.pth or .onnx)
        data_path: Path to validation data (.pkl)
        model_config: Model size
        device: Device to run on
        max_samples: Maximum number of samples to evaluate
    """
    print("="*80)
    print("knight0 Model Evaluation")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Data: {data_path}")
    print(f"Device: {device}")
    print()
    
    # Load model
    print("Loading model...")
    config = MODEL_CONFIGS[model_config]
    model = ChessNet(**config).to(device)
    
    if model_path.endswith('.onnx'):
        print("⚠ ONNX evaluation not yet implemented, load PyTorch model")
        return
    else:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    print(f"✓ Loaded model from {model_path}")
    
    # Load validation data
    print("\nLoading validation data...")
    with open(data_path, 'rb') as f:
        positions = pickle.load(f)
    
    # Take only a subset if too large
    if len(positions) > max_samples:
        import random
        random.seed(42)
        positions = random.sample(positions, max_samples)
    
    print(f"✓ Loaded {len(positions)} positions")
    
    # Create dataset
    dataset = ChessDataset(positions)
    
    # Evaluate
    print("\nEvaluating...")
    top1_correct = 0
    top3_correct = 0
    top5_correct = 0
    ranks = []
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            board_tensor, move_idx, value_target = dataset[i]
            
            # Forward pass
            board_tensor = board_tensor.unsqueeze(0).to(device)
            policy_logits, value_pred = model(board_tensor)
            
            # Get top-k predictions
            policy_probs = torch.softmax(policy_logits[0], dim=0)
            top_k = torch.topk(policy_probs, k=10)
            
            # Check if ground truth is in top-k
            ground_truth_idx = move_idx.item()
            
            if ground_truth_idx in top_k.indices[:1]:
                top1_correct += 1
            if ground_truth_idx in top_k.indices[:3]:
                top3_correct += 1
            if ground_truth_idx in top_k.indices[:5]:
                top5_correct += 1
            
            # Find rank of ground truth
            sorted_indices = torch.argsort(policy_probs, descending=True)
            rank = (sorted_indices == ground_truth_idx).nonzero(as_tuple=True)[0].item() + 1
            ranks.append(rank)
    
    # Compute metrics
    n = len(dataset)
    top1_acc = 100.0 * top1_correct / n
    top3_acc = 100.0 * top3_correct / n
    top5_acc = 100.0 * top5_correct / n
    avg_rank = np.mean(ranks)
    median_rank = np.median(ranks)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Samples evaluated: {n}")
    print()
    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-3 Accuracy: {top3_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")
    print()
    print(f"Average Rank: {avg_rank:.1f}")
    print(f"Median Rank: {median_rank:.0f}")
    print("="*80)
    
    # Interpretation
    print("\n📊 Interpretation:")
    if top1_acc > 50:
        print("✓ EXCELLENT: Top-1 accuracy >50% is very strong!")
    elif top1_acc > 35:
        print("✓ GOOD: Top-1 accuracy >35% is decent")
    elif top1_acc > 20:
        print("⚠ FAIR: Top-1 accuracy >20% is okay but needs improvement")
    else:
        print("⚠ WEAK: Top-1 accuracy <20% indicates the model needs more training/data")
    
    if top3_acc > 70:
        print("✓ Good positional understanding (Stockfish move usually in top-3)")
    
    return {
        'top1_accuracy': top1_acc,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'average_rank': avg_rank,
        'median_rank': median_rank,
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate knight0 model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to validation data (.pkl)")
    parser.add_argument("--config", type=str, default="medium", choices=["small", "medium", "large"])
    parser.add_argument("--max-samples", type=int, default=10000, help="Max samples to evaluate")
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        data_path=args.data,
        model_config=args.config,
        max_samples=args.max_samples
    )

