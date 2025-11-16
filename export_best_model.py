"""
Export the best trained model to ONNX format for inference.
"""

import torch
from pathlib import Path
from knight0.model import ChessNet, CONFIGS
from knight0.export_onnx import export_to_onnx

def main():
    print("="*80)
    print("EXPORTING BEST MODEL TO ONNX")
    print("="*80)
    print("")
    
    # Load checkpoint
    checkpoint_path = Path("knight0_best_model.pth")
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Val Loss: {checkpoint['val_loss']:.6f}")
    print("")
    
    # Create model
    print("Creating LARGE model...")
    config = CONFIGS["large"]
    model = ChessNet(**config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    print("")
    
    # Export to ONNX
    output_path = Path("knight0_model.onnx")
    print(f"Exporting to: {output_path}")
    print("")
    
    export_to_onnx(model, str(output_path))
    
    print("")
    print("="*80)
    print("EXPORT COMPLETE!")
    print("="*80)
    print(f"Model ready at: {output_path}")
    print(f"Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    print("")
    print("Next steps:")
    print("  1. Copy knight0_model.onnx to your Rust engine directory")
    print("  2. Use it as the evaluation function in your search engine")
    print("  3. Deploy and dominate! 🚀")

if __name__ == "__main__":
    main()

