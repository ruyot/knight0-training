"""
Simple training script

For 10MB model size requirement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import chess
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import encoding
import sys
sys.path.append('/Users/tahmeed_t/Documents/knight0-training')
from knight0.encoding import board_to_tensor

# ============================================================================
# TINY MODEL (under 10MB!)
# ============================================================================

class TinyChessNet(nn.Module):
    """
    Tiny chess network - ~1-2M parameters = ~8MB model size.
    """
    def __init__(self):
        super().__init__()
        
        # Keep it SMALL for 10MB limit
        filters = 64  # Much smaller than before!
        
        # Initial conv
        self.conv1 = nn.Conv2d(21, filters, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters)
        
        # Just 4 residual blocks (vs 20 before)
        self.res_blocks = nn.ModuleList([
            self._make_res_block(filters) for _ in range(4)
        ])
        
        # Value head (NO DROPOUT!)
        self.value_conv = nn.Conv2d(filters, 16, 1)
        self.value_bn = nn.BatchNorm2d(16)
        self.value_fc1 = nn.Linear(16 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
    def _make_res_block(self, filters):
        return nn.Sequential(
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.BatchNorm2d(filters),
        )
    
    def forward(self, x):
        # Initial
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = F.relu(x + residual)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return value


# ============================================================================
# SIMPLE DATASET
# ============================================================================

class SimpleDataset(Dataset):
    def __init__(self, positions):
        self.positions = positions
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        pos = self.positions[idx]
        board = chess.Board(pos["board"])
        board_tensor = torch.from_numpy(board_to_tensor(board)).float()
        value = torch.tensor([pos["value"]], dtype=torch.float32)
        return board_tensor, value


# ============================================================================
# TRAINING
# ============================================================================

def train():
    print("="*80)
    print("TINY MODEL TRAINING - Simple and Guaranteed to Work")
    print("="*80)
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()
    
    # Load data
    data_path = Path("training_data.pkl")
    if not data_path.exists():
        print("ERROR: training_data.pkl not found!")
        print("Run this first:")
        print("  python3 -c \"from knight0.data_sources import setup_quick_test_data; setup_quick_test_data()\"")
        return
    
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        all_positions = pickle.load(f)
    
    # Use first 10k positions (enough for tiny model)
    positions = all_positions[:10000]
    print(f"Using {len(positions)} positions")
    print()
    
    # Split
    split = int(0.9 * len(positions))
    train_data = SimpleDataset(positions[:split])
    val_data = SimpleDataset(positions[split:])
    
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64)
    
    print(f"Train: {len(train_data)} positions")
    print(f"Val:   {len(val_data)} positions")
    print()
    
    # Model
    model = TinyChessNet().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
    
    print(f"Model: {total_params:,} parameters (~{model_size_mb:.1f} MB)")
    print()
    
    if model_size_mb > 10:
        print(f"WARNING: Model is {model_size_mb:.1f}MB (limit is 10MB)")
        print("Need to make it smaller!")
        return
    
    # Simple optimizer (NO fancy tricks!)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Test positions to validate training is working
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0.0, "Start"),
        ("rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1", 0.9, "White +Q"),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1", 0.0, "Equal"),
    ]
    
    def test_model_understanding(model, device):
        """Check if model learned positions."""
        model.eval()
        evals = []
        with torch.no_grad():
            for fen, expected, name in test_positions:
                board = chess.Board(fen)
                tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).float().to(device)
                pred = model(tensor).item()
                evals.append(pred)
                print(f"    {name:10s}: {pred:+.3f} (expect ~{expected:+.1f})")
        
        # Check variance (should NOT be constant like before!)
        variance = np.var(evals)
        if variance < 0.01:
            print(f"    ⚠️  WARNING: Low variance ({variance:.4f}) - model may not be learning!")
        else:
            print(f"    ✓ Variance: {variance:.4f} (good!)")
        return variance
    
    # Train for 20 epochs
    print("Training for 20 epochs...")
    print()
    
    best_val_loss = float('inf')
    
    for epoch in range(20):
        # Train
        model.train()
        train_loss = 0
        
        for boards, values in tqdm(train_loader, desc=f"Epoch {epoch+1}/20"):
            boards = boards.to(device)
            values = values.to(device)
            
            optimizer.zero_grad()
            pred = model(boards)
            loss = criterion(pred, values)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for boards, values in val_loader:
                boards = boards.to(device)
                values = values.to(device)
                pred = model(boards)
                loss = criterion(pred, values)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:2d}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")
        
        # Test on known positions
        print(f"  Testing understanding:")
        test_model_understanding(model, device)
        
        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "tiny_model_best.pth")
            print(f"  → Saved best model (val_loss={best_val_loss:.6f})")
    
    print()
    print("="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"Best val loss: {best_val_loss:.6f}")
    print()
    
    # Export to ONNX
    print("Exporting to ONNX...")
    model.load_state_dict(torch.load("tiny_model_best.pth"))
    model.eval()
    
    dummy_input = torch.randn(1, 21, 8, 8).to(device)
    torch.onnx.export(
        model,
        dummy_input,
        "tiny_model.onnx",
        input_names=['input'],
        output_names=['value'],
        dynamic_axes={'input': {0: 'batch_size'}, 'value': {0: 'batch_size'}}
    )
    
    onnx_size = Path("tiny_model.onnx").stat().st_size / (1024 * 1024)
    print(f"✓ Exported: tiny_model.onnx ({onnx_size:.1f} MB)")
    print()
    
    # Quick test
    print("Quick test on different positions...")
    import onnxruntime as ort
    session = ort.InferenceSession("tiny_model.onnx")
    
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Start
        "rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1",  # White up queen
    ]
    
    for fen in test_positions:
        board = chess.Board(fen)
        board_tensor = torch.from_numpy(board_to_tensor(board)).unsqueeze(0).float().numpy()
        result = session.run(None, {'input': board_tensor})
        print(f"  {fen[:30]}... → {result[0][0][0]:+.4f}")
    
    print()
    print("Done! Use tiny_model.onnx in your bot.")


if __name__ == "__main__":
    train()

