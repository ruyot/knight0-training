"""
Train knight0 on the Kaggle dataset (109M positions) - VALUE-ONLY TRAINING.

IMPORTANT: This dataset has position evaluations but NO move data.
We train ONLY the value head to learn position evaluation.
For move selection, we use traditional search (minimax/alpha-beta) with the NN as eval function.

This is actually a STRONG approach used by many chess engines:
- Deep evaluation network (109M positions)
- Traditional search for move selection
- No need for policy head with random/arbitrary moves
"""

import modal
import pickle
from pathlib import Path

app = modal.App("knight0-kaggle-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "python-chess>=1.999",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
        "onnxscript>=0.1.0"
    )
    .add_local_dir("knight0", remote_path="/root/knight0_pkg/knight0")
)

volume = modal.Volume.from_name("knight0-volume", create_if_missing=True)
VOLUME_PATH = "/root/knight0"


@app.function(
    image=image,
    gpu="A100", 
    timeout=43200,  # 12 hours for full 109M dataset
    volumes={VOLUME_PATH: volume},
)
def train_on_kaggle_data():
    """
    Train on Kaggle chess dataset (109M positions) with large model.
    
    Configuration:
    - 109M positions (entire dataset)
    - Large model (~50M params)
    - A100 GPU (80GB VRAM)
    - 30 epochs with early stopping
    - Checkpoints saved every 4 epochs
    """
    import sys
    sys.path.insert(0, "/root/knight0_pkg")
    
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader, random_split
    from knight0.model import ChessNet, CONFIGS
    from knight0.encoding import board_to_tensor, move_to_index
    import chess
    import pickle
    from pathlib import Path
    from tqdm import tqdm
    
    print("="*80)
    print("TRAINING KNIGHT0 ON KAGGLE DATASET")
    print("="*80)
    print("")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("")
    
    # Load Kaggle dataset
    data_path = Path(VOLUME_PATH) / "kaggle_training_data.pkl"
    print(f"Loading dataset from {data_path}...")
    
    with open(data_path, 'rb') as f:
        all_positions = pickle.load(f)
    
    print(f"Loaded {len(all_positions):,} positions")
    print("")
    
    # Create train/val split (90/10)
    train_size = int(0.9 * len(all_positions))
    val_size = len(all_positions) - train_size
    
    print(f"Dataset split:")
    print(f"   Training:   {train_size:,} positions (90%)")
    print(f"   Validation: {val_size:,} positions (10%)")
    print("")
    
    # Dataset class
    class KaggleChessDataset(Dataset):
        def __init__(self, positions):
            self.positions = positions
        
        def __len__(self):
            return len(self.positions)
        
        def __getitem__(self, idx):
            pos = self.positions[idx]
            board = chess.Board(pos["board"])
            board_np = board_to_tensor(board)
            board_tensor = torch.from_numpy(board_np).float()
            
            # VALUE-ONLY TRAINING: No policy head training
            # We only need the evaluation for search-based move selection
            value = pos["value"]
            return board_tensor, value
    
    # Split data
    train_positions = all_positions[:train_size]
    val_positions = all_positions[train_size:]
    
    train_dataset = KaggleChessDataset(train_positions)
    val_dataset = KaggleChessDataset(val_positions)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=1024,  # HUGE batch for A100! (80GB VRAM)
        shuffle=True, 
        num_workers=8,  # More workers for 109M dataset
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"Created dataloaders")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Val batches: {len(val_loader)}")
    print("")
    
    # Create model (LARGE)
    print("Creating model (LARGE)...")
    config = CONFIGS["large"]
    model = ChessNet(**config).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model created: {total_params:,} parameters")
    print("")
    
    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,  # Higher learning rate for faster convergence
        weight_decay=1e-3,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=30,  # More epochs for 109M positions!
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Training loop
    print("Starting training (up to 30 epochs with early stopping)...")
    print("Checkpoints will be saved every 4 epochs")
    print("")
    
    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0
    checkpoint_interval = 4  # Save checkpoint every 4 epochs
    
    for epoch in range(30):
        print(f"{'='*80}")
        print(f"EPOCH {epoch + 1}/30")
        print(f"{'='*80}")
        
        # TRAINING (VALUE-ONLY)
        model.train()
        train_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for boards, values in progress_bar:
            boards = boards.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True).float()
            
            optimizer.zero_grad()
            
            # Only use value head, ignore policy
            _, value_pred = model(boards)
            
            # VALUE-ONLY LOSS
            loss = value_criterion(value_pred.squeeze(), values)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            
            progress_bar.set_postfix({
                'value_loss': f'{loss.item():.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        
        # VALIDATION (VALUE-ONLY)
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for boards, values in tqdm(val_loader, desc="Validation"):
                boards = boards.to(device, non_blocking=True)
                values = values.to(device, non_blocking=True).float()
                
                # Only use value head
                _, value_pred = model(boards)
                
                # VALUE-ONLY LOSS
                loss = value_criterion(value_pred.squeeze(), values)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print("")
        print(f"Epoch {epoch + 1} Results:")
        print(f"   Train Value Loss: {avg_train_loss:.4f}")
        print(f"   Val Value Loss:   {avg_val_loss:.4f}")
        print(f"   LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            model_path = Path(VOLUME_PATH) / "knight0_best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
            }, model_path)
            volume.commit()
            print(f"   Best model saved (val_loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"   No improvement ({patience_counter}/{patience})")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_path = Path(VOLUME_PATH) / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'best_val_loss': best_val_loss,
            }, checkpoint_path)
            volume.commit()
            print(f"   Checkpoint saved: checkpoint_epoch_{epoch + 1}.pth")
        
        print("")
        
        if patience_counter >= patience:
            print(f"Early stopping triggered (no improvement for {patience} epochs)")
            break
    
    # Load best model and export to ONNX
    print("")
    print("="*80)
    print("EXPORTING TO ONNX")
    print("="*80)
    print("")
    
    model_path = Path(VOLUME_PATH) / "knight0_best_model.pth"
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded best model from epoch {checkpoint['epoch']}")
    print(f"Val loss: {checkpoint['val_loss']:.4f}")
    
    # Export
    dummy_input = torch.randn(1, 21, 8, 8).to(device)
    onnx_path = Path(VOLUME_PATH) / "knight0_model.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['policy', 'value'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'policy': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    
    volume.commit()
    
    print(f"ONNX model exported: {onnx_path}")
    print(f"  Size: {onnx_path.stat().st_size / (1024**2):.1f} MB")
    print("")
    
    # Verify
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path))
    test_input = dummy_input.cpu().numpy()
    outputs = session.run(None, {'input': test_input})
    
    print("ONNX verification passed")
    print(f"  Policy shape: {outputs[0].shape}")
    print(f"  Value shape: {outputs[1].shape}")
    print("")
    
    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("")
    print("Model ready at: knight0_model.onnx")
    print("")
    
    return {
        "best_val_loss": best_val_loss,
        "total_positions": len(all_positions),
        "model_params": total_params
    }


@app.local_entrypoint()
def main():
    """Run the training!"""
    print("Starting Kaggle dataset training on Modal GPU...")
    print("")
    result = train_on_kaggle_data.remote()
    print("")
    print("Training complete")
    print(f"   Best val loss: {result['best_val_loss']:.4f}")
    print(f"   Total positions: {result['total_positions']:,}")
    print(f"   Model params: {result['model_params']:,}")

