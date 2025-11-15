"""
Training orchestration for ChessNet.

This module contains the main training loop and related functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import time
import logging
from typing import Optional

from .model import ChessNet, create_model
from .dataset import create_dataset
from .extract_positions import create_training_data
from .export_onnx import export_to_onnx
from .config import TRAINING_CONFIG, CONFIGS
from .utils import format_time, setup_logging

logger = logging.getLogger(__name__)


class ChessTrainer:
    """
    Trainer class for ChessNet models.
    """
    
    def __init__(
        self,
        model: ChessNet,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = TRAINING_CONFIG["learning_rate"],
        value_loss_weight: float = TRAINING_CONFIG["value_loss_weight"],
        gradient_clip: float = TRAINING_CONFIG["gradient_clip"],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model: ChessNet model
            train_loader: Training data loader
            val_loader: Optional validation data loader
            learning_rate: Learning rate
            value_loss_weight: Weight for value loss (alpha)
            gradient_clip: Gradient clipping threshold
            device: Device to train on
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.value_loss_weight = value_loss_weight
        self.gradient_clip = gradient_clip
        
        # Loss functions
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=50,
            eta_min=1e-6
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self) -> dict:
        """
        Train for one epoch.
        
        Returns:
            Dict with average losses
        """
        self.model.train()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        for batch_idx, (boards, move_indices, value_targets) in enumerate(self.train_loader):
            # Move to device
            boards = boards.to(self.device)
            move_indices = move_indices.to(self.device)
            value_targets = value_targets.to(self.device)
            
            # Forward pass
            policy_out, value_out = self.model(boards)
            
            # Compute losses
            policy_loss = self.policy_criterion(policy_out, move_indices)
            value_loss = self.value_criterion(value_out, value_targets)
            total_batch_loss = policy_loss + self.value_loss_weight * value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            
            self.optimizer.step()
            
            # Accumulate losses
            total_loss += total_batch_loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
            
            # Log progress
            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    f"  Batch {batch_idx + 1}/{len(self.train_loader)}: "
                    f"loss={total_batch_loss.item():.4f} "
                    f"policy={policy_loss.item():.4f} "
                    f"value={value_loss.item():.4f}"
                )
        
        return {
            "loss": total_loss / num_batches,
            "policy_loss": total_policy_loss / num_batches,
            "value_loss": total_value_loss / num_batches,
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """
        Run validation.
        
        Returns:
            Dict with average validation losses
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        num_batches = 0
        
        for boards, move_indices, value_targets in self.val_loader:
            # Move to device
            boards = boards.to(self.device)
            move_indices = move_indices.to(self.device)
            value_targets = value_targets.to(self.device)
            
            # Forward pass
            policy_out, value_out = self.model(boards)
            
            # Compute losses
            policy_loss = self.policy_criterion(policy_out, move_indices)
            value_loss = self.value_criterion(value_out, value_targets)
            total_batch_loss = policy_loss + self.value_loss_weight * value_loss
            
            # Accumulate losses
            total_loss += total_batch_loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        return {
            "val_loss": total_loss / num_batches,
            "val_policy_loss": total_policy_loss / num_batches,
            "val_value_loss": total_value_loss / num_batches,
        }
    
    def train(
        self,
        num_epochs: int,
        checkpoint_dir: Path,
        checkpoint_every: int = TRAINING_CONFIG["checkpoint_every"],
    ):
        """
        Train for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            checkpoint_dir: Directory to save checkpoints
            checkpoint_every: Save checkpoint every N epochs
        """
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {self.model.get_num_parameters():,}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            logger.info(f"Learning rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics["loss"])
            
            # Validate
            val_metrics = self.validate()
            if val_metrics:
                self.val_losses.append(val_metrics["val_loss"])
            
            # Log epoch results
            epoch_time = time.time() - epoch_start
            logger.info(f"\nEpoch {epoch} completed in {format_time(epoch_time)}")
            logger.info(f"  Train loss: {train_metrics['loss']:.4f} "
                       f"(policy: {train_metrics['policy_loss']:.4f}, "
                       f"value: {train_metrics['value_loss']:.4f})")
            
            if val_metrics:
                logger.info(f"  Val loss: {val_metrics['val_loss']:.4f} "
                           f"(policy: {val_metrics['val_policy_loss']:.4f}, "
                           f"value: {val_metrics['val_value_loss']:.4f})")
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            if epoch % checkpoint_every == 0:
                self.save_checkpoint(checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")
            
            # Save best model
            current_val_loss = val_metrics.get("val_loss", train_metrics["loss"])
            if current_val_loss < self.best_val_loss:
                self.best_val_loss = current_val_loss
                self.save_checkpoint(checkpoint_dir / "best_model.pth")
                logger.info(f"  ✓ Saved best model (val_loss: {current_val_loss:.4f})")
        
        total_time = time.time() - start_time
        logger.info(f"\nTraining completed in {format_time(total_time)}")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final model
        final_path = checkpoint_dir / "final_model.pth"
        self.save_checkpoint(final_path)
        logger.info(f"Saved final model to {final_path}")
    
    def save_checkpoint(self, path: Path):
        """
        Save a checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }, path)
    
    def load_checkpoint(self, path: Path):
        """
        Load a checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")


def train_main(
    root_dir: str,
    model_config: str = "medium",
    batch_size: int = TRAINING_CONFIG["batch_size"],
    num_epochs: int = TRAINING_CONFIG["num_epochs"],
    learning_rate: float = TRAINING_CONFIG["learning_rate"],
    use_test_data: bool = False,
    export_onnx_after: bool = True,
):
    """
    Main training function.
    
    This is the entry point called from modal_train.py.
    
    Args:
        root_dir: Root directory (e.g., /root/knight0 on Modal)
        model_config: Model size ("small", "medium", "large")
        batch_size: Training batch size
        num_epochs: Number of epochs
        learning_rate: Initial learning rate
        use_test_data: If True, use small test dataset
        export_onnx_after: If True, export ONNX after training
    """
    # Setup logging
    root_path = Path(root_dir)
    log_file = root_path / "training.log"
    setup_logging(str(log_file))
    
    logger.info("="*80)
    logger.info("knight0 Training Pipeline")
    logger.info("="*80)
    logger.info(f"Root directory: {root_dir}")
    logger.info(f"Model config: {model_config}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Num epochs: {num_epochs}")
    logger.info(f"Learning rate: {learning_rate}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Step 1: Ensure training data exists
    data_path = root_path / "training_data.pkl"
    
    if not data_path.exists():
        logger.info("\nTraining data not found. Creating...")
        
        if use_test_data:
            # Use built-in test data
            data_path = create_training_data(
                root_dir=root_dir,
                use_test_data=True,
                max_games_per_file=100
            )
        else:
            # Look for real PGN files in the mounted data directory
            # The repo is mounted at /root/knight0_pkg, data/ folder is there
            pgn_search_paths = [
                Path("/root/knight0_pkg/data"),  # Look in data/ folder
            ]
            
            found_pgns = []
            for search_dir in pgn_search_paths:
                if search_dir.exists() and search_dir.is_dir():
                    # Find all .pgn files in this directory
                    found_pgns.extend(search_dir.glob("*.pgn"))
                    logger.info(f"Searching for PGNs in {search_dir}...")
            
            if found_pgns:
                logger.info(f"Found {len(found_pgns)} PGN file(s): {[p.name for p in found_pgns]}")
                data_path = create_training_data(
                    root_dir=root_dir,
                    pgn_paths=found_pgns,  # Use all found PGNs
                    use_test_data=False,
                    max_games_per_file=5000,  # Limit for 30-min run with depth 15
                )
            else:
                logger.warning("No PGN files found. Using test data instead.")
                data_path = create_training_data(
                    root_dir=root_dir,
                    use_test_data=True,
                    max_games_per_file=100
                )
    else:
        logger.info(f"\nUsing existing training data: {data_path}")
    
    # Step 2: Create dataset and data loaders
    logger.info("\nCreating dataset...")
    dataset = create_dataset(
        str(data_path),
        in_memory=True,
        max_samples=1000 if use_test_data else None
    )
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Train size: {train_size:,}")
    logger.info(f"Val size: {val_size:,}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Use 0 for Modal compatibility
        pin_memory=(device == "cuda")
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device == "cuda")
    )
    
    # Step 3: Create model
    logger.info("\nCreating model...")
    model = create_model(model_config, device=device)
    
    # Step 4: Create trainer
    logger.info("\nInitializing trainer...")
    trainer = ChessTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        device=device
    )
    
    # Step 5: Train
    checkpoint_dir = root_path / "checkpoints"
    trainer.train(
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir
    )
    
    # Step 6: Export to ONNX
    if export_onnx_after:
        logger.info("\nExporting model to ONNX...")
        onnx_path = root_path / "knight0_model.onnx"
        export_to_onnx(model, str(onnx_path))
        logger.info(f"ONNX model saved to {onnx_path}")
    
    logger.info("\n" + "="*80)
    logger.info("Training completed successfully!")
    logger.info("="*80)


if __name__ == "__main__":
    import tempfile
    
    # Test training with a tiny dataset
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Testing training in {tmpdir}")
        
        train_main(
            root_dir=tmpdir,
            model_config="small",
            batch_size=32,
            num_epochs=2,
            use_test_data=True,
            export_onnx_after=True
        )
        
        print("\nTraining test completed!")

