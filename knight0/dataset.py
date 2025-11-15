"""
PyTorch Dataset for chess positions.

The dataset loads preprocessed positions from disk and converts them
to tensors suitable for training.
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import chess

from .encoding import board_to_tensor, move_to_index


class ChessDataset(Dataset):
    """
    PyTorch Dataset for chess positions.
    
    Each sample contains:
    - board_tensor: float32 [C, 8, 8] - encoded board state
    - move_idx: int - target move index for policy
    - value_target: float32 - target value for value head
    
    The dataset expects a pickle file with a list of dicts, each containing:
    - 'fen': FEN string of the position
    - 'move': UCI string of the best move
    - 'value': normalized score in [-1, 1]
    """
    
    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        """
        Args:
            data_path: Path to pickle file containing processed positions
            max_samples: Optional limit on number of samples to load
        """
        self.data_path = Path(data_path)
        
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        print(f"Loading dataset from {data_path}...")
        with open(self.data_path, 'rb') as f:
            self.positions = pickle.load(f)
        
        if max_samples is not None and max_samples < len(self.positions):
            self.positions = self.positions[:max_samples]
        
        print(f"Loaded {len(self.positions):,} positions")
    
    def __len__(self) -> int:
        return len(self.positions)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (board_tensor, move_idx, value_target):
            - board_tensor: [C, 8, 8] float32
            - move_idx: scalar int64 (for CrossEntropyLoss)
            - value_target: [1] float32
        """
        position = self.positions[idx]
        
        # Parse the board
        board = chess.Board(position['fen'])
        
        # Encode board to tensor
        board_tensor = board_to_tensor(board)
        board_tensor = torch.from_numpy(board_tensor).float()
        
        # Parse move and convert to index
        move = chess.Move.from_uci(position['move'])
        move_idx = move_to_index(move, board)
        move_idx = torch.tensor(move_idx, dtype=torch.long)
        
        # Get value target
        value_target = torch.tensor([position['value']], dtype=torch.float32)
        
        return board_tensor, move_idx, value_target


class ChessDatasetFromMemory(Dataset):
    """
    In-memory version of ChessDataset for faster iteration.
    
    This version pre-converts all positions to tensors at initialization time,
    trading memory for speed.
    """
    
    def __init__(self, data_path: str, max_samples: Optional[int] = None):
        """
        Args:
            data_path: Path to pickle file containing processed positions
            max_samples: Optional limit on number of samples to load
        """
        print(f"Loading and preprocessing dataset from {data_path}...")
        with open(data_path, 'rb') as f:
            positions = pickle.load(f)
        
        if max_samples is not None and max_samples < len(positions):
            positions = positions[:max_samples]
        
        # Pre-process all positions
        self.board_tensors = []
        self.move_indices = []
        self.value_targets = []
        
        print(f"Preprocessing {len(positions):,} positions...")
        for i, position in enumerate(positions):
            if i % 10000 == 0:
                print(f"  Processed {i:,} / {len(positions):,}")
            
            try:
                board = chess.Board(position['fen'])
                board_tensor = board_to_tensor(board)
                
                move = chess.Move.from_uci(position['move'])
                move_idx = move_to_index(move, board)
                
                self.board_tensors.append(board_tensor)
                self.move_indices.append(move_idx)
                self.value_targets.append(position['value'])
            except Exception as e:
                # Skip invalid positions
                print(f"Warning: Skipping position {i}: {e}")
                continue
        
        # Convert to numpy arrays for faster indexing
        self.board_tensors = np.array(self.board_tensors, dtype=np.float32)
        self.move_indices = np.array(self.move_indices, dtype=np.int64)
        self.value_targets = np.array(self.value_targets, dtype=np.float32)
        
        print(f"Preprocessed {len(self.board_tensors):,} valid positions")
    
    def __len__(self) -> int:
        return len(self.board_tensors)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Tuple of (board_tensor, move_idx, value_target)
        """
        board_tensor = torch.from_numpy(self.board_tensors[idx])
        move_idx = torch.tensor(self.move_indices[idx], dtype=torch.long)
        value_target = torch.tensor([self.value_targets[idx]], dtype=torch.float32)
        
        return board_tensor, move_idx, value_target


def create_dataset(
    data_path: str,
    in_memory: bool = True,
    max_samples: Optional[int] = None
) -> Dataset:
    """
    Factory function to create a ChessDataset.
    
    Args:
        data_path: Path to pickle file
        in_memory: If True, use ChessDatasetFromMemory (faster but uses more RAM)
        max_samples: Optional limit on number of samples
        
    Returns:
        ChessDataset instance
    """
    if in_memory:
        return ChessDatasetFromMemory(data_path, max_samples)
    else:
        return ChessDataset(data_path, max_samples)


if __name__ == "__main__":
    # Test the dataset with a dummy pickle file
    import tempfile
    
    # Create dummy data
    dummy_positions = [
        {
            'fen': chess.STARTING_FEN,
            'move': 'e2e4',
            'value': 0.1
        },
        {
            'fen': 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1',
            'move': 'e7e5',
            'value': 0.05
        }
    ]
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
        pickle.dump(dummy_positions, f)
        temp_path = f.name
    
    # Test dataset
    print("Testing ChessDataset:")
    dataset = ChessDataset(temp_path)
    print(f"Dataset size: {len(dataset)}")
    
    board_tensor, move_idx, value_target = dataset[0]
    print(f"Sample 0:")
    print(f"  Board tensor shape: {board_tensor.shape}")
    print(f"  Move index: {move_idx}")
    print(f"  Value target: {value_target}")
    
    # Clean up
    import os
    os.unlink(temp_path)

