"""
ChessNet: ResNet-style neural network for chess with policy and value heads.

Architecture:
- Initial convolutional layer
- N residual blocks
- Policy head: conv -> flatten -> linear -> logits
- Value head: conv -> flatten -> linear -> linear -> tanh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .config import CONFIGS


class ResidualBlock(nn.Module):
    """
    A single residual block with two conv layers and a skip connection.
    Includes dropout for regularization to prevent overfitting.
    """
    
    def __init__(self, filters: int, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)  # Dropout after activation
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out


class ChessNet(nn.Module):
    """
    ResNet-style neural network for chess with AlphaZero-like architecture.
    
    Args:
        input_channels: Number of input planes (default: 21)
        filters: Number of filters in conv layers (default: 128)
        num_blocks: Number of residual blocks (default: 10)
        n_moves: Number of possible moves for policy head (default: 4096)
    """
    
    def __init__(
        self,
        input_channels: int = 21,
        filters: int = 128,
        num_blocks: int = 10,
        n_moves: int = 4096,
        dropout: float = 0.15,  # Dropout rate (0.15 = 15%)
    ):
        super().__init__()
        self.input_channels = input_channels
        self.filters = filters
        self.num_blocks = num_blocks
        self.n_moves = n_moves
        
        # Initial convolutional layer
        self.conv_init = nn.Conv2d(
            input_channels, filters, kernel_size=3, padding=1, bias=False
        )
        self.bn_init = nn.BatchNorm2d(filters)
        
        # Residual tower with dropout
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(filters, dropout=dropout) for _ in range(num_blocks)
        ])
        
        # Policy head with dropout
        self.policy_conv = nn.Conv2d(filters, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_dropout = nn.Dropout(dropout * 1.5)  # Higher dropout in policy head
        self.policy_fc = nn.Linear(32 * 8 * 8, n_moves)
        
        # Value head with MINIMAL dropout (was too aggressive before!)
        self.value_conv = nn.Conv2d(filters, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_dropout1 = nn.Dropout(0.05)  # Much lower!
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_dropout2 = nn.Dropout(0.05)  # Much lower!
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, input_channels, 8, 8]
            
        Returns:
            Tuple of (policy_logits, value):
            - policy_logits: [batch, n_moves]
            - value: [batch, 1] in range [-1, 1]
        """
        # Initial convolution
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = F.relu(x)
        
        # Residual tower
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy head with dropout
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_dropout(policy)
        policy = self.policy_fc(policy)
        
        # Value head with dropout
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)  # Flatten
        value = self.value_dropout1(value)
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_dropout2(value)
        value = self.value_fc2(value)
        value = torch.tanh(value)
        
        return policy, value
    
    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(config_name: str = "medium", device: str = "cpu") -> ChessNet:
    """
    Create a ChessNet model from a configuration.
    
    Args:
        config_name: Name of the configuration ("small", "medium", or "large")
        device: Device to place the model on
        
    Returns:
        ChessNet model
    """
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(CONFIGS.keys())}")
    
    config = CONFIGS[config_name]
    model = ChessNet(**config).to(device)
    
    print(f"Created {config_name} model with {model.get_num_parameters():,} parameters")
    print(f"  - Input channels: {config['input_channels']}")
    print(f"  - Filters: {config['filters']}")
    print(f"  - Residual blocks: {config['num_blocks']}")
    print(f"  - Policy outputs: {config['n_moves']}")
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing ChessNet architectures...\n")
    
    for config_name in ["small", "medium", "large"]:
        print(f"Testing {config_name} model:")
        model = create_model(config_name)
        
        # Create dummy input
        batch_size = 4
        dummy_input = torch.randn(batch_size, 21, 8, 8)
        
        # Forward pass
        policy, value = model(dummy_input)
        
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Policy output shape: {policy.shape}")
        print(f"  Value output shape: {value.shape}")
        print(f"  Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")
        print()

