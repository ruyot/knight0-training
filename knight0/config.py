"""
Global configuration for the knight0 training pipeline.
"""

# Board encoding dimensions
INPUT_CHANNELS = 21  # 12 pieces + 1 turn + 4 castling + 1 en-passant + 2 extras + 1 fifty-move

# Move encoding
N_MOVES = 4096  # 64 from_squares * 64 to_squares (promotions handled separately)

# Model configurations
CONFIGS = {
    "small": {
        "input_channels": INPUT_CHANNELS,
        "filters": 64,
        "num_blocks": 5,
        "n_moves": N_MOVES,
    },
    "medium": {
        "input_channels": INPUT_CHANNELS,
        "filters": 128,
        "num_blocks": 10,
        "n_moves": N_MOVES,
    },
    "large": {
        "input_channels": INPUT_CHANNELS,
        "filters": 256,
        "num_blocks": 20,
        "n_moves": N_MOVES,
    },
}

# Training hyperparameters (defaults)
# Optimized for ~5M position training run (Option A)
TRAINING_CONFIG = {
    "batch_size": 512,  # Larger batch for better gradient estimates with more data
    "learning_rate": 5e-4,  # Conservative LR
    "num_epochs": 100,  # Early stopping will kick in when needed
    "value_loss_weight": 0.5,  # alpha in total_loss = policy_loss + alpha * value_loss
    "checkpoint_every": 5,  # Save checkpoint every N epochs
    "gradient_clip": 1.0,
    "weight_decay": 1e-3,  # L2 regularization (CRITICAL for preventing overfit)
    "dropout": 0.15,  # Dropout rate for model
}

# Stockfish labeling config (SLOW - not recommended)
STOCKFISH_CONFIG = {
    "depth": 5,
    "time_limit": 0.01,
    "sample_rate": 8,
    "min_move": 15,
    "max_move": 50,
}

# LC0 (Leela Chess Zero) config - MUCH FASTER! (10-100x)
LC0_CONFIG = {
    "visits": 50,  # Number of NN evaluations (very fast with GPU!)
    "sample_rate": 8,  # Label every 8th move
    "min_move": 15,  # Start from move 15
    "max_move": 50,  # End at move 50
}

# Data filtering
DATA_FILTERS = {
    "min_elo": 0,  # No Elo filtering - datasets are already high quality!
    "min_moves": 20,  # Minimum number of moves in a game
    "max_moves": 200,  # Maximum number of moves in a game
}

