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
TRAINING_CONFIG = {
    "batch_size": 256,
    "learning_rate": 1e-3,
    "num_epochs": 50,
    "value_loss_weight": 0.5,  # alpha in total_loss = policy_loss + alpha * value_loss
    "checkpoint_every": 5,  # Save checkpoint every N epochs
    "gradient_clip": 1.0,
}

# Stockfish labeling config
STOCKFISH_CONFIG = {
    "depth": 15,
    "time_limit": 0.1,  # seconds
    "sample_rate": 5,  # Label every Nth move
    "min_move": 10,  # Start labeling from move 10
    "max_move": 60,  # Stop labeling after move 60
}

# Data filtering
DATA_FILTERS = {
    "min_elo": 2200,  # Minimum Elo for games
    "min_moves": 20,  # Minimum number of moves in a game
    "max_moves": 200,  # Maximum number of moves in a game
}

