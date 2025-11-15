# knight0 Training Pipeline

This repository contains the training pipeline for **knight0**, a chess engine trained using supervised learning and engine distillation. The trained model is exported to ONNX format for deployment on the ChessHacks platform.

## Architecture Overview

### Training Approach
- **Supervised learning + engine distillation**: High-quality games are labeled using Stockfish analysis
- **AlphaZero-style neural network**: ResNet with policy and value heads
- **No reinforcement learning**: Pure supervised learning for hackathon speed

### Neural Network
- **Architecture**: ResNet with residual blocks
- **Input**: `[batch, 21, 8, 8]` tensor encoding board state
  - 12 planes for piece positions (6 white + 6 black)
  - 1 plane for side to move
  - 4 planes for castling rights
  - 1 plane for en passant
  - 3 planes for auxiliary features (fifty-move, move number, game phase)
- **Outputs**:
  - **Policy head**: `[batch, 4096]` logits over all possible moves
  - **Value head**: `[batch, 1]` position evaluation in `[-1, 1]`

### Training Data Sources

We support multiple high-quality chess datasets:

1. **Lichess Elite Database** - High-rated online games (2400+ Elo)
2. **Lumbra's Gigabase** - Over-the-board classical games
3. **TCEC Games** - Top engine vs engine matches
4. **CCRL Games** - Computer chess rating list games
5. **Hugging Face Dataset** - 14M games with mean Elo ~2388

## Repository Structure

```
knight0-training/
├── modal_train.py              # Modal entrypoint (GPU training)
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── knight0/                    # Main package
    ├── __init__.py
    ├── config.py               # Configuration and hyperparameters
    ├── model.py                # ChessNet ResNet implementation
    ├── encoding.py             # Board/move encoding
    ├── dataset.py              # PyTorch Dataset
    ├── data_sources.py         # Download and manage PGN files
    ├── extract_positions.py    # Stockfish labeling
    ├── train_loop.py           # Training orchestration
    ├── export_onnx.py          # ONNX export
    └── utils.py                # Utility functions
```

## Setup

### Prerequisites

1. **Python 3.11+**
2. **Modal account** - Sign up at [modal.com](https://modal.com)
3. **Stockfish** - Install locally for testing (optional)

### Installation

```bash
# Clone this repository
git clone <your-repo-url>
cd knight0-training

# Install dependencies (for local testing)
pip install -r requirements.txt

# Install Modal CLI
pip install modal

# Authenticate with Modal
modal token new
```

### Stockfish Installation (for local testing)

```bash
# macOS
brew install stockfish

# Ubuntu/Debian
sudo apt-get install stockfish

# Or download from https://stockfishchess.org/download/
```

## Quick Start - Testing Before Training

### Dry Test #1: Local CPU Test (No Modal, No Stockfish)

Test all components with dummy data on your local machine:

```bash
# Run the local dry test
python3 test_local.py
```

This will verify:
- ✅ ChessDataset loads dummy positions
- ✅ ChessNet forward pass works
- ✅ Loss computation works
- ✅ Optimizer step works
- ✅ Encoding functions work
- ✅ Mini training loop runs

**Expected output**: All tests pass with green checkmarks.

### Dry Test #2: Modal Smoke Test

Test that Modal infrastructure is set up correctly:

```bash
# Run the Modal smoke test
modal run test_modal_smoke.py
```

This will verify:
- ✅ Modal image builds successfully
- ✅ GPU allocation works
- ✅ Volume mounts and is writable
- ✅ knight0 package imports correctly
- ✅ All dependencies are available
- ✅ Stockfish is accessible

**Expected output**: Remote logs show "SMOKE TEST PASSED!" with green checkmarks.

---

## Usage

### Quick Start - Training on Modal

```bash
# Run training with default settings (medium model, 50 epochs)
modal run modal_train.py

# Use a larger model
modal run modal_train.py --config large --epochs 100

# Quick test with small dataset (debugging)
modal run modal_train.py --test true --epochs 2
```

### Download Trained Model

```bash
# Download the ONNX model from Modal volume to local directory
modal run modal_train.py --action download
```

### Manage Modal Volume

```bash
# List all files in the Modal volume
modal run modal_train.py --action list

# Clear the volume (use with caution!)
modal run modal_train.py --action clear
```

## Training Pipeline

The training process consists of several stages:

### 1. Data Acquisition

Place PGN files in the Modal volume under `/root/knight0/raw/`:
- `lichess/` - Lichess Elite games
- `lumbra/` - Lumbra OTB games
- `tcec/` - TCEC engine games
- `ccrl/` - CCRL engine games

Or use the built-in test data generator for quick experiments.

### 2. Position Extraction & Labeling

The pipeline automatically:
1. Parses PGN files
2. Filters low-quality games (by Elo, length, etc.)
3. Samples positions from each game
4. Analyzes positions with Stockfish (depth 15, ~0.1s per position)
5. Extracts best move and evaluation
6. Saves to `training_data.pkl`

Configuration in `knight0/config.py`:
```python
STOCKFISH_CONFIG = {
    "depth": 15,
    "time_limit": 0.1,
    "sample_rate": 5,      # Label every 5th move
    "min_move": 10,        # Start from move 10
    "max_move": 60,        # Stop at move 60
}
```

### 3. Training

The trainer:
- Loads preprocessed positions
- Splits into train/val (90/10)
- Trains with Adam optimizer
- Uses combined loss: `policy_loss + 0.5 * value_loss`
- Saves checkpoints every 5 epochs
- Exports best model to ONNX

Key hyperparameters in `knight0/config.py`:
```python
TRAINING_CONFIG = {
    "batch_size": 256,
    "learning_rate": 1e-3,
    "num_epochs": 50,
    "value_loss_weight": 0.5,
}
```

### 4. ONNX Export

The trained model is automatically exported to ONNX format:
- Input: `[batch, 21, 8, 8]` float32
- Outputs:
  - `policy`: `[batch, 4096]` float32
  - `value`: `[batch, 1]` float32

## Model Configurations

Three pre-configured model sizes:

| Config | Filters | Blocks | Parameters | Use Case |
|--------|---------|--------|------------|----------|
| small  | 64      | 5      | ~500K      | Fast iteration/testing |
| medium | 128     | 10     | ~2M        | Balanced performance |
| large  | 256     | 20     | ~10M       | Maximum strength |

## Testing & Debugging

### Recommended Testing Flow

1. **First**: Run local dry test to verify components work
   ```bash
   python test_local.py
   ```

2. **Second**: Run Modal smoke test to verify infrastructure
   ```bash
   modal run test_modal_smoke.py
   ```

3. **Third**: Run quick training test with dummy data
   ```bash
   modal run modal_train.py --test true --epochs 2
   ```

4. **Finally**: Run full training with real data
   ```bash
   modal run modal_train.py --config medium --epochs 50
   ```

### Individual Component Testing

You can test components locally:

```bash
# Test model architecture
python -m knight0.model

# Test encoding
python -m knight0.encoding

# Test dataset
python -m knight0.dataset

# Test ONNX export
python -m knight0.export_onnx

# Test full pipeline (requires Stockfish)
python -m knight0.train_loop
```

## Advanced Usage

### Custom Training Script

```python
from knight0.train_loop import train_main

train_main(
    root_dir="/path/to/data",
    model_config="large",
    batch_size=512,
    num_epochs=100,
    learning_rate=1e-3,
    use_test_data=False,
    export_onnx_after=True,
)
```

### Custom Model Configuration

Edit `knight0/config.py`:

```python
CONFIGS["custom"] = {
    "input_channels": 21,
    "filters": 192,
    "num_blocks": 15,
    "n_moves": 4096,
}
```

### Process Your Own PGN Files

```python
from knight0.extract_positions import create_training_data
from pathlib import Path

pgn_files = [Path("my_games.pgn")]

create_training_data(
    root_dir="/path/to/output",
    pgn_paths=pgn_files,
    stockfish_path="stockfish",
    max_games_per_file=1000,
)
```

## Deployment to Inference Repo

Once training is complete:

1. Download the ONNX model:
   ```bash
   modal run modal_train.py --action download
   ```

2. Copy `knight0_model.onnx` to your inference repo

3. Use the **same encoding functions** in inference:
   - Copy `knight0/encoding.py` to your inference repo
   - Ensure `board_to_tensor()` and `move_to_index()` are identical

4. In your inference code:
   ```python
   import onnxruntime as ort
   import numpy as np
   from encoding import board_to_tensor, index_to_move
   import chess

   session = ort.InferenceSession("knight0_model.onnx")
   
   def get_move(pgn: str) -> str:
       board = chess.Board()
       # ... parse PGN to board ...
       
       # Encode board
       input_tensor = board_to_tensor(board)
       input_tensor = np.expand_dims(input_tensor, 0)  # Add batch dim
       
       # Run inference
       policy, value = session.run(None, {"input": input_tensor})
       
       # Mask illegal moves
       legal_moves = list(board.legal_moves)
       legal_indices = [move_to_index(m, board) for m in legal_moves]
       
       # Find best legal move
       masked_policy = policy[0].copy()
       masked_policy[~np.isin(np.arange(4096), legal_indices)] = -np.inf
       best_idx = np.argmax(masked_policy)
       best_move = index_to_move(best_idx, board)
       
       return best_move.uci()
   ```

## Troubleshooting

### Stockfish Not Found

```bash
# Install Stockfish or specify path
export STOCKFISH_PATH=/path/to/stockfish
```

### Out of Memory During Training

- Reduce batch size: `modal run modal_train.py --batch_size 128`
- Use smaller model: `modal run modal_train.py --config small`

### Modal Volume Issues

```bash
# Clear and restart
modal run modal_train.py --action clear
```

### Slow Position Extraction

- Reduce Stockfish depth in `config.py` (e.g., depth=12)
- Increase sample rate (e.g., every 10th move)
- Use fewer games: edit `extract_positions.py`

## Performance Expectations

### Training Time (on A10G GPU)

| Model  | Dataset Size | Epochs | Time   |
|--------|--------------|--------|--------|
| small  | 100K pos     | 50     | ~30min |
| medium | 1M pos       | 50     | ~3h    |
| large  | 10M pos      | 100    | ~24h   |

### Inference Speed (ONNX Runtime)

- Small model: ~1ms per position
- Medium model: ~3ms per position
- Large model: ~10ms per position

## Future Improvements

- [ ] Add reinforcement learning / self-play
- [ ] Multi-GPU training
- [ ] Model quantization (INT8) for faster inference
- [ ] Attention-based architectures
- [ ] Opening book integration
- [ ] Endgame tablebase integration

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **Stockfish** - Engine used for position labeling
- **python-chess** - Chess library
- **Modal** - Serverless GPU infrastructure
- **AlphaZero** - Architectural inspiration

---

