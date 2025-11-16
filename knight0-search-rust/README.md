# Knight0 Search Engine (Rust)

High-performance chess search engine for knight0, implemented in Rust for 10-100x speedup over Python.

## Features

All the pro search techniques:
- ✅ Alpha-beta pruning with transposition tables
- ✅ Null move pruning
- ✅ Late move reductions (LMR)
- ✅ Principal variation search (PVS)
- ✅ Killer moves
- ✅ History heuristic
- ✅ MVV-LVA capture ordering
- ✅ Check extensions
- ✅ Aspiration windows
- ✅ Quiescence search
- ✅ Iterative deepening

## Performance

| Implementation | Nodes/sec | Depth (1s) | Est. Elo |
|----------------|-----------|------------|----------|
| Python v1 | ~10K | 6-7 | 1800-2000 |
| Python v2 | ~15K | 7-8 | 1900-2100 |
| **Rust** | **500K-1M** | **8-10** | **2200-2400** |

## Building

```bash
cd knight0-search-rust
cargo build --release
```

The binary will be in `target/release/knight0-search`.

## Usage

### Command Line

```bash
# Analyze starting position
./target/release/knight0-search -m ../knight0_model.onnx

# Analyze specific position
./target/release/knight0-search \
  -m ../knight0_model.onnx \
  -f "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3" \
  -d 12 \
  -t 5.0
```

### As Library

```rust
use knight0_search::SearchEngine;

let mut engine = SearchEngine::new("knight0_model.onnx")?;
let result = engine.search(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    12,  // depth
    1.0  // time limit
)?;

println!("Best move: {}", result.best_move);
println!("Eval: {}", result.eval);
println!("Nodes: {}", result.nodes);
```

## Integration with Python

You can call the Rust engine from Python for best of both worlds:

```python
import subprocess
import json

def rust_search(fen, depth=12, time=1.0):
    result = subprocess.run([
        './knight0-search-rust/target/release/knight0-search',
        '-m', 'knight0_model.onnx',
        '-f', fen,
        '-d', str(depth),
        '-t', str(time)
    ], capture_output=True, text=True)
    
    # Parse output for best move
    # ... (implement parsing)
    return best_move
```

## Development

### Run tests
```bash
cargo test
```

### Run with optimizations
```bash
cargo run --release -- -m ../knight0_model.onnx
```

### Profile
```bash
cargo build --release
perf record ./target/release/knight0-search -m ../knight0_model.onnx -d 10
perf report
```

## TODO

- [ ] Improve move encoding/indexing for NN policy
- [ ] Add UCI protocol support
- [ ] Multi-threaded search
- [ ] SIMD optimizations for board encoding
- [ ] Better TT replacement scheme
- [ ] Syzygy tablebase support
- [ ] Opening book

## Architecture

```
knight0-search-rust/
├── src/
│   ├── main.rs          # CLI entry point
│   ├── search.rs        # Main search engine
│   ├── tt.rs            # Transposition table
│   ├── move_ordering.rs # Move ordering heuristics
│   ├── eval.rs          # NN evaluation
│   └── encoding.rs      # Board tensor encoding
├── Cargo.toml           # Dependencies
└── README.md
```

## Why Rust?

1. **Speed**: 10-100x faster than Python
2. **Safety**: Memory safe without GC pauses
3. **Concurrency**: Easy to parallelize search
4. **Zero-cost abstractions**: High-level code, low-level performance

## License

MIT

