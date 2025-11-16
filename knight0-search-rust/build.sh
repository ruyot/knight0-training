#!/bin/bash
# Build knight0 search engine (Rust)

set -e

echo "Building knight0 search engine (Rust)..."
echo "========================================"
echo ""

# Check if cargo is installed
if ! command -v cargo &> /dev/null; then
    echo "ERROR: Rust/Cargo not installed!"
    echo "Install from: https://rustup.rs/"
    exit 1
fi

echo "Rust version:"
rustc --version
cargo --version
echo ""

# Build release version
echo "Building with optimizations (this takes 2-5 minutes)..."
cargo build --release

echo ""
echo "✅ Build complete!"
echo ""
echo "Binary location: target/release/knight0-search"
echo "Size: $(du -h target/release/knight0-search | cut -f1)"
echo ""

# Test run
echo "Testing binary..."
echo ""

if [ -f "../knight0_model.onnx" ]; then
    echo "Running quick test..."
    ./target/release/knight0-search -m ../knight0_model.onnx -d 6 -t 0.5
    echo ""
    echo "✅ Test successful!"
else
    echo "⚠️  knight0_model.onnx not found - skipping test"
    echo "   (Place model in parent directory to test)"
fi

echo ""
echo "To use:"
echo "  ./target/release/knight0-search -m ../knight0_model.onnx -d 12 -t 1.0"
echo ""

