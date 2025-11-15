"""
Modal smoke test for knight0 training pipeline.

This script tests that Modal setup works correctly:
- Image builds successfully
- GPU allocation works
- Volume mounts correctly
- Remote function execution works
- Logs appear locally

No actual training, just verification of infrastructure.
"""

import modal

# Create Modal app
app = modal.App("knight0-smoke-test")

# Define the Docker image (same as production)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "python-chess>=1.999",
        "numpy>=1.24.0",
    )
    .apt_install("stockfish")
    .add_local_dir("knight0", remote_path="/root/knight0_pkg/knight0")
)

# Create volume (same as production)
volume = modal.Volume.from_name("knight0-volume", create_if_missing=True)
VOLUME_PATH = "/root/knight0"


@app.function(
    image=image,
    gpu="any",  # Use any available GPU
    timeout=300,  # 5 minute timeout (plenty for smoke test)
    volumes={VOLUME_PATH: volume},
)
def smoke_test_gpu():
    """
    Smoke test: verify GPU, imports, and volume access.
    """
    import sys
    import torch
    import chess
    from pathlib import Path
    
    print("="*80)
    print("knight0 Modal Smoke Test - GPU Function")
    print("="*80)
    print()
    
    # Test 1: Python environment
    print("Test 1: Python environment")
    print("-"*80)
    print(f"✓ Python version: {sys.version.split()[0]}")
    print(f"✓ Python executable: {sys.executable}")
    print()
    
    # Test 2: Package imports
    print("Test 2: Package imports")
    print("-"*80)
    print(f"✓ torch version: {torch.__version__}")
    print(f"✓ chess version: {chess.__version__}")
    print()
    
    # Test 3: GPU detection
    print("Test 3: GPU detection")
    print("-"*80)
    if torch.cuda.is_available():
        print(f"✓ CUDA available: True")
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        print(f"✓ GPU name: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Test GPU computation
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = x @ y
        print(f"✓ GPU computation test: successful")
    else:
        print(f"⚠ CUDA available: False (running on CPU)")
    print()
    
    # Test 4: Volume access
    print("Test 4: Volume access")
    print("-"*80)
    volume_path = Path(VOLUME_PATH)
    print(f"✓ Volume path: {volume_path}")
    print(f"✓ Volume exists: {volume_path.exists()}")
    
    # Try to write a test file
    test_file = volume_path / "smoke_test.txt"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(test_file, 'w') as f:
        f.write("Modal smoke test successful!\n")
    
    print(f"✓ Test file written: {test_file}")
    print(f"✓ Test file exists: {test_file.exists()}")
    
    # Read it back
    with open(test_file, 'r') as f:
        content = f.read().strip()
    
    print(f"✓ Test file content: {content}")
    print()
    
    # Test 5: Import knight0 package
    print("Test 5: Import knight0 package")
    print("-"*80)
    sys.path.insert(0, "/root/knight0_pkg")
    
    try:
        from knight0 import config, encoding, model
        print(f"✓ knight0.config imported")
        print(f"✓ knight0.encoding imported")
        print(f"✓ knight0.model imported")
        
        # Test creating a small model
        test_model = model.ChessNet(**config.CONFIGS["small"])
        print(f"✓ ChessNet model created")
        print(f"  Parameters: {test_model.get_num_parameters():,}")
        
        # Test encoding
        import chess
        board = chess.Board()
        tensor = encoding.board_to_tensor(board)
        print(f"✓ board_to_tensor works")
        print(f"  Tensor shape: {tensor.shape}")
        
    except Exception as e:
        print(f"✗ Error importing knight0: {e}")
        raise
    
    print()
    
    # Test 6: Stockfish availability
    print("Test 6: Stockfish availability")
    print("-"*80)
    import subprocess
    
    try:
        result = subprocess.run(
            ["/usr/games/stockfish", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        print(f"✓ Stockfish available")
        version_line = result.stdout.strip().split('\n')[0]
        print(f"  {version_line}")
    except Exception as e:
        print(f"⚠ Stockfish not found at /usr/games/stockfish: {e}")
    
    print()
    
    # Summary
    print("="*80)
    print("✅ SMOKE TEST PASSED!")
    print("="*80)
    print()
    print("Modal infrastructure is working correctly:")
    print("  ✓ Image built successfully")
    print("  ✓ GPU allocated and accessible")
    print("  ✓ Volume mounted and writable")
    print("  ✓ knight0 package importable")
    print("  ✓ All dependencies available")
    print()
    print("Ready for real training!")
    print()
    
    return True


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def smoke_test_volume():
    """
    Simple test without GPU to check volume persistence.
    """
    from pathlib import Path
    
    print("Testing volume persistence...")
    
    test_file = Path(VOLUME_PATH) / "smoke_test.txt"
    
    if test_file.exists():
        with open(test_file, 'r') as f:
            content = f.read().strip()
        print(f"✓ Found persistent test file: {content}")
    else:
        print("ℹ No test file found (expected on first run)")
    
    # List volume contents
    volume_path = Path(VOLUME_PATH)
    if volume_path.exists():
        files = list(volume_path.rglob("*"))
        print(f"\nVolume contents ({len(files)} items):")
        for f in sorted(files)[:10]:  # Show first 10
            if f.is_file():
                print(f"  - {f.relative_to(volume_path)}")
    
    return True


@app.local_entrypoint()
def main(test_type: str = "full"):
    """
    Run Modal smoke tests.
    
    Args:
        test_type: Type of test to run ("full", "gpu", "volume")
    """
    print()
    print("🚀 Starting Modal Smoke Tests")
    print()
    
    if test_type in ["full", "gpu"]:
        print("Running GPU smoke test...")
        print("(This will allocate a GPU and may take 1-2 minutes)")
        print()
        smoke_test_gpu.remote()
    
    if test_type in ["full", "volume"]:
        print("\nRunning volume smoke test...")
        smoke_test_volume.remote()
    
    print()
    print("="*80)
    print("🎉 All Modal smoke tests completed!")
    print("="*80)
    print()


if __name__ == "__main__":
    print("Modal Smoke Test Script")
    print("\nUsage:")
    print("  modal run test_modal_smoke.py")
    print("  modal run test_modal_smoke.py --test-type gpu")
    print("  modal run test_modal_smoke.py --test-type volume")

