"""
ONNX export functionality for ChessNet models.

This module exports trained PyTorch models to ONNX format for deployment.
"""

import torch
import onnx
import onnxruntime as ort
import numpy as np
from pathlib import Path
import logging

from .model import ChessNet

logger = logging.getLogger(__name__)


def export_to_onnx(
    model: ChessNet,
    output_path: str,
    input_channels: int = 21,
    opset_version: int = 14,
    verify: bool = True
) -> Path:
    """
    Export a ChessNet model to ONNX format.
    
    Args:
        model: Trained ChessNet model
        output_path: Path to save ONNX file
        input_channels: Number of input channels (default: 21)
        opset_version: ONNX opset version
        verify: If True, verify the exported model
        
    Returns:
        Path to exported ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Exporting model to ONNX: {output_path}")
    
    # Set model to eval mode
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, input_channels, 8, 8)
    
    # Move to same device as model
    device = next(model.parameters()).device
    dummy_input = dummy_input.to(device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy", "value"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"}
        }
    )
    
    logger.info(f"Successfully exported ONNX model to {output_path}")
    
    # Verify the exported model
    if verify:
        verify_onnx_model(output_path, input_channels)
    
    return output_path


def verify_onnx_model(onnx_path: Path, input_channels: int = 21):
    """
    Verify that an ONNX model is valid and produces correct output shapes.
    
    Args:
        onnx_path: Path to ONNX file
        input_channels: Number of input channels
    """
    logger.info(f"Verifying ONNX model: {onnx_path}")
    
    # Check model validity
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("  ✓ ONNX model is valid")
    
    # Test with ONNX Runtime
    ort_session = ort.InferenceSession(str(onnx_path))
    
    # Create dummy input
    dummy_input = np.random.randn(1, input_channels, 8, 8).astype(np.float32)
    
    # Run inference
    outputs = ort_session.run(None, {"input": dummy_input})
    
    policy_output, value_output = outputs
    
    logger.info(f"  ✓ Policy output shape: {policy_output.shape}")
    logger.info(f"  ✓ Value output shape: {value_output.shape}")
    
    # Check output shapes
    assert policy_output.shape[0] == 1, "Policy batch size mismatch"
    assert policy_output.shape[1] == 4096, "Policy output size should be 4096"
    assert value_output.shape == (1, 1), "Value output shape should be (1, 1)"
    
    # Check value range
    assert -1.0 <= value_output[0, 0] <= 1.0, "Value should be in [-1, 1]"
    
    logger.info("  ✓ Output shapes and ranges are correct")
    logger.info("ONNX model verification successful!")


def load_and_export_checkpoint(
    checkpoint_path: str,
    output_path: str,
    model_config: dict,
    verify: bool = True
) -> Path:
    """
    Load a PyTorch checkpoint and export it to ONNX.
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        output_path: Path to save ONNX file
        model_config: Dict with model configuration (filters, num_blocks, etc.)
        verify: If True, verify the exported model
        
    Returns:
        Path to exported ONNX file
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Create model
    model = ChessNet(**model_config)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    logger.info("Checkpoint loaded successfully")
    
    # Export to ONNX
    return export_to_onnx(
        model,
        output_path,
        input_channels=model_config.get('input_channels', 21),
        verify=verify
    )


def compare_pytorch_onnx_outputs(
    pytorch_model: ChessNet,
    onnx_path: Path,
    num_tests: int = 10,
    tolerance: float = 1e-5
):
    """
    Compare outputs between PyTorch model and ONNX model to ensure consistency.
    
    Args:
        pytorch_model: PyTorch ChessNet model
        onnx_path: Path to ONNX file
        num_tests: Number of random inputs to test
        tolerance: Maximum allowed difference
    """
    logger.info(f"Comparing PyTorch and ONNX outputs ({num_tests} tests)...")
    
    pytorch_model.eval()
    device = next(pytorch_model.parameters()).device
    input_channels = pytorch_model.input_channels
    
    # Load ONNX model
    ort_session = ort.InferenceSession(str(onnx_path))
    
    max_policy_diff = 0.0
    max_value_diff = 0.0
    
    with torch.no_grad():
        for i in range(num_tests):
            # Create random input
            test_input = torch.randn(1, input_channels, 8, 8)
            
            # PyTorch inference
            pytorch_policy, pytorch_value = pytorch_model(test_input.to(device))
            pytorch_policy = pytorch_policy.cpu().numpy()
            pytorch_value = pytorch_value.cpu().numpy()
            
            # ONNX inference
            onnx_outputs = ort_session.run(None, {"input": test_input.numpy()})
            onnx_policy, onnx_value = onnx_outputs
            
            # Compare
            policy_diff = np.max(np.abs(pytorch_policy - onnx_policy))
            value_diff = np.max(np.abs(pytorch_value - onnx_value))
            
            max_policy_diff = max(max_policy_diff, policy_diff)
            max_value_diff = max(max_value_diff, value_diff)
    
    logger.info(f"  Max policy difference: {max_policy_diff:.2e}")
    logger.info(f"  Max value difference: {max_value_diff:.2e}")
    
    if max_policy_diff < tolerance and max_value_diff < tolerance:
        logger.info("  ✓ PyTorch and ONNX outputs match!")
    else:
        logger.warning(f"  ⚠ Outputs differ by more than tolerance ({tolerance})")


if __name__ == "__main__":
    import logging
    from .config import CONFIGS
    
    logging.basicConfig(level=logging.INFO)
    
    print("Testing ONNX export...\n")
    
    # Create a small model for testing
    print("Creating test model...")
    model = ChessNet(**CONFIGS["small"])
    
    # Export to ONNX
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "test_model.onnx"
        
        print(f"\nExporting to {onnx_path}...")
        export_to_onnx(model, str(onnx_path), verify=True)
        
        print("\nComparing PyTorch and ONNX outputs...")
        compare_pytorch_onnx_outputs(model, onnx_path, num_tests=5)
        
        print("\nONNX export test completed successfully!")

