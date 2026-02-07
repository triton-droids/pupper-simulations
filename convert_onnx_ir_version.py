#!/usr/bin/env python3
"""
Convert ONNX model to a lower IR version for compatibility with older ONNX Runtime.
"""

import sys
import onnx
from pathlib import Path


def convert_model_ir_version(input_path: str, output_path: str, target_ir_version: int = 9):
    """
    Load an ONNX model and save it with a lower IR version.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save converted model
        target_ir_version: Target IR version (default: 9 for maximum compatibility)
    """
    print(f"Loading ONNX model from {input_path}...")
    model = onnx.load(input_path)

    print(f"Current IR version: {model.ir_version}")
    print(f"Current opset version: {model.opset_import[0].version if model.opset_import else 'unknown'}")

    # Set the target IR version
    model.ir_version = target_ir_version
    print(f"Setting IR version to: {target_ir_version}")

    # Validate the model
    print("Validating converted model...")
    try:
        onnx.checker.check_model(model)
        print("Model validation successful!")
    except Exception as e:
        print(f"Warning: Model validation failed: {e}")
        print("Proceeding anyway...")

    # Save the converted model
    print(f"Saving converted model to {output_path}...")
    onnx.save(model, output_path)
    print("Conversion complete!")


if __name__ == "__main__":
    input_path = "locomotion/sim-outputs/policies/policy.onnx"
    output_path = "locomotion/sim-outputs/policies/policy_ir9.onnx"

    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]

    convert_model_ir_version(input_path, output_path)
    print(f"\nYou can now use the converted model at: {output_path}")
