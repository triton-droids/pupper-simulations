"""Test script to validate ONNX export with normalization.

This script creates dummy parameters and tests the ONNX export function
to ensure it works correctly with the normalization fix.
"""

import numpy as np
import onnx
from dataclasses import dataclass
from typing import Any


@dataclass
class DummyRunningStatisticsState:
    """Dummy class mimicking Brax's RunningStatisticsState."""
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray
    summed_variance: np.ndarray


def create_dummy_params():
    """Create dummy Brax PPO parameters for testing."""
    # Create dummy normalizer params
    obs_size = 510
    normalizer_mean = np.random.randn(obs_size).astype(np.float32) * 0.1
    normalizer_std = np.random.rand(obs_size).astype(np.float32) * 0.5 + 0.5  # std in [0.5, 1.0]

    normalizer_params = DummyRunningStatisticsState(
        mean=normalizer_mean,
        std=normalizer_std,
        count=np.array(10000.0),
        summed_variance=np.ones(obs_size)
    )

    # Create dummy policy params
    # Architecture: 510 -> 256 -> 256 -> 256 -> 256 -> 18
    layer_sizes = [510, 256, 256, 256, 256, 18]

    policy_params = {'params': {}}
    for i in range(len(layer_sizes) - 1):
        in_size = layer_sizes[i]
        out_size = layer_sizes[i + 1]
        policy_params['params'][f'hidden_{i}'] = {
            'kernel': np.random.randn(in_size, out_size).astype(np.float32) * 0.1,
            'bias': np.zeros(out_size).astype(np.float32)
        }

    # Dummy value params (not used in export)
    value_params = {}

    return (normalizer_params, policy_params, value_params)


def test_onnx_export():
    """Test ONNX export with dummy parameters."""
    from onnx_export import export_policy_to_onnx
    import tempfile
    import os

    print("Creating dummy parameters...")
    params = create_dummy_params()

    print("Testing ONNX export...")
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
        output_path = f.name

    try:
        export_policy_to_onnx(params, output_path, deterministic=True)
        print(f"✓ ONNX export successful: {output_path}")

        # Load and inspect the model
        model = onnx.load(output_path)
        print("\n✓ ONNX model loaded successfully")

        # Check graph structure
        graph = model.graph
        print(f"\nGraph info:")
        print(f"  Nodes: {len(graph.node)}")
        print(f"  Initializers: {len(graph.initializer)}")
        print(f"  Inputs: {[inp.name for inp in graph.input]}")
        print(f"  Outputs: {[out.name for out in graph.output]}")

        # Verify normalization nodes exist
        node_types = [node.op_type for node in graph.node]
        assert 'Sub' in node_types, "Missing Sub node for normalization"
        assert 'Div' in node_types, "Missing Div node for normalization"
        print("\n✓ Normalization nodes (Sub, Div) found in graph")

        # Verify initializers include normalizer_mean and normalizer_std
        initializer_names = [init.name for init in graph.initializer]
        assert 'normalizer_mean' in initializer_names, "Missing normalizer_mean initializer"
        assert 'normalizer_std' in initializer_names, "Missing normalizer_std initializer"
        print("✓ Normalizer parameters (mean, std) found in initializers")

        # Find first two nodes (should be Sub and Div)
        first_nodes = graph.node[:3]
        print(f"\nFirst 3 nodes:")
        for i, node in enumerate(first_nodes):
            print(f"  {i+1}. {node.op_type}: {node.input} -> {node.output}")

        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        print("\nThe ONNX export now includes observation normalization.")
        print("The exported model will automatically normalize inputs using:")
        print("  normalized_obs = (observation - mean) / std")

    finally:
        # Clean up
        if os.path.exists(output_path):
            os.remove(output_path)
            print(f"\nCleaned up test file: {output_path}")


if __name__ == '__main__':
    test_onnx_export()
