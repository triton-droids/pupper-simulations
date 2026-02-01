"""ONNX export utility for Brax/JAX policies.

Exports trained Brax PPO policies to ONNX format for cross-platform inference
without requiring JAX runtime.
"""

import numpy as np
import onnx
from onnx import helper, TensorProto
import logging

logger = logging.getLogger(__name__)


def export_policy_to_onnx(params, output_path: str, deterministic: bool = True):
    """Export Brax policy to ONNX format.

    Args:
        params: Brax PPO network parameters (nested PyTree)
        output_path: Path to save .onnx file
        deterministic: If True, export deterministic policy (mean actions only)

    The exported ONNX model accepts:
        - Input: "observation" [1, 510] float32
        - Output: "action" [1, 9] float32 in range [-1, 1]
    """
    logger.info("Extracting policy weights from Brax params...")

    # Brax PPO returns params as tuple: (normalizer_params, policy_params, value_params)
    normalizer_params, policy_params, value_params = params

    # Policy params structure: {'params': {'hidden_0': {...}, 'hidden_1': {...}, ...}}
    network_params = policy_params['params']

    # Convert JAX arrays to numpy and extract weights
    # Note: output layer is 'hidden_4', not 'output'
    weights = {}
    for layer_name in ['hidden_0', 'hidden_1', 'hidden_2', 'hidden_3', 'hidden_4']:
        layer_params = network_params[layer_name]
        kernel = np.array(layer_params['kernel'], dtype=np.float32)
        bias = np.array(layer_params['bias'], dtype=np.float32)
        weights[layer_name] = {'kernel': kernel, 'bias': bias}
        logger.info(f"  {layer_name}: kernel {kernel.shape}, bias {bias.shape}")

    logger.info("Building ONNX graph...")

    # Create ONNX graph nodes
    nodes = []
    initializers = []

    # Input
    input_tensor = helper.make_tensor_value_info('observation', TensorProto.FLOAT, [1, 510])

    # Add all weights as initializers
    for layer_name, layer_weights in weights.items():
        # Weight matrix
        initializers.append(
            helper.make_tensor(
                name=f'{layer_name}_weight',
                data_type=TensorProto.FLOAT,
                dims=layer_weights['kernel'].shape,
                vals=layer_weights['kernel'].flatten().tolist()
            )
        )
        # Bias vector
        initializers.append(
            helper.make_tensor(
                name=f'{layer_name}_bias',
                data_type=TensorProto.FLOAT,
                dims=layer_weights['bias'].shape,
                vals=layer_weights['bias'].flatten().tolist()
            )
        )

    # Build network layers
    current_input = 'observation'

    # Hidden layers with Swish activation
    for i, layer_name in enumerate(['hidden_0', 'hidden_1', 'hidden_2', 'hidden_3']):
        # MatMul: x @ W
        matmul_out = f'{layer_name}_matmul'
        nodes.append(helper.make_node(
            'MatMul',
            inputs=[current_input, f'{layer_name}_weight'],
            outputs=[matmul_out]
        ))

        # Add: (x @ W) + b
        add_out = f'{layer_name}_add'
        nodes.append(helper.make_node(
            'Add',
            inputs=[matmul_out, f'{layer_name}_bias'],
            outputs=[add_out]
        ))

        # Swish activation: x * sigmoid(x)
        sigmoid_out = f'{layer_name}_sigmoid'
        nodes.append(helper.make_node(
            'Sigmoid',
            inputs=[add_out],
            outputs=[sigmoid_out]
        ))

        swish_out = f'{layer_name}_swish'
        nodes.append(helper.make_node(
            'Mul',
            inputs=[add_out, sigmoid_out],
            outputs=[swish_out]
        ))

        current_input = swish_out

    # Output layer (hidden_4, no activation yet)
    # MatMul
    nodes.append(helper.make_node(
        'MatMul',
        inputs=[current_input, 'hidden_4_weight'],
        outputs=['hidden_4_matmul']
    ))

    # Add bias
    nodes.append(helper.make_node(
        'Add',
        inputs=['hidden_4_matmul', 'hidden_4_bias'],
        outputs=['logits']
    ))

    # Slice to extract mean actions (first 9 elements)
    # logits shape: [1, 18] -> action_mean shape: [1, 9]
    nodes.append(helper.make_node(
        'Slice',
        inputs=['logits', 'slice_starts', 'slice_ends', 'slice_axes'],
        outputs=['action_mean']
    ))

    # Add slice parameters as initializers
    initializers.extend([
        helper.make_tensor('slice_starts', TensorProto.INT64, [1], [0]),
        helper.make_tensor('slice_ends', TensorProto.INT64, [1], [9]),
        helper.make_tensor('slice_axes', TensorProto.INT64, [1], [1]),
    ])

    # Tanh activation to get actions in [-1, 1]
    nodes.append(helper.make_node(
        'Tanh',
        inputs=['action_mean'],
        outputs=['action']
    ))

    # Output
    output_tensor = helper.make_tensor_value_info('action', TensorProto.FLOAT, [1, 9])

    # Create graph
    graph_def = helper.make_graph(
        nodes=nodes,
        name='BraxPPOPolicy',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=initializers
    )

    # Create model
    model_def = helper.make_model(
        graph_def,
        producer_name='brax-onnx-exporter',
        opset_imports=[helper.make_opsetid('', 13)]
    )

    # Check model validity
    logger.info("Validating ONNX model...")
    onnx.checker.check_model(model_def)

    # Save model
    logger.info(f"Saving ONNX model to {output_path}...")
    onnx.save(model_def, output_path)

    # Log model info
    file_size_kb = len(onnx._serialize(model_def)) / 1024
    logger.info(f"ONNX export complete! Model size: {file_size_kb:.1f} KB")
    logger.info(f"  Input: observation [1, 510] float32")
    logger.info(f"  Output: action [1, 9] float32 in range [-1, 1]")
