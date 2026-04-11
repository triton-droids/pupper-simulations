"""
Export a trained Brax PPO policy to ONNX.

The exported model is intentionally small and deterministic. It contains only
the policy MLP needed for inference and emits the mean action vector passed
through ``tanh``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper


logger = logging.getLogger(__name__)

POLICY_LAYER_NAMES = ["hidden_0", "hidden_1", "hidden_2", "hidden_3", "hidden_4"]
OBSERVATION_SIZE = 510
ACTION_SIZE = 9
LOGIT_SIZE = ACTION_SIZE * 2
ONNX_OPSET = 11
ONNX_IR_VERSION = 9


@dataclass(slots=True)
class LayerWeights:
    """Dense-layer parameters extracted from the Brax policy network."""

    kernel: np.ndarray
    bias: np.ndarray


def _extract_policy_weights(params: Any) -> dict[str, LayerWeights]:
    """
    Extract dense-layer weights from the Brax PPO parameter pytree.

    Brax PPO currently returns parameters as:

    ``(normalizer_params, policy_params, value_params)``

    Only the policy MLP is needed for inference export.
    """
    _normalizer_params, policy_params, _value_params = params
    network_params = policy_params["params"]

    weights: dict[str, LayerWeights] = {}
    for layer_name in POLICY_LAYER_NAMES:
        layer_params = network_params[layer_name]
        weights[layer_name] = LayerWeights(
            kernel=np.asarray(layer_params["kernel"], dtype=np.float32),
            bias=np.asarray(layer_params["bias"], dtype=np.float32),
        )
        logger.info(
            "  %s: kernel %s, bias %s",
            layer_name,
            weights[layer_name].kernel.shape,
            weights[layer_name].bias.shape,
        )
    return weights


def _make_weight_initializers(weights: dict[str, LayerWeights]) -> list[onnx.TensorProto]:
    """Convert layer weights into ONNX graph initializers."""
    initializers: list[onnx.TensorProto] = []

    for layer_name, layer_weights in weights.items():
        initializers.append(
            helper.make_tensor(
                name=f"{layer_name}_weight",
                data_type=TensorProto.FLOAT,
                dims=layer_weights.kernel.shape,
                vals=layer_weights.kernel.flatten().tolist(),
            )
        )
        initializers.append(
            helper.make_tensor(
                name=f"{layer_name}_bias",
                data_type=TensorProto.FLOAT,
                dims=layer_weights.bias.shape,
                vals=layer_weights.bias.flatten().tolist(),
            )
        )

    initializers.extend(
        [
            helper.make_tensor("slice_starts", TensorProto.INT64, [1], [0]),
            helper.make_tensor("slice_ends", TensorProto.INT64, [1], [ACTION_SIZE]),
            helper.make_tensor("slice_axes", TensorProto.INT64, [1], [1]),
        ]
    )
    return initializers


def _build_policy_nodes() -> list[onnx.NodeProto]:
    """Create the ONNX node list for the policy MLP."""
    nodes: list[onnx.NodeProto] = []
    current_input = "observation"

    for layer_name in POLICY_LAYER_NAMES[:-1]:
        matmul_out = f"{layer_name}_matmul"
        add_out = f"{layer_name}_add"
        sigmoid_out = f"{layer_name}_sigmoid"
        swish_out = f"{layer_name}_swish"

        nodes.extend(
            [
                helper.make_node(
                    "MatMul",
                    inputs=[current_input, f"{layer_name}_weight"],
                    outputs=[matmul_out],
                ),
                helper.make_node(
                    "Add",
                    inputs=[matmul_out, f"{layer_name}_bias"],
                    outputs=[add_out],
                ),
                helper.make_node(
                    "Sigmoid",
                    inputs=[add_out],
                    outputs=[sigmoid_out],
                ),
                helper.make_node(
                    "Mul",
                    inputs=[add_out, sigmoid_out],
                    outputs=[swish_out],
                ),
            ]
        )
        current_input = swish_out

    nodes.extend(
        [
            helper.make_node(
                "MatMul",
                inputs=[current_input, "hidden_4_weight"],
                outputs=["hidden_4_matmul"],
            ),
            helper.make_node(
                "Add",
                inputs=["hidden_4_matmul", "hidden_4_bias"],
                outputs=["logits"],
            ),
            helper.make_node(
                "Slice",
                inputs=["logits", "slice_starts", "slice_ends", "slice_axes"],
                outputs=["action_mean"],
            ),
            helper.make_node(
                "Tanh",
                inputs=["action_mean"],
                outputs=["action"],
            ),
        ]
    )
    return nodes


def export_policy_to_onnx(
    params: Any,
    output_path: str,
    deterministic: bool = True,
) -> None:
    """
    Export a trained Brax PPO policy to a deterministic ONNX graph.

    Args:
        params: Brax PPO parameter pytree.
        output_path: Destination ``.onnx`` file path.
        deterministic: Kept for API compatibility. Only deterministic export is
            implemented; the ONNX graph emits mean actions only.
    """
    if not deterministic:
        logger.warning(
            "Stochastic ONNX export is not implemented; exporting deterministic "
            "mean-action policy instead."
        )

    logger.info("Extracting policy weights from Brax params...")
    weights = _extract_policy_weights(params)

    logger.info("Building ONNX graph...")
    input_tensor = helper.make_tensor_value_info(
        "observation",
        TensorProto.FLOAT,
        [1, OBSERVATION_SIZE],
    )
    output_tensor = helper.make_tensor_value_info(
        "action",
        TensorProto.FLOAT,
        [1, ACTION_SIZE],
    )

    graph_def = helper.make_graph(
        nodes=_build_policy_nodes(),
        name="BraxPPOPolicy",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=_make_weight_initializers(weights),
    )

    model_def = helper.make_model(
        graph_def,
        producer_name="brax-onnx-exporter",
        opset_imports=[helper.make_opsetid("", ONNX_OPSET)],
    )
    model_def.ir_version = ONNX_IR_VERSION
    logger.info("Set model IR version to %s", model_def.ir_version)

    logger.info("Validating ONNX model...")
    onnx.checker.check_model(model_def)

    logger.info("Saving ONNX model to %s...", output_path)
    onnx.save(model_def, output_path)

    logger.info("ONNX export complete")
    logger.info("  Input:  observation [1, %s] float32", OBSERVATION_SIZE)
    logger.info("  Output: action [1, %s] float32 in range [-1, 1]", ACTION_SIZE)
