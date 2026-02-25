"""Tests for locomotion/onnx_export.py (1 test)."""

import numpy as np
import onnx
import onnxruntime as ort
import pytest


def test_export_policy_to_onnx(tmp_path, fake_policy_params):
    """Exported ONNX model is valid and produces correct-shape bounded output."""
    from onnx_export import export_policy_to_onnx

    out = tmp_path / "policy.onnx"
    export_policy_to_onnx(fake_policy_params, str(out))

    # File exists and is a valid ONNX model
    assert out.exists()
    model = onnx.load(str(out))
    onnx.checker.check_model(model)

    # Inference produces [1, 9] output in [-1, 1]
    sess = ort.InferenceSession(str(out))
    obs = np.random.randn(1, 510).astype(np.float32)
    (action,) = sess.run(None, {"observation": obs})
    assert action.shape == (1, 9)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
