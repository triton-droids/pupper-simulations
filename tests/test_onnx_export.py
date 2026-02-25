import numpy as np
import onnx
import onnxruntime as ort

from onnx_export import export_policy_to_onnx


def test_export_policy_to_onnx(mock_policy_params, tmp_path):
    onnx_path = str(tmp_path / "test_policy.onnx")

    export_policy_to_onnx(mock_policy_params, onnx_path)

    # Validate model structure
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)

    # Run inference and check shapes/ranges
    session = ort.InferenceSession(onnx_path)
    obs = np.random.randn(1, 510).astype(np.float32)
    (action,) = session.run(None, {"observation": obs})

    assert action.shape == (1, 9)
    assert np.all(action >= -1.0) and np.all(action <= 1.0)
