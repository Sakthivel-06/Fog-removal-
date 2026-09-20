"""
export_onnx.py — Export TinyHazePerceptionModule to ONNX
=========================================================
Loads the saved PyTorch weights (perception_model.pth) and exports to ONNX format
so that onnxruntime can run it for fast CPU inference inside the main pipeline.

Usage:
    cd fog-removal-project/novel_method/perception_model
    python export_onnx.py
    # OR from project root:
    python novel_method/perception_model/export_onnx.py
"""

import os
import sys
import torch

# Ensure model module is importable regardless of working directory
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from model import TinyHazePerceptionModule


def export(
    pth_path: str = None,
    onnx_path: str = None,
    opset: int = 11,
):
    if pth_path is None:
        pth_path = os.path.join(_SCRIPT_DIR, "weights", "perception_model.pth")
    if onnx_path is None:
        onnx_path = os.path.join(_SCRIPT_DIR, "weights", "perception_model.onnx")

    if not os.path.exists(pth_path):
        print(f"[export_onnx] ERROR: .pth not found at {pth_path}")
        print("  -> Run train.py first to generate the weights.")
        sys.exit(1)

    print(f"[export_onnx] Loading weights from: {pth_path}")
    model = TinyHazePerceptionModule()
    state = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[export_onnx] Model loaded  {total_params:,} parameters")

    # Dummy input matching training resolution; dynamic_axes allow any H x W at inference
    dummy_input = torch.randn(1, 3, 128, 128)

    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    print(f"[export_onnx] Exporting to ONNX (opset {opset})...")
    # Use dynamo=False to force the legacy TorchScript export path.
    # PyTorch 2.x defaults to the dynamo exporter which needs onnxscript+onnx installed.
    # The legacy path works out-of-the-box with the existing venv.
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        },
        dynamo=False,   # force legacy TorchScript path (no onnxscript required)
    )

    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"[export_onnx] ONNX model saved to: {onnx_path}  ({size_kb:.1f} KB)")

    # Quick sanity-check with onnxruntime if available
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        dummy_np = np.random.randn(1, 3, 128, 128).astype(np.float32)
        out = sess.run(["output"], {"input": dummy_np})[0]
        print(f"[export_onnx] ORT inference check passed  output shape: {out.shape}")
    except ImportError:
        print("[export_onnx] onnxruntime not installed  skipping ORT check.")
    except Exception as e:
        print(f"[export_onnx] ORT check failed: {e}")


if __name__ == "__main__":
    export()
