"""
novel_method/nighttime_diagnostics.py — Diagnostic & Metric Utilities for Session 12
===================================================================================
CPU-friendly, NumPy/SciPy/Pillow based metrics and diagnostics for Nighttime Dehazing.
No OpenCV. Strictly adheres to project constraints.
"""

import os
import sys
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import metrics


def compute_frame_telemetry(in_f: np.ndarray, out_f: np.ndarray, prev_out_lum: np.ndarray = None):
    """
    Computes all standard Session 11 & Session 12 telemetry for a single output frame.
    """
    out_h, out_w = out_f.shape[:2]
    if in_f.shape[:2] != (out_h, out_w):
        in_res = np.array(Image.fromarray(in_f).resize((out_w, out_h), Image.BILINEAR))
    else:
        in_res = in_f

    lum_out = 0.299 * out_f[:, :, 0] + 0.587 * out_f[:, :, 1] + 0.114 * out_f[:, :, 2]
    lum_in = 0.299 * in_res[:, :, 0] + 0.587 * in_res[:, :, 1] + 0.114 * in_res[:, :, 2]

    black_clip = float(np.count_nonzero(lum_out < 1.0) / lum_out.size * 100.0)
    white_clip = float(np.count_nonzero(lum_out > 254.0) / lum_out.size * 100.0)
    mean_lum = float(np.mean(lum_out))
    b_minus_r = float(np.mean(out_f[:, :, 2].astype(float) - out_f[:, :, 0].astype(float)))
    colorfulness = float(metrics.colorfulness(out_f))

    # NIQE
    try:
        niqe_val = float(metrics.niqe(out_f))
    except Exception:
        niqe_val = 0.0

    # FDI
    fdi_val = float(metrics.fog_density_index(out_f))
    fdi_red = float(metrics.fog_reduction_percent_PROXY(in_res, out_f))

    # Structural / Edge Energy
    gy, gx = np.gradient(lum_out)
    edge_energy = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

    # Temporal flicker
    flicker = 0.0
    if prev_out_lum is not None and prev_out_lum.shape == lum_out.shape:
        flicker = float(np.mean(np.abs(lum_out - prev_out_lum)))

    # Halo ratio: ratio of high gradient to mean edge
    high_grad_mask = np.sqrt(gx ** 2 + gy ** 2) > 25.0
    halo_ratio = float(np.count_nonzero(high_grad_mask) / lum_out.size * 100.0)

    return {
        "black_clip": black_clip,
        "white_clip": white_clip,
        "mean_lum": mean_lum,
        "b_minus_r": b_minus_r,
        "colorfulness": colorfulness,
        "niqe": niqe_val,
        "fdi_val": fdi_val,
        "fdi_reduction": fdi_red,
        "edge_energy": edge_energy,
        "flicker": flicker,
        "halo_ratio": halo_ratio,
        "lum_out": lum_out,
    }
