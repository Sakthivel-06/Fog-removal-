"""
scratch/test_transmission_candidates.py — Test Models A, B, C, D on foggy_video3.mp4
=====================================================================================
Evaluates:
  Model A: Baseline Session 12 (t_A = t)
  Model B: Physical Non-Negativity Floor (t_B = max(t, 1 - I/A))
  Model C: Soft Underflow-Aware Floor (smooth continuous transition + subtle texture buffer)
  Model D: Dark-Region Adaptive Transmission Confidence (C_trans * t + (1 - C_trans) * 1.0)
"""

import os
import sys
import imageio
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nighttime_experimental import ExperimentalNighttimePipeline
from nighttime_diagnostics import compute_frame_telemetry

def evaluate_frame_candidates(frame_idx=30):
    reader = imageio.get_reader("test_videos/foggy_video3.mp4")
    pipe = ExperimentalNighttimePipeline()
    for i, f in enumerate(reader):
        if i < frame_idx:
            pipe.process_frame(f)
        elif i == frame_idx:
            target_frame = f
            break
    reader.close()

    # Process frame to warm up caches and get base outputs
    out_a, fps_a, diag = pipe.process_frame(target_frame, return_diagnostics=True)
    proc_float = pipe._proc_float_buf
    orig_h, orig_w = proc_float.shape[:2]

    # Reconstruct exact base terms
    half_img = proc_float[::2, ::2]
    gray_h = 0.299 * half_img[:, :, 0] + 0.587 * half_img[:, :, 1] + 0.114 * half_img[:, :, 2]
    A = pipe.last_A

    t_cached = pipe._cached_t
    t_full = np.array(Image.fromarray(t_cached, mode="F").resize((orig_w, orig_h), Image.BILINEAR))[:, :, None]

    V_full = np.max(proc_float, axis=2)
    L_smooth = ndimage.uniform_filter(V_full, size=pipe.local_illum_kernel_size)
    a_max_ref = max(float(np.max(A)), 0.25)
    eta = np.clip(L_smooth / a_max_ref, pipe.eta_min, 1.0)[:, :, None]
    A_spatial = A[None, None, :] * eta

    in_lum = 0.299 * proc_float[:, :, 0] + 0.587 * proc_float[:, :, 1] + 0.114 * proc_float[:, :, 2]
    native_black_pct = float(np.count_nonzero(in_lum < (1.0 / 255.0)) / in_lum.size * 100.0)

    # Calculate required t for non-negativity
    eps = 1e-4
    A_eff = np.maximum(A_spatial, eps)
    req_t_channels = 1.0 - proc_float / A_eff
    t_req = np.max(req_t_channels, axis=2, keepdims=True)  # (H, W, 1)

    candidates = {}

    # MODEL A: Baseline Session 12
    t_A = t_full.copy()
    J_raw_A = (proc_float - A_spatial) / t_A + A_spatial
    J_A = np.clip(J_raw_A, 0.0, 1.0)
    out_A = (J_A * 255.0).astype(np.uint8)
    candidates["Model A (Session 12 Baseline)"] = {
        "t": t_A, "J_raw": J_raw_A, "out": out_A,
    }

    # MODEL B: Hard Physical Floor
    t_B = np.clip(np.maximum(t_full, t_req), 0.05, 1.0)
    J_raw_B = (proc_float - A_spatial) / t_B + A_spatial
    J_B = np.clip(J_raw_B, 0.0, 1.0)
    out_B = (J_B * 255.0).astype(np.uint8)
    candidates["Model B (Hard Non-Negativity Floor)"] = {
        "t": t_B, "J_raw": J_raw_B, "out": out_B,
    }

    # MODEL C: Soft Underflow-Aware Floor
    # Continuous smooth formulation using softplus transition with a subtle safety buffer delta = 0.03
    k = 20.0
    delta = 0.03
    diff = t_req - t_full + delta
    # Stable softplus: 1/k * log(1 + exp(k*diff))
    # When k*diff > 30, softplus ~ diff
    soft_excess = np.where(k * diff > 30.0, diff, (1.0 / k) * np.log1p(np.exp(np.clip(k * diff, -30.0, 30.0))))
    t_C = np.clip(t_full + soft_excess, 0.05, 1.0)
    J_raw_C = (proc_float - A_spatial) / t_C + A_spatial
    J_C = np.clip(J_raw_C, 0.0, 1.0)
    out_C = (J_C * 255.0).astype(np.uint8)
    candidates["Model C (Soft Underflow Floor)"] = {
        "t": t_C, "J_raw": J_raw_C, "out": out_C,
    }

    # MODEL D: Dark-Region Adaptive Transmission Confidence
    # Confidence in transmission is high where illumination eta is strong and local luminance > 0.05
    # Where scene is unlit/dark, confidence -> 0, transmission t -> 1.0 (identity preservation)
    # Evidence: eta(x) and local luminance V_full
    conf_lum = np.clip((L_smooth - 0.02) / 0.15, 0.0, 1.0)[:, :, None]
    conf_eta = np.clip((eta - pipe.eta_min) / (0.40 - pipe.eta_min), 0.0, 1.0)
    C_trans = conf_lum * conf_eta
    # Smooth confidence map
    t_D = np.clip(1.0 - C_trans * (1.0 - t_full), 0.05, 1.0)
    J_raw_D = (proc_float - A_spatial) / t_D + A_spatial
    J_D = np.clip(J_raw_D, 0.0, 1.0)
    out_D = (J_D * 255.0).astype(np.uint8)
    candidates["Model D (Dark-Region Confidence)"] = {
        "t": t_D, "J_raw": J_raw_D, "out": out_D,
    }

    print(f"=== EVALUATION ON FOGGY_VIDEO3 (FRAME {frame_idx}) ===")
    print(f"Native Input Black (< 1/255): {native_black_pct:.2f}%\n")

    results = {}
    for name, data in candidates.items():
        t = data["t"]
        J_raw = data["J_raw"]
        out = data["out"]
        tel = compute_frame_telemetry(target_frame, out)
        
        underflow_pct = float(np.count_nonzero(np.any(J_raw < 0.0, axis=2)) / J_raw[:, :, 0].size * 100.0)
        min_J = float(np.min(J_raw))
        black_clip = tel["black_clip"]
        induced_crush = black_clip - native_black_pct
        
        # Dark region mean luminance (input < 38/255)
        dark_mask = (in_lum < (38.0 / 255.0))
        out_lum = tel["lum_out"]
        dark_mean_lum = float(np.mean(out_lum[dark_mask])) if np.any(dark_mask) else 0.0
        
        # Road region mean luminance (y: 60%..85%, x: 30%..70%)
        ry0, ry1 = int(0.60 * orig_h), int(0.85 * orig_h)
        rx0, rx1 = int(0.30 * orig_w), int(0.70 * orig_w)
        road_mean_lum = float(np.mean(out_lum[ry0:ry1, rx0:rx1]))

        print(f"[{name}]")
        print(f"  t mean: {np.mean(t):.3f} | t min: {np.min(t):.3f}")
        print(f"  J_raw < 0: {underflow_pct:5.2f}% | min J_raw: {min_J:+.4f}")
        print(f"  Black Clip: {black_clip:5.2f}% | Induced Crush: {induced_crush:+.2f}%")
        print(f"  White Clip: {tel['white_clip']:4.2f}% | Mean Lum: {tel['mean_lum']:5.1f}")
        print(f"  Dark Region Lum: {dark_mean_lum:5.1f} | Road Lum: {road_mean_lum:5.1f}")
        print(f"  Edge Energy: {tel['edge_energy']:5.2f} | NIQE: {tel['niqe']:4.2f} | Halo: {tel['halo_ratio']:4.2f}%\n")
        
        results[name] = {
            "underflow_pct": underflow_pct,
            "min_J": min_J,
            "black_clip": black_clip,
            "induced_crush": induced_crush,
            "white_clip": tel["white_clip"],
            "mean_lum": tel["mean_lum"],
            "dark_mean_lum": dark_mean_lum,
            "road_mean_lum": road_mean_lum,
            "edge_energy": tel["edge_energy"],
            "niqe": tel["niqe"],
            "halo": tel["halo_ratio"],
        }
    return results

if __name__ == "__main__":
    evaluate_frame_candidates(30)
