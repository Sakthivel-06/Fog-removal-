"""
scratch/test_inversion_comparison.py — Comparison of Model A vs Model B Inversion
================================================================================
Compares Single-Source (Model A) vs Multi-Source (Model B) physical inversion:
  - Artifact inspection (halos around light sources, edge discoloration)
  - Color balance and contrast
  - Quantitative metrics
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

def evaluate_frame(pipe, frame, multi_source=False, filter_type="box"):
    """Runs pipeline and applies either single-source or multi-source airlight."""
    # Process with pipe
    out_single, fps, diag = pipe.process_frame(frame, return_diagnostics=True)
    if not multi_source:
        return out_single, fps, diag, compute_frame_telemetry(frame, out_single)
        
    proc_float = pipe._proc_float_buf
    orig_h, orig_w = proc_float.shape[:2]
    t_smooth = pipe._cached_t
    t_full = np.array(Image.fromarray(t_smooth, mode="F").resize((orig_w, orig_h), Image.BILINEAR))[:, :, None]
    A_default = pipe.last_A
    
    # Build multi-source field
    half_float = proc_float[::2, ::2]
    gh, gw = half_float.shape[:2]
    val = np.max(half_float, axis=2)
    local_mean = ndimage.uniform_filter(val, size=15)
    local_excess = val - local_mean
    core_mask = (val > 0.85) | ((val > 0.65) & (local_excess > 0.20))
    labeled, num_features = ndimage.label(core_mask)
    
    if num_features > 0:
        sizes = ndimage.sum(core_mask, labeled, range(1, num_features + 1))
        top_indices = np.argsort(sizes)[::-1][:4]
        A_k_list, eta_half_list = [], []
        for idx in top_indices:
            c_id = idx + 1
            c_mask = (labeled == c_id)
            if np.count_nonzero(c_mask) < 4:
                continue
            source_pixels = half_float[c_mask]
            A_k = np.percentile(source_pixels, 85, axis=0)
            A_k = np.clip(A_k, 0.15, 1.0)
            A_k_list.append(A_k)
            c_float = c_mask.astype(np.float32)
            if filter_type == "gaussian":
                eta_raw = ndimage.gaussian_filter(c_float, sigma=35 / 3.0)
            else:
                eta_raw = ndimage.uniform_filter(c_float, size=35)
            max_eta = np.max(eta_raw)
            if max_eta > 1e-6:
                eta_raw = eta_raw / max_eta
            eta_half_list.append(eta_raw)
            
        if A_k_list:
            eta_stack = np.stack(eta_half_list, axis=-1)
            eta_sum = np.sum(eta_stack, axis=-1, keepdims=True)
            eta_norm = eta_stack / np.maximum(eta_sum, 1.0)
            A_stack = np.array(A_k_list)
            A_half = np.sum(eta_norm[:, :, :, None] * A_stack[None, None, :, :], axis=2)
            eta_total = np.sum(eta_norm, axis=-1)[:, :, None]
            A_half = A_half + A_default[None, None, :] * 0.05 * np.maximum(1.0 - eta_total, 0.0)
            
            A_spatial = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
            for c in range(3):
                A_spatial[:, :, c] = np.array(Image.fromarray(A_half[:, :, c].astype(np.float32), mode='F').resize((orig_w, orig_h), Image.BILINEAR))
        else:
            V_full = np.max(proc_float, axis=2)
            L_smooth = ndimage.uniform_filter(V_full, size=31)
            eta = np.clip(L_smooth / max(float(np.max(A_default)), 0.25), 0.05, 1.0)[:, :, None]
            A_spatial = A_default[None, None, :] * eta
    else:
        V_full = np.max(proc_float, axis=2)
        L_smooth = ndimage.uniform_filter(V_full, size=31)
        eta = np.clip(L_smooth / max(float(np.max(A_default)), 0.25), 0.05, 1.0)[:, :, None]
        A_spatial = A_default[None, None, :] * eta
        
    J_raw = (proc_float - A_spatial) / t_full + A_spatial
    J = np.clip(J_raw, 0.0, 1.0)
    out_multi = (J * 255.0).astype(np.uint8)
    
    return out_multi, fps, diag, compute_frame_telemetry(frame, out_multi)

test_vids = ["foggy_video3.mp4", "foggy_video4.mp4", "foggy_video5.mp4"]

print("=" * 90)
print("PRELIMINARY COMPARISON: MODEL A (SINGLE-SOURCE) VS MODEL B (MULTI-SOURCE)")
print("=" * 90)

for vname in test_vids:
    vpath = os.path.join(r"C:\Users\sakth\Desktop\Fog\fog-removal-project\test_videos", vname)
    reader = imageio.get_reader(vpath)
    # Warm up 15 frames
    pipe = ExperimentalNighttimePipeline(night_mode="prototype")
    for i in range(15):
        pipe.process_frame(reader.get_data(i))
        
    target_frame = reader.get_data(16)
    reader.close()
    
    # Model A: Single-source
    out_a, _, _, m_a = evaluate_frame(pipe, target_frame, multi_source=False)
    # Model B: Multi-source (Box)
    out_b_box, _, _, m_b_box = evaluate_frame(pipe, target_frame, multi_source=True, filter_type="box")
    # Model B: Multi-source (Gaussian)
    out_b_gauss, _, _, m_b_gauss = evaluate_frame(pipe, target_frame, multi_source=True, filter_type="gaussian")
    
    print(f"\nVideo: {vname}")
    print(f"  Model A (Single-Source):    Black={m_a['black_clip']:4.2f}% | White={m_a['white_clip']:4.2f}% | Lum={m_a['mean_lum']:5.1f} | NIQE={m_a['niqe']:4.2f} | FDI_Red={m_a['fdi_reduction']:4.1f}% | Halo={m_a['halo_ratio']:4.2f}%")
    print(f"  Model B (Multi-Src Box):    Black={m_b_box['black_clip']:4.2f}% | White={m_b_box['white_clip']:4.2f}% | Lum={m_b_box['mean_lum']:5.1f} | NIQE={m_b_box['niqe']:4.2f} | FDI_Red={m_b_box['fdi_reduction']:4.1f}% | Halo={m_b_box['halo_ratio']:4.2f}%")
    print(f"  Model B (Multi-Src Gauss):  Black={m_b_gauss['black_clip']:4.2f}% | White={m_b_gauss['white_clip']:4.2f}% | Lum={m_b_gauss['mean_lum']:5.1f} | NIQE={m_b_gauss['niqe']:4.2f} | FDI_Red={m_b_gauss['fdi_reduction']:4.1f}% | Halo={m_b_gauss['halo_ratio']:4.2f}%")
