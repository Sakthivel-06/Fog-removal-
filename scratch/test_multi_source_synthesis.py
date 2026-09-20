"""
scratch/test_multi_source_synthesis.py — Phase 4 & Phase 5 Multi-Source Synthesis
==================================================================================
Implements and evaluates:
  Model A: Session 12 Single-Source A(x) = A_source * eta(x)
  Model B (Gaussian): Multi-Source sum_k A_k * eta_k(x) with Gaussian diffusion
  Model B (Box): Multi-Source sum_k A_k * eta_k(x) with Box-filter diffusion
"""

import os
import sys
import time
import imageio
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage

def build_multi_source_airlight(proc_float, A_default, max_clusters=4, filter_type="box", kernel_size=45):
    """
    Constructs multi-source airlight field A(x) = sum_k A_k * eta_k(x) + A_amb * eta_amb(x).
    100% vectorized, CPU-friendly.
    """
    orig_h, orig_w = proc_float.shape[:2]
    
    # Work at half resolution for ultra-fast light source extraction
    half_float = proc_float[::2, ::2]
    gh, gw = half_float.shape[:2]
    
    val = np.max(half_float, axis=2)
    local_mean = ndimage.uniform_filter(val, size=15)
    local_excess = val - local_mean
    
    # Conservative detector
    core_mask = (val > 0.85) | ((val > 0.65) & (local_excess > 0.20))
    
    labeled, num_features = ndimage.label(core_mask)
    if num_features == 0:
        # Fallback to single-source Session 12
        L_smooth = ndimage.uniform_filter(val, size=15)
        eta = np.clip(L_smooth / (np.max(A_default) + 1e-4), 0.05, 1.0)[:, :, None]
        A_half = A_default[None, None, :] * eta
        A_spatial = np.array(Image.fromarray((A_half * 255.0).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)) / 255.0
        return A_spatial, 1, 0.0
        
    # Gather top clusters by size/prominence
    sizes = ndimage.sum(core_mask, labeled, range(1, num_features + 1))
    top_indices = np.argsort(sizes)[::-1][:max_clusters]
    
    # Diffuse each cluster to create eta_k
    K = len(top_indices)
    A_k_list = []
    eta_half_list = []
    
    for idx in top_indices:
        c_id = idx + 1
        c_mask = (labeled == c_id)
        if np.count_nonzero(c_mask) < 4:
            continue
            
        # Robust chromaticity of source (90th percentile to capture glow rather than clipped core)
        source_pixels = half_float[c_mask]
        A_k = np.percentile(source_pixels, 85, axis=0)
        A_k = np.clip(A_k, 0.15, 1.0)
        A_k_list.append(A_k)
        
        # Spatial influence field
        c_float = c_mask.astype(np.float32)
        if filter_type == "gaussian":
            eta_raw = ndimage.gaussian_filter(c_float, sigma=kernel_size / 3.0)
        else:
            eta_raw = ndimage.uniform_filter(c_float, size=kernel_size)
            
        # Normalize peak
        max_eta = np.max(eta_raw)
        if max_eta > 1e-6:
            eta_raw = eta_raw / max_eta
        eta_half_list.append(eta_raw)
        
    if not A_k_list:
        # Fallback to single source
        L_smooth = ndimage.uniform_filter(val, size=15)
        eta = np.clip(L_smooth / (np.max(A_default) + 1e-4), 0.05, 1.0)[:, :, None]
        A_half = A_default[None, None, :] * eta
        A_spatial = np.array(Image.fromarray((A_half * 255.0).astype(np.uint8)).resize((orig_w, orig_h), Image.BILINEAR)) / 255.0
        return A_spatial, 1, 0.0
        
    # Combine influence fields
    eta_stack = np.stack(eta_half_list, axis=-1) # (gh, gw, K)
    eta_sum = np.sum(eta_stack, axis=-1, keepdims=True)
    
    # Scale so sum <= 1.0
    scale = np.maximum(eta_sum, 1.0)
    eta_norm = eta_stack / scale
    
    # Construct A(x) at half resolution
    A_stack = np.array(A_k_list) # (K, 3)
    A_half = np.sum(eta_norm[:, :, :, None] * A_stack[None, None, :, :], axis=2) # (gh, gw, 3)
    
    # Residual ambient component in unlit areas
    eta_total = np.sum(eta_norm, axis=-1)[:, :, None]
    eta_amb = np.maximum(1.0 - eta_total, 0.0)
    # Ambient floor is scaled down A_default
    A_half = A_half + A_default[None, None, :] * 0.05 * eta_amb
    
    # Upsample A(x) to processing resolution
    A_spatial = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
    for c in range(3):
        A_spatial[:, :, c] = np.array(Image.fromarray(A_half[:, :, c].astype(np.float32), mode='F').resize((orig_w, orig_h), Image.BILINEAR))
        
    return A_spatial, len(A_k_list), float(np.mean(eta_total))

# Test on foggy_video5.mp4
vid_path = r"C:\Users\sakth\Desktop\Fog\fog-removal-project\test_videos\foggy_video5.mp4"
reader = imageio.get_reader(vid_path)
raw_frame = reader.get_data(15)
reader.close()

# Resize to standard max res
scale = min(426.0 / raw_frame.shape[1], 240.0 / raw_frame.shape[0], 1.0)
cur_w, cur_h = int(raw_frame.shape[1] * scale), int(raw_frame.shape[0] * scale)
img = np.array(Image.fromarray(raw_frame).resize((cur_w, cur_h), Image.BILINEAR)).astype(np.float32) / 255.0

A_default = np.array([0.45, 0.45, 0.45], dtype=np.float32)

print("=" * 80)
print("PHASE 4 & 5: SPEED & CHROMATICITY EVALUATION")
print("=" * 80)

for f_type in ["box", "gaussian"]:
    t0 = time.perf_counter()
    for _ in range(50):
        A_sp, n_k, cov = build_multi_source_airlight(img, A_default, max_clusters=4, filter_type=f_type, kernel_size=35)
    el = (time.perf_counter() - t0) / 50.0 * 1000.0
    print(f"Filter: {f_type:<10} | Time: {el:5.2f} ms/frame ({1000.0/el:5.1f} FPS) | Sources: {n_k} | Mean Illum Coverage: {cov*100:.1f}%")
    print(f"  A_spatial stats: Mean={np.mean(A_sp, axis=(0,1))}, Min={np.min(A_sp, axis=(0,1))}, Max={np.max(A_sp, axis=(0,1))}")
