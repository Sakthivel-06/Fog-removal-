"""
scratch/explore_distance_illumination.py — Phase 2 & 3 Proximity & Candidate Fields
===================================================================================
Investigates:
  1. Unified conservative active-light support mask M_light (no clustering)
  2. Euclidean distance transform D(x) = EDT(~M_light)
  3. Distance falloff functions:
       f_exp(x) = exp(-D(x) / d_0)
       f_inv(x) = 1 / (1 + (D(x) / d_0)^2)
       f_gauss(x) = exp(-(D(x) / sigma)^2)
  4. Hybrid distance + local illumination:
       eta_C1 = clip(0.5 * eta_A + 0.5 * f_decay, eta_min, 1.0)
       eta_C2 = clip(eta_A * (0.3 + 0.7 * f_decay), eta_min, 1.0)
       eta_C3 = clip(L_smooth / A_max * f_decay^0.5, eta_min, 1.0)
"""

import os
import sys
import imageio
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage

test_clips = [
    {"name": "foggy_video3.mp4", "frame_idx": 25, "label": "Night Driving Headlights"},
    {"name": "foggy_video14.mp4", "frame_idx": 15, "label": "Stadium Floodlight Glare"},
    {"name": "foggy_video4.mp4", "frame_idx": 15, "label": "4K Aerial Night City"},
]

def compute_candidate_fields(proc_float, A_default):
    orig_h, orig_w = proc_float.shape[:2]
    half_float = proc_float[::2, ::2]
    gh, gw = half_float.shape[:2]
    
    val_h = np.max(half_float, axis=2)
    local_mean_h = ndimage.uniform_filter(val_h, size=15)
    local_excess_h = val_h - local_mean_h
    
    # Model A: Baseline Session 12
    L_smooth_h = ndimage.uniform_filter(val_h, size=15)
    a_max = max(float(np.max(A_default)), 0.25)
    eta_A_h = np.clip(L_smooth_h / a_max, 0.05, 1.0)
    
    # Conservative Active Light Support (Single Unified Mask)
    core_mask = (val_h > 0.85) | ((val_h > 0.65) & (local_excess_h > 0.20))
    # Remove tiny noise specs
    labeled, num_features = ndimage.label(core_mask)
    if num_features > 0:
        sizes = ndimage.sum(core_mask, labeled, range(1, num_features + 1))
        keep_mask = np.isin(labeled, np.where(sizes >= 4)[0] + 1)
        m_light = core_mask & keep_mask
    else:
        m_light = np.zeros_like(core_mask)
        
    num_light_px = np.count_nonzero(m_light)
    
    # Distance Transform
    if num_light_px > 0:
        D_h = ndimage.distance_transform_edt(~m_light)
    else:
        # If no active light detected, distance is large
        D_h = np.full((gh, gw), fill_value=max(gh, gw), dtype=np.float32)
        
    # Scale parameter d_0 relative to guide diagonal
    diag = np.sqrt(gh**2 + gw**2)
    d_scale = 0.25 * diag # ~30-50 pixels
    
    # Candidate B1: Exponential Distance Decay
    f_exp_h = np.exp(-D_h / (d_scale * 0.7))
    eta_B1_h = np.clip(f_exp_h, 0.05, 1.0)
    
    # Candidate B2: Inverse Square Distance Decay
    f_inv_h = 1.0 / (1.0 + (D_h / (d_scale * 0.5))**2)
    eta_B2_h = np.clip(f_inv_h, 0.05, 1.0)
    
    # Candidate C1: Distance-Modulated Local Illumination (Multiplicative)
    # Light illuminates only regions that BOTH receive distance flux AND exhibit local luminance
    eta_C1_h = np.clip(eta_A_h * (0.20 + 0.80 * f_exp_h), 0.05, 1.0)
    
    # Candidate C2: Additive Soft Blend
    eta_C2_h = np.clip(0.60 * eta_A_h + 0.40 * f_inv_h, 0.05, 1.0)
    
    # Candidate C3: Normalized Physics-Guided Incident Illumination
    # Incident flux E(x) ~ f_inv(x) * (0.3 + 0.7 * L_smooth)
    eta_C3_h = np.clip((0.30 + 0.70 * f_inv_h) * (L_smooth_h / a_max), 0.05, 1.0)
    
    return {
        "num_light_px": num_light_px,
        "light_pct": (num_light_px / (gh * gw)) * 100.0,
        "D_mean": float(np.mean(D_h)),
        "D_max": float(np.max(D_h)),
        "eta_A": eta_A_h,
        "eta_B1": eta_B1_h,
        "eta_B2": eta_B2_h,
        "eta_C1": eta_C1_h,
        "eta_C2": eta_C2_h,
        "eta_C3": eta_C3_h,
    }

print("=" * 90)
print("PHASE 2 & 3: LIGHT-SOURCE DISTANCE AND CANDIDATE ILLUMINATION FIELDS")
print("=" * 90)

for item in test_clips:
    vpath = os.path.join(r"C:\Users\sakth\Desktop\Fog\fog-removal-project\test_videos", item["name"])
    reader = imageio.get_reader(vpath)
    raw = reader.get_data(item["frame_idx"])
    reader.close()
    
    # Resize to standard max res
    scale = min(426.0 / raw.shape[1], 240.0 / raw.shape[0], 1.0)
    cur_w, cur_h = int(raw.shape[1] * scale), int(raw.shape[0] * scale)
    proc_float = np.array(Image.fromarray(raw).resize((cur_w, cur_h), Image.BILINEAR)).astype(np.float32) / 255.0
    
    A_default = np.array([0.45, 0.45, 0.45], dtype=np.float32)
    stats = compute_candidate_fields(proc_float, A_default)
    
    print(f"\nSequence: {item['label']} ({item['name']}) [Frame {item['frame_idx']}, Size: {cur_w}x{cur_h}]")
    print(f"  Active Light Support: {stats['num_light_px']} px ({stats['light_pct']:.2f}%) | Mean Dist: {stats['D_mean']:.1f} px, Max Dist: {stats['D_max']:.1f} px")
    print(f"  Field Mean / Min / Max:")
    print(f"    - Model A (Baseline Session 12) : Mean={np.mean(stats['eta_A']):.3f}, Min={np.min(stats['eta_A']):.3f}, Max={np.max(stats['eta_A']):.3f}")
    print(f"    - Model B1 (Exp Distance Decay) : Mean={np.mean(stats['eta_B1']):.3f}, Min={np.min(stats['eta_B1']):.3f}, Max={np.max(stats['eta_B1']):.3f}")
    print(f"    - Model B2 (Inv-Sq Dist Decay)  : Mean={np.mean(stats['eta_B2']):.3f}, Min={np.min(stats['eta_B2']):.3f}, Max={np.max(stats['eta_B2']):.3f}")
    print(f"    - Model C1 (Dist * Local Illum) : Mean={np.mean(stats['eta_C1']):.3f}, Min={np.min(stats['eta_C1']):.3f}, Max={np.max(stats['eta_C1']):.3f}")
    print(f"    - Model C2 (Soft Blend)         : Mean={np.mean(stats['eta_C2']):.3f}, Min={np.min(stats['eta_C2']):.3f}, Max={np.max(stats['eta_C2']):.3f}")
    print(f"    - Model C3 (Physics Incident)   : Mean={np.mean(stats['eta_C3']):.3f}, Min={np.min(stats['eta_C3']):.3f}, Max={np.max(stats['eta_C3']):.3f}")

print("\nCandidate field generation complete.")
