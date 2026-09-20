"""
scratch/test_candidate_inversions.py — Evaluation of Models A, B, C, D Inversion
================================================================================
Compares:
  Model A: Session 12 Baseline (Box Filter L_smooth)
  Model B: Pure Distance Falloff (Exp decay from active lights)
  Model C: Distance-Modulated Local Illumination (Multiplicative)
  Model D: Physics Incident Illumination (Inverse-square decay * Local L)

Measures:
  - Black clipping %
  - White clipping %
  - Mean luminance
  - Raw negative inversion % (J_raw < 0)
  - NIQE
  - FDI proxy reduction %
  - Halo ratio
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

def evaluate_frame_candidates(pipe, raw_frame):
    # Run pipeline to get internal buffers: t_full, A, proc_float
    out_a, fps, diag = pipe.process_frame(raw_frame, return_diagnostics=True)
    
    proc_float = pipe._proc_float_buf
    orig_h, orig_w = proc_float.shape[:2]
    half_float = proc_float[::2, ::2]
    gh, gw = half_float.shape[:2]
    
    t_smooth = pipe._cached_t
    t_full = np.array(Image.fromarray(t_smooth, mode="F").resize((orig_w, orig_h), Image.BILINEAR))[:, :, None]
    A_source = pipe.last_A
    a_max = max(float(np.max(A_source)), 0.25)
    
    val_h = np.max(half_float, axis=2)
    local_mean_h = ndimage.uniform_filter(val_h, size=15)
    local_excess_h = val_h - local_mean_h
    L_smooth_h = ndimage.uniform_filter(val_h, size=15)
    
    # Active light support
    core_mask = (val_h > 0.85) | ((val_h > 0.65) & (local_excess_h > 0.20))
    labeled, num_features = ndimage.label(core_mask)
    if num_features > 0:
        sizes = ndimage.sum(core_mask, labeled, range(1, num_features + 1))
        keep = np.isin(labeled, np.where(sizes >= 4)[0] + 1)
        m_light = core_mask & keep
    else:
        m_light = np.zeros_like(core_mask)
        
    if np.count_nonzero(m_light) > 0:
        D_h = ndimage.distance_transform_edt(~m_light)
    else:
        D_h = np.full((gh, gw), fill_value=max(gh, gw), dtype=np.float32)
        
    diag_len = np.sqrt(gh**2 + gw**2)
    d_scale = 0.25 * diag_len
    
    # Candidate fields at half resolution
    # Model A: Baseline
    eta_A_h = np.clip(L_smooth_h / a_max, 0.05, 1.0)
    
    # Model B: Pure Distance Falloff
    f_exp_h = np.exp(-D_h / (d_scale * 0.7))
    eta_B_h = np.clip(f_exp_h, 0.05, 1.0)
    
    # Model C: Distance-Modulated Local Illumination (Multiplicative)
    eta_C_h = np.clip(eta_A_h * (0.20 + 0.80 * f_exp_h), 0.05, 1.0)
    
    # Model D: Physics Incident Illumination (Inverse-Square * Local L)
    f_inv_h = 1.0 / (1.0 + (D_h / (d_scale * 0.5))**2)
    eta_D_h = np.clip((0.30 + 0.70 * f_inv_h) * (L_smooth_h / a_max), 0.05, 1.0)
    
    candidates = {"Model A (Baseline)": eta_A_h, "Model B (Pure Distance)": eta_B_h, "Model C (Dist * Local)": eta_C_h, "Model D (Physics Inv-Sq)": eta_D_h}
    
    res = {}
    for name, eta_h in candidates.items():
        eta_full = np.array(Image.fromarray(eta_h.astype(np.float32), mode="F").resize((orig_w, orig_h), Image.BILINEAR))[:, :, None]
        A_sp = A_source[None, None, :] * eta_full
        J_raw = (proc_float - A_sp) / t_full + A_sp
        pct_neg = float(np.count_nonzero(np.any(J_raw < 0.0, axis=2)) / J_raw[:, :, 0].size * 100.0)
        J = np.clip(J_raw, 0.0, 1.0)
        out = (J * 255.0).astype(np.uint8)
        m = compute_frame_telemetry(raw_frame, out)
        m["pct_neg"] = pct_neg
        res[name] = m
    return res

test_vids = [
    {"name": "foggy_video3.mp4", "frame_idx": 25, "label": "Night Driving Headlights"},
    {"name": "foggy_video14.mp4", "frame_idx": 15, "label": "Stadium Floodlight Glare"},
    {"name": "foggy_video4.mp4", "frame_idx": 15, "label": "4K Aerial Night City"},
]

print("=" * 95)
print("COMPARATIVE EVALUATION: CANDIDATE ILLUMINATION FIELDS A, B, C, D")
print("=" * 95)

for v in test_vids:
    vpath = os.path.join(r"C:\Users\sakth\Desktop\Fog\fog-removal-project\test_videos", v["name"])
    reader = imageio.get_reader(vpath)
    pipe = ExperimentalNighttimePipeline(night_mode="prototype")
    for i in range(v["frame_idx"]):
        pipe.process_frame(reader.get_data(i))
    target = reader.get_data(v["frame_idx"])
    reader.close()
    
    results = evaluate_frame_candidates(pipe, target)
    print(f"\nSequence: {v['label']} ({v['name']}) [Frame {v['frame_idx']}]:")
    for m_name, m in results.items():
        print(f"  {m_name:<26} | Black: {m['black_clip']:5.2f}% | White: {m['white_clip']:4.2f}% | Lum: {m['mean_lum']:5.1f} | NegRaw: {m['pct_neg']:5.2f}% | NIQE: {m['niqe']:4.2f} | Halo: {m['halo_ratio']:4.2f}%")

print("\nEvaluation complete.")
