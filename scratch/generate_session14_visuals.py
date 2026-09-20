"""
scratch/generate_session14_visuals.py — Visual Contact Sheet for Session 14
============================================================================
Generates side-by-side visual comparison across:
  Col 1: Raw Input
  Col 2: Config A (Frozen Daytime Pipeline)
  Col 3: Config B (Session 12 Baseline Model A)
  Col 4: Config C (Distance-Modulated Model C)
  Col 5: Config D (Physics Inv-Square Model D)
"""

import os
import sys
import imageio
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nighttime_experimental import ExperimentalNighttimePipeline

test_scenes = [
    {"name": "foggy_video3.mp4", "frame_idx": 25, "label": "Night Driving (Headlights)"},
    {"name": "foggy_video4.mp4", "frame_idx": 15, "label": "4K Aerial Night City"},
    {"name": "foggy_video5.mp4", "frame_idx": 15, "label": "4K Neon City Lights"},
    {"name": "foggy_video14.mp4", "frame_idx": 15, "label": "Stadium Floodlight Glare"},
]

cols = [
    "Raw Input",
    "Config A (Frozen Daytime)",
    "Config B (Session 12 Model A)",
    "Config C (Dist-Modulated C)",
    "Config D (Physics Inv-Sq D)",
]

fig, axes = plt.subplots(len(test_scenes), len(cols), figsize=(22, 16))
plt.subplots_adjust(wspace=0.04, hspace=0.16)

for row_idx, scene in enumerate(test_scenes):
    vpath = os.path.join(r"C:\Users\sakth\Desktop\Fog\fog-removal-project\test_videos", scene["name"])
    reader = imageio.get_reader(vpath)
    frames = [reader.get_data(k) for k in range(scene["frame_idx"] + 1)]
    reader.close()
    
    raw_frame = frames[scene["frame_idx"]]
    
    # Run pipeline to get base buffers
    pipe = ExperimentalNighttimePipeline()
    for f in frames:
        out_b, _ = pipe.process_frame(f)
        
    # Config A: Frozen daytime
    pipe_a = ExperimentalNighttimePipeline(night_mode="frozen_daytime")
    for f in frames:
        out_a, _ = pipe_a.process_frame(f)
        
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
    
    eta_A_h = np.clip(L_smooth_h / a_max, 0.05, 1.0)
    
    # Model C
    f_exp_h = np.exp(-D_h / (d_scale * 0.7))
    eta_C_h = np.clip(eta_A_h * (0.20 + 0.80 * f_exp_h), 0.05, 1.0)
    eta_C_full = np.array(Image.fromarray(eta_C_h.astype(np.float32), mode="F").resize((orig_w, orig_h), Image.BILINEAR))[:, :, None]
    A_C = A_source[None, None, :] * eta_C_full
    out_c = (np.clip((proc_float - A_C) / t_full + A_C, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    # Model D
    f_inv_h = 1.0 / (1.0 + (D_h / (d_scale * 0.5))**2)
    eta_D_h = np.clip((0.30 + 0.70 * f_inv_h) * (L_smooth_h / a_max), 0.05, 1.0)
    eta_D_full = np.array(Image.fromarray(eta_D_h.astype(np.float32), mode="F").resize((orig_w, orig_h), Image.BILINEAR))[:, :, None]
    A_D = A_source[None, None, :] * eta_D_full
    out_d = (np.clip((proc_float - A_D) / t_full + A_D, 0.0, 1.0) * 255.0).astype(np.uint8)
    
    row_images = [raw_frame, out_a, out_b, out_c, out_d]
    
    for col_idx, img_disp in enumerate(row_images):
        ax = axes[row_idx, col_idx]
        ax.imshow(img_disp)
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(cols[col_idx], fontsize=12, fontweight='bold', pad=8)
        if col_idx == 0:
            ax.text(-0.05, 0.5, scene["label"], transform=ax.transAxes,
                    fontsize=11, fontweight='bold', rotation=90,
                    va='center', ha='right')

plt.suptitle("Session 14: Distance-Aware Illumination Field Refinement Comparison",
             fontsize=16, fontweight='bold', y=0.96)

out_path = r"C:\Users\sakth\.gemini\antigravity-ide\brain\a110838c-9ebf-4af3-a443-b35b20d1e5d7\session14_distance_illum_comparison.jpg"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved visual contact sheet to {out_path}")
