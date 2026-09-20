"""
scratch/generate_session13_visuals.py — Visual Contact Sheet for Session 13
============================================================================
Generates side-by-side visual comparison across:
  Col 1: Raw Input
  Col 2: Config A (Frozen Daytime Pipeline)
  Col 3: Config B (Single-Source Model A)
  Col 4: Config C (Multi-Source Model B)
  Col 5: Multi-Source Airlight Field A(x) RGB
"""

import os
import sys
import imageio
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    "Config B (Single-Source A)",
    "Config C (Multi-Source B)",
    "Multi-Source Airlight Map A(x)",
]

fig, axes = plt.subplots(len(test_scenes), len(cols), figsize=(22, 16))
plt.subplots_adjust(wspace=0.04, hspace=0.16)

for row_idx, scene in enumerate(test_scenes):
    vpath = os.path.join(r"C:\Users\sakth\Desktop\Fog\fog-removal-project\test_videos", scene["name"])
    reader = imageio.get_reader(vpath)
    frames = [reader.get_data(k) for k in range(scene["frame_idx"] + 1)]
    reader.close()
    
    raw_frame = frames[scene["frame_idx"]]
    
    # 1. Config A: Frozen daytime
    pipe_a = ExperimentalNighttimePipeline(night_mode="frozen_daytime")
    for f in frames:
        out_a, _ = pipe_a.process_frame(f)
        
    # 2. Config B: Single-source
    pipe_b = ExperimentalNighttimePipeline(night_mode="prototype")
    for f in frames:
        out_b, _ = pipe_b.process_frame(f)
        
    # 3. Config C: Multi-source
    pipe_c = ExperimentalNighttimePipeline(night_mode="multi_source")
    for f in frames:
        out_c, _ = pipe_c.process_frame(f)
        
    # Airlight Map A(x) from pipe_c
    proc_float = pipe_c._proc_float_buf
    half_float = proc_float[::2, ::2]
    A_sp, _ = pipe_c.construct_multi_source_airlight(half_float, proc_float.shape, pipe_c.last_A)
    # Brighten airlight map slightly for visualization if dark
    A_vis = np.clip(A_sp * 1.5, 0.0, 1.0)
    
    row_images = [
        raw_frame,
        out_a,
        out_b,
        out_c,
        (A_vis * 255.0).astype(np.uint8),
    ]
    
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

plt.suptitle("Session 13: Single-Source (Model A) vs Multi-Source (Model B) Nighttime Airlight Comparison",
             fontsize=16, fontweight='bold', y=0.96)

out_path = r"C:\Users\sakth\.gemini\antigravity-ide\brain\a110838c-9ebf-4af3-a443-b35b20d1e5d7\session13_multi_source_comparison.jpg"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved visual contact sheet to {out_path}")
