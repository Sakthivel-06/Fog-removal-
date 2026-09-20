"""
scratch/generate_session12_visuals.py — Visual Contact Sheet for Session 12
============================================================================
Generates side-by-side visual comparison across all 4 ablation configurations:
  Col 1: Raw Input
  Col 2: Config A (Frozen Daytime Pipeline)
  Col 3: Config B (Proposed Nighttime Prototype)
  Col 4: Config C (Prototype w/o Illum Correction)
  Col 5: Config D (Prototype w/o Modified Airlight)

Rows:
  Row 1: Night Driving Headlights (foggy_video3.mp4, Frame 25)
  Row 2: Night Driving Dark Highway (foggy_video3.mp4, Frame 35)
  Row 3: 4K Aerial Night City (foggy_video4.mp4, Frame 15)
  Row 4: Stadium Floodlight Glare (foggy_video14.mp4, Frame 15)
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
    {"name": "foggy_video3.mp4", "frame_idx": 25, "label": "Night Driving (Near Lights)"},
    {"name": "foggy_video3.mp4", "frame_idx": 35, "label": "Night Driving (Dark Verges)"},
    {"name": "foggy_video4.mp4", "frame_idx": 15, "label": "4K Aerial Night City"},
    {"name": "foggy_video14.mp4", "frame_idx": 15, "label": "Stadium Floodlight Glare"},
]

modes = [
    ("Raw Input", None),
    ("Config A (Frozen Daytime)", "frozen_daytime"),
    ("Config B (Night Prototype)", "prototype"),
    ("Config C (w/o Illum Corr)", "no_illum_correction"),
    ("Config D (w/o Airlight Mod)", "no_airlight_mod"),
]

fig, axes = plt.subplots(len(test_scenes), len(modes), figsize=(20, 15))
plt.subplots_adjust(wspace=0.03, hspace=0.15)

for row_idx, scene in enumerate(test_scenes):
    vpath = os.path.join("test_videos", scene["name"])
    if not os.path.exists(vpath):
        vpath = os.path.join(r"C:\Users\sakth\Desktop\Fog\fog-removal-project\test_videos", scene["name"])
    
    reader = imageio.get_reader(vpath)
    frames = []
    for k, f in enumerate(reader):
        if k > scene["frame_idx"]:
            break
        frames.append(f)
    reader.close()
    
    raw_frame = frames[scene["frame_idx"]]
    
    for col_idx, (m_label, mode) in enumerate(modes):
        ax = axes[row_idx, col_idx]
        if mode is None:
            # Raw Input
            out_img = raw_frame
        else:
            pipe = ExperimentalNighttimePipeline(night_mode=mode)
            for k in range(scene["frame_idx"] + 1):
                out_img, _ = pipe.process_frame(frames[k])
                
        ax.imshow(out_img)
        ax.axis('off')
        if row_idx == 0:
            ax.set_title(m_label, fontsize=12, fontweight='bold', pad=8)
        if col_idx == 0:
            ax.text(-0.05, 0.5, scene["label"], transform=ax.transAxes,
                    fontsize=11, fontweight='bold', rotation=90,
                    va='center', ha='right')

plt.suptitle("Session 12: Controlled Ablation Comparison on Nighttime Video Sequences",
             fontsize=16, fontweight='bold', y=0.96)

out_path = r"C:\Users\sakth\.gemini\antigravity-ide\brain\a110838c-9ebf-4af3-a443-b35b20d1e5d7\session12_night_ablation_comparison.jpg"
plt.savefig(out_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Saved visual contact sheet to {out_path}")
