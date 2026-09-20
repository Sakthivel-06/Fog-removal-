"""
scratch/run_phase7_daytime_check.py — Phase 7 Failure Safety Check
===================================================================
Verifies that the new nighttime branch does NOT degrade daytime sequences.
Compares ExperimentalNighttimePipeline against Session 10 frozen pipeline on:
  - foggy_video1.mp4 (Mountain Road Fog)
  - foggy_video6.mp4 (Valley Cliff Fog)
  - foggy_video7.mp4 (Hill Station Bus)
  - foggy_video2.mp4 (Urban Traffic Signal Bus)
  - car-detection_patchy_haze.mp4 (Synthetic Patchy Haze Benchmark)
"""

import os
import sys
import time
import json
import imageio
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import RealTimeDehazingPipeline
from nighttime_experimental import ExperimentalNighttimePipeline
from nighttime_diagnostics import compute_frame_telemetry

DAY_VIDEOS = [
    {"name": "foggy_video1.mp4", "title": "Mountain Road Fog", "n_frames": 35},
    {"name": "foggy_video6.mp4", "title": "Valley Cliff Fog", "n_frames": 35},
    {"name": "foggy_video7.mp4", "title": "Hill Station Flag & Bus", "n_frames": 35},
    {"name": "foggy_video2.mp4", "title": "Urban Traffic Signal Bus", "n_frames": 35},
    {"name": "car-detection_patchy_haze.mp4", "title": "Patchy Haze Cars", "n_frames": 35},
]

results = []

print("=" * 90)
print("SESSION 12 — PHASE 7: DAYTIME FAILURE SAFETY CHECK")
print("=" * 90)

for vid in DAY_VIDEOS:
    vpath = os.path.join("test_videos", vid["name"])
    if not os.path.exists(vpath):
        vpath = os.path.join(r"C:\Users\sakth\Desktop\Fog\fog-removal-project\test_videos", vid["name"])
    
    reader = imageio.get_reader(vpath)
    frames = []
    for idx, f in enumerate(reader):
        if idx >= vid["n_frames"]:
            break
        frames.append(f)
    reader.close()
    
    pipe_frozen = RealTimeDehazingPipeline()
    pipe_experimental = ExperimentalNighttimePipeline(enable_nighttime_branch=True)
    
    max_diffs = []
    mean_diffs = []
    gate_decisions = []
    
    for i, frame in enumerate(frames):
        out_froz, fps_froz = pipe_frozen.process_frame(frame)
        out_exp, fps_exp, diag = pipe_experimental.process_frame(frame, return_diagnostics=True)
        
        diff = np.abs(out_froz.astype(np.float32) - out_exp.astype(np.float32))
        max_diffs.append(float(np.max(diff)))
        mean_diffs.append(float(np.mean(diff)))
        gate_decisions.append(diag.get("gate_decision", "N/A"))
        
    all_bit_exact = (max(max_diffs) == 0.0)
    print(f"Sequence: {vid['title']:<28} ({vid['name']})")
    print(f"  Gate Decisions:      {set(gate_decisions)}")
    print(f"  Max Pixel Delta:     {max(max_diffs):.4f} / 255")
    print(f"  Mean Pixel Delta:    {np.mean(mean_diffs):.6f}")
    print(f"  Bit-Exact Match:     {'YES (100% BIT-EXACT)' if all_bit_exact else 'DEVIATION DETECTED'}")
    print("-" * 75)
    
    results.append({
        "video_name": vid["name"],
        "video_title": vid["title"],
        "gate_decisions": list(set(gate_decisions)),
        "max_pixel_delta": max(max_diffs),
        "mean_pixel_delta": float(np.mean(mean_diffs)),
        "bit_exact": all_bit_exact,
    })

out_json = r"C:\Users\sakth\.gemini\antigravity-ide\brain\a110838c-9ebf-4af3-a443-b35b20d1e5d7\scratch\phase7_daytime_check_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nPhase 7 safety check completed. Results saved to {out_json}")
