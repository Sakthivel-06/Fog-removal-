"""
scratch/run_phase7_color_tests.py — Phase 7 Color-Source-Specific Targeted Tests
================================================================================
Evaluates:
  TEST 1: Amber streetlights + blue/white LEDs (foggy_video4.mp4)
  TEST 2: Red traffic light + green traffic light (foggy_video2.mp4)
  TEST 3: Headlights + dark road (foggy_video3.mp4)
  TEST 4: Neon pink/cyan city (foggy_video5.mp4)
  TEST 5: Stadium floodlights (foggy_video14.mp4)
  TEST 6: Colored object without colored illumination (vehicle body)
"""

import os
import sys
import json
import imageio
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nighttime_experimental import ExperimentalNighttimePipeline
from nighttime_diagnostics import compute_frame_telemetry

TARGET_TESTS = [
    {
        "test_id": "TEST 1",
        "name": "Amber streetlights + blue/white LEDs",
        "video": "foggy_video4.mp4",
        "frame_idx": 15,
        "region_roi": (100, 150, 200, 250), # crop around mixed city street lights
        "desc": "Mixed sodium amber lamps and cool LED building lights"
    },
    {
        "test_id": "TEST 2",
        "name": "Red traffic light + green traffic light",
        "video": "foggy_video2.mp4",
        "frame_idx": 20,
        "region_roi": (80, 120, 100, 140), # pedestrian/traffic signal area
        "desc": "Emerald green signal light in urban mist"
    },
    {
        "test_id": "TEST 3",
        "name": "Headlights + dark road",
        "video": "foggy_video3.mp4",
        "frame_idx": 30,
        "region_roi": (120, 60, 200, 180), # illuminated road surface and verge
        "desc": "Vehicle headlight cone spreading across dark wet asphalt"
    },
    {
        "test_id": "TEST 4",
        "name": "Neon pink/cyan city",
        "video": "foggy_video5.mp4",
        "frame_idx": 15,
        "region_roi": (50, 150, 150, 250), # neon signs cluster
        "desc": "Commercial storefront neon signs (cyan, magenta, yellow)"
    },
    {
        "test_id": "TEST 5",
        "name": "Stadium floodlights",
        "video": "foggy_video14.mp4",
        "frame_idx": 15,
        "region_roi": (20, 70, 80, 130), # floodlight bloom core and halo
        "desc": "Intense white floodlight emitter with atmospheric halo"
    },
    {
        "test_id": "TEST 6",
        "name": "Colored object without colored illumination",
        "video": "foggy_video3.mp4",
        "frame_idx": 25,
        "region_roi": (180, 0, 240, 180), # car dashboard and wiper
        "desc": "Dark car hood/dashboard under ambient darkness (passive object)"
    },
]

results = []

print("=" * 95)
print("SESSION 13 — PHASE 7: COLOR-SOURCE-SPECIFIC TARGETED TESTS")
print("=" * 95)

for t in TARGET_TESTS:
    vpath = os.path.join(r"C:\Users\sakth\Desktop\Fog\fog-removal-project\test_videos", t["video"])
    reader = imageio.get_reader(vpath)
    frames = [reader.get_data(i) for i in range(t["frame_idx"] + 1)]
    reader.close()
    
    target_raw = frames[t["frame_idx"]]
    
    # Process under:
    # 1. Model A (Session 12 Single-Source)
    pipe_a = ExperimentalNighttimePipeline(night_mode="prototype")
    for f in frames:
        out_a, _, diag_a = pipe_a.process_frame(f, return_diagnostics=True)
    m_a = compute_frame_telemetry(target_raw, out_a)
    
    # 2. Model B (Session 13 Multi-Source)
    pipe_b = ExperimentalNighttimePipeline(night_mode="multi_source")
    for f in frames:
        out_b, _, diag_b = pipe_b.process_frame(f, return_diagnostics=True)
    m_b = compute_frame_telemetry(target_raw, out_b)
    
    # Crop ROI
    y0, x0, y1, x1 = t["region_roi"]
    # Ensure bounds match output shape
    oh, ow = out_a.shape[:2]
    y0 = min(y0, oh-1)
    y1 = min(y1, oh)
    x0 = min(x0, ow-1)
    x1 = min(x1, ow)
    
    crop_raw = target_raw[y0:y1, x0:x1] if target_raw.shape[:2] == (oh, ow) else out_a[y0:y1, x0:x1]
    crop_a = out_a[y0:y1, x0:x1]
    crop_b = out_b[y0:y1, x0:x1]
    
    # ROI color balance (Mean RGB and B - R)
    rgb_a = np.mean(crop_a, axis=(0, 1))
    rgb_b = np.mean(crop_b, axis=(0, 1))
    b_minus_r_a = float(rgb_a[2] - rgb_a[0])
    b_minus_r_b = float(rgb_b[2] - rgb_b[0])
    
    # Check for false recoloring or desaturation
    delta_color = float(np.mean(np.abs(rgb_b - rgb_a)))
    recolored = bool(delta_color > 15.0)
    
    record = {
        "test_id": t["test_id"],
        "name": t["name"],
        "video": t["video"],
        "desc": t["desc"],
        "model_a": {
            "black_clip": m_a["black_clip"],
            "white_clip": m_a["white_clip"],
            "halo": m_a["halo_ratio"],
            "rgb_roi": [float(c) for c in rgb_a],
            "b_minus_r": b_minus_r_a,
        },
        "model_b": {
            "black_clip": m_b["black_clip"],
            "white_clip": m_b["white_clip"],
            "halo": m_b["halo_ratio"],
            "rgb_roi": [float(c) for c in rgb_b],
            "b_minus_r": b_minus_r_b,
        },
        "delta_color_roi": delta_color,
        "object_recoloring_detected": recolored,
    }
    results.append(record)
    
    print(f"\n{t['test_id']}: {t['name']} ({t['video']})")
    print(f"  Description:           {t['desc']}")
    print(f"  Model A ROI RGB:       [{rgb_a[0]:.1f}, {rgb_a[1]:.1f}, {rgb_a[2]:.1f}] | B-R: {b_minus_r_a:5.1f} | Black: {m_a['black_clip']:4.1f}% | Halo: {m_a['halo_ratio']:4.2f}%")
    print(f"  Model B ROI RGB:       [{rgb_b[0]:.1f}, {rgb_b[1]:.1f}, {rgb_b[2]:.1f}] | B-R: {b_minus_r_b:5.1f} | Black: {m_b['black_clip']:4.1f}% | Halo: {m_b['halo_ratio']:4.2f}%")
    print(f"  Color Delta:           {delta_color:.2f} | False Object Recoloring: {'YES (UNSTABLE)' if recolored else 'NO (STABLE)'}")

out_json = r"C:\Users\sakth\.gemini\antigravity-ide\brain\a110838c-9ebf-4af3-a443-b35b20d1e5d7\scratch\session13_color_tests_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nPhase 7 Color tests completed. Saved to {out_json}")
