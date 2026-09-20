"""
scratch/run_phase6_phase7_session13.py — Session 13 Controlled Ablation & Color Tests
=====================================================================================
Executes:
  1. Controlled Ablation:
       Config A: Frozen Daytime Model
       Config B: Session 12 Single-Source Model
       Config C: Multi-Source Model (Box)
       Config C_smooth: Multi-Source Model with Temporal Smoothing
     Across nighttime sequences:
       - foggy_video3.mp4 (Night Driving Headlights)
       - foggy_video4.mp4 (4K Aerial Night City)
       - foggy_video5.mp4 (4K Neon City Lights)
       - foggy_video14.mp4 (Stadium Floodlight Glare)

  2. Targeted Color-Source Specific Tests:
       - Test 1: Amber streetlights + blue/white LEDs (foggy_video4.mp4)
       - Test 2: Red + Green traffic signals (foggy_video2.mp4)
       - Test 3: Headlights + dark road (foggy_video3.mp4)
       - Test 4: Neon pink/cyan city (foggy_video5.mp4)
       - Test 5: Stadium floodlights (foggy_video14.mp4)
       - Test 6: Colored object without colored illumination (vehicle body in foggy_video3/2)
"""

import os
import sys
import time
import json
import imageio
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nighttime_experimental import ExperimentalNighttimePipeline
from nighttime_diagnostics import compute_frame_telemetry

NIGHT_VIDEOS = [
    {"id": "B1", "name": "foggy_video3.mp4", "title": "Night Driving Headlights", "n_frames": 45, "warm_up": 15},
    {"id": "B2", "name": "foggy_video4.mp4", "title": "4K Aerial Night City", "n_frames": 35, "warm_up": 10},
    {"id": "D1", "name": "foggy_video5.mp4", "title": "4K Neon City Lights", "n_frames": 35, "warm_up": 10},
    {"id": "D2", "name": "foggy_video14.mp4", "title": "Stadium Floodlight Glare", "n_frames": 35, "warm_up": 10},
]

CONFIGS = [
    {"key": "A", "label": "A. Frozen Daytime Model", "mode": "frozen_daytime"},
    {"key": "B", "label": "B. Single-Source Nighttime (Session 12)", "mode": "prototype"},
    {"key": "C", "label": "C. Multi-Source Nighttime (Session 13 Raw)", "mode": "multi_source"},
    {"key": "C_smooth", "label": "C_smooth. Multi-Source Nighttime (Temporal Smooth)", "mode": "multi_source_smooth"},
]

ablation_results = []

print("=" * 95)
print("SESSION 13 — PHASE 6: CONTROLLED ABLATION (MODEL A VS MODEL B)")
print("=" * 95)

for vid in NIGHT_VIDEOS:
    vpath = os.path.join("test_videos", vid["name"])
    if not os.path.exists(vpath):
        vpath = os.path.join(r"C:\Users\sakth\Desktop\Fog\fog-removal-project\test_videos", vid["name"])
    
    reader = imageio.get_reader(vpath)
    all_frames = []
    for idx, f in enumerate(reader):
        if idx >= vid["n_frames"]:
            break
        all_frames.append(f)
    reader.close()
    
    print(f"\nEvaluating Sequence [{vid['id']}] {vid['title']} ({vid['name']}) — Measured: {len(all_frames) - vid['warm_up']} frames")
    
    for cfg in CONFIGS:
        pipe = ExperimentalNighttimePipeline(night_mode=cfg["mode"])
        
        # Warm up
        for w in range(vid["warm_up"]):
            pipe.process_frame(all_frames[w])
            
        frame_metrics = []
        fps_list = []
        prev_out_lum = None
        t_variations = []
        A_variations = []
        prev_t = None
        prev_A = None
        
        r_means, g_means, b_means = [], [], []
        
        for k in range(vid["warm_up"], len(all_frames)):
            in_frame = all_frames[k]
            t_f0 = time.perf_counter()
            out_frame, fps, diag = pipe.process_frame(in_frame, return_diagnostics=True)
            t_f1 = time.perf_counter()
            fps_list.append(1.0 / (t_f1 - t_f0 + 1e-9))
            
            # Telemetry
            m = compute_frame_telemetry(in_frame, out_frame, prev_out_lum=prev_out_lum)
            prev_out_lum = m["lum_out"]
            frame_metrics.append(m)
            
            # Channel means
            r_means.append(float(np.mean(out_frame[:, :, 0])))
            g_means.append(float(np.mean(out_frame[:, :, 1])))
            b_means.append(float(np.mean(out_frame[:, :, 2])))
            
            # Variations in t and A
            cur_t = pipe._cached_t
            cur_A = getattr(pipe, "_cached_A_spatial", getattr(pipe, "last_A", None))
            if prev_t is not None and cur_t is not None and prev_t.shape == cur_t.shape:
                t_variations.append(float(np.mean(np.abs(cur_t - prev_t))))
            if prev_A is not None and cur_A is not None and prev_A.shape == cur_A.shape:
                A_variations.append(float(np.mean(np.abs(cur_A - prev_A))))
            if cur_t is not None:
                prev_t = cur_t.copy()
            if cur_A is not None:
                prev_A = cur_A.copy()

        # Aggregations
        black_clip = float(np.mean([m["black_clip"] for m in frame_metrics]))
        white_clip = float(np.mean([m["white_clip"] for m in frame_metrics]))
        mean_lum = float(np.mean([m["mean_lum"] for m in frame_metrics]))
        b_minus_r = float(np.mean([m["b_minus_r"] for m in frame_metrics]))
        colorfulness = float(np.mean([m["colorfulness"] for m in frame_metrics]))
        niqe = float(np.mean([m["niqe"] for m in frame_metrics]))
        fdi_val = float(np.mean([m["fdi_val"] for m in frame_metrics]))
        fdi_red = float(np.mean([m["fdi_reduction"] for m in frame_metrics]))
        edge_energy = float(np.mean([m["edge_energy"] for m in frame_metrics]))
        flicker = float(np.mean([m["flicker"] for m in frame_metrics[1:]])) if len(frame_metrics) > 1 else 0.0
        halo = float(np.mean([m["halo_ratio"] for m in frame_metrics]))
        mean_fps = float(np.mean(fps_list))
        delta_t = float(np.mean(t_variations)) if t_variations else 0.0
        delta_A = float(np.mean(A_variations)) if A_variations else 0.0
        
        record = {
            "video_id": vid["id"],
            "video_name": vid["name"],
            "video_title": vid["title"],
            "config_key": cfg["key"],
            "config_label": cfg["label"],
            "config_mode": cfg["mode"],
            "black_clip": black_clip,
            "white_clip": white_clip,
            "mean_lum": mean_lum,
            "rgb_means": [float(np.mean(r_means)), float(np.mean(g_means)), float(np.mean(b_means))],
            "b_minus_r": b_minus_r,
            "colorfulness": colorfulness,
            "niqe": niqe,
            "fdi_val": fdi_val,
            "fdi_reduction": fdi_red,
            "edge_energy": edge_energy,
            "flicker": flicker,
            "delta_t": delta_t,
            "delta_A": delta_A,
            "halo_indicator": halo,
            "fps": mean_fps,
            "psnr": "Requires Ground Truth",
            "ssim": "Requires Ground Truth",
            "lpips": "Requires Ground Truth",
        }
        ablation_results.append(record)
        print(f"  [{cfg['key']:<8}] Black: {black_clip:5.2f}% | White: {white_clip:4.2f}% | Lum: {mean_lum:5.1f} | B-R: {b_minus_r:5.1f} | NIQE: {niqe:4.2f} | FDI_Red: {fdi_red:4.1f}% | Flick: {flicker:4.2f} | d_A: {delta_A:6.4f} | FPS: {mean_fps:5.1f}")

out_json = r"C:\Users\sakth\.gemini\antigravity-ide\brain\a110838c-9ebf-4af3-a443-b35b20d1e5d7\scratch\session13_ablation_results.json"
with open(out_json, "w") as f:
    json.dump(ablation_results, f, indent=2)

print(f"\nPhase 6 Ablation completed. Results saved to {out_json}")
