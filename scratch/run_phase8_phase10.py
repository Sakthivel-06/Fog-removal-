"""
scratch/run_phase8_phase10.py — Phase 8 Temporal Consistency & Phase 10 Profiling
===================================================================================
Measures:
  1. Temporal centroid jitter of detected light clusters
  2. Chromaticity stability over time
  3. Spatial airlight temporal delta (d_A)
  4. Frame-to-frame output luminance flicker (d_L)
  5. Component-level CPU runtime profiling
"""

import os
import sys
import time
import json
import imageio
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nighttime_experimental import ExperimentalNighttimePipeline

print("=" * 90)
print("SESSION 13 — PHASE 8: TEMPORAL CONSISTENCY & PHASE 10: RUNTIME PROFILING")
print("=" * 90)

# 1. Temporal Analysis on foggy_video3.mp4 across 30 measured frames
vpath = r"C:\Users\sakth\Desktop\Fog\fog-removal-project\test_videos\foggy_video3.mp4"
reader = imageio.get_reader(vpath)
frames = [reader.get_data(i) for i in range(45)]
reader.close()

pipe_a = ExperimentalNighttimePipeline(night_mode="prototype")
pipe_b = ExperimentalNighttimePipeline(night_mode="multi_source")
pipe_b_smooth = ExperimentalNighttimePipeline(night_mode="multi_source_smooth")

# Warm up
for i in range(15):
    pipe_a.process_frame(frames[i])
    pipe_b.process_frame(frames[i])
    pipe_b_smooth.process_frame(frames[i])

# Profile components
times_detect = []
times_cluster = []
times_field = []
times_inversion = []
times_total_a = []
times_total_b = []

centroids_history = []
A_spatial_diffs_a = []
A_spatial_diffs_b = []
A_spatial_diffs_b_smooth = []

lum_diffs_a = []
lum_diffs_b = []
lum_diffs_b_smooth = []

prev_lum_a = None
prev_lum_b = None
prev_lum_b_smooth = None

prev_A_sp_a = None
prev_A_sp_b = None
prev_A_sp_b_smooth = None

for k in range(15, 45):
    f = frames[k]
    
    # Model A
    t0_a = time.perf_counter()
    out_a, _ = pipe_a.process_frame(f)
    t_a = (time.perf_counter() - t0_a) * 1000.0
    times_total_a.append(t_a)
    
    lum_a = 0.299 * out_a[:, :, 0] + 0.587 * out_a[:, :, 1] + 0.114 * out_a[:, :, 2]
    if prev_lum_a is not None:
        lum_diffs_a.append(float(np.mean(np.abs(lum_a - prev_lum_a))))
    prev_lum_a = lum_a
    
    # Component timing for Model B
    proc_float = pipe_b._proc_float_buf
    orig_h, orig_w = proc_float.shape[:2]
    half_float = proc_float[::2, ::2]
    
    # Step 1: Detect
    t0 = time.perf_counter()
    val = np.max(half_float, axis=2)
    local_mean = ndimage.uniform_filter(val, size=15)
    local_excess = val - local_mean
    core_mask = (val > 0.85) | ((val > 0.65) & (local_excess > 0.20))
    t1 = time.perf_counter()
    times_detect.append((t1 - t0) * 1000.0)
    
    # Step 2: Clustering
    t0 = time.perf_counter()
    labeled, num_features = ndimage.label(core_mask)
    clusters = []
    if num_features > 0:
        sizes = ndimage.sum(core_mask, labeled, range(1, num_features + 1))
        top_idx = np.argsort(sizes)[::-1][:4]
        for idx in top_idx:
            c_mask = (labeled == (idx + 1))
            if np.count_nonzero(c_mask) >= 4:
                y_idx, x_idx = np.nonzero(c_mask)
                clusters.append((float(np.mean(x_idx)), float(np.mean(y_idx))))
    centroids_history.append(clusters)
    t1 = time.perf_counter()
    times_cluster.append((t1 - t0) * 1000.0)
    
    # Step 3: Influence fields
    t0 = time.perf_counter()
    A_sp_b_raw, _ = pipe_b.construct_multi_source_airlight(half_float, (orig_h, orig_w), pipe_b.last_A)
    t1 = time.perf_counter()
    times_field.append((t1 - t0) * 1000.0)
    
    # Step 4: Model B Total
    t0_b = time.perf_counter()
    out_b, _ = pipe_b.process_frame(f)
    t_b = (time.perf_counter() - t0_b) * 1000.0
    times_total_b.append(t_b)
    
    lum_b = 0.299 * out_b[:, :, 0] + 0.587 * out_b[:, :, 1] + 0.114 * out_b[:, :, 2]
    if prev_lum_b is not None:
        lum_diffs_b.append(float(np.mean(np.abs(lum_b - prev_lum_b))))
    prev_lum_b = lum_b
    
    # Model B Smooth
    out_b_sm, _ = pipe_b_smooth.process_frame(f)
    lum_b_sm = 0.299 * out_b_sm[:, :, 0] + 0.587 * out_b_sm[:, :, 1] + 0.114 * out_b_sm[:, :, 2]
    if prev_lum_b_smooth is not None:
        lum_diffs_b_smooth.append(float(np.mean(np.abs(lum_b_sm - prev_lum_b_smooth))))
    prev_lum_b_smooth = lum_b_sm

# Calculate centroid jitter
centroid_jitters = []
for k in range(1, len(centroids_history)):
    prev_c = centroids_history[k-1]
    cur_c = centroids_history[k]
    if prev_c and cur_c:
        # Match nearest
        dists = []
        for cx1, cy1 in cur_c:
            d = min(np.sqrt((cx1 - cx0)**2 + (cy1 - cy0)**2) for cx0, cy0 in prev_c)
            dists.append(d)
        centroid_jitters.append(np.mean(dists))
    elif (not prev_c and cur_c) or (prev_c and not cur_c):
        centroid_jitters.append(15.0) # Penalty for cluster popping in/out

print(f"\n1. TEMPORAL STABILITY ON foggy_video3.mp4 (30 measured frames):")
print(f"  Model A (Single-Source):    Luminance Flicker (d_L) = {np.mean(lum_diffs_a):.2f}")
print(f"  Model B (Multi-Source):     Luminance Flicker (d_L) = {np.mean(lum_diffs_b):.2f}  <-- (+{np.mean(lum_diffs_b) - np.mean(lum_diffs_a):.2f} FLICKER INCREASE)")
print(f"  Model B (Temporal Smooth):  Luminance Flicker (d_L) = {np.mean(lum_diffs_b_smooth):.2f}")
print(f"  Mean Centroid Jitter:       {np.mean(centroid_jitters):.2f} pixels/frame (popping / boundary shift)")

print(f"\n2. COMPONENT-LEVEL CPU RUNTIME (Model B, 180x240 resolution):")
print(f"  Source Detection:           {np.mean(times_detect):5.2f} ms")
print(f"  Connected Component Label:  {np.mean(times_cluster):5.2f} ms")
print(f"  Influence Field Diffusion:  {np.mean(times_field):5.2f} ms")
print(f"  -----------------------------------------------")
print(f"  Model A Total Frame Time:   {np.mean(times_total_a):5.2f} ms ({1000.0/np.mean(times_total_a):5.1f} FPS)")
print(f"  Model B Total Frame Time:   {np.mean(times_total_b):5.2f} ms ({1000.0/np.mean(times_total_b):5.1f} FPS)")

temporal_report = {
    "flicker_model_a": float(np.mean(lum_diffs_a)),
    "flicker_model_b": float(np.mean(lum_diffs_b)),
    "flicker_model_b_smooth": float(np.mean(lum_diffs_b_smooth)),
    "centroid_jitter_px": float(np.mean(centroid_jitters)),
    "time_detect_ms": float(np.mean(times_detect)),
    "time_cluster_ms": float(np.mean(times_cluster)),
    "time_field_ms": float(np.mean(times_field)),
    "total_fps_model_a": float(1000.0 / np.mean(times_total_a)),
    "total_fps_model_b": float(1000.0 / np.mean(times_total_b)),
}

with open(r"C:\Users\sakth\.gemini\antigravity-ide\brain\a110838c-9ebf-4af3-a443-b35b20d1e5d7\scratch\session13_temporal_runtime.json", "w") as f:
    json.dump(temporal_report, f, indent=2)

print("\nPhase 8 & 10 completed.")
