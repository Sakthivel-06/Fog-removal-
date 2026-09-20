"""
scratch/run_session14_ablation.py — Session 14 Full Multi-Video Controlled Benchmark
=====================================================================================
Evaluates across 30 measured frames on:
  - foggy_video3.mp4 (Night Driving Headlights)
  - foggy_video4.mp4 (4K Aerial Night City)
  - foggy_video5.mp4 (4K Neon City Lights)
  - foggy_video14.mp4 (Stadium Floodlight Glare)

Compares:
  Config A: Frozen Daytime Pipeline
  Config B: Session 12 Model A (Baseline Single-Source Box)
  Config C: Distance-Aware Model (Model C: Dist-Modulated Local Illum)
  Config D: Physics Distance Model (Model D: Inverse-Square * Local L)
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
from nighttime_diagnostics import compute_frame_telemetry

NIGHT_VIDEOS = [
    {"id": "B1", "name": "foggy_video3.mp4", "title": "Night Driving Headlights", "n_frames": 45, "warm_up": 15},
    {"id": "B2", "name": "foggy_video4.mp4", "title": "4K Aerial Night City", "n_frames": 35, "warm_up": 10},
    {"id": "D1", "name": "foggy_video5.mp4", "title": "4K Neon City Lights", "n_frames": 35, "warm_up": 10},
    {"id": "D2", "name": "foggy_video14.mp4", "title": "Stadium Floodlight Glare", "n_frames": 35, "warm_up": 10},
]

CONFIGS = [
    {"key": "A", "label": "A. Frozen Daytime Model"},
    {"key": "B", "label": "B. Session 12 Model A (Baseline)"},
    {"key": "C", "label": "C. Distance-Modulated Local (Model C)"},
    {"key": "D", "label": "D. Physics Inv-Square Dist (Model D)"},
]

def run_pipeline_with_mode(pipe, frame, mode_key):
    """Processes frame with custom spatial illumination model."""
    if mode_key == "A":
        # Force daytime frozen branch
        pipe.enable_nighttime_branch = False
        out, fps, diag = pipe.process_frame(frame, return_diagnostics=True)
        pipe.enable_nighttime_branch = True
        return out, fps, compute_frame_telemetry(frame, out), getattr(pipe, "last_A", None)
        
    # Run standard nighttime pipeline preparation up to inversion
    pipe.enable_nighttime_branch = True
    pipe.night_mode = "prototype"
    out_base, fps, diag = pipe.process_frame(frame, return_diagnostics=True)
    
    if mode_key == "B":
        return out_base, fps, compute_frame_telemetry(frame, out_base), getattr(pipe, "last_A", None)
        
    # For C and D, compute modified eta(x)
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
    
    eta_A_h = np.clip(L_smooth_h / a_max, 0.05, 1.0)
    
    if mode_key == "C":
        f_exp_h = np.exp(-D_h / (d_scale * 0.7))
        eta_h = np.clip(eta_A_h * (0.20 + 0.80 * f_exp_h), 0.05, 1.0)
    else: # mode_key == "D"
        f_inv_h = 1.0 / (1.0 + (D_h / (d_scale * 0.5))**2)
        eta_h = np.clip((0.30 + 0.70 * f_inv_h) * (L_smooth_h / a_max), 0.05, 1.0)
        
    eta_full = np.array(Image.fromarray(eta_h.astype(np.float32), mode="F").resize((orig_w, orig_h), Image.BILINEAR))[:, :, None]
    A_sp = A_source[None, None, :] * eta_full
    J_raw = (proc_float - A_sp) / t_full + A_sp
    J = np.clip(J_raw, 0.0, 1.0)
    out = (J * 255.0).astype(np.uint8)
    return out, fps, compute_frame_telemetry(frame, out), A_sp

results = []

print("=" * 95)
print("SESSION 14 — FULL MULTI-VIDEO CONTROLLED BENCHMARK (PHASE 10)")
print("=" * 95)

for vid in NIGHT_VIDEOS:
    vpath = os.path.join(r"C:\Users\sakth\Desktop\Fog\fog-removal-project\test_videos", vid["name"])
    reader = imageio.get_reader(vpath)
    all_frames = []
    for idx, f in enumerate(reader):
        if idx >= vid["n_frames"]:
            break
        all_frames.append(f)
    reader.close()
    
    print(f"\nSequence [{vid['id']}] {vid['title']} ({vid['name']}) — Measured: {len(all_frames) - vid['warm_up']} frames:")
    
    for cfg in CONFIGS:
        pipe = ExperimentalNighttimePipeline()
        
        # Warm up
        for w in range(vid["warm_up"]):
            pipe.process_frame(all_frames[w])
            
        frame_metrics = []
        fps_list = []
        prev_lum = None
        t_variations = []
        A_variations = []
        prev_t = None
        prev_A = None
        
        for k in range(vid["warm_up"], len(all_frames)):
            f = all_frames[k]
            t0 = time.perf_counter()
            out, fps, m, A_curr = run_pipeline_with_mode(pipe, f, cfg["key"])
            t1 = time.perf_counter()
            fps_list.append(1.0 / (t1 - t0 + 1e-9))
            
            lum_out = m["lum_out"]
            if prev_lum is not None:
                m["flicker"] = float(np.mean(np.abs(lum_out - prev_lum)))
            else:
                m["flicker"] = 0.0
            prev_lum = lum_out.copy()
            frame_metrics.append(m)
            
            cur_t = pipe._cached_t
            if prev_t is not None and cur_t is not None and prev_t.shape == cur_t.shape:
                t_variations.append(float(np.mean(np.abs(cur_t - prev_t))))
            if prev_A is not None and A_curr is not None and prev_A.shape == A_curr.shape:
                A_variations.append(float(np.mean(np.abs(A_curr - prev_A))))
            if cur_t is not None:
                prev_t = cur_t.copy()
            if A_curr is not None:
                prev_A = A_curr.copy()

        # Aggregate metrics
        black_clip = float(np.mean([m["black_clip"] for m in frame_metrics]))
        white_clip = float(np.mean([m["white_clip"] for m in frame_metrics]))
        mean_lum = float(np.mean([m["mean_lum"] for m in frame_metrics]))
        b_minus_r = float(np.mean([m["b_minus_r"] for m in frame_metrics]))
        niqe = float(np.mean([m["niqe"] for m in frame_metrics]))
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
            "black_clip": black_clip,
            "white_clip": white_clip,
            "mean_lum": mean_lum,
            "b_minus_r": b_minus_r,
            "niqe": niqe,
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
        results.append(record)
        print(f"  [{cfg['key']}] Black: {black_clip:5.2f}% | White: {white_clip:4.2f}% | Lum: {mean_lum:5.1f} | NIQE: {niqe:4.2f} | Flick: {flicker:4.2f} | Halo: {halo:4.2f}% | FPS: {mean_fps:5.1f}")

out_json = r"C:\Users\sakth\.gemini\antigravity-ide\brain\a110838c-9ebf-4af3-a443-b35b20d1e5d7\scratch\session14_ablation_results.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nPhase 10 Benchmark complete. Saved to {out_json}")
