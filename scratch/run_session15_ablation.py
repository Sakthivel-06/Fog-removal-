"""
scratch/run_session15_ablation.py — Full Session 15 Benchmark Across Nighttime Videos
===================================================================================
Compares:
  Model A: Session 12 Baseline (t_A = t)
  Model B: Hard Physical Non-Negativity Floor (t_B = max(t, 1 - I/A))
  Model C: Soft Underflow-Aware Floor (C^1 continuous softplus transition)
  Model D: Dark-Region Adaptive Transmission Confidence (C_trans * t + (1 - C_trans))
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
    {"key": "A", "label": "Model A (Session 12 Baseline)"},
    {"key": "B", "label": "Model B (Hard Physical Floor)"},
    {"key": "C", "label": "Model C (Soft Underflow Floor)"},
    {"key": "D", "label": "Model D (Dark-Region Confidence)"},
]

def run_pipeline_with_candidate(pipe, frame, candidate_key):
    """Processes frame and applies candidate transmission safety formulation."""
    t0 = time.perf_counter()
    # Execute standard pipeline process_frame to maintain temporal state
    out_base, fps_base, diag = pipe.process_frame(frame, return_diagnostics=True)
    
    # Check if gate classified as daytime
    if not diag.get("is_night", False):
        t1 = time.perf_counter()
        fps = 1.0 / (t1 - t0 + 1e-9)
        tel = compute_frame_telemetry(frame, out_base)
        lum_in = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        native_black = float(np.count_nonzero(lum_in < 1.0) / lum_in.size * 100.0)
        tel["native_black"] = native_black
        tel["induced_crush"] = tel["black_clip"] - native_black
        tel["dark_mean_lum"] = tel["mean_lum"]
        return out_base, fps, tel, getattr(pipe, "last_A", None), pipe._cached_t

    proc_float = pipe._proc_float_buf
    orig_h, orig_w = proc_float.shape[:2]
    
    # Intermediate representations
    t_smooth = pipe._cached_t
    t_full = np.array(Image.fromarray(t_smooth, mode="F").resize((orig_w, orig_h), Image.BILINEAR))[:, :, None]
    A_source = pipe.last_A
    a_max_ref = max(float(np.max(A_source)), 0.25)

    V_full = np.max(proc_float, axis=2)
    L_smooth = ndimage.uniform_filter(V_full, size=pipe.local_illum_kernel_size)
    eta = np.clip(L_smooth / a_max_ref, pipe.eta_min, 1.0)[:, :, None]
    A_spatial = A_source[None, None, :] * eta

    # Compute candidate transmission t_cand
    eps = 1e-4
    A_eff = np.maximum(A_spatial, eps)
    t_req = np.max(1.0 - proc_float / A_eff, axis=2, keepdims=True)

    if candidate_key == "A":
        t_cand = t_full
    elif candidate_key == "B":
        t_cand = np.clip(np.maximum(t_full, t_req), 0.05, 1.0)
    elif candidate_key == "C":
        k = 20.0
        delta = 0.03
        diff = t_req - t_full + delta
        soft_excess = np.where(k * diff > 30.0, diff, (1.0 / k) * np.log1p(np.exp(np.clip(k * diff, -30.0, 30.0))))
        t_cand = np.clip(t_full + soft_excess, 0.05, 1.0)
    elif candidate_key == "D":
        conf_lum = np.clip((L_smooth - 0.02) / 0.15, 0.0, 1.0)[:, :, None]
        conf_eta = np.clip((eta - pipe.eta_min) / (0.40 - pipe.eta_min), 0.0, 1.0)
        C_trans = conf_lum * conf_eta
        t_cand = np.clip(1.0 - C_trans * (1.0 - t_full), 0.05, 1.0)
    else:
        raise ValueError(f"Unknown candidate key: {candidate_key}")

    # Physical scene radiance inversion
    J_raw = (proc_float - A_spatial) / t_cand + A_spatial
    J = np.clip(J_raw, 0.0, 1.0)
    out = (J * 255.0).astype(np.uint8)

    t1 = time.perf_counter()
    fps = 1.0 / (t1 - t0 + 1e-9)
    
    tel = compute_frame_telemetry(frame, out)
    in_lum = 0.299 * proc_float[:, :, 0] + 0.587 * proc_float[:, :, 1] + 0.114 * proc_float[:, :, 2]
    native_black = float(np.count_nonzero(in_lum < (1.0 / 255.0)) / in_lum.size * 100.0)
    tel["native_black"] = native_black
    tel["induced_crush"] = tel["black_clip"] - native_black

    # Dark-region mean luminance (< 38/255)
    dark_mask = (in_lum < (38.0 / 255.0))
    tel["dark_mean_lum"] = float(np.mean(tel["lum_out"][dark_mask])) if np.any(dark_mask) else 0.0

    return out, fps, tel, A_spatial, t_cand

def run_full_benchmark():
    all_results = []

    print("=" * 100)
    print("SESSION 15 — TRANSMISSION-SAFETY CONTROLLED ABLATION (PHASE 11)")
    print("=" * 100)

    for vid in NIGHT_VIDEOS:
        vpath = os.path.join("test_videos", vid["name"])
        reader = imageio.get_reader(vpath)
        all_frames = []
        for idx, f in enumerate(reader):
            if idx >= vid["n_frames"]:
                break
            all_frames.append(f)
        reader.close()

        measured_frames = len(all_frames) - vid["warm_up"]
        print(f"\nSequence [{vid['id']}] {vid['title']} ({vid['name']}) — Measured: {measured_frames} frames:")

        for cfg in CONFIGS:
            pipe = ExperimentalNighttimePipeline()
            # Warm up
            for w in range(vid["warm_up"]):
                pipe.process_frame(all_frames[w])

            frame_metrics = []
            fps_list = []
            prev_lum = None
            prev_t = None
            prev_A = None
            t_variations = []
            A_variations = []

            for i in range(vid["warm_up"], len(all_frames)):
                f = all_frames[i]
                out, fps, m, A_curr, t_curr = run_pipeline_with_candidate(pipe, f, cfg["key"])
                fps_list.append(fps)

                lum_out = m["lum_out"]
                if prev_lum is not None:
                    m["flicker"] = float(np.mean(np.abs(lum_out - prev_lum)))
                else:
                    m["flicker"] = 0.0
                prev_lum = lum_out.copy()
                frame_metrics.append(m)

                if prev_t is not None and t_curr is not None and prev_t.shape == t_curr.shape:
                    t_variations.append(float(np.mean(np.abs(t_curr - prev_t))))
                if prev_A is not None and A_curr is not None and prev_A.shape == A_curr.shape:
                    A_variations.append(float(np.mean(np.abs(A_curr - prev_A))))
                if t_curr is not None:
                    prev_t = t_curr.copy()
                if A_curr is not None:
                    prev_A = A_curr.copy()

            # Aggregate
            black_clip = float(np.mean([m["black_clip"] for m in frame_metrics]))
            native_black = float(np.mean([m["native_black"] for m in frame_metrics]))
            induced_crush = float(np.mean([m["induced_crush"] for m in frame_metrics]))
            white_clip = float(np.mean([m["white_clip"] for m in frame_metrics]))
            mean_lum = float(np.mean([m["mean_lum"] for m in frame_metrics]))
            dark_mean_lum = float(np.mean([m["dark_mean_lum"] for m in frame_metrics]))
            b_minus_r = float(np.mean([m["b_minus_r"] for m in frame_metrics]))
            niqe = float(np.mean([m["niqe"] for m in frame_metrics]))
            fdi_red = float(np.mean([m["fdi_reduction"] for m in frame_metrics]))
            edge_energy = float(np.mean([m["edge_energy"] for m in frame_metrics]))
            flicker = float(np.mean([m["flicker"] for m in frame_metrics[1:]])) if len(frame_metrics) > 1 else 0.0
            halo = float(np.mean([m["halo_ratio"] for m in frame_metrics]))
            mean_fps = float(np.mean(fps_list))
            delta_t = float(np.mean(t_variations)) if t_variations else 0.0
            delta_A = float(np.mean(A_variations)) if A_variations else 0.0

            rec = {
                "video_id": vid["id"],
                "video_name": vid["name"],
                "video_title": vid["title"],
                "config_key": cfg["key"],
                "config_label": cfg["label"],
                "black_clip": black_clip,
                "native_black": native_black,
                "induced_crush": induced_crush,
                "white_clip": white_clip,
                "mean_lum": mean_lum,
                "dark_mean_lum": dark_mean_lum,
                "b_minus_r": b_minus_r,
                "niqe": niqe,
                "fdi_reduction": fdi_red,
                "edge_energy": edge_energy,
                "flicker": flicker,
                "delta_t": delta_t,
                "delta_A": delta_A,
                "halo_ratio": halo,
                "fps": mean_fps,
                "psnr": "Requires Ground Truth",
                "ssim": "Requires Ground Truth",
                "lpips": "Requires Ground Truth",
            }
            all_results.append(rec)
            print(f"  [{cfg['key']}] Black: {black_clip:5.2f}% (Nat: {native_black:5.2f}%, Crush: {induced_crush:+5.2f}%) | "
                  f"White: {white_clip:4.2f}% | DarkLum: {dark_mean_lum:5.1f} | NIQE: {niqe:4.2f} | Flick: {flicker:4.2f} | "
                  f"Dt: {delta_t:.4f} | Halo: {halo:4.2f}% | FPS: {mean_fps:5.1f}")

    out_json = r"C:\Users\sakth\.gemini\antigravity-ide\brain\a110838c-9ebf-4af3-a443-b35b20d1e5d7\scratch\session15_ablation_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved ablation benchmark to {out_json}")

if __name__ == "__main__":
    run_full_benchmark()
