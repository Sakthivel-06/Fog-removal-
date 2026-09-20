import os
import sys
import time
import json
import numpy as np
from PIL import Image
import imageio.v3 as iio

# Add root to python path
sys.path.insert(0, os.path.abspath("."))

import metrics
from novel_method.pipeline import RealTimeDehazingPipeline
from novel_method.pipeline_learned_experimental import LearnedResidualDehazingPipeline
import algorithms.dcp as alg_dcp
import algorithms.cap as alg_cap
import algorithms.clahe_baseline as alg_clahe
import algorithms.ceef as alg_ceef_mod
import algorithms.ceef_original as alg_ceef_orig
from algorithms.light_dehazenet_wrapper import LightDehazeNetWrapper

def run_comparison(video_path="test_videos/foggy_video.mp4", num_frames=30):
    print("=" * 80)
    print(f"BENCHMARKING 10 ALGORITHMS ON: {video_path}")
    print(f"Sampling {num_frames} frames under identical processing conditions")
    print("=" * 80)
    
    # 1. Read test frames
    reader = iio.imiter(video_path)
    frames_raw = []
    for idx, f in enumerate(reader):
        frames_raw.append(f)
        if len(frames_raw) >= num_frames:
            break
            
    print(f"Loaded {len(frames_raw)} frames from {video_path}. Native resolution: {frames_raw[0].shape}")

    # Scale to standard fast evaluation resolution (238 x 426, aspect-ratio preserved, even dims)
    target_w, target_h = 238, 426
    eval_frames = [
        np.array(Image.fromarray(f).resize((target_w, target_h), Image.Resampling.BILINEAR))
        for f in frames_raw
    ]
    print(f"Standardized test resolution: {eval_frames[0].shape} (WxH: {target_w}x{target_h})")

    # 2. Instantiate all methods
    p_a1 = RealTimeDehazingPipeline(
        downscale_res=(512, 288), max_process_res=None,
        beta=1.0, kappa=1.25, omega=0.85, enable_sky_model=True, preserve_resolution=False
    )
    p_a2 = LearnedResidualDehazingPipeline(
        checkpoint_path="checkpoints/delta_t_tiny_unet_no_temp.pt", ablation_mode="A2",
        downscale_res=(512, 288), max_process_res=None,
        beta=1.0, kappa=1.25, omega=0.85, enable_sky_model=True, preserve_resolution=False
    )
    p_a3 = LearnedResidualDehazingPipeline(
        checkpoint_path="checkpoints/delta_t_tiny_unet_best.pt", ablation_mode="A3",
        downscale_res=(512, 288), max_process_res=None,
        beta=1.0, kappa=1.25, omega=0.85, enable_sky_model=True, preserve_resolution=False
    )
    p_a4 = LearnedResidualDehazingPipeline(
        checkpoint_path="checkpoints/delta_t_tiny_unet_best.pt", ablation_mode="A4",
        downscale_res=(512, 288), max_process_res=None,
        beta=1.0, kappa=1.25, omega=0.85, enable_sky_model=True, preserve_resolution=False
    )
    ldnet = LightDehazeNetWrapper()

    methods = [
        ("DCP (He et al. 2009)", lambda f: alg_dcp.process_frame(f), "Heuristic Prior"),
        ("CAP (Zhu et al. 2015)", lambda f: alg_cap.process_frame(f), "Depth Prior"),
        ("CLAHE (Local HistEq)", lambda f: alg_clahe.process_frame(f), "Contrast Stretch"),
        ("Original CEEF (Kaplan 2023)", lambda f: alg_ceef_orig.process_frame(f), "Multi-Scale Fusion"),
        ("Modified CEEF (Fast Approx)", lambda f: alg_ceef_mod.process_frame(f), "Single-Scale Fusion"),
        ("Light-DehazeNet (Li et al.)", lambda f: ldnet.process_frame(f)[0], "Lightweight CNN"),
        ("A1: Physics Baseline", lambda f: p_a1.process_frame(f)[0], "Closed-Form Physics"),
        ("A2: Physics + Delta-t (No Temp)", lambda f: p_a2.process_frame(f)[0], "Learned Delta-t"),
        ("A3: Physics + Delta-t (Temp Loss)", lambda f: p_a3.process_frame(f)[0], "Learned Delta-t + L_temp"),
        ("A4: Proposed Full Method", lambda f: p_a4.process_frame(f)[0], "Physics + Delta-t + Temp Coherence"),
    ]

    all_results = {}

    for name, fn, paradigm in methods:
        print(f"\nProcessing with {name}...")
        dehazed_frames = []
        latencies = []
        
        # Warm-up run
        try:
            _ = fn(eval_frames[0])
        except Exception as e:
            print(f"Error in warmup for {name}: {e}")

        # Benchmark run
        for frame in eval_frames:
            t0 = time.perf_counter()
            out = fn(frame)
            lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)
            dehazed_frames.append(out)

        # Compute Metrics
        ent_vals = [metrics.entropy(df) for df in dehazed_frames]
        col_vals = [metrics.colorfulness(df) for df in dehazed_frames]
        fog_rem_vals = [metrics.fog_reduction_percent_PROXY(orig, df) for orig, df in zip(eval_frames, dehazed_frames)]
        black_vals = [metrics.black_pixel_rate(df) for df in dehazed_frames]
        white_vals = [metrics.white_clipping_rate(df) for df in dehazed_frames]
        
        # Temporal flicker index between consecutive frames
        flicker_vals = []
        for i in range(1, len(dehazed_frames)):
            flicker_vals.append(metrics.temporal_flicker_index(dehazed_frames[i], dehazed_frames[i-1]))

        # NIQE samples (every 5 frames)
        niqe_vals = []
        for i in range(0, len(dehazed_frames), 5):
            try:
                niqe_vals.append(metrics.niqe(dehazed_frames[i]))
            except Exception:
                pass

        avg_lat = float(np.mean(latencies))
        avg_fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

        res_dict = {
            "name": name,
            "paradigm": paradigm,
            "fps": round(avg_fps, 1),
            "latency_ms": round(avg_lat, 1),
            "fog_removal_fdi_pct": round(float(np.mean(fog_rem_vals)), 1),
            "entropy": round(float(np.mean(ent_vals)), 2),
            "colorfulness": round(float(np.mean(col_vals)), 1),
            "niqe": round(float(np.mean(niqe_vals)), 2) if niqe_vals else None,
            "flicker": round(float(np.mean(flicker_vals)), 2) if flicker_vals else None,
            "black_crush_pct": round(float(np.mean(black_vals)), 3),
            "white_clip_pct": round(float(np.mean(white_vals)), 3),
        }
        all_results[name] = res_dict
        print(f" -> FPS: {res_dict['fps']} | FDI FogRem: {res_dict['fog_removal_fdi_pct']}% | Flicker: {res_dict['flicker']} | NIQE: {res_dict['niqe']} | BlackCrush: {res_dict['black_crush_pct']}%")

    # Save results JSON
    os.makedirs("scratch", exist_ok=True)
    with open("scratch/foggy_video_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nBenchmark completed successfully! Saved to scratch/foggy_video_benchmark_results.json")

if __name__ == "__main__":
    run_comparison()
