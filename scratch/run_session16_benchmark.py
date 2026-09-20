"""
scratch/run_session16_benchmark.py — Comprehensive Session 16 Benchmark
=======================================================================
Executes:
  1. Classical Baselines vs. Proposed Method across all 6 categories (13 videos):
     - Baseline 1: DCP (algorithms/dcp.py)
     - Baseline 2: CAP (algorithms/cap.py)
     - Baseline 3: CLAHE (algorithms/clahe_baseline.py)
     - Baseline 4: Corrected Gamma (algorithms/gamma_baseline.py)
     - Baseline 5: Original CEEF (algorithms/ceef_original.py)
     - Baseline 6: Modified CEEF (algorithms/ceef.py)
     - Proposed Method (Frozen Daytime Pipeline + Session 12 Model A Nighttime)
  2. Component Ablation (7 configurations) on representative GT and natural sequences:
     - Config A: Full Proposed Method
     - Config B: Full - adaptive spatial t_min(x)
     - Config C: Full - DCP/CAP spatial fusion
     - Config D: Full - robust atmospheric-light estimation
     - Config E: Full - chromaticity-preserving atmospheric-light bound
     - Config F: Full - guided transmission refinement
     - Config G: Full - motion-aware temporal consistency
  3. Ground Truth Evaluation:
     - car-detection_patchy_haze.mp4 vs car-detection_clean.mp4 (PSNR, SSIM)
     - person-bicycle_uniform_haze.mp4 vs person-bicycle_clean.mp4 (PSNR, SSIM)
"""

import os
import sys
import time
import json
import imageio
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "algorithms")))

# Import Baselines
import dcp
import cap
import clahe_baseline
import gamma_baseline
import ceef
import ceef_original

# Import Proposed Pipeline
from pipeline import RealTimeDehazingPipeline, _resize_np
from nighttime_experimental import ExperimentalNighttimePipeline
import metrics

# Benchmark Datasets across 6 categories
BENCHMARK_VIDEOS = [
    # Category A: Dense Natural Fog
    {"cat": "A", "cat_name": "Dense Natural Fog", "name": "foggy_video1.mp4", "gt": None},
    {"cat": "A", "cat_name": "Dense Natural Fog", "name": "foggy_video6.mp4", "gt": None},
    # Category B: Night / Low Light
    {"cat": "B", "cat_name": "Night / Low Light", "name": "foggy_video3.mp4", "gt": None},
    {"cat": "B", "cat_name": "Night / Low Light", "name": "foggy_video4.mp4", "gt": None},
    # Category C: Dynamic Urban
    {"cat": "C", "cat_name": "Dynamic Urban", "name": "foggy_video2.mp4", "gt": None},
    {"cat": "C", "cat_name": "Dynamic Urban", "name": "car-detection_patchy_haze.mp4", "gt": "car-detection_clean.mp4"},
    {"cat": "C", "cat_name": "Dynamic Urban", "name": "person-bicycle_uniform_haze.mp4", "gt": "person-bicycle_clean.mp4"},
    # Category D: Extreme / Nonuniform Lighting
    {"cat": "D", "cat_name": "Extreme / Nonuniform Lighting", "name": "foggy_video5.mp4", "gt": None},
    {"cat": "D", "cat_name": "Extreme / Nonuniform Lighting", "name": "foggy_video14.mp4", "gt": None},
    # Category E: General Outdoor
    {"cat": "E", "cat_name": "General Outdoor", "name": "foggy_video7.mp4", "gt": None},
    {"cat": "E", "cat_name": "General Outdoor", "name": "foggy_video10.mp4", "gt": None},
    # Category F: Indoor / Artificial
    {"cat": "F", "cat_name": "Indoor / Artificial", "name": "foggy_video9.mp4", "gt": None},
    {"cat": "F", "cat_name": "Indoor / Artificial", "name": "foggy_video11.mp4", "gt": None},
]

METHODS = [
    {"key": "DCP", "name": "DCP", "type": "Faithful reimplementation", "temporal": False},
    {"key": "CAP", "name": "CAP", "type": "Faithful reimplementation", "temporal": False},
    {"key": "CLAHE", "name": "CLAHE", "type": "Standard LAB baseline", "temporal": False},
    {"key": "Gamma", "name": "Corrected Gamma", "type": "Corrected power-law baseline", "temporal": False},
    {"key": "OrigCEEF", "name": "Original CEEF", "type": "Faithful reimplementation", "temporal": False},
    {"key": "ModCEEF", "name": "Modified CEEF", "type": "Modified/approximate implementation", "temporal": False},
    {"key": "Proposed", "name": "Proposed Method", "type": "Current frozen proposed pipeline", "temporal": True},
]

def compute_all_metrics(in_f: np.ndarray, out_f: np.ndarray, gt_f: np.ndarray = None, prev_out_lum: np.ndarray = None):
    """Computes reference-free, photometric, structural, temporal, and GT metrics."""
    out_h, out_w = out_f.shape[:2]
    if in_f.shape[:2] != (out_h, out_w):
        in_res = np.array(Image.fromarray(in_f).resize((out_w, out_h), Image.BILINEAR))
    else:
        in_res = in_f

    # Luminance
    lum_out = 0.299 * out_f[:, :, 0] + 0.587 * out_f[:, :, 1] + 0.114 * out_f[:, :, 2]
    lum_in = 0.299 * in_res[:, :, 0] + 0.587 * in_res[:, :, 1] + 0.114 * in_res[:, :, 2]

    # Range / Photometric
    black_clip = float(np.count_nonzero(lum_out < 1.0) / lum_out.size * 100.0)
    white_clip = float(np.count_nonzero(lum_out > 254.0) / lum_out.size * 100.0)
    mean_lum = float(np.mean(lum_out))
    p5_lum = float(np.percentile(lum_out, 5))
    p50_lum = float(np.percentile(lum_out, 50))
    p95_lum = float(np.percentile(lum_out, 95))
    
    # Entropy & Colorfulness
    entropy_val = float(metrics.entropy(out_f))
    colorfulness_val = float(metrics.colorfulness(out_f))
    b_minus_r = float(np.mean(out_f[:, :, 2].astype(float) - out_f[:, :, 0].astype(float)))

    # NIQE & FDI Proxy
    try:
        niqe_val = float(metrics.niqe(out_f))
    except Exception:
        niqe_val = 0.0
    fdi_val = float(metrics.fog_density_index(out_f))

    # Structural
    gy, gx = np.gradient(lum_out)
    edge_energy = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))
    grad_energy = float(np.mean(gx ** 2 + gy ** 2))
    high_grad_mask = np.sqrt(gx ** 2 + gy ** 2) > 25.0
    halo_ratio = float(np.count_nonzero(high_grad_mask) / lum_out.size * 100.0)

    # Dark-region preservation (input < 38/255)
    dark_mask = (lum_in < 38.0)
    dark_lum = float(np.mean(lum_out[dark_mask])) if np.any(dark_mask) else mean_lum

    # Temporal flicker
    flicker = 0.0
    if prev_out_lum is not None and prev_out_lum.shape == lum_out.shape:
        flicker = float(np.mean(np.abs(lum_out - prev_out_lum)))

    # Ground Truth metrics
    if gt_f is not None:
        if gt_f.shape[:2] != (out_h, out_w):
            gt_res = np.array(Image.fromarray(gt_f).resize((out_w, out_h), Image.BILINEAR))
        else:
            gt_res = gt_f
        psnr_val = float(sk_psnr(gt_res, out_f, data_range=255))
        ssim_val = float(sk_ssim(gt_res, out_f, channel_axis=-1, data_range=255))
    else:
        psnr_val = "Requires Ground Truth"
        ssim_val = "Requires Ground Truth"

    return {
        "black_clip": black_clip,
        "white_clip": white_clip,
        "mean_lum": mean_lum,
        "p5_lum": p5_lum,
        "p50_lum": p50_lum,
        "p95_lum": p95_lum,
        "entropy": entropy_val,
        "colorfulness": colorfulness_val,
        "b_minus_r": b_minus_r,
        "niqe": niqe_val,
        "fdi_proxy": fdi_val,
        "fade": "FADE: unavailable",
        "edge_energy": edge_energy,
        "grad_energy": grad_energy,
        "halo_ratio": halo_ratio,
        "dark_preservation": dark_lum,
        "flicker": flicker,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "lpips": "Requires Ground Truth" if gt_f is None else "Requires learned model",
        "lum_out": lum_out,
    }

def run_method_on_frame(method_key, frame, proposed_pipe, target_res=(426, 239)):
    """Runs a single method on a frame with normalized resolution."""
    raw_h, raw_w = frame.shape[:2]
    # For fair comparison across all methods, resize input to target resolution
    tw, th = target_res
    if (raw_w, raw_h) != (tw, th):
        s = min(tw / float(raw_w), th / float(raw_h))
        cur_w, cur_h = int(raw_w * s), int(raw_h * s)
        in_frame = _resize_np(frame, (cur_w, cur_h))
    else:
        in_frame = frame

    t0 = time.perf_counter()
    if method_key == "DCP":
        out = dcp.process_frame(in_frame)
    elif method_key == "CAP":
        out = cap.process_frame(in_frame)
    elif method_key == "CLAHE":
        out = clahe_baseline.process_frame(in_frame)
    elif method_key == "Gamma":
        out = gamma_baseline.process_frame(in_frame, gamma=2.0)
    elif method_key == "OrigCEEF":
        out = ceef_original.process_frame(in_frame)
    elif method_key == "ModCEEF":
        out = ceef.process_frame(in_frame)
    elif method_key == "Proposed":
        out, _ = proposed_pipe.process_frame(frame)  # Proposed handles internal resolution scaling
        # Ensure output matches in_frame shape
        if out.shape[:2] != in_frame.shape[:2]:
            out = _resize_np(out, (in_frame.shape[1], in_frame.shape[0]))
    else:
        raise ValueError(f"Unknown method key: {method_key}")
    dt = time.perf_counter() - t0
    return in_frame, out, dt

def run_baseline_benchmark():
    """Section 1-7: Full multi-method benchmark across 13 videos."""
    n_warmup = 8
    n_measure = 15
    total_frames = n_warmup + n_measure

    results = []

    print("=" * 110)
    print("SESSION 16 — CLASSICAL BASELINE COMPARISON BENCHMARK (13 VIDEOS, 6 CATEGORIES)")
    print("=" * 110)

    for vid in BENCHMARK_VIDEOS:
        vpath = os.path.join("test_videos", vid["name"])
        if not os.path.exists(vpath):
            print(f"Warning: {vpath} not found, skipping.")
            continue

        rh = imageio.get_reader(vpath)
        frames_haze = []
        for idx, f in enumerate(rh):
            if idx >= total_frames:
                break
            frames_haze.append(f)
        rh.close()

        frames_gt = None
        if vid["gt"]:
            gtpath = os.path.join("test_videos", vid["gt"])
            rg = imageio.get_reader(gtpath)
            frames_gt = []
            for idx, f in enumerate(rg):
                if idx >= total_frames:
                    break
                frames_gt.append(f)
            rg.close()

        print(f"\nEvaluating Category [{vid['cat']}] {vid['cat_name']} -> {vid['name']} ({len(frames_haze)-n_warmup} measured frames):")

        for m_info in METHODS:
            m_key = m_info["key"]
            m_name = m_info["name"]

            # Initialize proposed pipeline fresh per video
            if m_key == "Proposed":
                pipe = ExperimentalNighttimePipeline(enable_nighttime_branch=True, night_mode="prototype")
            else:
                pipe = None

            # Warm-up
            for w in range(n_warmup):
                f_w = frames_haze[w]
                run_method_on_frame(m_key, f_w, pipe)

            # Measurement loop
            latencies = []
            frame_records = []
            prev_lum = None

            for i in range(n_warmup, len(frames_haze)):
                f_haze = frames_haze[i]
                f_gt = frames_gt[i] if frames_gt else None

                in_f, out_f, dt = run_method_on_frame(m_key, f_haze, pipe)
                latencies.append(dt)

                met = compute_all_metrics(in_f, out_f, f_gt, prev_lum)
                prev_lum = met["lum_out"].copy()
                frame_records.append(met)

            # Aggregate stats
            mean_ms = float(np.mean(latencies) * 1000.0)
            median_ms = float(np.median(latencies) * 1000.0)
            fps = float(1.0 / np.mean(latencies))
            std_ms = float(np.std(latencies) * 1000.0)

            mean_niqe = float(np.mean([r["niqe"] for r in frame_records]))
            mean_fdi = float(np.mean([r["fdi_proxy"] for r in frame_records]))
            mean_black = float(np.mean([r["black_clip"] for r in frame_records]))
            mean_white = float(np.mean([r["white_clip"] for r in frame_records]))
            mean_lum = float(np.mean([r["mean_lum"] for r in frame_records]))
            p5_lum = float(np.mean([r["p5_lum"] for r in frame_records]))
            p50_lum = float(np.mean([r["p50_lum"] for r in frame_records]))
            p95_lum = float(np.mean([r["p95_lum"] for r in frame_records]))
            entropy_val = float(np.mean([r["entropy"] for r in frame_records]))
            colorfulness_val = float(np.mean([r["colorfulness"] for r in frame_records]))
            b_r = float(np.mean([r["b_minus_r"] for r in frame_records]))
            edge_eng = float(np.mean([r["edge_energy"] for r in frame_records]))
            flicker = float(np.mean([r["flicker"] for r in frame_records[1:]])) if len(frame_records) > 1 else 0.0
            halo = float(np.mean([r["halo_ratio"] for r in frame_records]))

            if vid["gt"]:
                mean_psnr = float(np.mean([r["psnr"] for r in frame_records]))
                mean_ssim = float(np.mean([r["ssim"] for r in frame_records]))
            else:
                mean_psnr = "Requires Ground Truth"
                mean_ssim = "Requires Ground Truth"

            rec = {
                "category": vid["cat"],
                "category_name": vid["cat_name"],
                "video_name": vid["name"],
                "has_gt": bool(vid["gt"]),
                "method_key": m_key,
                "method_name": m_name,
                "method_type": m_info["type"],
                "temporal_model": "Motion-aware temporal consistency" if m_info["temporal"] else "Frame-independent baseline",
                "mean_ms": mean_ms,
                "median_ms": median_ms,
                "std_ms": std_ms,
                "fps": fps,
                "niqe": mean_niqe,
                "fdi_proxy": mean_fdi,
                "fade": "FADE: unavailable",
                "black_clip": mean_black,
                "white_clip": mean_white,
                "mean_lum": mean_lum,
                "p5_lum": p5_lum,
                "p50_lum": p50_lum,
                "p95_lum": p95_lum,
                "entropy": entropy_val,
                "colorfulness": colorfulness_val,
                "b_minus_r": b_r,
                "edge_energy": edge_eng,
                "flicker": flicker,
                "halo_ratio": halo,
                "psnr": mean_psnr,
                "ssim": mean_ssim,
                "lpips": "Requires Ground Truth" if not vid["gt"] else "Requires learned model",
            }
            results.append(rec)
            psnr_str = f"{mean_psnr:5.2f} dB" if isinstance(mean_psnr, float) else "No GT"
            ssim_str = f"{mean_ssim:.4f}" if isinstance(mean_ssim, float) else "No GT"
            print(f"  [{m_name:16s}] FPS: {fps:5.1f} | NIQE: {mean_niqe:4.2f} | FDI: {mean_fdi:.3f} | "
                  f"Black: {mean_black:5.2f}% | Flick: {flicker:4.2f} | PSNR: {psnr_str} | SSIM: {ssim_str}")

    return results

def run_ablation_benchmark():
    """Section 9-10: Component-by-component ablation study on proposed method."""
    print("\n" + "=" * 110)
    print("SESSION 16 — PROPOSED METHOD COMPONENT ABLATION (7 CONFIGURATIONS)")
    print("=" * 110)

    # Test on a representative GT video and representative natural videos
    ablation_videos = [
        {"cat": "C", "name": "car-detection_patchy_haze.mp4", "gt": "car-detection_clean.mp4"},
        {"cat": "A", "name": "foggy_video1.mp4", "gt": None},
        {"cat": "B", "name": "foggy_video3.mp4", "gt": None},
    ]

    ABLATION_CONFIGS = [
        {"key": "A_Full", "label": "A. Full Proposed Method"},
        {"key": "B_NoTmin", "label": "B. Full - Adaptive t_min(x)"},
        {"key": "C_NoFusion", "label": "C. Full - DCP/CAP Spatial Fusion"},
        {"key": "D_NoAirlight", "label": "D. Full - Robust Airlight Estimator"},
        {"key": "E_NoABound", "label": "E. Full - Chromaticity Airlight Bound"},
        {"key": "F_NoRefine", "label": "F. Full - Guided Transmission Refinement"},
        {"key": "G_NoTemporal", "label": "G. Full - Motion-Aware Temporal"},
    ]

    n_warmup = 8
    n_measure = 15
    total_frames = n_warmup + n_measure

    ablation_results = []

    for vid in ablation_videos:
        vpath = os.path.join("test_videos", vid["name"])
        rh = imageio.get_reader(vpath)
        frames_haze = [rh.get_data(i) for i in range(total_frames)]
        rh.close()

        frames_gt = None
        if vid["gt"]:
            rg = imageio.get_reader(os.path.join("test_videos", vid["gt"]))
            frames_gt = [rg.get_data(i) for i in range(total_frames)]
            rg.close()

        print(f"\nAblation on {vid['name']} ({n_measure} measured frames):")

        for cfg in ABLATION_CONFIGS:
            cfg_key = cfg["key"]
            cfg_label = cfg["label"]

            # Configure pipeline specifically for this ablation
            pipe = ExperimentalNighttimePipeline(enable_nighttime_branch=True, night_mode="prototype")

            if cfg_key == "B_NoTmin":
                # Disable adaptive t_min, use static fixed floor 0.10
                pipe.t_min = 0.10
                pipe.t_min_adaptive = False
            elif cfg_key == "C_NoFusion":
                # Pure CAP transmission alone (weight for DCP = 0.0)
                pipe.fusion_cfg["w_min"] = 0.0
                pipe.fusion_cfg["w_max"] = 0.0
                pipe.fusion_cfg["w_base"] = 0.0
            elif cfg_key == "D_NoAirlight":
                # Naive top brightest in dark channel without robust scoring
                pipe.airlight_cfg["top_candidate_ratio"] = 0.001
                pipe.airlight_cfg["min_candidates"] = 1
                pipe.airlight_cfg["blue_sky_lum_thresh"] = 0.0
            elif cfg_key == "E_NoABound":
                # Remove chromaticity-preserving lower bound
                pipe.airlight_cfg["a_min"] = 0.20
                pipe.airlight_cfg["a_min_hard"] = 0.20
            elif cfg_key == "F_NoRefine":
                # Disable guided filter refinement (set eps huge or skip)
                pipe.refinement_cfg["eps"] = 1e9  # Degenerates guided filter to uniform mean
            elif cfg_key == "G_NoTemporal":
                # Disable temporal consistency (frame-independent processing)
                pipe.temporal_cfg["alpha_static"] = 1.0
                pipe.temporal_cfg["alpha_motion"] = 1.0
                pipe.temporal_cfg["alpha_airlight"] = 1.0

            # Warm-up
            for w in range(n_warmup):
                pipe.process_frame(frames_haze[w])

            latencies = []
            records = []
            prev_lum = None
            t_variations = []
            A_variations = []
            prev_t = None
            prev_A = None

            for i in range(n_warmup, total_frames):
                f_haze = frames_haze[i]
                f_gt = frames_gt[i] if frames_gt else None

                t0 = time.perf_counter()
                out, _, diag = pipe.process_frame(f_haze, return_diagnostics=True)
                latencies.append(time.perf_counter() - t0)

                met = compute_all_metrics(f_haze, out, f_gt, prev_lum)
                prev_lum = met["lum_out"].copy()
                records.append(met)

                cur_t = getattr(pipe, "_cached_t", None)
                cur_A = getattr(pipe, "last_A", None)
                if prev_t is not None and cur_t is not None and prev_t.shape == cur_t.shape:
                    t_variations.append(float(np.mean(np.abs(cur_t - prev_t))))
                if prev_A is not None and cur_A is not None and prev_A.shape == cur_A.shape:
                    A_variations.append(float(np.mean(np.abs(cur_A - prev_A))))
                if cur_t is not None:
                    prev_t = cur_t.copy()
                if cur_A is not None:
                    prev_A = cur_A.copy()

            mean_fps = float(1.0 / np.mean(latencies))
            mean_niqe = float(np.mean([r["niqe"] for r in records]))
            mean_fdi = float(np.mean([r["fdi_proxy"] for r in records]))
            mean_black = float(np.mean([r["black_clip"] for r in records]))
            mean_white = float(np.mean([r["white_clip"] for r in records]))
            b_r = float(np.mean([r["b_minus_r"] for r in records]))
            edge_eng = float(np.mean([r["edge_energy"] for r in records]))
            flicker = float(np.mean([r["flicker"] for r in records[1:]])) if len(records) > 1 else 0.0
            delta_t = float(np.mean(t_variations)) if t_variations else 0.0
            delta_A = float(np.mean(A_variations)) if A_variations else 0.0

            if vid["gt"]:
                mean_psnr = float(np.mean([r["psnr"] for r in records]))
                mean_ssim = float(np.mean([r["ssim"] for r in records]))
            else:
                mean_psnr = "Requires Ground Truth"
                mean_ssim = "Requires Ground Truth"

            rec = {
                "video_name": vid["name"],
                "has_gt": bool(vid["gt"]),
                "config_key": cfg_key,
                "config_label": cfg_label,
                "fps": mean_fps,
                "niqe": mean_niqe,
                "fdi_proxy": mean_fdi,
                "black_clip": mean_black,
                "white_clip": mean_white,
                "b_minus_r": b_r,
                "edge_energy": edge_eng,
                "flicker": flicker,
                "delta_t": delta_t,
                "delta_A": delta_A,
                "psnr": mean_psnr,
                "ssim": mean_ssim,
                "lpips": "Requires Ground Truth" if not vid["gt"] else "Requires learned model",
            }
            ablation_results.append(rec)
            psnr_str = f"{mean_psnr:5.2f} dB" if isinstance(mean_psnr, float) else "No GT"
            ssim_str = f"{mean_ssim:.4f}" if isinstance(mean_ssim, float) else "No GT"
            print(f"  [{cfg_label:38s}] FPS: {mean_fps:5.1f} | NIQE: {mean_niqe:4.2f} | Black: {mean_black:5.2f}% | "
                  f"Flick: {flicker:4.2f} | Dt: {delta_t:.4f} | PSNR: {psnr_str} | SSIM: {ssim_str}")

    return ablation_results

if __name__ == "__main__":
    t_start = time.perf_counter()
    baseline_data = run_baseline_benchmark()
    ablation_data = run_ablation_benchmark()

    all_data = {
        "baseline_comparison": baseline_data,
        "ablation_study": ablation_data,
        "benchmark_metadata": {
            "num_videos": len(BENCHMARK_VIDEOS),
            "categories": 6,
            "methods": len(METHODS),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_runtime_sec": round(time.perf_counter() - t_start, 2),
        }
    }

    out_file = os.path.join("scratch", "session16_results.json")
    with open(out_file, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSaved complete Session 16 benchmark results to {out_file} (Total time: {time.perf_counter() - t_start:.1f}s)")
