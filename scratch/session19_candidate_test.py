"""
scratch/session19_candidate_test.py — Phase 10 & 11: Candidate Corrections & Real-World Safety Audit
====================================================================================================
Tests:
  - Baseline Proposed Pipeline (Frozen)
  - Candidate 1: Geometric Sky Gating (vertical decay: sky floor only applies to top half of frame)
  - Candidate 2: Contrast-Gated Bright Floor (prevents uniform fog over non-sky from forcing t >= 0.57)
  - Candidate 3: Analytical Dynamic Range Attenuation Calibration
Evaluates on BOTH Synthetic GT and 7 Real-World Videos.
"""

import os
import sys
import imageio
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
import metrics
from pipeline import RealTimeDehazingPipeline, _resize_np

SYNTHETIC_VIDEOS = [
    {
        "name": "person-bicycle_uniform_haze.mp4",
        "gt": "person-bicycle_clean.mp4",
        "t_true_scalar": float(np.exp(-1.5 * 0.8)) # 0.3012
    },
    {
        "name": "car-detection_patchy_haze.mp4",
        "gt": "car-detection_clean.mp4",
        "t_true_scalar": None
    }
]

REAL_VIDEOS = [
    "foggy_video1.mp4", # Dense fog
    "foggy_video6.mp4", # Dense fog
    "foggy_video7.mp4", # General outdoor
    "foggy_video2.mp4", # Dynamic urban
    "foggy_video10.mp4",# Outdoor
    "foggy_video3.mp4", # Night driving
    "foggy_video4.mp4", # Night traffic
]

def run_pipeline_variant(frame, pipe, variant="baseline", prev_t=None, prev_gray=None):
    """
    Runs pipeline with controlled offline candidate variants WITHOUT touching novel_method/pipeline.py.
    """
    orig_h, orig_w = frame.shape[:2]
    max_w, max_h = pipe.max_process_res
    s = min(max_w / float(orig_w), max_h / float(orig_h))
    cur_w, cur_h = int(orig_w * s), int(orig_h * s)
    proc_frame = _resize_np(frame, (cur_w, cur_h))
    proc_float = proc_frame.astype(np.float32) / 255.0

    half_img = proc_float[::2, ::2]
    gh, gw = half_img.shape[:2]
    gray_h = 0.299 * half_img[:, :, 0] + 0.587 * half_img[:, :, 1] + 0.114 * half_img[:, :, 2]

    v = np.max(half_img, axis=2)
    dc_img = np.min(half_img, axis=2)
    s_map = (v - dc_img) / (v + 1e-6)
    depth = ndimage.uniform_filter(0.121779 + 0.959710 * v - 0.780245 * s_map, size=3)

    A_raw, candidate_mask = pipe.estimate_atmospheric_light(
        half_img, depth, v=v, s=s_map, dc=dc_img, gray_h=gray_h
    )

    norm = half_img / (A_raw[None, None, :] + 1e-6)
    dc = np.min(norm, axis=2)
    dc_min = ndimage.minimum_filter(dc, size=5)
    h_local = ndimage.uniform_filter(dc, size=3)

    omega_map = np.clip(pipe.omega + 0.08 * h_local, pipe.omega, 0.96)
    t_dcp = 1.0 - omega_map * dc_min
    t_cap = np.exp(-pipe.beta * np.clip(depth, 0.0, 2.5))

    cfg = pipe.fusion_cfg
    var_h = ndimage.uniform_filter(gray_h ** 2, size=3) - ndimage.uniform_filter(gray_h, size=3) ** 2
    var_h = np.maximum(var_h, 0.0)

    b_factor = np.clip((gray_h - cfg.get("sky_lum_thresh", 0.55)) / cfg.get("sky_lum_range", 0.25), 0.0, 1.0)
    s_factor = np.clip(1.0 - var_h / cfg.get("sky_var_limit", 0.0005), 0.0, 1.0)
    bright_term = np.clip((gray_h - cfg.get("bright_lum_thresh", 0.75)) / cfg.get("bright_lum_range", 0.15), 0.0, 1.0)
    m_sky = np.clip(np.maximum(b_factor * s_factor, bright_term), 0.0, 1.0)

    text_conf = np.clip(var_h / cfg.get("texture_var_scale", 0.0008), 0.0, 1.0)
    dark_penalty = np.clip((cfg.get("dark_lum_thresh", 0.32) - gray_h) / cfg.get("dark_lum_range", 0.20), 0.0, 1.0)

    haze_lum_gate = (
        np.clip((gray_h - cfg.get("haze_lum_low_thresh", 0.20)) / cfg.get("haze_lum_low_range", 0.12), 0.0, 1.0)
        * np.clip((cfg.get("haze_lum_high_thresh", 0.82) - gray_h) / cfg.get("haze_lum_high_range", 0.15), 0.0, 1.0)
    )
    haze_dc_conf = np.clip((dc - cfg.get("haze_dc_thresh", 0.15)) / cfg.get("haze_dc_range", 0.35), 0.0, 1.0)
    h_conf = haze_lum_gate * haze_dc_conf

    w_raw = (
        cfg.get("w_base", 0.15)
        + cfg.get("w_haze_scale", 0.60) * h_conf
        - cfg.get("w_sky_penalty", 0.85) * m_sky
        - cfg.get("w_dark_penalty", 0.55) * dark_penalty
        + cfg.get("w_text_base", 0.60) * cfg.get("w_text_scale", 0.40) * text_conf
    )
    w_dcp = np.clip(w_raw, cfg.get("w_min", 0.05), cfg.get("w_max", 0.75))
    w_dcp = ndimage.gaussian_filter(w_dcp, sigma=cfg.get("w_smooth_sigma", 1.5))

    t_fused = w_dcp * t_dcp + (1.0 - w_dcp) * t_cap

    # -------------------------------------------------------------------------
    # VARIANT LOGIC IN ADAPTIVE T_MIN
    # -------------------------------------------------------------------------
    t_base = pipe.t_min
    w_bright = np.clip((gray_h - 0.65) / 0.25, 0.0, 1.0)
    w_dark = np.clip((0.35 - gray_h) / 0.25, 0.0, 1.0)
    w_smooth_dark = np.clip(1.0 - var_h / 0.001, 0.0, 1.0)
    floor_dark = 0.35 * w_dark * (0.6 + 0.4 * w_smooth_dark)
    haze_ind = np.clip((h_local - 0.20) / 0.30, 0.0, 1.0) * (1.0 - w_dark) * (1.0 - m_sky)
    relief = 0.06 * haze_ind

    if variant == "baseline":
        floor_bright = 0.50 * np.maximum(m_sky, w_bright * 0.6)
        t_min_map = np.clip(t_base + floor_dark + floor_bright - relief, t_base, 0.85)
        t_tmin = np.clip(t_fused, t_min_map, 1.0)

    elif variant == "cand1_geo_sky":
        # Candidate 1: Geometric Vertical Sky Decay
        # Y coordinate normalized [0, 1] from top to bottom
        y_grid = np.linspace(0.0, 1.0, gh)[:, None]
        y_decay = np.clip(1.0 - (y_grid - 0.20) / 0.35, 0.0, 1.0) # 1.0 in top 20%, decays to 0 by 55%
        floor_bright = 0.50 * np.maximum(m_sky * y_decay, w_bright * 0.6 * y_decay)
        t_min_map = np.clip(t_base + floor_dark + floor_bright - relief, t_base, 0.85)
        t_tmin = np.clip(t_fused, t_min_map, 1.0)

    elif variant == "cand2_contrast_gate":
        # Candidate 2: Contrast-gated bright floor
        # Do not force high floor if scene has dark channel contrast indicating ground fog
        dc_evidence = np.clip((dc_img - 0.15) / 0.40, 0.0, 1.0)
        floor_bright = 0.50 * np.maximum(m_sky, w_bright * 0.6) * (1.0 - 0.60 * dc_evidence)
        t_min_map = np.clip(t_base + floor_dark + floor_bright - relief, t_base, 0.85)
        t_tmin = np.clip(t_fused, t_min_map, 1.0)

    elif variant == "cand3_dynamic_scale":
        # Candidate 3: Analytical monotonic attenuation mapping
        floor_bright = 0.50 * np.maximum(m_sky, w_bright * 0.6)
        t_min_map = np.clip(t_base + floor_dark + floor_bright - relief, t_base, 0.85)
        t_tmin = np.clip(t_fused, t_min_map, 1.0)
        # Power-law compression towards lower transmission in heavy fog
        t_tmin = np.clip(t_tmin ** 1.35, 0.05, 1.0)

    # Guided smoothing
    mean_I = ndimage.uniform_filter(gray_h, size=5)
    mean_p = ndimage.uniform_filter(t_tmin, size=5)
    mean_Ip = ndimage.uniform_filter(gray_h * t_tmin, size=5)
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = ndimage.uniform_filter(gray_h * gray_h, size=5) - mean_I * mean_I
    a = cov_Ip / (var_I + 1e-3)
    b = mean_p - a * mean_I
    t_smooth = ndimage.uniform_filter(a, size=5) * gray_h + ndimage.uniform_filter(b, size=5)
    t_smooth = np.clip(t_smooth, t_base, 1.0)

    # Upsample t to proc resolution
    t_full = np.array(Image.fromarray(t_smooth).resize((cur_w, cur_h), Image.BILINEAR))

    # Physical inversion J = (I - A)/t + A
    A_3d = A_raw[None, None, :]
    t_3d = t_full[:, :, None]
    J_f = (proc_float - A_3d) / np.maximum(t_3d, 0.05) + A_3d
    J_u8 = np.clip(np.round(J_f * 255.0), 0, 255).astype(np.uint8)

    return J_u8, t_full

def evaluate_all():
    print("=" * 90)
    print("SESSION 19 — PHASE 10 & 11: CANDIDATE CORRECTIONS BENCHMARK")
    print("=" * 90)

    variants = ["baseline", "cand1_geo_sky", "cand2_contrast_gate", "cand3_dynamic_scale"]
    pipe = RealTimeDehazingPipeline()

    # 1. EVALUATE ON SYNTHETIC BENCHMARKS
    print("\n--- SYNTHETIC GROUND TRUTH BENCHMARK ---")
    for sv in SYNTHETIC_VIDEOS:
        rh = imageio.get_reader(os.path.join("test_videos", sv["name"]))
        rc = imageio.get_reader(os.path.join("test_videos", sv["gt"]))
        frames_h = [rh.get_data(i) for i in range(15)]
        frames_c = [rc.get_data(i) for i in range(15)]
        rh.close()
        rc.close()

        print(f"\nDataset: {sv['name']}")
        for v in variants:
            psnrs = []
            ssims = []
            t_means = []
            for i in range(5, 15):
                fh = frames_h[i]
                fc = frames_c[i]
                out, t_map = run_pipeline_variant(fh, pipe, variant=v)
                fc_res = np.array(Image.fromarray(fc).resize((out.shape[1], out.shape[0]), Image.BILINEAR))
                psnrs.append(sk_psnr(fc_res, out, data_range=255))
                ssims.append(sk_ssim(fc_res, out, channel_axis=-1, data_range=255))
                t_means.append(np.mean(t_map))
            print(f"  [{v:20s}] PSNR: {np.mean(psnrs):5.2f} dB | SSIM: {np.mean(ssims):.4f} | Mean t: {np.mean(t_means):.4f}")

    # 2. EVALUATE ON REAL-WORLD BENCHMARKS (SAFETY TEST)
    print("\n--- REAL-WORLD NATURAL FOG & NIGHT SAFETY AUDIT ---")
    for rv in REAL_VIDEOS:
        rpath = os.path.join("test_videos", rv)
        if not os.path.exists(rpath):
            continue
        reader = imageio.get_reader(rpath)
        frames = [reader.get_data(i) for i in range(15)]
        reader.close()

        print(f"\nReal Video: {rv}")
        for v in variants:
            niqes = []
            flickers = []
            black_clips = []
            prev_lum = None
            for i in range(5, 15):
                out, _ = run_pipeline_variant(frames[i], pipe, variant=v)
                lum = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
                niqes.append(metrics.niqe(out))
                black_clips.append(np.count_nonzero(lum < 1.0) / lum.size * 100.0)
                if prev_lum is not None:
                    flickers.append(np.mean(np.abs(lum - prev_lum)))
                prev_lum = lum
            print(f"  [{v:20s}] NIQE: {np.mean(niqes):4.2f} | Flicker: {np.mean(flickers):4.2f} | Black Clip: {np.mean(black_clips):5.2f}%")

if __name__ == "__main__":
    evaluate_all()
