"""
scratch/session19_stage_decomposition.py — Stage-by-Stage Transmission Error Audit
===================================================================================
Executes Phase 2, Phase 3, Phase 7, Phase 8, and Phase 9:
Decomposes transmission evolution stage-by-stage against the true reference transmission.
"""

import os
import sys
import imageio
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "algorithms")))

import dcp
import cap
from pipeline import RealTimeDehazingPipeline, _resize_np

def run_stage_audit():
    print("=" * 90)
    print("SESSION 19 — PHASE 2 & 3: STAGE-BY-STAGE TRANSMISSION ERROR DECOMPOSITION")
    print("=" * 90)

    test_cases = [
        {
            "id": "person-bicycle",
            "name": "Person Bicycle (Uniform Haze)",
            "hazy_file": "test_videos/person-bicycle_uniform_haze.mp4",
            "clean_file": "test_videos/person-bicycle_clean.mp4",
            "pattern": "uniform",
            "t_const": np.exp(-1.5 * 0.8), # 0.3011942
        },
        {
            "id": "car-detection",
            "name": "Car Detection (Patchy Haze)",
            "hazy_file": "test_videos/car-detection_patchy_haze.mp4",
            "clean_file": "test_videos/car-detection_clean.mp4",
            "pattern": "patchy",
            "t_const": None,
        }
    ]

    for tc in test_cases:
        print(f"\n==================================================================")
        print(f"EVALUATING: {tc['name']}")
        print(f"==================================================================")

        rh = imageio.get_reader(tc["hazy_file"])
        rc = imageio.get_reader(tc["clean_file"])
        frames_h = [rh.get_data(i) for i in range(20)]
        frames_c = [rc.get_data(i) for i in range(20)]
        rh.close()
        rc.close()

        frame_idx = 10
        fh = frames_h[frame_idx]
        fc = frames_c[frame_idx]

        # ---------------------------------------------------------------------
        # True Transmission Benchmark
        # ---------------------------------------------------------------------
        if tc["pattern"] == "uniform":
            t_true = np.ones(fh.shape[:2], dtype=np.float32) * tc["t_const"]
            valid_mask = np.ones(fh.shape[:2], dtype=bool)
        else:
            # For patchy, derive pixel-level reference from I and J using A=255
            A_true = np.array([255.0, 255.0, 255.0], dtype=np.float32)
            denom = fc.astype(np.float32) - A_true
            numer = fh.astype(np.float32) - A_true
            valid_mask = np.all(np.abs(denom) >= 20.0, axis=-1)
            t_channels = numer / (denom + 1e-6)
            t_true = np.median(t_channels, axis=-1)
            t_true = np.clip(t_true, 0.05, 1.0)

        # ---------------------------------------------------------------------
        # Pipeline Internal Stage Capture
        # ---------------------------------------------------------------------
        pipe = RealTimeDehazingPipeline()
        # Warmup temporal state
        for w in range(frame_idx):
            pipe.process_frame(frames_h[w])

        # Match exact pipeline.py resolution scaling
        orig_h, orig_w = fh.shape[:2]
        max_w, max_h = pipe.max_process_res
        s = min(max_w / float(orig_w), max_h / float(orig_h))
        cur_w, cur_h = int(orig_w * s), int(orig_h * s)
        proc_frame = _resize_np(fh, (cur_w, cur_h))

        proc_float = proc_frame.astype(np.float32) / 255.0
        half_img = proc_float[::2, ::2]
        gh, gw = half_img.shape[:2]
        gray_h = 0.299 * half_img[:, :, 0] + 0.587 * half_img[:, :, 1] + 0.114 * half_img[:, :, 2]

        v = np.max(half_img, axis=2)
        dc_img = np.min(half_img, axis=2)
        s = (v - dc_img) / (v + 1e-6)
        depth = ndimage.uniform_filter(0.121779 + 0.959710 * v - 0.780245 * s, size=3)

        A_raw, candidate_mask = pipe.estimate_atmospheric_light(
            half_img, depth, v=v, s=s, dc=dc_img, gray_h=gray_h
        )

        norm = half_img / (A_raw[None, None, :] + 1e-6)
        dc = np.min(norm, axis=2)
        dc_min = ndimage.minimum_filter(dc, size=5)
        h_local = ndimage.uniform_filter(dc, size=3)

        omega_map = np.clip(pipe.omega + 0.08 * h_local, pipe.omega, 0.96)
        t_dcp_raw = 1.0 - omega_map * dc_min
        t_cap_raw = np.exp(-pipe.beta * np.clip(depth, 0.0, 2.5))

        # Spatial fusion
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

        t_fused = w_dcp * t_dcp_raw + (1.0 - w_dcp) * t_cap_raw

        # 4. Adaptive t_min (Session 2 / 10 exact logic)
        w_sky = m_sky
        t_base = pipe.t_min
        w_bright = np.clip((gray_h - 0.65) / 0.25, 0.0, 1.0)
        floor_bright = 0.50 * np.maximum(w_sky, w_bright * 0.6)

        w_dark = np.clip((0.35 - gray_h) / 0.25, 0.0, 1.0)
        w_smooth_dark = np.clip(1.0 - var_h / 0.001, 0.0, 1.0)
        floor_dark = 0.35 * w_dark * (0.6 + 0.4 * w_smooth_dark)

        haze_ind = np.clip((h_local - 0.20) / 0.30, 0.0, 1.0) * (1.0 - w_dark) * (1.0 - w_sky)
        relief = 0.06 * haze_ind

        t_min_map = np.clip(t_base + floor_dark + floor_bright - relief, t_base, 0.85)
        t_tmin = np.clip(t_fused, t_min_map, 1.0)

        # 5. Guided Smoothing (inlined exact logic)
        mean_I = ndimage.uniform_filter(gray_h, size=5)
        mean_p = ndimage.uniform_filter(t_tmin, size=5)
        mean_Ip = ndimage.uniform_filter(gray_h * t_tmin, size=5)
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = ndimage.uniform_filter(gray_h * gray_h, size=5) - mean_I * mean_I
        a = cov_Ip / (var_I + 1e-3)
        b = mean_p - a * mean_I
        t_guided = ndimage.uniform_filter(a, size=5) * gray_h + ndimage.uniform_filter(b, size=5)
        t_guided = np.clip(t_guided, t_min_map, 1.0)

        # 6. Temporal Smoothing
        if pipe._cached_t is not None and pipe._prev_gray is not None:
            motion_mask = np.abs(gray_h - pipe._prev_gray) > 0.05
            alpha_map = np.where(motion_mask, 0.90, 0.15).astype(np.float32)
            t_temporal = alpha_map * t_guided + (1.0 - alpha_map) * pipe._cached_t
        else:
            t_temporal = t_guided

        # Resize all stages back to original resolution for comparison against t_true
        stages = [
            ("Stage A: Raw CAP", t_cap_raw),
            ("Stage B: Raw DCP", t_dcp_raw),
            ("Stage C: Spatial Fusion", t_fused),
            ("Stage D: Adaptive t_min", t_tmin),
            ("Stage E: Guided Refinement", t_guided),
            ("Stage F: Temporal Smoothing", t_temporal),
        ]

        print(f"\nStage-by-Stage Transmission Metrics (Evaluated vs t_true on {np.count_nonzero(valid_mask)/valid_mask.size*100:.1f}% valid pixels):")
        print(f"{'Stage Name':28s} | {'Mean t':7s} | {'Median':7s} | {'Bias':7s} | {'MAE':7s} | {'RMSE':7s} | {'Corr':6s} | {'[P5, P95]':15s}")
        print("-" * 96)

        t_true_val = t_true[valid_mask]
        print(f"{'Ground Truth (t_true)':28s} | {np.mean(t_true_val):7.4f} | {np.median(t_true_val):7.4f} | {'0.0000':7s} | {'0.0000':7s} | {'0.0000':7s} | {'1.0000':6s} | [{np.percentile(t_true_val, 5):.3f}, {np.percentile(t_true_val, 95):.3f}]")

        stage_maps = {}
        for s_name, s_map in stages:
            s_map_full = np.array(Image.fromarray(s_map).resize((orig_w, orig_h), Image.BILINEAR))
            stage_maps[s_name] = s_map_full
            s_val = s_map_full[valid_mask]
            
            bias = float(np.mean(s_val - t_true_val))
            mae = float(np.mean(np.abs(s_val - t_true_val)))
            rmse = float(np.sqrt(np.mean((s_val - t_true_val) ** 2)))
            corr = float(np.corrcoef(s_val, t_true_val)[0, 1]) if np.std(s_val) > 1e-6 and np.std(t_true_val) > 1e-6 else 0.0
            p5 = float(np.percentile(s_val, 5))
            p95 = float(np.percentile(s_val, 95))
            med = float(np.median(s_val))
            mean_t = float(np.mean(s_val))

            print(f"{s_name:28s} | {mean_t:7.4f} | {med:7.4f} | {bias:+7.4f} | {mae:7.4f} | {rmse:7.4f} | {corr:+6.3f} | [{p5:.3f}, {p95:.3f}]")

        # ---------------------------------------------------------------------
        # PHASE 7: FUSION WEIGHT AUDIT
        # ---------------------------------------------------------------------
        print(f"\n--- PHASE 7: SPATIAL FUSION WEIGHT AUDIT ---")
        w_dcp_full = np.array(Image.fromarray(w_dcp).resize((orig_w, orig_h), Image.BILINEAR))[valid_mask]
        cap_err = np.abs(stage_maps["Stage A: Raw CAP"][valid_mask] - t_true_val)
        dcp_err = np.abs(stage_maps["Stage B: Raw DCP"][valid_mask] - t_true_val)
        fused_err = np.abs(stage_maps["Stage C: Spatial Fusion"][valid_mask] - t_true_val)

        dcp_wins = np.count_nonzero(dcp_err < cap_err) / cap_err.size * 100.0
        cap_wins = np.count_nonzero(cap_err < dcp_err) / cap_err.size * 100.0

        print(f"Estimator Accuracy Competition:")
        print(f"  CAP Mean Absolute Error:  {np.mean(cap_err):.4f}")
        print(f"  DCP Mean Absolute Error:  {np.mean(dcp_err):.4f}")
        print(f"  Fused Mean Absolute Error:{np.mean(fused_err):.4f}")
        print(f"  DCP is more accurate in:  {dcp_wins:.1f}% of pixels")
        print(f"  CAP is more accurate in:  {cap_wins:.1f}% of pixels")
        print(f"  Mean Fusion Weight w_dcp: {np.mean(w_dcp_full):.4f} (Weight given to DCP)")
        print(f"  Mean Fusion Weight w_cap: {1.0 - np.mean(w_dcp_full):.4f} (Weight given to CAP)")

        # Check correlation between w_dcp and (cap_err - dcp_err)
        # If w_dcp is optimal, when cap_err > dcp_err (DCP is better), w_dcp should be high
        advantage_dcp = cap_err - dcp_err # positive when DCP is better
        corr_weight_opt = np.corrcoef(w_dcp_full, advantage_dcp)[0, 1]
        print(f"  Correlation between w_dcp and DCP Advantage (cap_err - dcp_err): {corr_weight_opt:+.4f}")
        if np.mean(dcp_err) < np.mean(cap_err) and np.mean(w_dcp_full) < 0.5:
            print(f"  -> CRITICAL AUDIT FINDING: DCP is significantly MORE accurate than CAP (MAE {np.mean(dcp_err):.4f} vs {np.mean(cap_err):.4f}), but Fusion assigns only {np.mean(w_dcp_full)*100:.1f}% weight to DCP and { (1-np.mean(w_dcp_full))*100:.1f}% to CAP!")

if __name__ == "__main__":
    run_stage_audit()
