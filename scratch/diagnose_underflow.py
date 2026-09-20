"""
scratch/diagnose_underflow.py — Deep Diagnostic for Session 15 Phases 2, 3, 4, 7
================================================================================
Analyzes:
  1. Input statistics (RGB, luminance, native black % < 1/255, dark % < 38/255)
  2. Estimated terms (A_source, eta(x), A(x), t(x), adaptive t_min(x))
  3. Pre-clipping J_raw (min, percentiles, % < 0, % > 1)
  4. Post-clipping output (black clip %, white clip %, dark region luminance, road luminance, edge energy)
  5. Required t(x) = 1 - I(x)/A(x) where J_raw < 0
  6. Underflow root-cause decomposition (A: t too low, B: A too high, C: eta too high, D: genuinely unlit, E: sensor zero, F: mixed)
  7. Native input black vs model-induced black crush
  8. Noise amplification (local variance in dark regions before and after inversion)
"""

import os
import sys
import json
import imageio
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nighttime_experimental import ExperimentalNighttimePipeline
from nighttime_diagnostics import compute_frame_telemetry

def extract_frame(video_path, frame_idx=30):
    reader = imageio.get_reader(video_path)
    frame = None
    for i, f in enumerate(reader):
        if i == frame_idx:
            frame = f
            break
    reader.close()
    return frame

def run_diagnostics_on_sequence(video_name, frame_idx=30):
    video_path = os.path.join("test_videos", video_name)
    frame = extract_frame(video_path, frame_idx)
    if frame is None:
        raise ValueError(f"Could not extract frame {frame_idx} from {video_path}")
    
    # Initialize pipeline with Session 12 Model A
    pipe = ExperimentalNighttimePipeline(enable_nighttime_branch=True, night_mode="prototype")
    
    # Warm up pipeline on frames prior to frame_idx to ensure temporal filters stabilize
    reader = imageio.get_reader(video_path)
    for i, f in enumerate(reader):
        if i < frame_idx:
            pipe.process_frame(f)
        else:
            break
    reader.close()
    
    # Now process target frame and tap internal intermediate representations
    raw_h, raw_w = frame.shape[:2]
    max_w, max_h = 1280, 720
    if raw_w > max_w or raw_h > max_h:
        stride = min(max(raw_w // max_w, raw_h // max_h, 1), 4)
        sub_f = frame[::stride, ::stride]
        s = min(max_w / float(sub_f.shape[1]), max_h / float(sub_f.shape[0]))
        cur_w, cur_h = int(sub_f.shape[1] * s), int(sub_f.shape[0] * s)
        from pipeline import _resize_np
        if (sub_f.shape[1], sub_f.shape[0]) == (cur_w, cur_h):
            proc_frame = np.ascontiguousarray(sub_f)
        else:
            proc_frame = _resize_np(sub_f, (cur_w, cur_h))
    else:
        proc_frame = frame

    orig_h, orig_w = proc_frame.shape[:2]
    proc_float = proc_frame.astype(np.float32) / 255.0

    # 1. INPUT STATS
    in_lum = 0.299 * proc_float[:, :, 0] + 0.587 * proc_float[:, :, 1] + 0.114 * proc_float[:, :, 2]
    native_black_mask = (in_lum < (1.0 / 255.0))
    native_black_pct = float(np.count_nonzero(native_black_mask) / in_lum.size * 100.0)
    dark_pixel_pct = float(np.count_nonzero(in_lum < (38.0 / 255.0)) / in_lum.size * 100.0)
    input_mean_lum = float(np.mean(in_lum * 255.0))

    # Process frame through pipeline to obtain estimates
    out, fps, diag = pipe.process_frame(frame, return_diagnostics=True)

    # Tap internal variables
    half_img = proc_float[::2, ::2]
    gray_h = 0.299 * half_img[:, :, 0] + 0.587 * half_img[:, :, 1] + 0.114 * half_img[:, :, 2]
    is_night, gate_stats = pipe.evaluate_illumination_gate(gray_h)

    A_source = pipe.last_A.copy()

    # Reconstruct exact t_smooth and t_full
    v = np.max(half_img, axis=2)
    dc_img = np.min(half_img, axis=2)
    s = (v - dc_img) / (v + 1e-6)
    depth = ndimage.uniform_filter(0.121779 + 0.959710 * v - 0.780245 * s, size=3)
    norm = half_img / (A_source[None, None, :] + 1e-6)
    dc = np.min(norm, axis=2)
    dc_min = ndimage.minimum_filter(dc, size=5)
    h_local = ndimage.uniform_filter(dc, size=3)
    omega_night = min(pipe.omega, 0.85)
    t_dcp = 1.0 - omega_night * dc_min
    t_cap = np.exp(-pipe.beta * np.clip(depth, 0.0, 2.5))
    dark_penalty = np.clip((0.30 - gray_h) / 0.20, 0.0, 1.0)
    w_raw = 0.20 * (1.0 - 0.70 * dark_penalty) + 0.30 * np.clip(h_local, 0.0, 1.0)
    w_dcp = ndimage.gaussian_filter(np.clip(w_raw, 0.05, 0.60), sigma=1.5)
    t_hybrid = w_dcp * t_dcp + (1.0 - w_dcp) * t_cap

    t_base = max(pipe.t_min, 0.15)
    floor_dark = 0.25 * dark_penalty
    t_min_map = np.clip(t_base + floor_dark, t_base, 0.85)
    t_hybrid = np.clip(t_hybrid, t_min_map, 1.0)

    mean_I = ndimage.uniform_filter(gray_h, size=5)
    mean_p = ndimage.uniform_filter(t_hybrid, size=5)
    mean_Ip = ndimage.uniform_filter(gray_h * t_hybrid, size=5)
    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = ndimage.uniform_filter(gray_h * gray_h, size=5) - mean_I * mean_I
    a = cov_Ip / (var_I + 1e-3)
    b = mean_p - a * mean_I
    t_smooth = ndimage.uniform_filter(a, size=5) * gray_h + ndimage.uniform_filter(b, size=5)
    t_smooth = np.clip(t_smooth, t_min_map, 1.0)

    # Upsample t_smooth and t_min_map to full processing resolution
    t_full = np.array(Image.fromarray(t_smooth, mode="F").resize((orig_w, orig_h), Image.BILINEAR))[:, :, None]
    t_min_full = np.array(Image.fromarray(t_min_map, mode="F").resize((orig_w, orig_h), Image.BILINEAR))[:, :, None]

    # Model A spatial illumination
    V_full = np.max(proc_float, axis=2)
    L_smooth = ndimage.uniform_filter(V_full, size=pipe.local_illum_kernel_size)
    a_max_ref = max(float(np.max(A_source)), 0.25)
    eta = np.clip(L_smooth / a_max_ref, pipe.eta_min, 1.0)[:, :, None]
    A_spatial = A_source[None, None, :] * eta

    # Raw physical scene radiance before clipping
    J_raw = (proc_float - A_spatial) / t_full + A_spatial

    # Underflow & Overflow analysis
    underflow_mask = np.any(J_raw < 0.0, axis=2)
    underflow_pct = float(np.count_nonzero(underflow_mask) / underflow_mask.size * 100.0)
    overflow_mask = np.any(J_raw > 1.0, axis=2)
    overflow_pct = float(np.count_nonzero(overflow_mask) / overflow_mask.size * 100.0)
    min_J_raw = float(np.min(J_raw))
    max_J_raw = float(np.max(J_raw))
    percentiles_J = {
        f"p{p}": float(np.percentile(J_raw, p)) for p in [0.1, 1, 5, 10, 25, 50, 75, 90, 99]
    }

    # Final clipped output
    J_clipped = np.clip(J_raw, 0.0, 1.0)
    out_lum = 0.299 * J_clipped[:, :, 0] + 0.587 * J_clipped[:, :, 1] + 0.114 * J_clipped[:, :, 2]
    final_black_mask = (out_lum < (1.0 / 255.0))
    final_black_pct = float(np.count_nonzero(final_black_mask) / out_lum.size * 100.0)
    white_clip_pct = float(np.count_nonzero(out_lum > (254.0 / 255.0)) / out_lum.size * 100.0)
    induced_black_crush = final_black_pct - native_black_pct

    # Dark-region mean luminance (< 38/255 in input)
    dark_mask_input = (in_lum < (38.0 / 255.0))
    dark_region_out_lum = float(np.mean(out_lum[dark_mask_input] * 255.0)) if np.any(dark_mask_input) else 0.0
    dark_region_in_lum = float(np.mean(in_lum[dark_mask_input] * 255.0)) if np.any(dark_mask_input) else 0.0

    # Road-region luminance (middle-lower region y: 60%..85%, x: 30%..70%)
    ry0, ry1 = int(0.60 * orig_h), int(0.85 * orig_h)
    rx0, rx1 = int(0.30 * orig_w), int(0.70 * orig_w)
    road_out_lum = float(np.mean(out_lum[ry0:ry1, rx0:rx1] * 255.0))
    road_in_lum = float(np.mean(in_lum[ry0:ry1, rx0:rx1] * 255.0))

    # Edge energy
    gy, gx = np.gradient(out_lum * 255.0)
    edge_energy = float(np.mean(np.sqrt(gx**2 + gy**2)))

    # PHASE 3: UNDERFLOW ROOT CAUSE ANALYSIS
    # Where J_raw < 0:
    # J_c < 0 <=> I_c < A_c * (1 - t) <=> t < 1 - I_c / A_c
    # We analyze pixels where any channel has J_raw < 0.
    eps = 1e-4
    A_eff = np.maximum(A_spatial, eps)
    required_t_per_channel = 1.0 - proc_float / A_eff
    required_t = np.max(required_t_per_channel, axis=2)  # highest t needed to prevent all channels from underflowing
    actual_t = t_full[:, :, 0]

    # Analyze specifically the underflow pixels
    underflow_count = np.count_nonzero(underflow_mask)
    categories = {}
    if underflow_count > 0:
        # Category criteria:
        # For underflow pixels:
        u_in_lum = in_lum[underflow_mask]
        u_actual_t = actual_t[underflow_mask]
        u_req_t = required_t[underflow_mask]
        u_eta = eta[:, :, 0][underflow_mask]
        u_A_mean = np.mean(A_spatial, axis=2)[underflow_mask]
        u_I_min = np.min(proc_float, axis=2)[underflow_mask]

        # D. Genuinely unlit / native black pixels (I < 2/255 in input)
        cat_D = (u_in_lum < (2.0 / 255.0))
        # E. Camera black / sensor quantization (2/255 <= I < 5/255, very low SNR)
        cat_E = (~cat_D) & (u_in_lum < (5.0 / 255.0))
        # Now for pixels with non-trivial signal (I >= 5/255):
        remaining = (~cat_D) & (~cat_E)
        
        # A. Transmission too low: actual_t < 0.35 while required_t is within physical range (0.35..0.85)
        cat_A = remaining & (u_actual_t < 0.35) & (u_req_t <= 0.85)
        # C. Illumination estimate eta too high: local smoothing U_31 bled headlight light into dark area
        cat_C = remaining & (~cat_A) & (u_eta > 0.30) & (u_in_lum < 0.15)
        # B. Airlight too high: A_spatial is inflated because A_source is high relative to local I
        cat_B = remaining & (~cat_A) & (~cat_C) & (u_A_mean > 2.0 * u_in_lum)
        # F. Mixed cause
        cat_F = remaining & (~cat_A) & (~cat_B) & (~cat_C)

        categories = {
            "cat_D_genuinely_unlit": float(np.count_nonzero(cat_D) / underflow_count * 100.0),
            "cat_E_sensor_black_limit": float(np.count_nonzero(cat_E) / underflow_count * 100.0),
            "cat_A_transmission_too_low": float(np.count_nonzero(cat_A) / underflow_count * 100.0),
            "cat_C_illumination_eta_bleed": float(np.count_nonzero(cat_C) / underflow_count * 100.0),
            "cat_B_airlight_too_high": float(np.count_nonzero(cat_B) / underflow_count * 100.0),
            "cat_F_mixed": float(np.count_nonzero(cat_F) / underflow_count * 100.0),
        }

    # PHASE 7: NOISE AMPLIFICATION ANALYSIS
    # Local variance in dark regions before and after inversion
    # Compute local 5x5 variance of input luminance vs output luminance
    local_mean_in = ndimage.uniform_filter(in_lum, size=5)
    local_sqmean_in = ndimage.uniform_filter(in_lum**2, size=5)
    local_var_in = np.maximum(local_sqmean_in - local_mean_in**2, 0.0)

    local_mean_out = ndimage.uniform_filter(out_lum, size=5)
    local_sqmean_out = ndimage.uniform_filter(out_lum**2, size=5)
    local_var_out = np.maximum(local_sqmean_out - local_mean_out**2, 0.0)

    # In dark regions that didn't clip to 0:
    dark_valid_mask = dark_mask_input & (~final_black_mask)
    if np.any(dark_valid_mask):
        mean_dark_var_in = float(np.mean(local_var_in[dark_valid_mask]))
        mean_dark_var_out = float(np.mean(local_var_out[dark_valid_mask]))
        mean_dark_t = float(np.mean(actual_t[dark_valid_mask]))
        amp_factor_empirical = mean_dark_var_out / (mean_dark_var_in + 1e-9)
        amp_factor_theoretical = (1.0 / (mean_dark_t + 1e-9)) ** 2
    else:
        mean_dark_var_in = 0.0
        mean_dark_var_out = 0.0
        mean_dark_t = 0.0
        amp_factor_empirical = 0.0
        amp_factor_theoretical = 0.0

    return {
        "video_name": video_name,
        "is_night": bool(is_night),
        "gate_decision": gate_stats.get("gate_decision", "UNKNOWN"),
        "input_mean_lum": input_mean_lum,
        "native_black_pct": native_black_pct,
        "dark_pixel_pct": dark_pixel_pct,
        "A_source": [float(x) for x in A_source],
        "eta_mean": float(np.mean(eta)),
        "eta_min": float(np.min(eta)),
        "eta_max": float(np.max(eta)),
        "A_spatial_mean": float(np.mean(A_spatial)),
        "t_mean": float(np.mean(t_full)),
        "t_min": float(np.min(t_full)),
        "adaptive_t_min_mean": float(np.mean(t_min_full)),
        "underflow_pct_J_raw": underflow_pct,
        "overflow_pct_J_raw": overflow_pct,
        "min_J_raw": min_J_raw,
        "max_J_raw": max_J_raw,
        "percentiles_J": percentiles_J,
        "final_black_clip_pct": final_black_pct,
        "white_clip_pct": white_clip_pct,
        "induced_black_crush_pct": induced_black_crush,
        "dark_region_in_lum": dark_region_in_lum,
        "dark_region_out_lum": dark_region_out_lum,
        "road_in_lum": road_in_lum,
        "road_out_lum": road_out_lum,
        "edge_energy": edge_energy,
        "underflow_categories": categories,
        "noise_analysis": {
            "mean_dark_var_in": mean_dark_var_in,
            "mean_dark_var_out": mean_dark_var_out,
            "mean_dark_t": mean_dark_t,
            "noise_amp_empirical": amp_factor_empirical,
            "noise_amp_theoretical": amp_factor_theoretical,
        }
    }

if __name__ == "__main__":
    targets = [
        "foggy_video3.mp4",
        "foggy_video4.mp4",
        "foggy_video5.mp4",
        "foggy_video14.mp4",
        "foggy_video1.mp4",
    ]
    results = {}
    for t in targets:
        print(f"--- Diagnosing {t} ---")
        res = run_diagnostics_on_sequence(t, frame_idx=30)
        results[t] = res
        print(f"  Gate: {res['gate_decision']}")
        print(f"  Native Black %: {res['native_black_pct']:.2f}%")
        print(f"  J_raw < 0 %:    {res['underflow_pct_J_raw']:.2f}% (min J_raw: {res['min_J_raw']:.3f})")
        print(f"  Final Black %:  {res['final_black_clip_pct']:.2f}%")
        print(f"  Induced Crush:  {res['induced_black_crush_pct']:+.2f}%")
        if res['underflow_categories']:
            print("  Root causes:")
            for k, v in res['underflow_categories'].items():
                print(f"    {k}: {v:.1f}%")
        print(f"  Noise Amp (Empirical): {res['noise_analysis']['noise_amp_empirical']:.2f}x (Theoretical 1/t^2: {res['noise_analysis']['noise_amp_theoretical']:.2f}x)")
        print()
        
    out_file = os.path.join("scratch", "session15_underflow_diagnostics.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full diagnostics to {out_file}")
