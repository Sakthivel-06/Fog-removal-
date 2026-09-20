"""
scratch/session19_depth_and_cap_dcp_audit.py — Depth Distribution, CAP/DCP Sensitivity, & Sky-Floor Audit
===========================================================================================================
Executes Phase 4, Phase 5, Phase 6, Phase 7, Phase 8, and Phase 9.
"""

import os
import sys
import imageio
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "algorithms")))

import dcp
import cap
from pipeline import RealTimeDehazingPipeline, _resize_np

def run_depth_and_sensitivity_audit():
    print("=" * 90)
    print("SESSION 19 — PHASE 4, 5, 6, 8, 9: DEPTH, CAP/DCP SENSITIVITY & SKY-FLOOR AUDIT")
    print("=" * 90)

    # -------------------------------------------------------------------------
    # PHASE 4: SYNTHETIC VS REAL-WORLD DEPTH/TRANSMISSION DISTRIBUTIONS
    # -------------------------------------------------------------------------
    print("\n--- PHASE 4: TRANSMISSION DISTRIBUTIONS (SYNTHETIC VS REAL FOG) ---")
    pipe = RealTimeDehazingPipeline()

    real_videos = ["foggy_video1.mp4", "foggy_video6.mp4", "foggy_video7.mp4"]
    for rv in real_videos:
        rpath = os.path.join("test_videos", rv)
        if os.path.exists(rpath):
            reader = imageio.get_reader(rpath)
            f = reader.get_data(15)
            reader.close()
            pipe.reset()
            for w in range(5):
                pipe.process_frame(f)
            out, _ = pipe.process_frame(f)
            t_real = pipe._cached_t
            print(f"Real Video [{rv:16s}]: Mean t = {np.mean(t_real):.3f} | Std = {np.std(t_real):.3f} | Range = [{np.min(t_real):.3f}, {np.max(t_real):.3f}] | P5-P95 = [{np.percentile(t_real, 5):.3f}, {np.percentile(t_real, 95):.3f}]")

    # -------------------------------------------------------------------------
    # PHASE 5: CAP PARAMETER SENSITIVITY (OFFLINE)
    # -------------------------------------------------------------------------
    print("\n--- PHASE 5: CAP PARAMETER SENSITIVITY SWEEP (Person Bicycle) ---")
    rc = imageio.get_reader("test_videos/person-bicycle_clean.mp4")
    rh = imageio.get_reader("test_videos/person-bicycle_uniform_haze.mp4")
    fc = rc.get_data(10)
    fh = rh.get_data(10)
    rc.close()
    rh.close()

    t_true_scalar = float(np.exp(-1.5 * 0.8)) # 0.3012
    proc_float = fh.astype(np.float32) / 255.0
    half_img = proc_float[::2, ::2]
    v = np.max(half_img, axis=2)
    s = (v - np.min(half_img, axis=2)) / (v + 1e-6)

    # Grid sweep of beta and (theta0, theta1, theta2)
    beta_candidates = [1.0, 1.2, 1.5, 1.8, 2.0]
    for b in beta_candidates:
        d_std = 0.121779 + 0.959710 * v - 0.780245 * s
        t_cap_test = np.exp(-b * np.clip(d_std, 0.0, 2.5))
        mae = float(np.mean(np.abs(t_cap_test - t_true_scalar)))
        print(f"  CAP beta = {b:.1f} (std thetas) -> Mean t_cap = {np.mean(t_cap_test):.4f} | Bias = {np.mean(t_cap_test)-t_true_scalar:+.4f} | MAE = {mae:.4f}")

    # -------------------------------------------------------------------------
    # PHASE 6: DCP PARAMETER SENSITIVITY (OFFLINE)
    # -------------------------------------------------------------------------
    print("\n--- PHASE 6: DCP PARAMETER SENSITIVITY SWEEP (Person Bicycle) ---")
    A_true = np.array([1.0, 1.0, 1.0])
    norm = half_img / A_true[None, None, :]
    dc = np.min(norm, axis=2)

    for p_size in [3, 5, 7, 15]:
        import scipy.ndimage as ndimage
        dc_min = ndimage.minimum_filter(dc, size=p_size)
        for om in [0.70, 0.85, 0.90, 0.95, 0.98]:
            t_dcp_test = 1.0 - om * dc_min
            mae = float(np.mean(np.abs(t_dcp_test - t_true_scalar)))
            print(f"  DCP patch = {p_size:2d}, omega = {om:.2f} -> Mean t_dcp = {np.mean(t_dcp_test):.4f} | Bias = {np.mean(t_dcp_test)-t_true_scalar:+.4f} | MAE = {mae:.4f}")

    # -------------------------------------------------------------------------
    # PHASE 8 & 9: GUIDED FILTER AND TEMPORAL SMOOTHING AUDIT
    # -------------------------------------------------------------------------
    print("\n--- PHASE 8 & 9: GUIDED FILTER AND TEMPORAL SMOOTHING DELTAS ---")
    # Check impact on car-detection_patchy_haze
    rh_car = imageio.get_reader("test_videos/car-detection_patchy_haze.mp4")
    rc_car = imageio.get_reader("test_videos/car-detection_clean.mp4")
    frames_h_car = [rh_car.get_data(i) for i in range(15)]
    frames_c_car = [rc_car.get_data(i) for i in range(15)]
    rh_car.close()
    rc_car.close()

    pipe.reset()
    for i in range(10):
        pipe.process_frame(frames_h_car[i])
    
    # Analyze frame 10
    f_car = frames_h_car[10]
    out, _ = pipe.process_frame(f_car)
    
    # Reconstruct true t for car frame 10
    A_true_255 = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    denom = frames_c_car[10].astype(np.float32) - A_true_255
    numer = frames_h_car[10].astype(np.float32) - A_true_255
    valid_mask = np.all(np.abs(denom) >= 20.0, axis=-1)
    t_true_car = np.clip(np.median(numer / (denom + 1e-6), axis=-1), 0.05, 1.0)
    
    t_cached = pipe._cached_t
    t_cached_full = np.array(Image.fromarray(t_cached).resize((f_car.shape[1], f_car.shape[0]), Image.BILINEAR))
    
    print(f"Car Detection Frame 10:")
    print(f"  True t Mean:           {np.mean(t_true_car[valid_mask]):.4f}")
    print(f"  Proposed Cached t Mean:{np.mean(t_cached_full[valid_mask]):.4f}")
    print(f"  Temporal Lag Bias:     {np.mean(t_cached_full[valid_mask] - t_true_car[valid_mask]):+.4f}")

if __name__ == "__main__":
    run_depth_and_sensitivity_audit()
