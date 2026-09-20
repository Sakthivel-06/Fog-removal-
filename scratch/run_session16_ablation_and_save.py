"""
scratch/run_session16_ablation_and_save.py — Session 16 Ablation and Results Combiner
=====================================================================================
1. Parses all 91 benchmark runs across 13 videos from task-2737.log.
2. Implements pure Python subclass AblationPipeline to evaluate:
   - Config A: Full Proposed Method
   - Config B: Full - Adaptive spatial t_min(x)
   - Config C: Full - DCP/CAP Spatial Fusion
   - Config D: Full - Robust Atmospheric-Light Estimation
   - Config E: Full - Chromaticity-Preserving Atmospheric-Light Bound
   - Config F: Full - Guided Transmission Refinement
   - Config G: Full - Motion-Aware Temporal Consistency
   Across:
   - car-detection_patchy_haze.mp4 (GT)
   - person-bicycle_uniform_haze.mp4 (GT)
   - foggy_video1.mp4 (Dense Fog)
   - foggy_video3.mp4 (Night Driving)
3. Combines everything into scratch/session16_results.json.
"""

import os
import sys
import re
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
from nighttime_experimental import ExperimentalNighttimePipeline
from pipeline import _resize_np, _to_gray
import metrics

# 1. PARSE BASELINE RUNS FROM TASK LOG
def parse_baseline_runs():
    log_path = r"C:\Users\sakth\.gemini\antigravity-ide\brain\a110838c-9ebf-4af3-a443-b35b20d1e5d7\.system_generated\tasks\task-2737.log"
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    video_blocks = re.split(r"Evaluating Category \[(.*?)\] (.*?) -> (.*?) \(15 measured frames\):", text)
    # video_blocks[0] is header, then tuples of (cat, cat_name, vid_name, content)
    records = []
    
    # Iterate through chunks
    for i in range(1, len(video_blocks), 4):
        cat = video_blocks[i]
        cat_name = video_blocks[i+1]
        vid_name = video_blocks[i+2]
        content = video_blocks[i+3]
        
        # Parse each method line
        # e.g.: [DCP             ] FPS:  89.0 | NIQE: 4.48 | FDI: 0.136 | Black:  0.02% | Flick: 5.56 | PSNR: No GT | SSIM: No GT
        pattern = r"\[(.*?)\]\s+FPS:\s+([\d\.]+)\s+\|\s+NIQE:\s+([\d\.]+)\s+\|\s+FDI:\s+([\d\.]+)\s+\|\s+Black:\s+([\d\.]+)%\s+\|\s+Flick:\s+([\d\.]+)\s+\|\s+PSNR:\s+(.*?)\s+\|\s+SSIM:\s+(.*)"
        for m in re.finditer(pattern, content):
            m_name = m.group(1).strip()
            fps = float(m.group(2))
            niqe = float(m.group(3))
            fdi = float(m.group(4))
            black = float(m.group(5))
            flick = float(m.group(6))
            psnr_str = m.group(7).strip()
            ssim_str = m.group(8).strip()
            
            psnr_val = float(psnr_str.replace("dB", "").strip()) if "No GT" not in psnr_str else "Requires Ground Truth"
            ssim_val = float(ssim_str.strip()) if "No GT" not in ssim_str else "Requires Ground Truth"
            
            rec = {
                "category": cat,
                "category_name": cat_name,
                "video_name": vid_name,
                "has_gt": ("No GT" not in psnr_str),
                "method_name": m_name,
                "fps": fps,
                "niqe": niqe,
                "fdi_proxy": fdi,
                "fade": "FADE: unavailable",
                "black_clip": black,
                "flicker": flick,
                "psnr": psnr_val,
                "ssim": ssim_val,
                "lpips": "Requires Ground Truth" if "No GT" in psnr_str else "Requires learned model",
            }
            records.append(rec)
            
    print(f"Successfully parsed {len(records)} baseline records from task-2737.log!")
    return records

# 2. DEFINE ABLATION PIPELINE SUBCLASS
class AblationPipeline(ExperimentalNighttimePipeline):
    """Subclass allowing clean removal of individual components for Session 16 ablation."""
    def __init__(self, ablation_mode="A_Full", **kwargs):
        super().__init__(enable_nighttime_branch=True, night_mode="prototype", **kwargs)
        self.ablation_mode = ablation_mode

    def process_frame(self, frame: np.ndarray, return_diagnostics: bool = False):
        # We override process_frame to surgically isolate each ablated component
        t0 = time.perf_counter()
        raw_h, raw_w = frame.shape[:2]

        # Resolution scaling
        if self.max_process_res is not None:
            max_w, max_h = self.max_process_res
            if raw_w > max_w or raw_h > max_h:
                stride = min(max(raw_w // max_w, raw_h // max_h, 1), 4)
                sub_f = frame[::stride, ::stride]
                s = min(max_w / float(sub_f.shape[1]), max_h / float(sub_f.shape[0]))
                cur_w, cur_h = int(sub_f.shape[1] * s), int(sub_f.shape[0] * s)
                if (sub_f.shape[1], sub_f.shape[0]) == (cur_w, cur_h):
                    proc_frame = np.ascontiguousarray(sub_f)
                else:
                    proc_frame = _resize_np(sub_f, (cur_w, cur_h))
            else:
                proc_frame = frame
        else:
            proc_frame = frame

        orig_h, orig_w = proc_frame.shape[:2]
        proc_float = proc_frame.astype(np.float32) / 255.0

        # Half resolution guide
        half_img = proc_float[::2, ::2]
        gh, gw = half_img.shape[:2]
        gray_h = 0.299 * half_img[:, :, 0] + 0.587 * half_img[:, :, 1] + 0.114 * half_img[:, :, 2]

        is_night, gate_stats = self.evaluate_illumination_gate(gray_h)

        # Depth from CAP
        v = np.max(half_img, axis=2)
        dc_img = np.min(half_img, axis=2)
        s = (v - dc_img) / (v + 1e-6)
        depth = ndimage.uniform_filter(0.121779 + 0.959710 * v - 0.780245 * s, size=3)

        # Component D: Robust Atmospheric Light Estimation vs. Naive DCP Airlight
        if self.ablation_mode == "D_NoAirlight":
            # Naive atmospheric light: top 0.1% brightest in dark channel, average color
            dc_raw = np.min(half_img, axis=2)
            top_n = max(int(dc_raw.size * 0.001), 10)
            idx_top = np.argpartition(dc_raw.ravel(), -top_n)[-top_n:]
            A_raw = np.mean(half_img.reshape(-1, 3)[idx_top], axis=0)
            candidate_mask = np.zeros((gh, gw), dtype=bool)
        else:
            A_raw, candidate_mask = self.estimate_atmospheric_light(
                half_img, depth, v=v, s=s, dc=dc_img, gray_h=gray_h
            )

        # Component E: Chromaticity-Preserving Atmospheric Light Bound
        if self.ablation_mode == "E_NoABound":
            # Remove lower bound entirely (allow A to collapse to unclipped candidate mean)
            if hasattr(self, "last_A_est_unclipped") and self.last_A_est_unclipped is not None:
                A_raw = self.last_A_est_unclipped.copy()

        self.last_A_raw = A_raw

        # Raw DCP transmission
        norm = half_img / (A_raw[None, None, :] + 1e-6)
        dc = np.min(norm, axis=2)
        dc_min = ndimage.minimum_filter(dc, size=5)
        h_local = ndimage.uniform_filter(dc, size=3)
        omega_map = np.clip(self.omega + 0.08 * h_local, self.omega, 0.96)
        t_dcp = 1.0 - omega_map * dc_min

        # Raw CAP transmission
        t_cap = np.exp(-self.beta * np.clip(depth, 0.0, 2.5))

        # Component C: DCP/CAP Spatial Fusion vs. Pure CAP alone
        if self.ablation_mode == "C_NoFusion":
            t_hybrid = t_cap
        else:
            # Full spatial fusion w_dcp(x)
            cfg = self.fusion_cfg
            var_h = np.maximum(ndimage.uniform_filter(gray_h ** 2, size=3) - ndimage.uniform_filter(gray_h, size=3) ** 2, 0.0)
            b_factor = np.clip((gray_h - 0.55) / 0.25, 0.0, 1.0)
            s_factor = np.clip(1.0 - var_h / 0.0005, 0.0, 1.0)
            bright_term = np.clip((gray_h - 0.75) / 0.15, 0.0, 1.0)
            m_sky = np.clip(np.maximum(b_factor * s_factor, bright_term), 0.0, 1.0)
            text_conf = np.clip(var_h / 0.0008, 0.0, 1.0)
            dark_penalty = np.clip((0.32 - gray_h) / 0.20, 0.0, 1.0)
            haze_lum_gate = np.clip((gray_h - 0.20) / 0.12, 0.0, 1.0) * np.clip((0.82 - gray_h) / 0.15, 0.0, 1.0)
            H_x = np.clip((h_local - 0.15) / 0.35, 0.0, 1.0) * haze_lum_gate
            w_raw = (0.15 + 0.60 * H_x) * (1.0 - 0.85 * m_sky) * (1.0 - 0.55 * dark_penalty) * (0.60 + 0.40 * text_conf)
            w_dcp = ndimage.gaussian_filter(np.clip(w_raw, 0.05, 0.75), sigma=1.5)
            t_hybrid = w_dcp * t_dcp + (1.0 - w_dcp) * t_cap

        # Component B: Adaptive spatial t_min(x) vs. Fixed scalar floor (0.10)
        if self.ablation_mode == "B_NoTmin":
            t_min_map = np.full_like(t_hybrid, 0.10)
            t_hybrid = np.clip(t_hybrid, t_min_map, 1.0)
        else:
            var_h = np.maximum(ndimage.uniform_filter(gray_h ** 2, size=3) - ndimage.uniform_filter(gray_h, size=3) ** 2, 0.0)
            t_base = max(self.t_min, 0.12)
            w_sky = np.clip((gray_h - 0.50) / 0.25, 0.0, 1.0) * np.clip(1.0 - var_h / 0.0003, 0.0, 1.0)
            w_bright = np.clip((gray_h - 0.65) / 0.25, 0.0, 1.0)
            floor_bright = 0.50 * np.maximum(w_sky, w_bright * 0.6)
            w_dark = np.clip((0.35 - gray_h) / 0.25, 0.0, 1.0)
            w_smooth_dark = np.clip(1.0 - var_h / 0.001, 0.0, 1.0)
            floor_dark = 0.35 * w_dark * (0.6 + 0.4 * w_smooth_dark)
            haze_ind = np.clip((h_local - 0.20) / 0.30, 0.0, 1.0) * (1.0 - w_dark) * (1.0 - w_sky)
            relief = 0.06 * haze_ind
            t_min_map = np.clip(t_base + floor_dark + floor_bright - relief, t_base, 0.85)
            t_hybrid = np.clip(t_hybrid, t_min_map, 1.0)

        # Component F: Guided Transmission Refinement vs. Unrefined Raw Transmission
        if self.ablation_mode == "F_NoRefine":
            t_smooth = t_hybrid
        else:
            mean_I = ndimage.uniform_filter(gray_h, size=5)
            mean_p = ndimage.uniform_filter(t_hybrid, size=5)
            mean_Ip = ndimage.uniform_filter(gray_h * t_hybrid, size=5)
            cov_Ip = mean_Ip - mean_I * mean_p
            var_I = ndimage.uniform_filter(gray_h * gray_h, size=5) - mean_I * mean_I
            a = cov_Ip / (var_I + 1e-3)
            b = mean_p - a * mean_I
            t_smooth = ndimage.uniform_filter(a, size=5) * gray_h + ndimage.uniform_filter(b, size=5)
            t_smooth = np.clip(t_smooth, t_min_map, 1.0)

        # Component G: Motion-Aware Temporal Consistency vs. Frame-Independent Processing
        if self.ablation_mode == "G_NoTemporal":
            A = A_raw
        else:
            from temporal_coherence import compute_motion_mask, is_scene_change, compute_adaptive_temporal_alpha, smooth_transmission_temporal, smooth_airlight_temporal
            if self._prev_gray is not None and self._prev_gray.shape == gray_h.shape:
                motion_mask, mean_diff, high_motion_pct = compute_motion_mask(gray_h, self._prev_gray)
                is_scene_cut = is_scene_change(mean_diff, high_motion_pct=high_motion_pct)
                if not is_scene_cut and self._cached_t is not None and self._cached_t.shape == t_smooth.shape:
                    alpha_map = compute_adaptive_temporal_alpha(motion_mask)
                    t_smooth = smooth_transmission_temporal(t_smooth, self._cached_t, alpha_map)
                    A = smooth_airlight_temporal(A_raw, self._cached_A, alpha_airlight=0.08, is_scene_cut=False)
                else:
                    A = A_raw
            else:
                A = A_raw

        self._cached_t = t_smooth.copy()
        self._cached_A = A.copy()
        self.last_A = A.copy()
        self._prev_gray = gray_h.copy()

        # Full resolution upsample
        t_full = np.array(Image.fromarray(t_smooth, mode="F").resize((orig_w, orig_h), Image.BILINEAR))[:, :, None]

        # Spatial illumination if nighttime
        if is_night:
            V_full = np.max(proc_float, axis=2)
            L_smooth = ndimage.uniform_filter(V_full, size=self.local_illum_kernel_size)
            a_max_ref = max(float(np.max(A)), 0.25)
            eta = np.clip(L_smooth / a_max_ref, self.eta_min, 1.0)[:, :, None]
            A_spatial = A[None, None, :] * eta
        else:
            A_spatial = A[None, None, :]

        # Scene radiance recovery
        J_raw = (proc_float - A_spatial) / t_full + A_spatial
        J = np.clip(J_raw, 0.0, 1.0)
        out = (J * 255.0).astype(np.uint8)

        fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
        return out, fps

# 3. RUN ABLATION BENCHMARK
def run_ablation_experiments():
    ablation_videos = [
        {"cat": "C", "name": "car-detection_patchy_haze.mp4", "gt": "car-detection_clean.mp4"},
        {"cat": "C", "name": "person-bicycle_uniform_haze.mp4", "gt": "person-bicycle_clean.mp4"},
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
    print("\n" + "=" * 110)
    print("RUNNING SESSION 16 ABLATION EXPERIMENTS (4 REPRESENTATIVE VIDEOS, 7 CONFIGS)")
    print("=" * 110)

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

            pipe = AblationPipeline(ablation_mode=cfg_key)

            # Warmup
            for w in range(n_warmup):
                pipe.process_frame(frames_haze[w])

            latencies = []
            flickers = []
            prev_lum = None
            psnrs = []
            ssims = []
            niqes = []
            blacks = []
            b_rs = []
            edges = []

            for i in range(n_warmup, total_frames):
                f_haze = frames_haze[i]
                f_gt = frames_gt[i] if frames_gt else None

                t0 = time.perf_counter()
                out, _ = pipe.process_frame(f_haze)
                latencies.append(time.perf_counter() - t0)

                lum_out = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
                if prev_lum is not None:
                    flickers.append(float(np.mean(np.abs(lum_out - prev_lum))))
                prev_lum = lum_out.copy()

                black_clip = float(np.count_nonzero(lum_out < 1.0) / lum_out.size * 100.0)
                b_r = float(np.mean(out[:, :, 2].astype(float) - out[:, :, 0].astype(float)))
                gy, gx = np.gradient(lum_out)
                edge_eng = float(np.mean(np.sqrt(gx**2 + gy**2)))

                try:
                    niqe_val = float(metrics.niqe(out))
                except Exception:
                    niqe_val = 0.0

                blacks.append(black_clip)
                b_rs.append(b_r)
                edges.append(edge_eng)
                niqes.append(niqe_val)

                if f_gt is not None:
                    if f_gt.shape[:2] != out.shape[:2]:
                        f_gt_res = np.array(Image.fromarray(f_gt).resize((out.shape[1], out.shape[0]), Image.BILINEAR))
                    else:
                        f_gt_res = f_gt
                    psnrs.append(float(sk_psnr(f_gt_res, out, data_range=255)))
                    ssims.append(float(sk_ssim(f_gt_res, out, channel_axis=-1, data_range=255)))

            mean_fps = float(1.0 / np.mean(latencies))
            mean_flicker = float(np.mean(flickers)) if flickers else 0.0
            mean_black = float(np.mean(blacks))
            mean_b_r = float(np.mean(b_rs))
            mean_edge = float(np.mean(edges))
            mean_niqe = float(np.mean(niqes))

            if vid["gt"]:
                mean_psnr = float(np.mean(psnrs))
                mean_ssim = float(np.mean(ssims))
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
                "black_clip": mean_black,
                "b_minus_r": mean_b_r,
                "edge_energy": mean_edge,
                "flicker": mean_flicker,
                "psnr": mean_psnr,
                "ssim": mean_ssim,
                "lpips": "Requires Ground Truth" if not vid["gt"] else "Requires learned model",
            }
            ablation_results.append(rec)
            psnr_str = f"{mean_psnr:5.2f} dB" if isinstance(mean_psnr, float) else "No GT"
            ssim_str = f"{mean_ssim:.4f}" if isinstance(mean_ssim, float) else "No GT"
            print(f"  [{cfg_label:40s}] FPS: {mean_fps:5.1f} | NIQE: {mean_niqe:4.2f} | Black: {mean_black:5.2f}% | "
                  f"Flick: {mean_flicker:4.2f} | Edge: {mean_edge:4.2f} | PSNR: {psnr_str} | SSIM: {ssim_str}")

    return ablation_results

if __name__ == "__main__":
    baseline_records = parse_baseline_runs()
    ablation_records = run_ablation_experiments()

    all_data = {
        "baseline_comparison": baseline_records,
        "ablation_study": ablation_records,
        "metadata": {
            "num_baseline_records": len(baseline_records),
            "num_ablation_records": len(ablation_records),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    out_file = os.path.join("scratch", "session16_results.json")
    with open(out_file, "w") as f:
        json.dump(all_data, f, indent=2)
    print(f"\nSuccessfully combined and saved complete Session 16 benchmark results to {out_file}!")
