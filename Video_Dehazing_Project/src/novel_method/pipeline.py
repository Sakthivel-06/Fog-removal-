"""
novel_method/pipeline.py — Novel Real-Time Video Dehazing Pipeline
==================================================================
Based on:
  - Base Paper: Kaplan (2023) "Real-world image dehazing with improved joint enhancement and exposure fusion"
    (Smoothing-Sharpening Image Filter SSIF + Adaptive Dynamic Range Expansion)
  - Color Attenuation Prior (Zhu et al. 2015) & Dark Channel Prior (He et al. 2009)
  - Spatially Adaptive Density-Weighted Fusion for Non-Uniform / Patchy Fog
  - Transmission-Level Temporal Coherence for Flicker-Free Video with Zero Ghosting

No OpenCV dependency. 100% NumPy + SciPy + Pillow.
Optimized for real-time CPU performance (25–30+ FPS) with maximum fog penetration.
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "algorithms")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from temporal_coherence import (
    compute_motion_mask,
    is_scene_change,
    compute_adaptive_temporal_alpha,
    smooth_transmission_temporal,
    smooth_airlight_temporal,
)


def _resize_np(img_np, target_wh):
    return np.array(Image.fromarray(img_np).resize(target_wh, Image.BILINEAR))


def _to_gray(img):
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def load_pipeline_config(config_path=None):
    """
    Loads YAML configuration from config/*.yaml.
    Supports PyYAML if available, with a dependency-free built-in parser fallback.
    """
    default_config = {
        "fusion": {
            "sky_lum_thresh": 0.55,
            "sky_lum_range": 0.25,
            "sky_var_limit": 0.0005,
            "bright_lum_thresh": 0.75,
            "bright_lum_range": 0.15,
            "texture_var_scale": 0.0008,
            "dark_lum_thresh": 0.32,
            "dark_lum_range": 0.20,
            "haze_lum_low_thresh": 0.20,
            "haze_lum_low_range": 0.12,
            "haze_lum_high_thresh": 0.82,
            "haze_lum_high_range": 0.15,
            "haze_dc_thresh": 0.15,
            "haze_dc_range": 0.35,
            "w_base": 0.15,
            "w_haze_scale": 0.60,
            "w_sky_penalty": 0.85,
            "w_dark_penalty": 0.55,
            "w_text_base": 0.60,
            "w_text_scale": 0.40,
            "w_min": 0.05,
            "w_max": 0.75,
            "w_smooth_sigma": 1.5,
        },
        "airlight": {
            "highlight_thresh": 0.95,
            "highlight_max_channel": 0.96,
            "blue_sky_diff_thresh": 0.16,
            "blue_sky_lum_thresh": 0.50,
            "upper_sky_lum_thresh": 0.65,
            "upper_sky_var_limit": 0.0003,
            "upper_sky_y_max": 0.28,
            "vehicle_lum_thresh": 0.60,
            "vehicle_var_min": 0.0010,
            "ground_y_min": 0.65,
            "top_candidate_ratio": 0.001,
            "min_candidates": 30,
            "a_min": 0.65,
            "a_max": 0.94,
        },
        "temporal": {
            "alpha_static": 0.15,
            "alpha_motion": 0.90,
            "alpha_airlight": 0.08,
            "motion_thresh_low": 0.03,
            "motion_thresh_high": 0.15,
            "motion_filter_size": 3,
            "scene_cut_thresh": 0.20,
            "scene_cut_motion_pct": 60.0,
        }
    }

    if config_path is None:
        candidates = [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "pipeline_config.yaml")),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "config", "pipeline_config.yaml")),
            os.path.abspath("config/pipeline_config.yaml"),
        ]
        for c in candidates:
            if os.path.exists(c):
                config_path = c
                break

    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    for sec, vals in loaded.items():
                        if isinstance(vals, dict):
                            default_config.setdefault(sec, {}).update(vals)
            return default_config
        except ImportError:
            try:
                cur_sec = None
                with open(config_path, "r") as f:
                    for line in f:
                        line = line.split("#")[0].strip()
                        if not line:
                            continue
                        if line.endswith(":") and not ":" in line[:-1]:
                            cur_sec = line[:-1].strip()
                            default_config.setdefault(cur_sec, {})
                        elif ":" in line:
                            k, v = line.split(":", 1)
                            k, v = k.strip(), v.strip()
                            try:
                                val = float(v) if "." in v else int(v)
                            except ValueError:
                                if v.lower() == "true":
                                    val = True
                                elif v.lower() == "false":
                                    val = False
                                else:
                                    val = v.strip("'\"")
                            if cur_sec:
                                default_config[cur_sec][k] = val
                            else:
                                default_config[k] = val
            except Exception:
                pass
    return default_config


class RealTimeDehazingPipeline:
    """
    Novel Real-Time Video Dehazing Pipeline.
    Combines Color Attenuation Prior (CAP) with Dark Channel Prior (DCP),
    spatially adaptive density-weighted fusion for non-uniform fog,
    pure physical atmospheric inversion with clamped dynamic range (Session 7),
    flat-sky protection, and transmission-level temporal coherence for clear,
    flicker-free video.
    """

    def __init__(
        self,
        downscale_res=(240, 135),
        omega=0.90,
        beta=1.20,
        t_min=0.07,
        kappa=1.25,
        temporal_alpha=0.30,
        enable_sky_model=True,
        preserve_resolution=True,
        max_process_res=(426, 240),
        **kwargs,
    ):
        self.downscale_res = downscale_res
        self.omega = omega
        self.beta = beta
        self.t_min = t_min
        self.kappa = kappa
        self.temporal_alpha = temporal_alpha
        self.enable_sky_model = enable_sky_model
        self.preserve_resolution = preserve_resolution
        self.max_process_res = max_process_res
        self.use_sky_cnn = kwargs.get("use_sky_cnn", False)

        # Load configuration from config/*.yaml
        self.config = load_pipeline_config(kwargs.get("config_path", None))
        self.fusion_cfg = self.config.get("fusion", {})
        self.airlight_cfg = self.config.get("airlight", {})
        self.temporal_cfg = self.config.get("temporal", {})
        self.last_candidate_mask = None
        self.last_A_raw = None
        self.last_temporal_diagnostics = {}

        # Deep Learning Sky Segmentation CNN (SkySegNet: ~24k params)
        self.sky_net = None
        if self.use_sky_cnn or kwargs.get("sky_cnn", False):
            try:
                import torch
                from sky_model.model import SkySegNet
                weight_path = os.path.join(os.path.dirname(__file__), "sky_model", "weights", "sky_seg_net.pth")
                if os.path.exists(weight_path):
                    self.sky_net = SkySegNet().to("cpu")
                    self.sky_net.load_state_dict(torch.load(weight_path, map_location="cpu"))
                    self.sky_net.eval()
            except Exception:
                self.sky_net = None

        self.frame_count = 0
        self._cached_t = None
        self._cached_A = None
        self._cached_sky_mask = None
        self._prev_gray = None
        self._prev_out_lum = None
        self._proc_float_buf = None

    def reset(self):
        """Resets temporal caches for scene cuts or new video."""
        self._cached_t = None
        self._cached_A = None
        self._prev_gray = None
        self._prev_out_lum = None
        self._proc_float_buf = None
        self.frame_count = 0
        self.last_candidate_mask = None
        self.last_A_raw = None
        self.last_A = None
        self.last_A_est_unclipped = None
        self._last_gray_h = None
        self.last_is_sky = None
        self.last_is_highlight = None
        self.last_temporal_diagnostics = {}
        self._cached_y_coords = None

    def estimate_atmospheric_light(
        self,
        half_img: np.ndarray,
        depth: np.ndarray,
        v: np.ndarray = None,
        s: np.ndarray = None,
        dc: np.ndarray = None,
        gray_h: np.ndarray = None,
    ):
        """
        Refined Atmospheric Light (A) Estimation (Session 3 / 9 Optimized).
        1. Considers brightness, haze confidence, local variance, saturation, depth.
        2. Explicitly excludes sky, snow, bright vehicles, headlights, specular highlights.
        3. Selects A from top ~0.1% candidates in filtered pool.
        4. Stores boolean candidate mask in self.last_candidate_mask for diagnostics.
        """
        cfg = self.airlight_cfg
        gh, gw = half_img.shape[:2]

        if gray_h is None:
            gray_h = 0.299 * half_img[:, :, 0] + 0.587 * half_img[:, :, 1] + 0.114 * half_img[:, :, 2]

        mean_gray = ndimage.uniform_filter(gray_h, size=5)
        var_h = np.maximum(ndimage.uniform_filter(gray_h * gray_h, size=5) - mean_gray * mean_gray, 0.0)

        if v is None:
            v = np.max(half_img, axis=2)
        if dc is None:
            dc = np.min(half_img, axis=2)
        if s is None:
            s = (v - dc) / (v + 1e-6)

        if not hasattr(self, "_cached_y_coords") or self._cached_y_coords is None or self._cached_y_coords.shape[0] != gh:
            self._cached_y_coords = np.linspace(0.0, 1.0, gh, dtype=np.float32)[:, None]
        y_coords = self._cached_y_coords

        # 1. Candidate Pool Exclusions
        hl_thresh = min(cfg.get("highlight_thresh", 0.95), cfg.get("highlight_max_channel", 0.96))
        is_highlight = v > hl_thresh
        is_blue_sky = (half_img[:, :, 2] - half_img[:, :, 0] > cfg.get("blue_sky_diff_thresh", 0.16)) & (gray_h > cfg.get("blue_sky_lum_thresh", 0.50))
        is_upper_sky = (gray_h > cfg.get("upper_sky_lum_thresh", 0.65)) & (var_h < cfg.get("upper_sky_var_limit", 0.0003)) & (y_coords < cfg.get("upper_sky_y_max", 0.28))
        is_sky = is_blue_sky | is_upper_sky
        is_vehicle_sign = (gray_h > cfg.get("vehicle_lum_thresh", 0.60)) & (var_h > cfg.get("vehicle_var_min", 0.0010))
        is_ground = (y_coords > cfg.get("ground_y_min", 0.65))

        # Strict exclusion mask that must NEVER be relaxed:
        invalid_mask = is_highlight | is_sky

        # Primary pool includes additional geometric/semantic soft exclusions:
        valid_pool = (~invalid_mask) & (~is_vehicle_sign) & (~is_ground)
        min_cand = cfg.get("min_candidates", 30)
        cand_ratio = cfg.get("top_candidate_ratio", 0.001)
        target_pool_size = max(int(gh * gw * 0.002), min_cand)

        # Robust multi-stage fallback: strictly preserves sky and highlight exclusions at all stages
        if np.count_nonzero(valid_pool) < target_pool_size:
            # Stage 1: relax ground boundary to y < 0.80
            valid_pool = (~invalid_mask) & (~is_vehicle_sign) & (y_coords < 0.80)
            if np.count_nonzero(valid_pool) < target_pool_size:
                # Stage 2: relax vehicle texture filter
                valid_pool = (~invalid_mask) & (y_coords < 0.80)
                if np.count_nonzero(valid_pool) < target_pool_size:
                    # Final safety fallback: strictly preserve sky and highlight exclusions
                    valid_pool = (~invalid_mask)

        # 2. Candidate Scoring in Filtered Pool
        score_dc = np.clip(dc / 0.75, 0.0, 1.0)
        score_bright = 1.0 - np.clip(np.abs(gray_h - 0.78) / 0.30, 0.0, 1.0)
        score_smooth = 1.0 - np.clip(var_h / 0.0012, 0.0, 1.0)
        score_neutral = 1.0 - np.clip(s / 0.25, 0.0, 1.0)
        score_depth = np.clip(depth / 1.5, 0.0, 1.0)

        candidate_score = (
            0.40 * score_dc +
            0.25 * score_bright +
            0.15 * score_smooth +
            0.10 * score_neutral +
            0.10 * score_depth
        )
        candidate_score[~valid_pool] = -1.0

        ntop = max(int(gh * gw * cand_ratio), min_cand)
        idx = np.argpartition(candidate_score.ravel(), -ntop)[-ntop:]
        idx = idx[candidate_score.ravel()[idx] >= 0]
        if len(idx) == 0:
            idx = np.argpartition(candidate_score.ravel(), -ntop)[-ntop:]

        selected_pixels = half_img.reshape(-1, 3)[idx]
        if len(selected_pixels) >= 10:
            p10, p90 = np.percentile(selected_pixels, (10, 90), axis=0)
            inliers = np.all((selected_pixels >= p10) & (selected_pixels <= p90), axis=1)
            if np.any(inliers):
                A_est = np.mean(selected_pixels[inliers], axis=0)
            else:
                A_est = np.median(selected_pixels, axis=0)
        else:
            A_est = np.median(selected_pixels, axis=0)

        # Data-driven chromaticity-preserving lower bound rule (Session 3B)
        a_min_hard = cfg.get("a_min_hard", cfg.get("a_min", 0.45))
        a_max = cfg.get("a_max", 0.94)

        # Candidate distribution luminance
        L_cand = 0.299 * A_est[0] + 0.587 * A_est[1] + 0.114 * A_est[2]

        # Chromaticity-preserving bounding: scales vector uniformly if below hard safety floor
        # or above maximum bound, avoiding artificial distortion of cool/neutral haze ratios
        if L_cand < a_min_hard:
            scale = a_min_hard / (L_cand + 1e-6)
            A_raw = A_est * scale
        elif L_cand > a_max:
            scale = a_max / (L_cand + 1e-6)
            A_raw = A_est * scale
        else:
            A_raw = A_est.copy()

        # Final fail-safe per-channel clamping
        A_raw = np.clip(A_raw, a_min_hard, a_max)

        candidate_mask = np.zeros((gh, gw), dtype=bool)
        candidate_mask.ravel()[idx] = True

        self.last_A_est_unclipped = A_est.copy()
        self._last_gray_h = gray_h
        self.last_is_sky = is_sky
        self.last_is_highlight = is_highlight

        return A_raw, candidate_mask

    def process_frame(self, frame: np.ndarray, return_diagnostics: bool = False):
        """
        Dehazes a single RGB uint8 frame with deep fog penetration and 25-30+ FPS real-time speed.
        Returns: (dehazed_frame_uint8, fps) or (dehazed_frame_uint8, fps, diagnostics) if return_diagnostics=True
        """
        t0 = time.perf_counter()
        raw_h, raw_w = frame.shape[:2]

        # 1. Processing Resolution Scaling (Optimized Session 10: aspect-aware striding & single-pass float conversion)
        if self.max_process_res is not None:
            max_w, max_h = self.max_process_res
            if raw_w > max_w or raw_h > max_h:
                # Aspect-ratio-aware striding: ensure intermediate sub_f is downscaled proportionally
                # without exceeding stride 4 to preserve 2x bilinear oversampling and prevent aliasing.
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
        if self._proc_float_buf is None or self._proc_float_buf.shape != proc_frame.shape:
            self._proc_float_buf = np.empty(proc_frame.shape, dtype=np.float32)
        np.multiply(proc_frame, 1.0 / 255.0, out=self._proc_float_buf, casting='unsafe')
        proc_float = self._proc_float_buf

        # 2. Guide at half resolution for ultra-fast physics parameter estimation
        half_img = proc_float[::2, ::2]
        gh, gw = half_img.shape[:2]
        gray_h = 0.299 * half_img[:, :, 0] + 0.587 * half_img[:, :, 1] + 0.114 * half_img[:, :, 2]

        # 3. Depth from CAP (Zhu et al.)
        v = np.max(half_img, axis=2)
        dc_img = np.min(half_img, axis=2)
        s = (v - dc_img) / (v + 1e-6)
        depth = ndimage.uniform_filter(0.121779 + 0.959710 * v - 0.780245 * s, size=3)

        # 4. Atmospheric Light A (Refined Candidate Scoring & Explicit Exclusion - Session 3 & 9)
        A_raw, candidate_mask = self.estimate_atmospheric_light(
            half_img, depth, v=v, s=s, dc=dc_img, gray_h=gray_h
        )
        self.last_candidate_mask = candidate_mask
        self.last_A_raw = A_raw

        # 5. Deep Fog Penetration Transmission
        norm = half_img / (A_raw[None, None, :] + 1e-6)
        dc = np.min(norm, axis=2)
        dc_min = ndimage.minimum_filter(dc, size=5)
        h_local = ndimage.uniform_filter(dc, size=3)

        # Adaptive omega: in dense fog pockets, boost omega up to 0.96 for aggressive fog cutting
        omega_map = np.clip(self.omega + 0.08 * h_local, self.omega, 0.96)
        t_dcp = 1.0 - omega_map * dc_min

        # Physical depth transmission
        t_cap = np.exp(-self.beta * np.clip(depth, 0.0, 2.5))

        # 5b. Spatially-Varying DCP/CAP Fusion Weight w_dcp(x) from config/*.yaml
        cfg = self.fusion_cfg
        var_h = ndimage.uniform_filter(gray_h ** 2, size=3) - ndimage.uniform_filter(gray_h, size=3) ** 2
        var_h = np.maximum(var_h, 0.0)

        # 1. Sky / bright-region mask
        b_factor = np.clip((gray_h - cfg.get("sky_lum_thresh", 0.55)) / cfg.get("sky_lum_range", 0.25), 0.0, 1.0)
        s_factor = np.clip(1.0 - var_h / cfg.get("sky_var_limit", 0.0005), 0.0, 1.0)
        bright_term = np.clip((gray_h - cfg.get("bright_lum_thresh", 0.75)) / cfg.get("bright_lum_range", 0.15), 0.0, 1.0)
        m_sky = np.clip(np.maximum(b_factor * s_factor, bright_term), 0.0, 1.0)

        # 2. Local contrast / texture confidence
        text_conf = np.clip(var_h / cfg.get("texture_var_scale", 0.0008), 0.0, 1.0)

        # 3. Dark region penalty: prevents dark asphalt from being falsely interpreted as haze
        dark_penalty = np.clip((cfg.get("dark_lum_thresh", 0.32) - gray_h) / cfg.get("dark_lum_range", 0.20), 0.0, 1.0)

        # 4. Haze density evidence H(x): elevated dark channel in mid-tone foggy range
        haze_lum_gate = (
            np.clip((gray_h - cfg.get("haze_lum_low_thresh", 0.20)) / cfg.get("haze_lum_low_range", 0.12), 0.0, 1.0)
            * np.clip((cfg.get("haze_lum_high_thresh", 0.82) - gray_h) / cfg.get("haze_lum_high_range", 0.15), 0.0, 1.0)
        )
        H_x = np.clip((h_local - cfg.get("haze_dc_thresh", 0.15)) / cfg.get("haze_dc_range", 0.35), 0.0, 1.0) * haze_lum_gate

        # 5. Composite spatially-varying DCP weight (bounded, smoothly filtered to avoid halos)
        w_raw = (
            (cfg.get("w_base", 0.15) + cfg.get("w_haze_scale", 0.60) * H_x)
            * (1.0 - cfg.get("w_sky_penalty", 0.85) * m_sky)
            * (1.0 - cfg.get("w_dark_penalty", 0.55) * dark_penalty)
            * (cfg.get("w_text_base", 0.60) + cfg.get("w_text_scale", 0.40) * text_conf)
        )
        w_min = cfg.get("w_min", 0.05)
        w_max = cfg.get("w_max", 0.75)
        w_sigma = cfg.get("w_smooth_sigma", 1.5)
        w_dcp = ndimage.gaussian_filter(np.clip(w_raw, w_min, w_max), sigma=w_sigma)
        w_dcp = np.clip(w_dcp, w_min, w_max)

        t_hybrid = w_dcp * t_dcp + (1.0 - w_dcp) * t_cap

        # 6. Flat-Sky Protection & Spatially Adaptive Transmission Floor (Session 1 Approved)

        if self.enable_sky_model:
            is_sky = np.clip((gray_h - 0.72) / 0.18, 0.0, 1.0) * np.clip(1.0 - var_h / 0.0012, 0.0, 1.0)
            t_hybrid = t_hybrid * (1.0 - is_sky) + np.maximum(t_hybrid, 0.65) * is_sky

        # Spatially Adaptive Transmission Floor t_min(x):
        # Replaces single global self.t_min to prevent black crush on roads and dead/dark sky
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

        # 7. Edge-Preserving Guided Smoothing
        mean_I = ndimage.uniform_filter(gray_h, size=5)
        mean_p = ndimage.uniform_filter(t_hybrid, size=5)
        mean_Ip = ndimage.uniform_filter(gray_h * t_hybrid, size=5)
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = ndimage.uniform_filter(gray_h * gray_h, size=5) - mean_I * mean_I
        a = cov_Ip / (var_I + 1e-3)
        b = mean_p - a * mean_I
        t_smooth = ndimage.uniform_filter(a, size=5) * gray_h + ndimage.uniform_filter(b, size=5)
        t_smooth = np.clip(t_smooth, t_min_map, 1.0)

        # 8. Motion-Aware Temporal Transmission & Airlight Smoothing (Session 5)
        # Prevents temporal flicker in static regions while avoiding ghost trails around moving objects and across scene cuts.
        t_cfg = self.temporal_cfg
        alpha_static = t_cfg.get("alpha_static", 0.15)
        alpha_motion = t_cfg.get("alpha_motion", 0.90)
        alpha_airlight = t_cfg.get("alpha_airlight", 0.08)
        motion_thresh_low = t_cfg.get("motion_thresh_low", 0.03)
        motion_thresh_high = t_cfg.get("motion_thresh_high", 0.15)
        motion_filter_size = int(t_cfg.get("motion_filter_size", 3))
        scene_cut_thresh = t_cfg.get("scene_cut_thresh", 0.20)
        scene_cut_motion_pct = t_cfg.get("scene_cut_motion_pct", 60.0)

        is_scene_cut = False
        mean_diff = 0.0
        high_motion_pct = 0.0
        motion_mask = None
        alpha_map = None
        t_before_temporal = t_smooth.copy()

        if self._prev_gray is not None and self._prev_gray.shape == gray_h.shape:
            motion_mask, mean_diff, high_motion_pct = compute_motion_mask(
                gray_h,
                self._prev_gray,
                motion_thresh_low=motion_thresh_low,
                motion_thresh_high=motion_thresh_high,
                filter_size=motion_filter_size,
            )
            is_scene_cut = is_scene_change(
                mean_diff,
                high_motion_pct=high_motion_pct,
                change_threshold=scene_cut_thresh,
                high_motion_pct_threshold=scene_cut_motion_pct,
            )

            if is_scene_cut:
                self._cached_t = None
                self._cached_A = None

            if not is_scene_cut and self._cached_t is not None and self._cached_t.shape == t_smooth.shape:
                alpha_map = compute_adaptive_temporal_alpha(
                    motion_mask,
                    alpha_static=alpha_static,
                    alpha_motion=alpha_motion,
                )
                t_smooth = smooth_transmission_temporal(t_smooth, self._cached_t, alpha_map)
                A = smooth_airlight_temporal(A_raw, self._cached_A, alpha_airlight=alpha_airlight, is_scene_cut=False)
            else:
                alpha_map = np.ones_like(t_smooth)
                A = A_raw
        else:
            motion_mask = np.zeros_like(t_smooth)
            alpha_map = np.ones_like(t_smooth)
            A = A_raw

        self._cached_t = t_smooth.copy()
        self._cached_A = A.copy()
        self.last_A = A.copy()
        self._prev_gray = gray_h.copy()

        # 9. Direct Bilinear Transfer to Processing Resolution
        t_full = np.array(Image.fromarray(t_smooth, mode='F').resize((orig_w, orig_h), Image.BILINEAR))[:, :, None]

        # 10. Physical Scene Inversion: J = (I - A) / t + A
        A3 = A[None, None, :]
        J = (proc_float - A3) / t_full + A3

        # 11. Physical Output Value Clamping (Session 6 & 7 Approved)
        # Final post-inversion physical output preserved directly without artificial sharpening or distortion.
        J = np.clip(J, 0.0, 1.0)

        # Inter-frame output luminance variation telemetry
        out_lum = 0.299 * J[:, :, 0] + 0.587 * J[:, :, 1] + 0.114 * J[:, :, 2]
        lum_diff = 0.0
        if self._prev_out_lum is not None and self._prev_out_lum.shape == out_lum.shape:
            lum_diff = float(np.mean(np.abs(out_lum - self._prev_out_lum)))
        self._prev_out_lum = out_lum.copy()

        self.last_temporal_diagnostics = {
            "mean_frame_diff": mean_diff,
            "high_motion_pct": high_motion_pct,
            "is_scene_cut": is_scene_cut,
            "mean_alpha": float(np.mean(alpha_map)),
            "min_alpha": float(np.min(alpha_map)),
            "max_alpha": float(np.max(alpha_map)),
            "t_temporal_delta": float(np.mean(np.abs(t_smooth - t_before_temporal))),
            "output_lum_diff": lum_diff,
            "motion_mask": motion_mask,
            "alpha_map": alpha_map,
        }

        out = (J * 255.0).astype(np.uint8)
        self.frame_count += 1
        fps = 1.0 / (time.perf_counter() - t0 + 1e-9)
        if return_diagnostics:
            return out, fps, self.get_airlight_diagnostics()
        return out, fps

    def get_airlight_diagnostics(self):
        """
        Returns diagnostic dictionary for the most recent frame.
        All metrics explicitly differentiate spatial vertical position from photometric luminance.
        """
        mask = self.last_candidate_mask
        cand_count = int(np.count_nonzero(mask)) if mask is not None else 0

        cand_mean_row_norm = 0.0
        cand_mean_lum = 0.0
        sky_pct = 0.0
        hl_pct = 0.0

        if mask is not None and cand_count > 0:
            gh, gw = mask.shape[:2]
            y_indices, _ = np.nonzero(mask)
            cand_mean_row_norm = float(np.mean(y_indices) / float(gh))
            if hasattr(self, "_last_gray_h") and self._last_gray_h is not None:
                cand_mean_lum = float(np.mean(self._last_gray_h[mask]))
            if hasattr(self, "last_is_sky") and self.last_is_sky is not None:
                sky_pct = float(np.count_nonzero(self.last_is_sky & mask) / cand_count * 100.0)
            if hasattr(self, "last_is_highlight") and self.last_is_highlight is not None:
                hl_pct = float(np.count_nonzero(self.last_is_highlight & mask) / cand_count * 100.0)

        a_min = self.airlight_cfg.get("a_min_hard", self.airlight_cfg.get("a_min", 0.45))
        a_max = self.airlight_cfg.get("a_max", 0.94)
        A_raw = self.last_A_raw
        A_est = getattr(self, "last_A_est_unclipped", A_raw)
        hits_min = bool(np.any(A_est <= a_min)) if A_est is not None else False
        hits_max = bool(np.any(A_est >= a_max)) if A_est is not None else False

        d = {
            "A_est_unclipped": A_est,
            "A_raw": A_raw,
            "A_final": getattr(self, "last_A", A_raw),
            "a_min": a_min,
            "a_max": a_max,
            "hits_a_min": hits_min,
            "hits_a_max": hits_max,
            "candidate_mask": mask,
            "candidate_count": cand_count,
            "candidate_mean_normalized_row": cand_mean_row_norm,
            "candidate_mean_luminance": cand_mean_lum,
            "candidate_sky_percentage": sky_pct,
            "candidate_highlight_percentage": hl_pct,
        }
        if hasattr(self, "last_temporal_diagnostics") and self.last_temporal_diagnostics:
            d.update(self.last_temporal_diagnostics)
        return d

    def get_temporal_diagnostics(self):
        """
        Returns diagnostic dictionary for the temporal consistency stage of the most recent frame.
        """
        return getattr(self, "last_temporal_diagnostics", {})



if __name__ == "__main__":
    import imageio
    print("Testing RealTimeDehazingPipeline on test video...")
    pipeline = RealTimeDehazingPipeline(max_process_res=(426, 240))
    video_path = sys.argv[1] if len(sys.argv) > 1 else "test_videos/foggy_video1.mp4"
    if not os.path.exists(video_path):
        alt = os.path.join(os.path.dirname(__file__), "..", "test_videos", video_path)
        if os.path.exists(alt):
            video_path = alt

    reader = imageio.get_reader(video_path)
    fps_list = []
    for i, frm in enumerate(reader):
        if i >= 30:
            break
        out, fps = pipeline.process_frame(frm)
        fps_list.append(fps)
        print(f"\rFrame {i+1:2d} | Shape: {out.shape} | FPS: {fps:5.1f}", end="", flush=True)
    reader.close()
    if fps_list:
        avg = len(fps_list) / sum(1.0 / f for f in fps_list)
        print(f"\nCompleted 30 frames! Average Throughput: {avg:.1f} FPS")
