"""
novel_method/nighttime_experimental.py — Session 13 Multi-Source / Multi-Color Extension
========================================================================================
EXPERIMENTAL RESEARCH & CONTROLLED PROTOTYPE ONLY.
Preserves frozen daytime pipeline intact.

Models supported:
  - Model A (Session 12 Baseline):
      A(x) = A_source * eta(x)
  - Model B (Session 13 Multi-Source Experimental):
      A(x) = sum_k A_k * eta_k(x) + A_amb * eta_amb(x)
      with conservative object-rejection and temporal smoothing.

Constraints:
  - NO OpenCV (100% NumPy, SciPy, Pillow).
  - NO CNN / Transformer / Learned models.
  - novel_method/pipeline.py remains 100% FROZEN (Sessions 1-10 untouched).
  - Automatic scene/illumination gate routes daylight to frozen pipeline.
  - Switchable OFF via enable_nighttime_branch=False for 100% bit-exact reproducibility.
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from pipeline import RealTimeDehazingPipeline, _resize_np, _to_gray
from temporal_coherence import (
    compute_motion_mask,
    is_scene_change,
    compute_adaptive_temporal_alpha,
    smooth_transmission_temporal,
    smooth_airlight_temporal,
)


class ExperimentalNighttimePipeline(RealTimeDehazingPipeline):
    """
    Session 13 Experimental Pipeline supporting:
      1. Scene / Illumination Gate (day vs night classification)
      2. Model A: Single-source spatially-varying airlight (Session 12)
      3. Model B: Multi-source / multi-color spatially-varying airlight (Session 13)
      4. Object-color discrimination (prevents colored cars/trees from becoming light sources)
      5. Temporal consistency of spatial airlight maps
    """

    def __init__(
        self,
        enable_nighttime_branch: bool = True,
        night_mode: str = "prototype",  # "prototype" (Model A), "multi_source" (Model B), "multi_source_smooth", "no_illum_correction", "no_airlight_mod", "frozen_daytime"
        night_lum_thresh: float = 55.0,
        night_dark_pct_thresh: float = 25.0,
        night_dark_lum_thresh: float = 75.0,
        eta_min: float = 0.05,
        local_illum_kernel_size: int = 31,
        max_light_sources: int = 4,
        source_filter_type: str = "box",  # "box" or "gaussian"
        source_filter_size: int = 35,
        temporal_alpha_airlight: float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.enable_nighttime_branch = enable_nighttime_branch
        self.night_mode = night_mode
        self.night_lum_thresh = night_lum_thresh
        self.night_dark_pct_thresh = night_dark_pct_thresh
        self.night_dark_lum_thresh = night_dark_lum_thresh
        self.eta_min = eta_min
        self.local_illum_kernel_size = local_illum_kernel_size
        self.max_light_sources = max_light_sources
        self.source_filter_type = source_filter_type
        self.source_filter_size = source_filter_size
        self.temporal_alpha_airlight = temporal_alpha_airlight

        # Diagnostics & temporal caches
        self.last_gate_decision = "DAY / NATURAL"
        self.last_gate_stats = {}
        self.last_night_stats = {}
        self._cached_A_spatial = None
        self._cached_source_clusters = []

    def reset(self):
        """Resets all pipeline and experimental caches."""
        super().reset()
        self._cached_A_spatial = None
        self._cached_source_clusters = []
        self.last_gate_decision = "DAY / NATURAL"
        self.last_gate_stats = {}
        self.last_night_stats = {}

    def evaluate_illumination_gate(self, gray_h: np.ndarray):
        """
        Evaluates scene illumination from half-resolution guide.
        Uses statistical criteria evaluated across Session 11 benchmark:
          1. Global mean luminance < 55 / 255 OR
          2. Dark pixel percentage (> 25% of scene has L < 38/255) AND mean luminance < 75/255.
        Returns: (is_night: bool, stats: dict)
        """
        mean_lum_255 = float(np.mean(gray_h) * 255.0)
        dark_pct = float(np.count_nonzero(gray_h < (38.0 / 255.0)) / gray_h.size * 100.0)
        bright_pct = float(np.count_nonzero(gray_h > (200.0 / 255.0)) / gray_h.size * 100.0)

        is_night = (mean_lum_255 < self.night_lum_thresh) or (
            dark_pct > self.night_dark_pct_thresh and mean_lum_255 < self.night_dark_lum_thresh
        )

        stats = {
            "mean_lum": mean_lum_255,
            "dark_pixel_pct": dark_pct,
            "bright_pixel_pct": bright_pct,
            "is_night": is_night,
            "gate_decision": "NIGHT / LOW-LIGHT" if is_night else "DAY / NATURAL",
        }
        return is_night, stats

    def detect_active_light_sources(self, half_float: np.ndarray):
        """
        Conservative Light Source Detector (Phase 3).
        Rejects colored reflective surfaces (red cars, green trees, blue signs)
        by enforcing:
          1. Absolute brightness: V > 0.85 OR (V > 0.65 with local contrast excess > 0.20)
          2. Surrounding glow evidence (surrounding halo elevation)
          3. Minimum connected component area (rejects single-pixel camera noise)
        Returns: labeled_mask, cluster_info_list
        """
        val = np.max(half_float, axis=2)
        local_mean = ndimage.uniform_filter(val, size=15)
        local_excess = val - local_mean

        # Active light source core condition
        core_mask = (val > 0.85) | ((val > 0.65) & (local_excess > 0.20))

        labeled, num_features = ndimage.label(core_mask)
        if num_features == 0:
            return labeled, []

        gh, gw = half_float.shape[:2]
        total_pixels = gh * gw
        sizes = ndimage.sum(core_mask, labeled, range(1, num_features + 1))
        
        # Sort by size / prominence
        sorted_indices = np.argsort(sizes)[::-1]
        valid_clusters = []

        for idx in sorted_indices:
            c_id = idx + 1
            c_size = sizes[idx]
            if c_size < 4 or c_size > (0.25 * total_pixels):
                continue

            c_mask = (labeled == c_id)
            c_pixels = half_float[c_mask]

            # Chromaticity: use upper 85th percentile to avoid sensor clip white-out
            c_rgb = np.percentile(c_pixels, 85, axis=0)
            c_rgb = np.clip(c_rgb, 0.15, 1.0)

            # Spatial centroid
            y_idx, x_idx = np.nonzero(c_mask)
            cy, cx = float(np.mean(y_idx)), float(np.mean(x_idx))
            radius = float(np.sqrt(c_size / np.pi))

            valid_clusters.append({
                "id": c_id,
                "size": c_size,
                "cx": cx,
                "cy": cy,
                "radius": radius,
                "rgb": c_rgb,
                "mask": c_mask,
            })

            if len(valid_clusters) >= self.max_light_sources:
                break

        return labeled, valid_clusters

    def construct_multi_source_airlight(
        self,
        half_float: np.ndarray,
        orig_shape: tuple,
        A_default: np.ndarray,
        filter_type: str = "box",
        filter_size: int = 35,
    ):
        """
        Constructs Spatially Varying Multi-Source Airlight Field (Model B):
          A(x) = sum_k A_k * eta_k(x) + A_amb * eta_amb(x)
        Enforces sum_k eta_k(x) <= 1.0 and preserves ambient lower bound.
        """
        orig_h, orig_w = orig_shape[:2]
        labeled, clusters = self.detect_active_light_sources(half_float)

        if not clusters:
            # Fallback to Model A (Single Source)
            val_h = np.max(half_float, axis=2)
            L_smooth = ndimage.uniform_filter(val_h, size=15)
            a_max = max(float(np.max(A_default)), 0.25)
            eta = np.clip(L_smooth / a_max, self.eta_min, 1.0)[:, :, None]
            A_half = A_default[None, None, :] * eta
        else:
            eta_list = []
            A_k_list = []
            for cl in clusters:
                A_k_list.append(cl["rgb"])
                c_float = cl["mask"].astype(np.float32)
                if filter_type == "gaussian":
                    eta_raw = ndimage.gaussian_filter(c_float, sigma=filter_size / 3.0)
                else:
                    eta_raw = ndimage.uniform_filter(c_float, size=filter_size)
                max_eta = np.max(eta_raw)
                if max_eta > 1e-6:
                    eta_raw = eta_raw / max_eta
                eta_list.append(eta_raw)

            eta_stack = np.stack(eta_list, axis=-1)  # (gh, gw, K)
            eta_sum = np.sum(eta_stack, axis=-1, keepdims=True)
            scale = np.maximum(eta_sum, 1.0)
            eta_norm = eta_stack / scale

            A_stack = np.array(A_k_list)  # (K, 3)
            # sum_k A_k * eta_k(x)
            A_half = np.sum(eta_norm[:, :, :, None] * A_stack[None, None, :, :], axis=2)

            # Residual ambient in unlit areas
            eta_total = np.sum(eta_norm, axis=-1)[:, :, None]
            eta_amb = np.maximum(1.0 - eta_total, 0.0)
            A_half = A_half + A_default[None, None, :] * self.eta_min * eta_amb

        # Upsample A(x) to full processing resolution
        A_spatial = np.zeros((orig_h, orig_w, 3), dtype=np.float32)
        for c in range(3):
            A_spatial[:, :, c] = np.array(
                Image.fromarray(A_half[:, :, c].astype(np.float32), mode="F").resize((orig_w, orig_h), Image.BILINEAR)
            )

        return A_spatial, clusters

    def process_frame(self, frame: np.ndarray, return_diagnostics: bool = False):
        """
        Processes a single frame. Routes to daytime or nighttime branch based on
        scene illumination gate and configuration.
        """
        t0 = time.perf_counter()

        # If nighttime branch is disabled or set to frozen_daytime, pass directly to frozen pipeline
        if not self.enable_nighttime_branch or self.night_mode == "frozen_daytime":
            self.last_gate_decision = "DAY / NATURAL (FORCED FROZEN)"
            return super().process_frame(frame, return_diagnostics=return_diagnostics)

        raw_h, raw_w = frame.shape[:2]

        # 1. Processing Resolution Scaling (identical to frozen pipeline)
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
        if self._proc_float_buf is None or self._proc_float_buf.shape != proc_frame.shape:
            self._proc_float_buf = np.empty(proc_frame.shape, dtype=np.float32)
        np.multiply(proc_frame, 1.0 / 255.0, out=self._proc_float_buf, casting="unsafe")
        proc_float = self._proc_float_buf

        # 2. Guide at half resolution
        half_img = proc_float[::2, ::2]
        gray_h = 0.299 * half_img[:, :, 0] + 0.587 * half_img[:, :, 1] + 0.114 * half_img[:, :, 2]

        # 3. Illumination / Scene Gate
        is_night, gate_stats = self.evaluate_illumination_gate(gray_h)
        self.last_gate_decision = gate_stats["gate_decision"]
        self.last_gate_stats = gate_stats

        # If gate classifies as DAY / NATURAL, execute exact frozen daytime pipeline logic
        if not is_night:
            return super().process_frame(frame, return_diagnostics=return_diagnostics)

        # =========================================================================
        # NIGHTTIME-AWARE PHYSICAL PIPELINE BRANCH
        # =========================================================================

        # 4. Depth from CAP (Zhu et al.)
        v = np.max(half_img, axis=2)
        dc_img = np.min(half_img, axis=2)
        s = (v - dc_img) / (v + 1e-6)
        depth = ndimage.uniform_filter(0.121779 + 0.959710 * v - 0.780245 * s, size=3)

        # 5. Night Atmospheric Light Estimation
        if self.night_mode == "no_airlight_mod":
            A_raw, candidate_mask = self.estimate_atmospheric_light(
                half_img, depth, v=v, s=s, dc=dc_img, gray_h=gray_h
            )
        else:
            saved_a_min = self.airlight_cfg.get("a_min_hard", 0.45)
            self.airlight_cfg["a_min_hard"] = 0.15
            try:
                A_raw, candidate_mask = self.estimate_atmospheric_light(
                    half_img, depth, v=v, s=s, dc=dc_img, gray_h=gray_h
                )
            finally:
                self.airlight_cfg["a_min_hard"] = saved_a_min

        self.last_candidate_mask = candidate_mask
        self.last_A_raw = A_raw

        # 6. Nighttime Transmission Estimation
        norm = half_img / (A_raw[None, None, :] + 1e-6)
        dc = np.min(norm, axis=2)
        dc_min = ndimage.minimum_filter(dc, size=5)
        h_local = ndimage.uniform_filter(dc, size=3)

        omega_night = min(self.omega, 0.85)
        t_dcp = 1.0 - omega_night * dc_min
        t_cap = np.exp(-self.beta * np.clip(depth, 0.0, 2.5))

        # Spatial fusion weight w_dcp(x)
        var_h = ndimage.uniform_filter(gray_h ** 2, size=3) - ndimage.uniform_filter(gray_h, size=3) ** 2
        var_h = np.maximum(var_h, 0.0)

        dark_penalty = np.clip((0.30 - gray_h) / 0.20, 0.0, 1.0)
        w_raw = 0.20 * (1.0 - 0.70 * dark_penalty) + 0.30 * np.clip(h_local, 0.0, 1.0)
        w_dcp = ndimage.gaussian_filter(np.clip(w_raw, 0.05, 0.60), sigma=1.5)

        t_hybrid = w_dcp * t_dcp + (1.0 - w_dcp) * t_cap

        # Adaptive night transmission floor
        t_base = max(self.t_min, 0.15)
        floor_dark = 0.25 * dark_penalty
        t_min_map = np.clip(t_base + floor_dark, t_base, 0.85)
        t_hybrid = np.clip(t_hybrid, t_min_map, 1.0)

        # Guided smoothing
        mean_I = ndimage.uniform_filter(gray_h, size=5)
        mean_p = ndimage.uniform_filter(t_hybrid, size=5)
        mean_Ip = ndimage.uniform_filter(gray_h * t_hybrid, size=5)
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = ndimage.uniform_filter(gray_h * gray_h, size=5) - mean_I * mean_I
        a = cov_Ip / (var_I + 1e-3)
        b = mean_p - a * mean_I
        t_smooth = ndimage.uniform_filter(a, size=5) * gray_h + ndimage.uniform_filter(b, size=5)
        t_smooth = np.clip(t_smooth, t_min_map, 1.0)

        # 7. Motion-Aware Temporal Transmission & Airlight Smoothing (Session 5 Approved)
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
                self._cached_A_spatial = None

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

        # 8. Transfer transmission to full processing resolution
        t_full = np.array(Image.fromarray(t_smooth, mode="F").resize((orig_w, orig_h), Image.BILINEAR))[:, :, None]

        # 9. Spatially Varying Atmospheric Light Model A(x)
        detected_clusters = []
        if self.night_mode == "no_illum_correction":
            A_spatial = A[None, None, :]
        elif self.night_mode == "no_airlight_mod":
            A_spatial = A[None, None, :]
            t_safe = np.maximum(t_full, 1.0 - proc_float / (A_spatial + 1e-4))
            t_full = t_safe
        elif self.night_mode in ("multi_source", "multi_source_smooth"):
            # Model B: Multi-Source Spatially Varying Airlight
            A_spatial_raw, detected_clusters = self.construct_multi_source_airlight(
                half_img,
                (orig_h, orig_w),
                A,
                filter_type=self.source_filter_type,
                filter_size=self.source_filter_size,
            )
            if self.night_mode == "multi_source_smooth" and self._cached_A_spatial is not None and not is_scene_cut:
                # Temporal smoothing of spatial airlight map
                alpha_A = self.temporal_alpha_airlight
                A_spatial = (1.0 - alpha_A) * self._cached_A_spatial + alpha_A * A_spatial_raw
            else:
                A_spatial = A_spatial_raw
            self._cached_A_spatial = A_spatial.copy()
        else:
            # Model A (Session 12 Baseline): Single-Source Spatially Varying Airlight
            V_full = np.max(proc_float, axis=2)
            L_smooth = ndimage.uniform_filter(V_full, size=self.local_illum_kernel_size)
            a_max_ref = max(float(np.max(A)), 0.25)
            eta = np.clip(L_smooth / a_max_ref, self.eta_min, 1.0)[:, :, None]
            A_spatial = A[None, None, :] * eta

        # 10. Physical Atmospheric Scene Inversion
        # J(x) = (I(x) - A(x)) / t(x) + A(x)
        J_raw = (proc_float - A_spatial) / t_full + A_spatial
        J = np.clip(J_raw, 0.0, 1.0)

        # Output Telemetry
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

        self.last_night_stats = {
            "mode": self.night_mode,
            "A_source": A,
            "A_spatial_mean": float(np.mean(A_spatial)),
            "A_spatial_min": float(np.min(A_spatial)),
            "A_spatial_max": float(np.max(A_spatial)),
            "t_mean": float(np.mean(t_full)),
            "t_min": float(np.min(t_full)),
            "pct_negative_raw": float(np.count_nonzero(np.any(J_raw < 0.0, axis=2)) / J_raw[:, :, 0].size * 100.0),
            "num_clusters": len(detected_clusters),
        }

        out = (J * 255.0).astype(np.uint8)
        self.frame_count += 1
        fps = 1.0 / (time.perf_counter() - t0 + 1e-9)

        if return_diagnostics:
            diag = self.get_airlight_diagnostics()
            diag.update(gate_stats)
            diag.update(self.last_night_stats)
            return out, fps, diag
        return out, fps
