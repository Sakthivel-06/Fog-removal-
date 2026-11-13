"""
novel_method/pipeline_learned_experimental.py
=============================================
Session 21: Experimental Dehazing Pipeline with Learned Transmission Residual (Delta-t).

Architecture:
    Existing physics transmission (CAP + DCP + Spatial Fusion)
              ↓
       Tiny-UNet Delta-t ∈ [-0.30, +0.30]
              ↓
     t_candidate = clip(t_fused + Delta-t, 0.05, 1.0)
              ↓
     t_constrained = max(t_candidate, EXISTING_ADAPTIVE_TMIN)   [NON-BYPASS RULE]
              ↓
      Existing guided refinement
              ↓
     Existing motion-aware temporal coherence (Session 5)
              ↓
       Existing physical inversion J = (I - A)/t + A

STRICT CONSTRAINTS:
1. Zero OpenCV (cv2) dependency.
2. novel_method/pipeline.py is 100% frozen and remains untouched.
3. Adaptive t_min(x, y) remains authoritative and cannot be bypassed.
4. Pure CPU real-time execution.
"""

import os
import sys
import time
import numpy as np
import torch
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
from learned_residual.tiny_unet import TinyUNet
from learned_residual.feature_extractor import extract_physics_features


class LearnedResidualDehazingPipeline(RealTimeDehazingPipeline):
    """
    Experimental pipeline augmenting frozen physics baseline with learned Delta-t.
    """
    def __init__(
        self,
        checkpoint_path: str = None,
        enable_learned_residual: bool = True,
        ablation_mode: str = "A4",
        downscale_res=(240, 135),
        max_process_res=(426, 240),
        **kwargs,
    ):
        super().__init__(
            downscale_res=downscale_res,
            max_process_res=max_process_res,
            **kwargs,
        )
        self.ablation_mode = ablation_mode
        if ablation_mode == "A1":
            self.enable_learned_residual = False
            self.enable_inference_temporal = True
        elif ablation_mode in ["A2", "A3"]:
            self.enable_learned_residual = True
            self.enable_inference_temporal = False
        else: # "A4" default
            self.enable_learned_residual = enable_learned_residual
            self.enable_inference_temporal = kwargs.get("enable_inference_temporal", True)

        self.checkpoint_path = checkpoint_path
        self.model = None

        if checkpoint_path and os.path.exists(checkpoint_path):
            self.load_learned_model(checkpoint_path)

        # Diagnostics cache
        self.last_delta_t = None
        self.last_t_fused = None
        self.last_t_candidate = None
        self.last_t_constrained = None
        self.last_t_temporal = None
        self.latency_profile = {}

    def load_learned_model(self, checkpoint_path: str):
        """Loads TinyUNet model checkpoint and uses GPU (CUDA) if available."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TinyUNet().to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        else:
            self.model.load_state_dict(ckpt)
        self.model.eval()
        self.checkpoint_path = checkpoint_path

    def process_frame(self, frame: np.ndarray, return_diagnostics: bool = False, skip_inversion: bool = False):
        """
        Processes single frame with learned residual Delta-t integrated into physics pipeline.
        """
        t_total_start = time.perf_counter()
        prof = {}

        # 1. Processing Resolution Scaling
        t0 = time.perf_counter()
        raw_h, raw_w = frame.shape[:2]
        if self.max_process_res is not None:
            max_w, max_h = self.max_process_res
            if raw_w > max_w or raw_h > max_h:
                stride = min(max(raw_w // max_w, raw_h // max_h, 1), 4)
                sub_f = frame[::stride, ::stride]
                s = min(max_w / float(sub_f.shape[1]), max_h / float(sub_f.shape[0]))
                cur_w = max(2, (int(sub_f.shape[1] * s) // 2) * 2)
                cur_h = max(2, (int(sub_f.shape[0] * s) // 2) * 2)
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
        prof["prep"] = (time.perf_counter() - t0) * 1000.0

        # 2. Physics Feature Extraction at Guide Resolution
        t0 = time.perf_counter()
        features, meta = extract_physics_features(proc_float, self)
        A_raw = meta["A_raw"]
        t_fused = features[3]
        t_min_map = meta["t_min_map"]
        gray_h = meta["gray_h"]
        prof["physics_features"] = (time.perf_counter() - t0) * 1000.0

        self.last_candidate_mask = None
        self.last_A_raw = A_raw
        self.last_t_fused = t_fused.copy()

        # 3. Learned Transmission Residual (Delta-t)
        t0 = time.perf_counter()
        if self.enable_learned_residual and self.model is not None:
            # Guide resolution tensor (1, 8, gh, gw)
            feat_tensor = torch.from_numpy(features).unsqueeze(0).to(getattr(self, 'device', torch.device('cpu')))
            with torch.no_grad():
                delta_t_tensor = self.model(feat_tensor)
            delta_t = delta_t_tensor.squeeze().cpu().numpy()

            # Physical coupling: t_candidate = t_fused + delta_t
            t_candidate = t_fused + delta_t
            t_clamped = np.clip(t_candidate, 0.05, 1.0)

            # Strict Subordination to Existing Adaptive t_min Floor (Non-Bypass Rule)
            t_constrained = np.maximum(t_clamped, t_min_map)
        else:
            delta_t = np.zeros_like(t_fused)
            t_candidate = t_fused.copy()
            t_clamped = np.clip(t_candidate, 0.05, 1.0)
            t_constrained = np.maximum(t_clamped, t_min_map)

        prof["cnn_inference"] = (time.perf_counter() - t0) * 1000.0

        self.last_delta_t = delta_t.copy()
        self.last_t_candidate = t_candidate.copy()
        self.last_t_constrained = t_constrained.copy()

        # 4. Existing Edge-Preserving Guided Smoothing
        t0 = time.perf_counter()
        mean_I = ndimage.uniform_filter(gray_h, size=5)
        mean_p = ndimage.uniform_filter(t_constrained, size=5)
        mean_Ip = ndimage.uniform_filter(gray_h * t_constrained, size=5)
        cov_Ip = mean_Ip - mean_I * mean_p
        var_I = ndimage.uniform_filter(gray_h * gray_h, size=5) - mean_I * mean_I
        a = cov_Ip / (var_I + 1e-3)
        b = mean_p - a * mean_I
        t_smooth = ndimage.uniform_filter(a, size=5) * gray_h + ndimage.uniform_filter(b, size=5)
        t_smooth = np.clip(t_smooth, t_min_map, 1.0)
        prof["guided_smoothing"] = (time.perf_counter() - t0) * 1000.0

        # 5. Existing Motion-Aware Temporal Coherence (Session 5)
        t0 = time.perf_counter()
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
        motion_mask = None
        alpha_map = None

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

            if not is_scene_cut and self.enable_inference_temporal and self._cached_t is not None and self._cached_t.shape == t_smooth.shape:
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
        self.last_t_temporal = t_smooth.copy()
        prof["temporal_coherence"] = (time.perf_counter() - t0) * 1000.0

        # 6. Physical Scene Inversion J = (I - A) / t + A
        t0 = time.perf_counter()
        
        if skip_inversion:
            dehazed_uint8 = np.zeros_like(frame) # Dummy return, ignored by high-res pipeline
            prof["inversion"] = 0.0
        else:
            if getattr(self, "preserve_resolution", False) and (raw_w != orig_w or raw_h != orig_h):
                # Native full-resolution inversion (High-res mode)
                t_full = np.array(Image.fromarray(t_smooth.astype(np.float32), mode="F").resize((raw_w, raw_h), Image.BILINEAR))[:, :, None]
                A3 = A[None, None, :]
                frame_float = frame.astype(np.float32) * (1.0 / 255.0)
                J = (frame_float - A3) / t_full + A3
                J = np.clip(J, 0.0, 1.0)
                dehazed_uint8 = (J * 255.0).astype(np.uint8)
            else:
                # Fast real-time inversion at processing resolution (Matches A1 speed: 60-80+ FPS)
                t_proc = np.array(Image.fromarray(t_smooth.astype(np.float32), mode="F").resize((orig_w, orig_h), Image.BILINEAR))[:, :, None]
                A3 = A[None, None, :]
                J = (proc_float - A3) / t_proc + A3
                J = np.clip(J, 0.0, 1.0)
                dehazed_uint8 = (J * 255.0).astype(np.uint8)
            prof["inversion"] = (time.perf_counter() - t0) * 1000.0

        total_frame_ms = (time.perf_counter() - t_total_start) * 1000.0
        fps = 1000.0 / max(total_frame_ms, 1e-3)
        prof["total_ms"] = total_frame_ms
        prof["fps"] = fps
        self.latency_profile = prof

        if return_diagnostics:
            diag = {
                "delta_t": delta_t,
                "t_fused": t_fused,
                "t_candidate": t_candidate,
                "t_constrained": t_constrained,
                "t_temporal": t_smooth,
                "A": A,
                "latency_profile": prof,
            }
            return dehazed_uint8, fps, diag

        return dehazed_uint8, fps
