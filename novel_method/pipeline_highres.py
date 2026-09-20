"""
novel_method/pipeline_highres.py — A4 High-Resolution Output Pipeline (Session 24C)
==================================================================================
Non-destructive experimental branch extending the frozen Session 23 A4 pipeline.

Architecture:
  1. Downscaled feature extraction & learned Delta-t residual estimation
     at frozen guide resolution (240x135) and processing limit (426x240).
  2. Temporal coherence filtering at low resolution (t_temporal).
  3. Reconstruct final transmission map at ORIGINAL input resolution:
     - Mode B1: Bilinear upsampling
     - Mode B2: Edge-preserving guided upsampling using full-resolution RGB/gray guide
  4. Physical scene inversion at ORIGINAL input resolution:
     J_full = (I_full - A) / t_full + A

Zero OpenCV (cv2) dependency. Uses NumPy, SciPy, Pillow, and PyTorch CPU only.
"""

import os
import sys
import time
import numpy as np
from PIL import Image
from skimage.color import rgb2gray

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from novel_method.pipeline_learned_experimental import LearnedResidualDehazingPipeline
from algorithms.filters import guided_filter, fast_guided_filter

class HighResLearnedResidualDehazingPipeline:
    """
    Experimental high-resolution output harness for A4.
    Reconstructs transmission at full native resolution and inverts raw camera sensor RGB.
    """
    def __init__(
        self,
        checkpoint_path: str = "checkpoints/delta_t_tiny_unet_best.pt",
        downscale_res=(240, 135),
        max_process_res=(426, 240),
        upsample_mode: str = "guided", # 'guided' (B2) or 'bilinear' (B1)
        guided_radius: int = 15,
        guided_eps: float = 1e-3,
        **kwargs
    ):
        self.upsample_mode = upsample_mode
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps
        
        # Frozen low-resolution A4 estimation engine
        self.base_pipeline = LearnedResidualDehazingPipeline(
            checkpoint_path=checkpoint_path,
            downscale_res=downscale_res,
            max_process_res=max_process_res,
            ablation_mode="A4",
            enable_inference_temporal=True,
            **kwargs
        )
        self.latency_profile = {}
        
    def reset(self):
        """Resets temporal filtering state."""
        self.base_pipeline.reset()
        self.latency_profile = {}
        
    def process_frame(
        self,
        frame_raw: np.ndarray,
        upsample_mode: str = None,
        return_diagnostics: bool = False
    ):
        """
        Processes frame with full native resolution physical inversion.
        
        Args:
            frame_raw: Raw input RGB image (uint8, [0, 255]), shape (H, W, 3)
            upsample_mode: 'guided' (B2) or 'bilinear' (B1). If None, uses self.upsample_mode.
            return_diagnostics: If True, returns detailed diagnostic maps.
            
        Returns:
            dehazed_full_u8: Native resolution restored image (uint8, shape (H, W, 3))
            fps: Total processing frames per second
            diag (optional): Diagnostic metadata dictionary
        """
        t_total_start = time.perf_counter()
        mode = upsample_mode or self.upsample_mode
        prof = {}
        
        raw_h, raw_w = frame_raw.shape[:2]
        
        # ── Step 1: Low-Resolution A4 Pipeline Execution ──
        t0 = time.perf_counter()
        out_lr, fps_lr, diag_lr = self.base_pipeline.process_frame(frame_raw, return_diagnostics=True, skip_inversion=True)
        prof["base_pipeline_ms"] = (time.perf_counter() - t0) * 1000.0
        
        t_temporal = diag_lr["t_temporal"]  # Shape (gh, gw)
        A = diag_lr["A"]                    # Shape (3,)
        
        # ── Step 2: Transmission Upsampling to Native Resolution ──
        t0 = time.perf_counter()
        
        I_full = frame_raw.astype(np.float32) / 255.0
        
        if mode == "guided":
            # B2: Edge-preserving FAST guided upsampling using full-resolution grayscale guide
            gray_full = rgb2gray(I_full)
            t_full = fast_guided_filter(
                guide_full=gray_full,
                src_low=t_temporal,
                radius=self.guided_radius,
                eps=self.guided_eps
            )
            # Physical safety clamp (enforce adaptive t_min floor)
            t_full = np.clip(t_full, 0.05, 1.0)
        else:
            # B1: Bilinear interpolation only
            # Coarse bilinear scaling to full resolution
            t_pil = Image.fromarray(t_temporal, mode="F").resize((raw_w, raw_h), Image.BILINEAR)
            t_coarse = np.array(t_pil)
            t_full = np.clip(t_coarse, 0.05, 1.0)
            
        prof["upsampling_ms"] = (time.perf_counter() - t0) * 1000.0
        
        # ── Step 3: Full-Resolution Physical Scene Inversion ──
        t0 = time.perf_counter()
        
        # CPU Optimization: Perform physical inversion entirely in-place to save memory bandwidth
        # J = (I - A) / t + A
        np.subtract(I_full, A[None, None, :], out=I_full)
        np.divide(I_full, t_full[:, :, None], out=I_full)
        np.add(I_full, A[None, None, :], out=I_full)
        np.clip(I_full, 0.0, 1.0, out=I_full)
        
        I_full *= 255.0
        dehazed_full_u8 = I_full.astype(np.uint8)
        prof["inversion_ms"] = (time.perf_counter() - t0) * 1000.0
        
        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        fps = 1000.0 / max(total_ms, 1e-3)
        prof["total_proc_ms"] = total_ms
        prof["fps"] = fps
        self.latency_profile = prof
        
        if return_diagnostics:
            diag = {
                "t_temporal_low": t_temporal,
                "t_temporal_full": t_full,
                "A": A,
                "out_lr": out_lr,
                "latency_profile": prof,
                "upsample_mode": mode,
                "native_resolution": (raw_w, raw_h),
                "low_resolution": (out_lr.shape[1], out_lr.shape[0])
            }
            return dehazed_full_u8, fps, diag
            
        return dehazed_full_u8, fps
