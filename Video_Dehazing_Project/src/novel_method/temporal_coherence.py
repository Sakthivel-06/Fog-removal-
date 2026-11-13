"""
temporal_coherence.py
=====================
SSIF-CEEF Motion-Aware Temporal Coherence Module (Session 5)
Project: "SSIF-CEEF using Temporal Coherence" -- Novel Video Dehazing

Strict Physical Control Rule:
  Temporal smoothing is applied ONLY to the physical transmission map t(x)
  and the atmospheric light vector A.
  Raw RGB image frames are NEVER blended.

Components:
  1. Lightweight Frame-Difference Motion Estimation at Guide Resolution.
  2. Scene-Change Detection with Instantaneous Temporal History Reset.
  3. Spatially Adaptive Temporal Alpha Map:
       - Static / low-motion regions: strong smoothing (alpha ~ 0.15) to eliminate flicker.
       - Moving object regions: rapid update (alpha ~ 0.90) to eliminate trailing ghost wakes.
  4. Stable Global Airlight Temporal Smoothing (alpha ~ 0.08).

All operations are CPU-only (NumPy + SciPy). Zero OpenCV dependencies.
"""

import numpy as np
import scipy.ndimage as ndimage


def compute_motion_mask(
    current_gray: np.ndarray,
    prev_gray: np.ndarray,
    motion_thresh_low: float = 0.03,
    motion_thresh_high: float = 0.15,
    filter_size: int = 3,
):
    """
    Computes a lightweight, noise-regularized motion confidence mask M(x) in [0, 1]
    using absolute frame difference at the guide resolution.

    Returns:
        motion_mask: 2D float32 array in [0.0, 1.0].
                     0.0 = static background, 1.0 = high motion / moving object.
        mean_diff: float, global mean frame difference across guide pixels.
        high_motion_pct: float, percentage of pixels with motion confidence > 0.5.
    """
    diff = np.abs(current_gray.astype(np.float32) - prev_gray.astype(np.float32))
    mean_diff = float(np.mean(diff))

    if filter_size > 1:
        diff_filt = ndimage.uniform_filter(diff, size=filter_size)
    else:
        diff_filt = diff

    denom = max(motion_thresh_high - motion_thresh_low, 1e-6)
    motion_mask = np.clip((diff_filt - motion_thresh_low) / denom, 0.0, 1.0)
    high_motion_pct = float(np.count_nonzero(motion_mask > 0.5) / motion_mask.size * 100.0)

    return motion_mask, mean_diff, high_motion_pct


def is_scene_change(
    mean_diff: float,
    high_motion_pct: float = 0.0,
    change_threshold: float = 0.20,
    high_motion_pct_threshold: float = 60.0,
) -> bool:
    """
    Detects abrupt scene change / cut.
    Returns True if temporal history should be reset immediately.
    """
    return bool(mean_diff > change_threshold or high_motion_pct > high_motion_pct_threshold)


def compute_adaptive_temporal_alpha(
    motion_mask: np.ndarray,
    alpha_static: float = 0.15,
    alpha_motion: float = 0.90,
) -> np.ndarray:
    """
    Calculates spatially varying alpha(x) map.
    - Static regions (M=0): alpha = alpha_static (strong smoothing, low flicker).
    - Moving regions (M=1): alpha = alpha_motion (near-instantaneous update, zero ghosting).
    """
    return alpha_static + motion_mask * (alpha_motion - alpha_static)


def smooth_transmission_temporal(
    current_t: np.ndarray,
    prev_t: np.ndarray,
    alpha_map: np.ndarray,
) -> np.ndarray:
    """
    Applies spatially adaptive temporal IIR smoothing to the transmission map:
        t_smooth(x) = (1 - alpha(x)) * prev_t(x) + alpha(x) * current_t(x)
    """
    if prev_t is None or prev_t.shape != current_t.shape:
        return current_t.copy()
    return (1.0 - alpha_map) * prev_t + alpha_map * current_t


def smooth_airlight_temporal(
    current_A: np.ndarray,
    prev_A: np.ndarray,
    alpha_airlight: float = 0.08,
    is_scene_cut: bool = False,
) -> np.ndarray:
    """
    Smooths the global atmospheric light vector A temporally across frames.
    Resets instantaneously to current_A on scene cut.
    """
    if prev_A is None or is_scene_cut:
        return current_A.copy()
    return (1.0 - alpha_airlight) * prev_A + alpha_airlight * current_A

