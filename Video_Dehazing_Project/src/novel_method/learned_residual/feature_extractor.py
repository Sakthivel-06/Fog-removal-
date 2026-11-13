"""
feature_extractor.py
====================
Session 21: Shared Physics Feature Extractor.
Extracts the identical 8-channel guide features directly using the exact
equations and parameters from the authoritative novel_method/pipeline.py.

Zero OpenCV dependency.
"""

import numpy as np
import scipy.ndimage as ndimage


def extract_physics_features(proc_float: np.ndarray, pipeline_inst) -> tuple:
    """
    Extracts the 8 guide features from a float32 RGB image in [0, 1]
    using the exact parameters and methods of pipeline_inst (RealTimeDehazingPipeline).

    Returns:
        feature_tensor_np: (8, gh, gw) float32 array
        metadata: dict containing A_raw, depth, t_min_map, etc.
    """
    # 1. Guide at half resolution
    half_img = proc_float[::2, ::2]
    gh, gw = half_img.shape[:2]
    gray_h = 0.299 * half_img[:, :, 0] + 0.587 * half_img[:, :, 1] + 0.114 * half_img[:, :, 2]

    # 2. Depth from CAP (Zhu et al.)
    v = np.max(half_img, axis=2)
    dc_img = np.min(half_img, axis=2)
    s = (v - dc_img) / (v + 1e-6)
    depth = ndimage.uniform_filter(0.121779 + 0.959710 * v - 0.780245 * s, size=3)

    # 3. Atmospheric Light A (exact pipeline method)
    A_raw, candidate_mask = pipeline_inst.estimate_atmospheric_light(
        half_img, depth, v=v, s=s, dc=dc_img, gray_h=gray_h
    )

    # 4. DCP Transmission
    norm = half_img / (A_raw[None, None, :] + 1e-6)
    dc = np.min(norm, axis=2)
    dc_min = ndimage.minimum_filter(dc, size=5)
    h_local = ndimage.uniform_filter(dc, size=3)

    omega_map = np.clip(pipeline_inst.omega + 0.08 * h_local, pipeline_inst.omega, 0.96)
    t_dcp = 1.0 - omega_map * dc_min

    # 5. CAP Transmission
    t_cap = np.exp(-pipeline_inst.beta * np.clip(depth, 0.0, 2.5))

    # 6. Spatially-Varying Fusion Weight w_dcp
    cfg = pipeline_inst.fusion_cfg
    var_h = ndimage.uniform_filter(gray_h**2, size=3) - ndimage.uniform_filter(gray_h, size=3)**2
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
    H_x = np.clip((h_local - cfg.get("haze_dc_thresh", 0.15)) / cfg.get("haze_dc_range", 0.35), 0.0, 1.0) * haze_lum_gate

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

    t_fused = w_dcp * t_dcp + (1.0 - w_dcp) * t_cap

    # Flat sky protection on t_fused if enabled
    if pipeline_inst.enable_sky_model:
        is_sky = np.clip((gray_h - 0.72) / 0.18, 0.0, 1.0) * np.clip(1.0 - var_h / 0.0012, 0.0, 1.0)
        t_fused = t_fused * (1.0 - is_sky) + np.maximum(t_fused, 0.65) * is_sky

    # 7. Adaptive t_min map (preserves exact Session 1 logic)
    t_base = max(pipeline_inst.t_min, 0.12)
    w_sky = np.clip((gray_h - 0.50) / 0.25, 0.0, 1.0) * np.clip(1.0 - var_h / 0.0003, 0.0, 1.0)
    w_bright = np.clip((gray_h - 0.65) / 0.25, 0.0, 1.0)
    floor_bright = 0.50 * np.maximum(w_sky, w_bright * 0.6)

    w_dark = np.clip((0.35 - gray_h) / 0.25, 0.0, 1.0)
    w_smooth_dark = np.clip(1.0 - var_h / 0.001, 0.0, 1.0)
    floor_dark = 0.35 * w_dark * (0.6 + 0.4 * w_smooth_dark)

    haze_ind = np.clip((h_local - 0.20) / 0.30, 0.0, 1.0) * (1.0 - w_dark) * (1.0 - w_sky)
    relief = 0.06 * haze_ind

    t_min_map = np.clip(t_base + floor_dark + floor_bright - relief, t_base, 0.85)

    # 8. Atmospheric veil proxy: max(gray - lum(A), 0)
    A_lum = 0.299 * A_raw[0] + 0.587 * A_raw[1] + 0.114 * A_raw[2]
    veil_proxy = np.maximum(gray_h - A_lum, 0.0).astype(np.float32)

    # 8-channel Guide Feature Tensor
    features = np.stack(
        [
            half_img[:, :, 0],  # 0: R guide
            half_img[:, :, 1],  # 1: G guide
            half_img[:, :, 2],  # 2: B guide
            t_fused,            # 3: t_fused
            t_cap,              # 4: t_cap
            t_dcp,              # 5: t_dcp
            w_dcp,              # 6: w_dcp
            veil_proxy,         # 7: veil proxy
        ],
        axis=0,
    ).astype(np.float32)

    metadata = {
        "A_raw": A_raw,
        "gray_h": gray_h,
        "t_min_map": t_min_map,
        "half_img": half_img,
        "depth": depth,
    }

    return features, metadata
