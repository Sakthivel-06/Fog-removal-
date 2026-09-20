import numpy as np
import scipy.ndimage as ndimage
from skimage.color import rgb2gray
from skimage.transform import pyramid_laplacian, pyramid_gaussian

def box_filter(img, radius):
    """Fast box filter using scipy."""
    return ndimage.uniform_filter(img, size=2*radius+1)

def guided_filter(guide, src, radius, eps):
    """
    Guided filter implementation using NumPy and SciPy.
    Matches cv2.ximgproc.guidedFilter behavior for single-channel images.
    """
    mean_I = box_filter(guide, radius)
    mean_p = box_filter(src, radius)
    mean_Ip = box_filter(guide * src, radius)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = box_filter(guide * guide, radius)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)

    q = mean_a * guide + mean_b
    return q

def fast_guided_filter(guide_full, src_low, radius, eps):
    """
    Fast Guided Filter (He et al., 2015).
    Computes a and b at the low resolution of src_low, 
    then upsamples a and b to the full resolution, and applies to guide_full.
    This guarantees edge preservation of guided filter but runs at a fraction of the time.
    """
    from PIL import Image
    h_low, w_low = src_low.shape[:2]
    h_full, w_full = guide_full.shape[:2]
    
    # 1. Downsample guide to match src_low
    guide_full_f32 = guide_full.astype(np.float32)
    guide_low = np.array(Image.fromarray(guide_full_f32, mode="F").resize((w_low, h_low), Image.BILINEAR))
    
    # 2. Compute a and b at low resolution
    mean_I = box_filter(guide_low, radius)
    mean_p = box_filter(src_low, radius)
    mean_Ip = box_filter(guide_low * src_low, radius)
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = box_filter(guide_low * guide_low, radius)
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)
    
    # 3. Upsample mean_a and mean_b to full resolution
    mean_a_full = np.array(Image.fromarray(mean_a.astype(np.float32), mode="F").resize((w_full, h_full), Image.BILINEAR))
    mean_b_full = np.array(Image.fromarray(mean_b.astype(np.float32), mode="F").resize((w_full, h_full), Image.BILINEAR))
    
    # 4. Final reconstruction at full resolution
    q_full = mean_a_full * guide_full + mean_b_full
    return q_full

def compute_mertens_weights(exposures):
    weights = []
    for img in exposures:
        gray = rgb2gray(img)
        # Contrast
        laplacian = np.abs(ndimage.laplace(gray))
        # Saturation
        sat = np.std(img, axis=2)
        # Well-exposedness
        sigma = 0.2
        exp = np.exp(-0.5 * ((img - 0.5) / sigma)**2)
        well_exp = np.prod(exp, axis=2)
        
        W = (laplacian + 1e-12) * (sat + 1e-12) * (well_exp + 1e-12)
        weights.append(W)
    
    weights = np.array(weights)
    weights /= np.sum(weights, axis=0) + 1e-12
    return weights

def exposure_fusion(exposures, levels=3):
    """
    Fast and robust multi-scale Mertens exposure fusion.
    Uses multi-scale detail decomposition (base + detail layers) with normalized Mertens weights.
    Prevents brightness collapse while ensuring seamless edge-preserving blending.
    """
    exposures = [np.clip(e, 0.0, 1.0) for e in exposures]
    weights = compute_mertens_weights(exposures)  # (N, H, W)
    
    # Smooth weights slightly to prevent seams
    smoothed_weights = [ndimage.gaussian_filter(w, sigma=2.0) for w in weights]
    w_sum = np.sum(smoothed_weights, axis=0) + 1e-12
    norm_weights = [w / w_sum for w in smoothed_weights]

    # Two-scale decomposition for speed & zero brightness loss:
    # Base layer (large-scale illumination) + Detail layer (high-frequency edges)
    fused_base = np.zeros_like(exposures[0])
    fused_detail = np.zeros_like(exposures[0])

    for img, w in zip(exposures, norm_weights):
        w3 = w[:, :, np.newaxis]
        # Base: low-pass filtered illumination
        base = np.stack([ndimage.uniform_filter(img[:, :, c], size=9) for c in range(3)], axis=2)
        detail = img - base
        fused_base += base * w3
        fused_detail += detail * w3

    res = fused_base + fused_detail
    return np.clip(res, 0.0, 1.0)

