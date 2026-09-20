"""
algorithms/ceef_original.py — Original CEEF Baseline Implementation
====================================================================
Faithful reimplementation of the published CEEF (Contrast Enhancement and Exposure Fusion)
dehazing framework (Guo et al. / Kaplan et al.).
OpenCV-free: purely uses NumPy, SciPy, scikit-image, and Pillow.

Pipeline:
  1. Underexposure sequence: Power-law transforms I^gamma for gamma in [2, 3, 4, 5] to eliminate atmospheric veil.
  2. Color-Preserving AHE (CP-AHE): Adapts local contrast while preserving color balance by interleaving R, G, B channels horizontally.
  3. FSPD-MEF (Fast Sequential Pyramid Decomposition Multi-Exposure Fusion): Multi-scale detail and base weight fusion.
"""

import numpy as np
import scipy.ndimage as ndimage
from skimage.exposure import equalize_adapthist

def box_filter(img: np.ndarray, r: int) -> np.ndarray:
    """Computes exact 2D box filter sum with radius r."""
    size = 2 * r + 1
    return ndimage.uniform_filter(img.astype(np.float64), size=size, mode='nearest') * (size * size)

def weight_cal_detail(I: np.ndarray, p: int, r: int):
    """Calculates multi-scale detail weights and local means."""
    R = 2 * r + 1
    h, w = I.shape[:2]
    N1 = box_filter(np.ones((h, w), dtype=np.float64), r)
    
    if I.ndim == 3:
        N = I.shape[2]
        C = np.zeros((h, w, N), dtype=np.float64)
        i_mean2 = np.zeros((h, w, N), dtype=np.float64)
        for i in range(N):
            img = I[:, :, i]
            i_mean2[:, :, i] = box_filter(img, r) / N1
            i_var2 = box_filter(img * img, r) / N1 - i_mean2[:, :, i] ** 2
            i_var2 = np.sqrt(np.maximum(i_var2, 0.0))
            C[:, :, i] = i_var2 * np.sqrt(R * R) + 1e-12
    else:
        N = I.shape[3]
        C = np.zeros((h, w, N), dtype=np.float64)
        i_mean2 = np.zeros((h, w, N), dtype=np.float64)
        for i in range(N):
            img = I[:, :, :, i]
            m = (box_filter(img[:, :, 0], r) / N1 +
                 box_filter(img[:, :, 1], r) / N1 +
                 box_filter(img[:, :, 2], r) / N1) / 3.0
            i_mean2[:, :, i] = m
            var = (box_filter(img[:, :, 0]**2, r) / N1 +
                   box_filter(img[:, :, 1]**2, r) / N1 +
                   box_filter(img[:, :, 2]**2, r) / N1) / 3.0 - m**2
            var = np.sqrt(np.maximum(var, 0.0))
            C[:, :, i] = var * np.sqrt(3 * R * R) + 1e-12

    Cmax = np.max(C, axis=2)
    sMap1 = C ** p
    sMap2 = C ** (p - 1)
    normalizer = np.sum(sMap1 + 1e-12, axis=2, keepdims=True)
    sMap = sMap2 / normalizer
    return sMap, Cmax, i_mean2

def meanfun_our(X: np.ndarray) -> np.ndarray:
    """Well-exposedness weighting function."""
    Y5 = (2.0 / np.pi) * np.arctan(20.0 * (X * (X <= 0.5)))
    Y6 = (2.0 / np.pi) * np.arctan(20.0 * ((1.0 - X) * (X > 0.5)))
    return Y5 + Y6

def weight_cal_base(I: np.ndarray) -> np.ndarray:
    """Calculates base layer weights."""
    h, w, N = I.shape
    W = np.ones((h, w, N), dtype=np.float64)
    for i in range(N):
        img = I[:, :, i]
        gy, gx = np.gradient(img)
        grad = np.sqrt(gx**2 + gy**2)
        well_exp = meanfun_our(img)
        W[:, :, i] = grad * well_exp + 1e-12
    W /= (np.sum(W, axis=2, keepdims=True) + 1e-12)
    return W

def scale_fine(imgSeqColor: np.ndarray, r: int):
    """Fine-scale detail fusion."""
    h, w, c, n = imgSeqColor.shape
    N = box_filter(np.ones((h, w), dtype=np.float64), r)
    tem = np.ones((h, w), dtype=np.float64)
    tem[:, 1::2] = 0
    tem[1::2, :] = 0
    N1 = box_filter(tem, r)
    p = 4
    WD, Cmax, i_mean2 = weight_cal_detail(imgSeqColor, p, r)
    WD = WD * Cmax[:, :, np.newaxis]
    F_temp2_detail = np.zeros((h, w, c, n), dtype=np.float64)
    i_meant = np.zeros((int(np.ceil(h / 2.0)), int(np.ceil(w / 2.0)), n), dtype=np.float64)
    for i in range(n):
        aa = i_mean2[:, :, i] * tem
        i_meant[:, :, i] = aa[0::2, 0::2]
        W_D1 = box_filter(i_mean2[:, :, i] * WD[:, :, i], r) / N
        W_D2 = box_filter(WD[:, :, i], r) / N
        for ch in range(3):
            F_temp2_detail[:, :, ch, i] = W_D2 * imgSeqColor[:, :, ch, i] - W_D1
    fI3 = np.sum(F_temp2_detail, axis=3)
    return fI3, i_meant, aa, N1

def scale_interm(imgSeqGray: np.ndarray, r: int):
    """Intermediate-scale detail fusion."""
    h, w, n = imgSeqGray.shape
    N = box_filter(np.ones((h, w), dtype=np.float64), r)
    tem = np.ones((h, w), dtype=np.float64)
    tem[:, 1::2] = 0
    tem[1::2, :] = 0
    N1 = box_filter(tem, r)
    p = 4
    WD, Cmax, i_mean2 = weight_cal_detail(imgSeqGray, p, r)
    WD = WD * Cmax[:, :, np.newaxis]
    F_temp2_detail = np.zeros((h, w, n), dtype=np.float64)
    i_meant = np.zeros((int(np.ceil(h / 2.0)), int(np.ceil(w / 2.0)), n), dtype=np.float64)
    for i in range(n):
        aa = i_mean2[:, :, i] * tem
        i_meant[:, :, i] = aa[0::2, 0::2]
        W_D1 = box_filter(i_mean2[:, :, i] * WD[:, :, i], r) / N
        W_D2 = box_filter(WD[:, :, i], r) / N
        F_temp2_detail[:, :, i] = W_D2 * imgSeqGray[:, :, i] - W_D1
    fI3 = np.sum(F_temp2_detail, axis=2)
    return fI3, i_meant, aa, N1

def scale_coarse(imgSeqGray: np.ndarray, r: int):
    """Coarse-scale base fusion."""
    h, w, n = imgSeqGray.shape
    N = box_filter(np.ones((h, w), dtype=np.float64), r)
    tem = np.ones((h, w), dtype=np.float64)
    tem[:, 1::2] = 0
    tem[1::2, :] = 0
    N1 = box_filter(tem, r)
    p = 6
    WD, Cmax, i_mean2 = weight_cal_detail(imgSeqGray, p, r)
    WD = WD * Cmax[:, :, np.newaxis]
    WB = weight_cal_base(imgSeqGray)
    F_temp2 = np.zeros((h, w, n), dtype=np.float64)
    i_meant = np.zeros((int(np.ceil(h / 2.0)), int(np.ceil(w / 2.0)), n), dtype=np.float64)
    for i in range(n):
        aa = i_mean2[:, :, i] * tem
        i_meant[:, :, i] = aa[0::2, 0::2]
        W_D1 = box_filter(i_mean2[:, :, i] * WD[:, :, i], r) / N
        W_D2 = box_filter(WD[:, :, i], r) / N
        W_B2 = box_filter(i_mean2[:, :, i] * WB[:, :, i], r) / N
        F_temp2[:, :, i] = (W_D2 * imgSeqGray[:, :, i] - W_D1) + W_B2
    fI3 = np.sum(F_temp2, axis=2)
    return fI3, i_meant, aa, N1

def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Original CEEF Baseline (OpenCV-free faithful implementation).
    Takes uint8 RGB frame, returns uint8 RGB dehazed frame.
    """
    img_haze = frame.astype(np.float64) / 255.0
    h, w, c = img_haze.shape
    I = np.zeros((h, w, c, 5), dtype=np.float64)
    
    # 1. Underexposure sequences: gamma = 2, 3, 4, 5
    for i, g in enumerate([2, 3, 4, 5], start=1):
        I[:, :, :, i] = img_haze ** g
        
    # 2. Color-Preserving AHE (CP-AHE)
    # The authors interleave R, G, B horizontally into (H, 3W) to adapt local contrast while preserving color balance
    in_comb = np.zeros((h, w * 3), dtype=np.float64)
    in_comb[:, 0::3] = img_haze[:, :, 0]
    in_comb[:, 1::3] = img_haze[:, :, 1]
    in_comb[:, 2::3] = img_haze[:, :, 2]
    out_comb = equalize_adapthist(in_comb, clip_limit=0.01)
    cp_ahe = np.zeros((h, w, 3), dtype=np.float64)
    cp_ahe[:, :, 0] = out_comb[:, 0::3]
    cp_ahe[:, :, 1] = out_comb[:, 1::3]
    cp_ahe[:, :, 2] = out_comb[:, 2::3]
    I[:, :, :, 0] = cp_ahe
    
    # 3. FSPD-MEF Multi-scale Fusion
    r1 = 5
    D1, i_mean1, aa1, N1 = scale_fine(I, r1)
    
    nlev = int(np.floor(np.log2(min(w, h)))) - 3
    nlev = max(1, nlev)
    D2 = []
    aa2 = []
    N2 = []
    r2_reconstruct = []
    r2 = 4
    for ii in range(nlev):
        r2_reconstruct.append(r2)
        d2_l, i_mean2, aa2_l, n2_l = scale_interm(i_mean1, r2)
        D2.append(d2_l)
        aa2.append(aa2_l)
        N2.append(n2_l)
        i_mean1 = i_mean2
        r2 = max(r2 - 1, 1)
        
    r3 = max(r2 - 1, 1)
    fI3, i_mean3, aa3, N3 = scale_coarse(i_mean2, r3)
    
    # Reconstruction across pyramid levels
    for ii in range(nlev - 1, -1, -1):
        temp = aa2[ii]
        fI = np.zeros_like(temp)
        th, tw = fI3.shape
        fI[0:2*th:2, 0:2*tw:2] = fI3
        B2 = box_filter(fI, r2_reconstruct[ii]) / N2[ii] + D2[ii]
        fI3 = B2
        
    fI = np.zeros_like(aa1)
    th, tw = B2.shape
    fI[0:2*th:2, 0:2*tw:2] = B2
    B1 = box_filter(fI, r1) / N1
    
    result = np.repeat(B1[:, :, np.newaxis], 3, axis=2) + D1
    result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
    return result
