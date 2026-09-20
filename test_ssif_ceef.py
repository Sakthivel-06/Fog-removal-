import numpy as np
import scipy.ndimage as ndimage
from skimage.exposure import equalize_adapthist
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from PIL import Image
import imageio
import time
import metrics

def ssif_filter(img, r=5, kappa=1.5, eps=0.1, s=1.0):
    """
    Smoothing-Sharpening Image Filter (SSIF) from Kaplan (2023) & Xiao et al. (2018).
    Self-guided edge-aware filter.
    img: float32 in [0, 1], shape (H, W) or (H, W, C)
    """
    sz = 2 * r + 1
    if img.ndim == 3:
        out = np.zeros_like(img)
        for c in range(img.shape[2]):
            out[:, :, c] = ssif_filter(img[:, :, c], r=r, kappa=kappa, eps=eps, s=s)
        return out

    # Single channel
    mean_I = ndimage.uniform_filter(img, size=sz)
    mean_II = ndimage.uniform_filter(img * img, size=sz)
    var_I = np.maximum(mean_II - mean_I * mean_I, 0.0)
    
    mean_var = np.mean(var_I) + 1e-7
    denom = var_I + eps
    t_k = var_I / denom
    alpha_k = 0.5 * (t_k + np.sqrt(t_k * t_k + (4.0 * kappa * eps) / denom))
    
    # Weighting factor omega_k
    omega_k = 1.0 / (1.0 + var_I / (s * mean_var))
    
    a_k = alpha_k
    b_k = (1.0 - alpha_k) * mean_I
    
    w_a = ndimage.uniform_filter(omega_k * a_k, size=sz)
    w_b = ndimage.uniform_filter(omega_k * b_k, size=sz)
    w_sum = ndimage.uniform_filter(omega_k, size=sz) + 1e-7
    
    mean_a = w_a / w_sum
    mean_b = w_b / w_sum
    
    q = mean_a * img + mean_b
    return np.clip(q, 0.0, 1.0)

def compute_weights(exposures):
    weights = []
    for img in exposures:
        gray = rgb2gray(img)
        # Contrast (Laplacian)
        lap = np.abs(ndimage.laplace(gray))
        # Saturation (standard deviation across color channels)
        sat = np.std(img, axis=2)
        # Well-exposedness
        sigma = 0.2
        well_exp = np.exp(-0.5 * ((img - 0.5) / sigma) ** 2)
        well_exp_prod = np.prod(well_exp, axis=2)
        
        W = (lap + 1e-6) * (sat + 1e-6) * (well_exp_prod + 1e-6)
        weights.append(W)
    
    weights = np.array(weights) # (N, H, W)
    w_sum = np.sum(weights, axis=0) + 1e-7
    return weights / w_sum

def mertens_fusion(exposures):
    """Multi-scale edge-preserving exposure fusion."""
    exposures = [np.clip(e, 0.0, 1.0) for e in exposures]
    weights = compute_weights(exposures)
    
    # Smooth weights
    smoothed_weights = [ndimage.gaussian_filter(w, sigma=1.5) for w in weights]
    w_sum = np.sum(smoothed_weights, axis=0) + 1e-7
    norm_weights = [w / w_sum for w in smoothed_weights]
    
    fused_base = np.zeros_like(exposures[0])
    fused_detail = np.zeros_like(exposures[0])
    
    for img, w in zip(exposures, norm_weights):
        w3 = w[:, :, np.newaxis]
        base = np.stack([ndimage.uniform_filter(img[:, :, c], size=7) for c in range(3)], axis=2)
        detail = img - base
        fused_base += base * w3
        fused_detail += detail * w3
        
    res = fused_base + fused_detail
    return np.clip(res, 0.0, 1.0)

def process_ssif_ceef(frame, kappa=1.5, clip_limit=0.015):
    img_float = frame.astype(np.float32) / 255.0
    
    # Step 1: SSIF Sharpening
    I_S = ssif_filter(img_float, r=5, kappa=kappa, eps=0.1, s=1.0)
    
    # Step 2: Generate multi-exposure sequence
    # Underexposed gamma images (remove haze)
    gammas = [1.5, 2.2, 3.2]
    exposures = []
    
    for g in gammas:
        exposures.append(np.power(I_S, g))
        
    # CLAHE on L channel of I_S
    lab = rgb2lab(I_S)
    l_chan = lab[:, :, 0] / 100.0
    cl = equalize_adapthist(l_chan, clip_limit=clip_limit)
    lab[:, :, 0] = cl * 100.0
    clahe_img = lab2rgb(lab)
    exposures.append(clahe_img)
    
    # Also include I_S itself
    exposures.append(I_S)
    
    # Step 3: Mertens Exposure Fusion
    fused = mertens_fusion(exposures)
    
    # Step 4: Natural contrast stretch
    p1, p99 = np.percentile(fused, (0.5, 99.5))
    if p99 - p1 > 0.01:
        fused = np.clip((fused - p1) / (p99 - p1), 0.0, 1.0)
        
    return (fused * 255.0).astype(np.uint8)

if __name__ == '__main__':
    r = imageio.get_reader('test_videos/car-detection_patchy_haze.mp4')
    f_car = r.get_data(15)
    r.close()
    
    r_clean = imageio.get_reader('test_videos/car-detection_clean.mp4')
    clean_car = r_clean.get_data(15)
    r_clean.close()
    
    t0 = time.perf_counter()
    dehazed_car = process_ssif_ceef(f_car, kappa=1.3, clip_limit=0.015)
    dt = time.perf_counter() - t0
    
    Image.fromarray(dehazed_car).save('results/comparison_test/ssif_ceef_car.jpg')
    
    psnr = metrics.calculate_psnr(clean_car, dehazed_car)
    ssim_val = metrics.calculate_ssim(clean_car, dehazed_car)
    fog = metrics.calculate_fog_density_proxy(dehazed_car)
    ent = metrics.calculate_entropy(dehazed_car)
    col = metrics.calculate_colorfulness(dehazed_car)
    
    print(f'SSIF-CEEF Car: PSNR={psnr:.2f} | SSIM={ssim_val:.3f} | Fog={fog:.3f} | Ent={ent:.2f} | Col={col:.1f} | Latency={dt*1000:.1f}ms')
    
    # Also test foggy video
    r_fog = imageio.get_reader('test_videos/foggy_video1.mp4')
    f_fog = r_fog.get_data(30)
    r_fog.close()
    
    h, w = f_fog.shape[:2]
    f_fog_sub = np.array(Image.fromarray(f_fog).resize((640, 360), Image.BILINEAR))
    t0 = time.perf_counter()
    dehazed_fog = process_ssif_ceef(f_fog_sub, kappa=1.8, clip_limit=0.02)
    dt = time.perf_counter() - t0
    Image.fromarray(dehazed_fog).save('results/comparison_test/ssif_ceef_foggy.jpg')
    
    fog_val = metrics.calculate_fog_density_proxy(dehazed_fog)
    ent_val = metrics.calculate_entropy(dehazed_fog)
    col_val = metrics.calculate_colorfulness(dehazed_fog)
    print(f'SSIF-CEEF Foggy (360p): Fog={fog_val:.3f} | Ent={ent_val:.2f} | Col={col_val:.1f} | Latency={dt*1000:.1f}ms ({1.0/dt:.1f} FPS)')
