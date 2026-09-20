"""
scratch/audit_psnr_ssim.py — Independent Recomputation and Audit of PSNR and SSIM
===================================================================================
Audits:
  1. Exact mathematical formulation of PSNR and SSIM
  2. Data range convention (uint8 [0, 255] vs float32 [0.0, 1.0])
  3. Frame-by-frame verification on:
     - car-detection_patchy_haze.mp4 vs car-detection_clean.mp4
     - person-bicycle_uniform_haze.mp4 vs person-bicycle_clean.mp4
  4. Root cause analysis of Proposed Method PSNR (9.66 dB / 10.11 dB)
"""

import os
import sys
import imageio
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "algorithms")))

import dcp
import cap
import clahe_baseline
import gamma_baseline
import ceef
import ceef_original
from pipeline import RealTimeDehazingPipeline, _resize_np
from nighttime_experimental import ExperimentalNighttimePipeline

PAIRS = [
    {
        "name": "car-detection_patchy_haze.mp4",
        "gt": "car-detection_clean.mp4",
        "title": "Car Detection (Patchy Haze)"
    },
    {
        "name": "person-bicycle_uniform_haze.mp4",
        "gt": "person-bicycle_clean.mp4",
        "title": "Person Bicycle (Uniform Haze)"
    }
]

METHODS = [
    ("DCP", lambda img, p: dcp.process_frame(img)),
    ("CAP", lambda img, p: cap.process_frame(img)),
    ("CLAHE", lambda img, p: clahe_baseline.process_frame(img)),
    ("Corrected Gamma", lambda img, p: gamma_baseline.process_frame(img, gamma=2.0)),
    ("Original CEEF", lambda img, p: ceef_original.process_frame(img)),
    ("Modified CEEF", lambda img, p: ceef.process_frame(img)),
    ("Proposed Method", lambda img, p: p.process_frame(img)[0]),
]

def audit_metrics_on_sequence(pair_info, n_warmup=8, n_measure=15):
    hazy_path = os.path.join("test_videos", pair_info["name"])
    clean_path = os.path.join("test_videos", pair_info["gt"])
    
    rh = imageio.get_reader(hazy_path)
    rc = imageio.get_reader(clean_path)
    
    total_frames = n_warmup + n_measure
    frames_h = [rh.get_data(i) for i in range(total_frames)]
    frames_c = [rc.get_data(i) for i in range(total_frames)]
    rh.close()
    rc.close()
    
    print(f"\n=======================================================")
    print(f"PSNR & SSIM AUDIT ON: {pair_info['title']}")
    print(f"=======================================================")
    
    for m_name, m_func in METHODS:
        pipe = ExperimentalNighttimePipeline(enable_nighttime_branch=True, night_mode="prototype")
        
        # Warm up
        for w in range(n_warmup):
            m_func(frames_h[w], pipe)
            
        psnr_uint8_list = []
        psnr_float_list = []
        ssim_rgb_list = []
        ssim_gray_list = []
        mse_list = []
        mean_pred_lums = []
        mean_gt_lums = []
        
        for i in range(n_warmup, total_frames):
            fh = frames_h[i]
            fc = frames_c[i]
            
            out = m_func(fh, pipe)
            if out.shape[:2] != fc.shape[:2]:
                fc_res = np.array(Image.fromarray(fc).resize((out.shape[1], out.shape[0]), Image.BILINEAR))
            else:
                fc_res = fc
                
            # Exact formula checks:
            # 1. uint8 MSE and PSNR (data_range=255)
            mse = float(np.mean((out.astype(np.float64) - fc_res.astype(np.float64)) ** 2))
            mse_list.append(mse)
            psnr_u8 = float(10.0 * np.log10((255.0 ** 2) / (mse + 1e-12)))
            psnr_uint8_list.append(psnr_u8)
            
            # 2. float32 MSE and PSNR (data_range=1.0)
            out_f = out.astype(np.float64) / 255.0
            fc_f = fc_res.astype(np.float64) / 255.0
            mse_f = float(np.mean((out_f - fc_f) ** 2))
            psnr_flt = float(10.0 * np.log10(1.0 / (mse_f + 1e-12)))
            psnr_float_list.append(psnr_flt)
            
            # 3. Multichannel SSIM (RGB, channel_axis=-1, data_range=255)
            ssim_rgb = float(sk_ssim(fc_res, out, channel_axis=-1, data_range=255))
            ssim_rgb_list.append(ssim_rgb)
            
            # 4. Grayscale SSIM
            g_out = (0.299*out[:,:,0] + 0.587*out[:,:,1] + 0.114*out[:,:,2]).astype(np.uint8)
            g_fc = (0.299*fc_res[:,:,0] + 0.587*fc_res[:,:,1] + 0.114*fc_res[:,:,2]).astype(np.uint8)
            ssim_gray = float(sk_ssim(g_fc, g_out, data_range=255))
            ssim_gray_list.append(ssim_gray)
            
            mean_pred_lums.append(float(np.mean(g_out)))
            mean_gt_lums.append(float(np.mean(g_fc)))
            
        print(f"[{m_name:16s}]")
        print(f"  MSE:              {np.mean(mse_list):8.2f} (Pred Lum: {np.mean(mean_pred_lums):5.1f}, GT Lum: {np.mean(mean_gt_lums):5.1f})")
        print(f"  PSNR (uint8):     {np.mean(psnr_uint8_list):6.2f} dB (Formula: 10*log10(255^2 / MSE))")
        print(f"  PSNR (float):     {np.mean(psnr_float_list):6.2f} dB (Formula: 10*log10(1 / MSE_float))")
        print(f"  SSIM (RGB):       {np.mean(ssim_rgb_list):.4f}")
        print(f"  SSIM (Grayscale): {np.mean(ssim_gray_list):.4f}")

if __name__ == "__main__":
    for p in PAIRS:
        audit_metrics_on_sequence(p)
