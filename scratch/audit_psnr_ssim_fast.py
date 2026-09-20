"""
scratch/audit_psnr_ssim_fast.py — Fast, Precise Audit of PSNR, SSIM, and Luminance Shift
=======================================================================================
"""

import os
import sys
import time
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

def run_audit():
    n_warmup = 4
    n_measure = 10
    total_frames = n_warmup + n_measure

    for p in PAIRS:
        hazy_path = os.path.join("test_videos", p["name"])
        clean_path = os.path.join("test_videos", p["gt"])

        rh = imageio.get_reader(hazy_path)
        rc = imageio.get_reader(clean_path)
        frames_h = [rh.get_data(i) for i in range(total_frames)]
        frames_c = [rc.get_data(i) for i in range(total_frames)]
        rh.close()
        rc.close()

        print(f"\n=======================================================")
        print(f"AUDIT ON: {p['title']}")
        print(f"=======================================================")

        for m_name, m_func in METHODS:
            pipe = ExperimentalNighttimePipeline(enable_nighttime_branch=True, night_mode="prototype")

            # Warmup (resized to 426x239 as in benchmark)
            for w in range(n_warmup):
                fh_w = _resize_np(frames_h[w], (426, 239))
                m_func(fh_w, pipe)

            psnr_list = []
            ssim_rgb_list = []
            ssim_gray_list = []
            mse_list = []
            pred_lums = []
            gt_lums = []

            for i in range(n_warmup, total_frames):
                fh = _resize_np(frames_h[i], (426, 239))
                fc = _resize_np(frames_c[i], (426, 239))

                out = m_func(fh, pipe)
                if out.shape[:2] != (239, 426):
                    out = _resize_np(out, (426, 239))

                # Compute MSE
                diff = out.astype(np.float64) - fc.astype(np.float64)
                mse = float(np.mean(diff ** 2))
                mse_list.append(mse)

                # PSNR (uint8 convention)
                psnr = float(10.0 * np.log10((255.0 ** 2) / (mse + 1e-12)))
                psnr_list.append(psnr)

                # SSIM (RGB)
                s_rgb = float(sk_ssim(fc, out, channel_axis=-1, data_range=255))
                ssim_rgb_list.append(s_rgb)

                # SSIM (Grayscale)
                g_out = (0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]).astype(np.uint8)
                g_fc = (0.299 * fc[:, :, 0] + 0.587 * fc[:, :, 1] + 0.114 * fc[:, :, 2]).astype(np.uint8)
                s_gray = float(sk_ssim(g_fc, g_out, data_range=255))
                ssim_gray_list.append(s_gray)

                pred_lums.append(float(np.mean(g_out)))
                gt_lums.append(float(np.mean(g_fc)))

            print(f"{m_name:16s} | MSE: {np.mean(mse_list):7.1f} | PSNR: {np.mean(psnr_list):5.2f} dB | SSIM(RGB): {np.mean(ssim_rgb_list):.4f} | SSIM(Gray): {np.mean(ssim_gray_list):.4f} | LumPred: {np.mean(pred_lums):5.1f} | LumGT: {np.mean(gt_lums):5.1f}")
            sys.stdout.flush()

if __name__ == "__main__":
    run_audit()
