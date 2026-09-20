import os
import sys
import time
import json
import numpy as np
import imageio
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))

from novel_method.pipeline import RealTimeDehazingPipeline
from metrics import calculate_niqe, calculate_ssim, calculate_psnr

def evaluate_video(pipeline, hazy_path, clean_path=None, max_frames=60):
    reader = imageio.get_reader(hazy_path)
    clean_reader = imageio.get_reader(clean_path) if clean_path and os.path.exists(clean_path) else None
    
    pipeline.reset()
    frames_processed = 0
    t_start = time.perf_counter()
    
    psnrs = []
    ssims = []
    niqes = []
    lum_diffs = []
    black_rates = []
    white_rates = []
    prev_lum = None
    
    for i, frame in enumerate(reader):
        if i >= max_frames:
            break
        
        out_f, fps = pipeline.process_frame(frame)
        frames_processed += 1
        
        # Black / White rates
        black_rates.append(float(np.count_nonzero(out_f <= 2) / out_f.size * 100.0))
        white_rates.append(float(np.count_nonzero(out_f >= 253) / out_f.size * 100.0))
        
        # Temporal flicker
        cur_lum = 0.299 * out_f[:, :, 0] + 0.587 * out_f[:, :, 1] + 0.114 * out_f[:, :, 2]
        if prev_lum is not None and prev_lum.shape == cur_lum.shape:
            lum_diffs.append(float(np.mean(np.abs(cur_lum - prev_lum))))
        prev_lum = cur_lum
        
        # NIQE on sampled frames
        if i % 10 == 0:
            try:
                niqes.append(float(calculate_niqe(out_f)))
            except Exception:
                pass
        
        # GT metrics if clean available
        if clean_reader is not None:
            try:
                clean_f = clean_reader.get_data(i)
                # Ensure same resolution
                if clean_f.shape != out_f.shape:
                    clean_f = np.array(Image.fromarray(clean_f).resize((out_f.shape[1], out_f.shape[0]), Image.BILINEAR))
                p = calculate_psnr(out_f, clean_f)
                s = calculate_ssim(out_f, clean_f)
                psnrs.append(float(p))
                ssims.append(float(s))
            except Exception:
                pass
                
    total_time = time.perf_counter() - t_start
    overall_fps = frames_processed / max_frames if total_time == 0 else frames_processed / total_time
    
    reader.close()
    if clean_reader:
        clean_reader.close()
        
    return {
        "video": os.path.basename(hazy_path),
        "frames": frames_processed,
        "fps": round(overall_fps, 2),
        "psnr": round(float(np.mean(psnrs)), 2) if psnrs else None,
        "ssim": round(float(np.mean(ssims)), 4) if ssims else None,
        "niqe": round(float(np.mean(niqes)), 2) if niqes else None,
        "temporal_flicker": round(float(np.mean(lum_diffs)), 3) if lum_diffs else None,
        "black_rate_pct": round(float(np.mean(black_rates)), 2),
        "white_rate_pct": round(float(np.mean(white_rates)), 2)
    }

if __name__ == '__main__':
    pipeline = RealTimeDehazingPipeline()
    base_tv = r'c:\Users\sakth\OneDrive\Desktop\Fog\fog-removal-project\test_videos'
    
    videos = [
        ("dense_fog", os.path.join(base_tv, "foggy_video1.mp4"), None),
        ("dynamic_urban", os.path.join(base_tv, "foggy_video2.mp4"), None),
        ("synthetic_gt", os.path.join(base_tv, "car-detection_patchy_haze.mp4"), os.path.join(base_tv, "car-detection_clean.mp4")),
        ("difficult_night", os.path.join(base_tv, "foggy_video3.mp4"), None),
    ]
    
    results = {}
    for tag, hazy_p, clean_p in videos:
        print(f"Evaluating {tag} ({os.path.basename(hazy_p)})...")
        res = evaluate_video(pipeline, hazy_p, clean_p, max_frames=50)
        results[tag] = res
        print(" ", res)
        
    out_path = r'c:\Users\sakth\OneDrive\Desktop\Fog\fog-removal-project\scratch\phase0_baseline_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved baseline reference to {out_path}")
