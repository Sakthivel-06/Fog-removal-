import os
import sys
import json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))

from novel_method.pipeline_learned_experimental import LearnedResidualDehazingPipeline
import imageio.v3 as iio

def analyze_regions():
    pipeline = LearnedResidualDehazingPipeline(checkpoint_path="checkpoints/delta_t_tiny_unet_debug.pt")
    frame = iio.imread("test_videos/foggy_video1.mp4", index=15)
    out, fps, diag = pipeline.process_frame(frame, return_diagnostics=True)
    
    dt = diag["delta_t"] # (gh, gw)
    tf = diag["t_fused"]
    tc = diag["t_constrained"]
    gh, gw = dt.shape
    
    # Regional segmentations based on geometric altitude and luminance
    # Top 30% = sky/distant fog
    sky_mask = np.zeros_like(dt, dtype=bool)
    sky_mask[:int(gh * 0.3), :] = True
    
    # Bottom 30% = road/foreground
    road_mask = np.zeros_like(dt, dtype=bool)
    road_mask[int(gh * 0.7):, :] = True
    
    # Mid 40% = vehicles / buildings / mid-ground
    mid_mask = ~(sky_mask | road_mask)
    
    report = {
        "overall": {
            "mean": float(np.mean(dt)),
            "std": float(np.std(dt)),
            "min": float(np.min(dt)),
            "max": float(np.max(dt)),
        },
        "sky_region": {
            "mean": float(np.mean(dt[sky_mask])),
            "std": float(np.std(dt[sky_mask])),
            "min": float(np.min(dt[sky_mask])),
            "max": float(np.max(dt[sky_mask])),
            "t_fused_mean": float(np.mean(tf[sky_mask])),
            "t_constrained_mean": float(np.mean(tc[sky_mask])),
        },
        "road_region": {
            "mean": float(np.mean(dt[road_mask])),
            "std": float(np.std(dt[road_mask])),
            "min": float(np.min(dt[road_mask])),
            "max": float(np.max(dt[road_mask])),
            "t_fused_mean": float(np.mean(tf[road_mask])),
            "t_constrained_mean": float(np.mean(tc[road_mask])),
        },
        "midground_vehicles": {
            "mean": float(np.mean(dt[mid_mask])),
            "std": float(np.std(dt[mid_mask])),
            "min": float(np.min(dt[mid_mask])),
            "max": float(np.max(dt[mid_mask])),
            "t_fused_mean": float(np.mean(tf[mid_mask])),
            "t_constrained_mean": float(np.mean(tc[mid_mask])),
        }
    }
    
    print("=== Regional Delta-t Analysis (foggy_video1, frame 15) ===")
    for k, v in report.items():
        print(f"Region: {k}")
        for param, val in v.items():
            print(f"  {param}: {val:.4f}")
            
    with open("scratch/session21_regional_delta_t.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == '__main__':
    analyze_regions()
