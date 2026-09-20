import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))

from novel_method.pipeline_learned_experimental import LearnedResidualDehazingPipeline
import imageio.v3 as iio

def analyze_fully_trained_regions():
    pipeline = LearnedResidualDehazingPipeline(checkpoint_path="checkpoints/delta_t_tiny_unet_best.pt")
    
    # Analyze across 2 videos: foggy_video1 (dense fog) and car-detection (synthetic)
    videos = [
        ("foggy_video1", "test_videos/foggy_video1.mp4", 15),
        ("car_detection", "test_videos/car-detection_patchy_haze.mp4", 15),
    ]
    
    report = {}
    for name, vpath, fidx in videos:
        frame = iio.imread(vpath, index=fidx)
        out, fps, diag = pipeline.process_frame(frame, return_diagnostics=True)
        dt = diag["delta_t"]
        tf = diag["t_fused"]
        tc = diag["t_constrained"]
        gh, gw = dt.shape
        
        sky_m = np.zeros_like(dt, dtype=bool)
        sky_m[:int(gh * 0.3), :] = True
        
        road_m = np.zeros_like(dt, dtype=bool)
        road_m[int(gh * 0.7):, :] = True
        
        mid_m = ~(sky_m | road_m)
        
        report[name] = {
            "overall": {"mean": float(np.mean(dt)), "std": float(np.std(dt)), "min": float(np.min(dt)), "max": float(np.max(dt))},
            "sky": {"mean_dt": float(np.mean(dt[sky_m])), "std_dt": float(np.std(dt[sky_m])), "t_fused": float(np.mean(tf[sky_m])), "t_constrained": float(np.mean(tc[sky_m]))},
            "road": {"mean_dt": float(np.mean(dt[road_m])), "std_dt": float(np.std(dt[road_m])), "t_fused": float(np.mean(tf[road_m])), "t_constrained": float(np.mean(tc[road_m]))},
            "midground": {"mean_dt": float(np.mean(dt[mid_m])), "std_dt": float(np.std(dt[mid_m])), "t_fused": float(np.mean(tf[mid_m])), "t_constrained": float(np.mean(tc[mid_m]))},
        }
        
    print("=== Regional Delta-t Analysis (Fully Trained TinyUNet) ===")
    for vname, vdata in report.items():
        print(f"\nVideo: {vname}")
        for reg, rstats in vdata.items():
            print(f"  [{reg}]: {rstats}")
            
    with open("scratch/session22_regional_delta_t.json", "w") as f:
        json.dump(report, f, indent=2)

if __name__ == '__main__':
    analyze_fully_trained_regions()
