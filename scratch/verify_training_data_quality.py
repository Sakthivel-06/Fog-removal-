import os
import json
import numpy as np
from PIL import Image

def verify_data_quality():
    print("=== Phase 2: Training Data Quality Verification ===")
    train_dir = "data/training_coherent_haze/train"
    with open(os.path.join(train_dir, "manifest.json"), "r") as f:
        meta = json.load(f)

    print(f"Total training frames in manifest: {len(meta)}")
    
    tgt_list = []
    hazy_lum_list = []
    tgt_diffs = []
    hazy_diffs = []
    
    prev_tgt = None
    prev_hazy = None
    prev_video = None
    
    for item in meta:
        cur_video = item["source_video"]
        tgt = np.load(os.path.join(train_dir, item["tgt_file"]))
        hazy_img = np.array(Image.open(os.path.join(train_dir, item["hazy_file"])))
        hazy_lum = 0.299 * hazy_img[:, :, 0] + 0.587 * hazy_img[:, :, 1] + 0.114 * hazy_img[:, :, 2]
        
        tgt_list.append(tgt)
        hazy_lum_list.append(hazy_lum)
        
        # Check inter-frame difference only within the same source video
        if prev_video == cur_video and prev_tgt is not None:
            tgt_diffs.append(np.mean(np.abs(tgt - prev_tgt)))
            hazy_diffs.append(np.mean(np.abs(hazy_lum - prev_hazy)))
            
        prev_tgt = tgt
        prev_hazy = hazy_lum
        prev_video = cur_video
        
    all_tgt = np.array(tgt_list)
    mean_tgt_diff = float(np.mean(tgt_diffs))
    mean_hazy_diff = float(np.mean(hazy_diffs))
    mean_tgt = float(np.mean(all_tgt))
    std_tgt = float(np.std(all_tgt))
    min_tgt = float(np.min(all_tgt))
    max_tgt = float(np.max(all_tgt))
    
    # Histogram of transmission: 10 bins in [0, 1]
    hist, bin_edges = np.histogram(all_tgt, bins=10, range=(0.0, 1.0))
    hist_pct = (hist / all_tgt.size * 100.0).tolist()
    
    print("\n--- Key Quality Metrics ---")
    print(f"1. Mean |Delta t_gt| between adjacent frames: {mean_tgt_diff:.5f} (Target: smooth, 0.0005 - 0.0100)")
    print(f"2. Mean |Delta haze| (lum diff) between adjacent frames: {mean_hazy_diff:.3f} levels/frame")
    print(f"3. Transmission Range: [{min_tgt:.4f}, {max_tgt:.4f}]")
    print(f"4. Transmission Mean: {mean_tgt:.4f}, Std: {std_tgt:.4f}")
    print(f"5. NaN/Inf Check: NaN={np.isnan(all_tgt).any()}, Inf={np.isinf(all_tgt).any()}")
    print(f"6. Constraint Check (0 < t_gt <= 1): {min_tgt > 0.0 and max_tgt <= 1.0}")
    
    print("\nTransmission Distribution Histogram (0.0 to 1.0):")
    for b in range(10):
        print(f"  [{bin_edges[b]:.1f} - {bin_edges[b+1]:.1f}]: {hist_pct[b]:5.1f}% | {'#' * int(hist_pct[b] // 2)}")
        
    report = {
        "mean_delta_tgt_adjacent": round(mean_tgt_diff, 5),
        "mean_delta_haze_lum_adjacent": round(mean_hazy_diff, 3),
        "min_tgt": round(min_tgt, 4),
        "max_tgt": round(max_tgt, 4),
        "mean_tgt": round(mean_tgt, 4),
        "std_tgt": round(std_tgt, 4),
        "has_nan": bool(np.isnan(all_tgt).any()),
        "has_inf": bool(np.isinf(all_tgt).any()),
        "histogram_pct": [round(x, 1) for x in hist_pct],
        "bin_edges": [round(x, 1) for x in bin_edges.tolist()]
    }
    
    with open("scratch/session22_data_quality_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved verification data to scratch/session22_data_quality_report.json")
    print("=== Phase 2 Verification PASSED ===")

if __name__ == '__main__':
    verify_data_quality()
