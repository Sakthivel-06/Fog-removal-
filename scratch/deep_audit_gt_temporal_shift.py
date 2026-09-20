"""
scratch/deep_audit_gt_temporal_shift.py — Temporal Alignment Audit
"""
import imageio
import numpy as np

def check_temporal(hazy_path, clean_path, name):
    print(f"\nChecking Temporal Alignment for: {name}")
    rh = imageio.get_reader(hazy_path)
    rc = imageio.get_reader(clean_path)
    
    # Check 5 frames: 10, 20, 30, 40, 50
    for target_t in [10, 25, 40]:
        fh = rh.get_data(target_t)
        gh = 0.299 * fh[:, :, 0] + 0.587 * fh[:, :, 1] + 0.114 * fh[:, :, 2]
        
        # Test clean frames from target_t - 2 to target_t + 2
        corrs = {}
        for offset in [-2, -1, 0, 1, 2]:
            fc = rc.get_data(target_t + offset)
            gc = 0.299 * fc[:, :, 0] + 0.587 * fc[:, :, 1] + 0.114 * fc[:, :, 2]
            corr = float(np.corrcoef(gh.flatten(), gc.flatten())[0, 1])
            corrs[offset] = corr
            
        best_offset = max(corrs, key=corrs.get)
        print(f"Frame {target_t:2d} -> Offset Correlations: {corrs}")
        print(f"Frame {target_t:2d} -> Peak Offset: {best_offset} (Correlation={corrs[best_offset]:.4f})")
    
    rh.close()
    rc.close()

check_temporal("test_videos/car-detection_patchy_haze.mp4", "test_videos/car-detection_clean.mp4", "Car Detection")
check_temporal("test_videos/person-bicycle_uniform_haze.mp4", "test_videos/person-bicycle_clean.mp4", "Person Bicycle")
