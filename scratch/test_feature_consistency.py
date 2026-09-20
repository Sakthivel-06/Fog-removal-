import os
import sys
import numpy as np
import imageio.v3 as iio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))

from novel_method.pipeline import RealTimeDehazingPipeline
from novel_method.learned_residual.feature_extractor import extract_physics_features

def test_feature_consistency():
    pipeline = RealTimeDehazingPipeline()
    # Read one test frame from foggy_video1
    r = iio.imread("test_videos/foggy_video1.mp4", index=0)
    
    # Run through pipeline up to inversion
    # We pass it to extract_physics_features
    # Convert uint8 to proc_float as pipeline does
    proc_float = r.astype(np.float32) / 255.0
    
    features, meta = extract_physics_features(proc_float, pipeline)
    print("Features shape:", features.shape)
    print("Feature channels:")
    names = ["R_guide", "G_guide", "B_guide", "t_fused", "t_cap", "t_dcp", "w_dcp", "veil_proxy"]
    for i, name in enumerate(names):
        ch = features[i]
        print(f"  Ch {i} ({name:>10}): min={ch.min():.4f}, max={ch.max():.4f}, mean={ch.mean():.4f}, has_nan={np.isnan(ch).any()}")
        assert not np.isnan(ch).any(), f"NaN in channel {name}"
        
    print(f"Airlight A_raw: {meta['A_raw']}")
    print(f"Adaptive t_min_map: min={meta['t_min_map'].min():.4f}, max={meta['t_min_map'].max():.4f}")
    print("=== Phase 4 Feature Consistency PASSED ===")

if __name__ == '__main__':
    test_feature_consistency()
