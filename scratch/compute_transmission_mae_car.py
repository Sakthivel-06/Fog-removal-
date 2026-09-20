import os
import sys
sys.path.insert(0, ".")
import numpy as np
import imageio
from novel_method.pipeline import RealTimeDehazingPipeline
from novel_method.pipeline_learned_experimental import LearnedResidualDehazingPipeline

def compute_t_mae():
    hazy_path = "test_videos/car-detection_patchy_haze.mp4"
    clean_path = "test_videos/car-detection_clean.mp4"
    
    rh = imageio.get_reader(hazy_path)
    rc = imageio.get_reader(clean_path)
    
    # Initialize pipelines
    pipe_base = RealTimeDehazingPipeline(downscale_res=(240, 135))
    pipe_learned = LearnedResidualDehazingPipeline(
        checkpoint_path="checkpoints/delta_t_tiny_unet_best.pt",
        downscale_res=(240, 135),
        ablation_mode="A4"
    )
    
    A_vec = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    
    mae_fused_list = []
    mae_candidate_list = []
    mae_constrained_list = []
    mae_temporal_list = []
    
    for i in range(25):
        fh = rh.get_data(i)
        fc = rc.get_data(i)
        
        # Recover reference t_gt at full resolution then downscale to 240x135 for fair comparison with internal t
        denom = fc.astype(np.float32) - A_vec
        numer = fh.astype(np.float32) - A_vec
        valid_mask = np.all(np.abs(denom) >= 25.0, axis=-1)
        t_ch = numer / (denom + 1e-6)
        t_ref = np.clip(np.median(t_ch, axis=-1), 0.05, 1.0)
        
        # Run learned pipeline
        pipe_learned.process_frame(fh)
        t_fused = pipe_learned.last_t_fused
        t_candidate = pipe_learned.last_t_candidate
        t_constrained = pipe_learned.last_t_constrained
        t_temporal = pipe_learned.last_t_temporal
        
        # Downscale t_ref and valid_mask to match t_fused.shape
        gh, gw = t_fused.shape
        from PIL import Image
        t_ref_img = Image.fromarray(t_ref).resize((gw, gh), Image.Resampling.BILINEAR)
        mask_img = Image.fromarray(valid_mask.astype(np.uint8)*255).resize((gw, gh), Image.Resampling.NEAREST)
        
        t_ref_small = np.array(t_ref_img)
        mask_small = np.array(mask_img) > 128
        
        if np.any(mask_small):
            mae_fused_list.append(np.mean(np.abs(t_fused[mask_small] - t_ref_small[mask_small])))
            mae_candidate_list.append(np.mean(np.abs(t_candidate[mask_small] - t_ref_small[mask_small])))
            mae_constrained_list.append(np.mean(np.abs(t_constrained[mask_small] - t_ref_small[mask_small])))
            mae_temporal_list.append(np.mean(np.abs(t_temporal[mask_small] - t_ref_small[mask_small])))
            
    rh.close()
    rc.close()
    
    print("=== Transmission MAE against Ground Truth Reference (car-detection) ===")
    print(f"t_fused MAE (Baseline A1):       {np.mean(mae_fused_list):.4f}")
    print(f"t_candidate MAE (Physics + CNN): {np.mean(mae_candidate_list):.4f}")
    print(f"t_constrained MAE (+ Adaptive):  {np.mean(mae_constrained_list):.4f}")
    print(f"t_temporal MAE (Full A4):        {np.mean(mae_temporal_list):.4f}")
    
    mean_t_ref = np.mean(t_ref_small[mask_small])
    mean_t_fused = np.mean(t_fused[mask_small])
    mean_t_cand = np.mean(t_candidate[mask_small])
    mean_t_const = np.mean(t_constrained[mask_small])
    print(f"\nMean Transmission Values:")
    print(f"  t_ref:        {mean_t_ref:.4f}")
    print(f"  t_fused:      {mean_t_fused:.4f} (Bias: {mean_t_fused - mean_t_ref:+.4f})")
    print(f"  t_candidate:  {mean_t_cand:.4f} (Bias: {mean_t_cand - mean_t_ref:+.4f})")
    print(f"  t_constrained:{mean_t_const:.4f} (Bias: {mean_t_const - mean_t_ref:+.4f})")

if __name__ == "__main__":
    compute_t_mae()
