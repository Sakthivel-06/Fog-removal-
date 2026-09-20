import os
import sys
import json
import numpy as np
import imageio
from PIL import Image
from skimage.metrics import structural_similarity as sk_ssim

sys.path.insert(0, ".")
from novel_method.pipeline import RealTimeDehazingPipeline
from novel_method.pipeline_learned_experimental import LearnedResidualDehazingPipeline
from novel_method.temporal_coherence import compute_motion_mask

def compute_temporal_study():
    print("=== Computing Session 23 In-Depth Temporal Consistency Study ===")
    
    videos = [
        ("car_detection", "test_videos/car-detection_patchy_haze.mp4", 40),
        ("dense_fog", "test_videos/foggy_video1.mp4", 40),
        ("dynamic_urban", "test_videos/foggy_video2.mp4", 40)
    ]
    
    modes = [
        ("A1_baseline", None, False, True),
        ("A2_no_temp_loss", "checkpoints/delta_t_tiny_unet_no_temp.pt", True, False),
        ("A3_temp_loss_no_inf", "checkpoints/delta_t_tiny_unet_best.pt", True, False),
        ("A4_proposed_full", "checkpoints/delta_t_tiny_unet_best.pt", True, True)
    ]
    
    study_results = {}
    
    for vid_key, vid_path, n_frames in videos:
        print(f"\nProcessing Video: {vid_key} ({n_frames} frames)...")
        study_results[vid_key] = {}
        
        # Pre-load frames
        reader = imageio.get_reader(vid_path)
        raw_frames = [reader.get_data(i) for i in range(n_frames)]
        reader.close()
        
        for mode_name, ckpt_path, enable_learned, enable_inf_temp in modes:
            # Initialize pipeline
            if mode_name == "A1_baseline":
                pipe = RealTimeDehazingPipeline(downscale_res=(240, 135))
            else:
                pipe = LearnedResidualDehazingPipeline(
                    checkpoint_path=ckpt_path,
                    enable_learned_residual=enable_learned,
                    ablation_mode="A4" if enable_inf_temp else ("A3" if "temp_loss" in mode_name else "A2"),
                    enable_inference_temporal=enable_inf_temp,
                    downscale_res=(240, 135)
                )
                
            out_frames = []
            for f in raw_frames:
                out, _ = pipe.process_frame(f)
                out_frames.append(out.astype(np.float32))
                
            # Now compute temporal metrics between consecutive frames k-1 and k
            overall_flicker_list = []
            static_flicker_list = []
            motion_flicker_list = []
            temporal_ssim_list = []
            
            for k in range(1, len(out_frames)):
                f_prev = out_frames[k-1]
                f_curr = out_frames[k]
                
                # Convert to luminance
                lum_prev = 0.299 * f_prev[:,:,0] + 0.587 * f_prev[:,:,1] + 0.114 * f_prev[:,:,2]
                lum_curr = 0.299 * f_curr[:,:,0] + 0.587 * f_curr[:,:,1] + 0.114 * f_curr[:,:,2]
                
                diff = np.abs(lum_curr - lum_prev)
                overall_flicker_list.append(np.mean(diff))
                
                # Motion mask: pixels where inter-frame luminance changes substantially (> 12 levels)
                # indicating genuine motion or lighting transients
                motion_mask = (diff > 12.0)
                static_mask = ~motion_mask
                
                if np.any(static_mask):
                    static_flicker_list.append(np.mean(diff[static_mask]))
                else:
                    static_flicker_list.append(np.mean(diff))
                    
                if np.any(motion_mask):
                    motion_flicker_list.append(np.mean(diff[motion_mask]))
                else:
                    motion_flicker_list.append(0.0)
                    
                # Temporal SSIM between consecutive frames
                # Downsample slightly for speed
                t_ssim = sk_ssim(lum_curr.astype(np.uint8), lum_prev.astype(np.uint8), data_range=255)
                temporal_ssim_list.append(t_ssim)
                
            res = {
                "overall_flicker": float(np.mean(overall_flicker_list)),
                "static_flicker": float(np.mean(static_flicker_list)),
                "motion_flicker": float(np.mean(motion_flicker_list)),
                "temporal_ssim": float(np.mean(temporal_ssim_list))
            }
            study_results[vid_key][mode_name] = res
            print(f"  [{mode_name}]: Overall={res['overall_flicker']:.3f}, Static={res['static_flicker']:.3f}, Motion={res['motion_flicker']:.3f}, TempSSIM={res['temporal_ssim']:.4f}")
            
    with open("scratch/session23_temporal_study.json", "w") as f:
        json.dump(study_results, f, indent=2)
    print("\nSaved temporal study to scratch/session23_temporal_study.json")

if __name__ == "__main__":
    compute_temporal_study()
