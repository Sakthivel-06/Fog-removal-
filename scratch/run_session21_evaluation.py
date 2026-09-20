import os
import sys
import time
import json
import numpy as np
import imageio
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))

from novel_method.pipeline_learned_experimental import LearnedResidualDehazingPipeline
from metrics import calculate_niqe, calculate_ssim, calculate_psnr

def run_evaluation(checkpoint_path="checkpoints/delta_t_tiny_unet_debug.pt", max_frames=50):
    print("=== Running Session 21 Integrated Evaluation ===")
    base_tv = r"c:\Users\sakth\OneDrive\Desktop\Fog\fog-removal-project\test_videos"
    
    videos = [
        ("dense_fog", os.path.join(base_tv, "foggy_video1.mp4"), None),
        ("dynamic_urban", os.path.join(base_tv, "foggy_video2.mp4"), None),
        ("synthetic_gt", os.path.join(base_tv, "car-detection_patchy_haze.mp4"), os.path.join(base_tv, "car-detection_clean.mp4")),
        ("difficult_night", os.path.join(base_tv, "foggy_video3.mp4"), None),
    ]

    pipeline = LearnedResidualDehazingPipeline(
        checkpoint_path=checkpoint_path,
        enable_learned_residual=True,
    )
    
    # Visual output directory
    vis_dir = r"c:\Users\sakth\OneDrive\Desktop\Fog\fog-removal-project\scratch\session21_visuals"
    os.makedirs(vis_dir, exist_ok=True)
    
    results = {}
    
    for tag, hazy_path, clean_path in videos:
        print(f"\nProcessing {tag} ({os.path.basename(hazy_path)})...")
        pipeline.reset()
        reader = imageio.get_reader(hazy_path)
        clean_reader = imageio.get_reader(clean_path) if clean_path and os.path.exists(clean_path) else None
        
        frames_proc = 0
        t_e2e_start = time.perf_counter()
        
        t_decode_list = []
        t_proc_list = []
        t_prep_list = []
        t_physics_list = []
        t_cnn_list = []
        t_guided_list = []
        t_temp_list = []
        t_inv_list = []
        
        psnrs = []
        ssims = []
        niqes = []
        lum_diffs = []
        black_rates = []
        white_rates = []
        delta_ts = []
        
        prev_lum = None
        saved_visuals = False
        
        for i in range(max_frames):
            t_d0 = time.perf_counter()
            try:
                frame = reader.get_data(i)
            except Exception:
                break
            t_decode_list.append((time.perf_counter() - t_d0) * 1000.0)
            
            t_p0 = time.perf_counter()
            out_f, fps, diag = pipeline.process_frame(frame, return_diagnostics=True)
            t_proc_list.append((time.perf_counter() - t_p0) * 1000.0)
            
            prof = diag["latency_profile"]
            t_prep_list.append(prof.get("prep", 0.0))
            t_physics_list.append(prof.get("physics_features", 0.0))
            t_cnn_list.append(prof.get("cnn_inference", 0.0))
            t_guided_list.append(prof.get("guided_smoothing", 0.0))
            t_temp_list.append(prof.get("temporal_coherence", 0.0))
            t_inv_list.append(prof.get("inversion", 0.0))
            
            frames_proc += 1
            
            # Record delta_t values
            dt = diag["delta_t"]
            delta_ts.extend(dt.flatten())
            
            # Black / White rates
            black_rates.append(float(np.count_nonzero(out_f <= 2) / out_f.size * 100.0))
            white_rates.append(float(np.count_nonzero(out_f >= 253) / out_f.size * 100.0))
            
            # Temporal flicker
            cur_lum = 0.299 * out_f[:, :, 0] + 0.587 * out_f[:, :, 1] + 0.114 * out_f[:, :, 2]
            if prev_lum is not None and prev_lum.shape == cur_lum.shape:
                lum_diffs.append(float(np.mean(np.abs(cur_lum - prev_lum))))
            prev_lum = cur_lum
            
            # NIQE every 10 frames
            if i % 10 == 0:
                try:
                    niqes.append(float(calculate_niqe(out_f)))
                except Exception:
                    pass
                    
            # GT metrics
            if clean_reader is not None:
                try:
                    clean_f = clean_reader.get_data(i)
                    if clean_f.shape != out_f.shape:
                        clean_f = np.array(Image.fromarray(clean_f).resize((out_f.shape[1], out_f.shape[0]), Image.BILINEAR))
                    p = calculate_psnr(out_f, clean_f)
                    s = calculate_ssim(out_f, clean_f)
                    psnrs.append(float(p))
                    ssims.append(float(s))
                except Exception:
                    pass
                    
            # Save visual multi-map on frame 15
            if i == 15 and not saved_visuals:
                saved_visuals = True
                # Normalize delta_t for visualization: map [-0.30, +0.30] -> [0, 255]
                dt_vis = np.clip((dt + 0.30) / 0.60 * 255.0, 0, 255).astype(np.uint8)
                tf_vis = np.clip(diag["t_fused"] * 255.0, 0, 255).astype(np.uint8)
                tc_vis = np.clip(diag["t_candidate"] * 255.0, 0, 255).astype(np.uint8)
                tcon_vis = np.clip(diag["t_constrained"] * 255.0, 0, 255).astype(np.uint8)
                ttemp_vis = np.clip(diag["t_temporal"] * 255.0, 0, 255).astype(np.uint8)
                
                Image.fromarray(tf_vis).save(os.path.join(vis_dir, f"{tag}_t_fused.png"))
                Image.fromarray(dt_vis).save(os.path.join(vis_dir, f"{tag}_delta_t.png"))
                Image.fromarray(tc_vis).save(os.path.join(vis_dir, f"{tag}_t_candidate.png"))
                Image.fromarray(tcon_vis).save(os.path.join(vis_dir, f"{tag}_t_constrained.png"))
                Image.fromarray(ttemp_vis).save(os.path.join(vis_dir, f"{tag}_t_temporal.png"))
                Image.fromarray(out_f).save(os.path.join(vis_dir, f"{tag}_dehazed_rgb.jpg"))
                Image.fromarray(frame).save(os.path.join(vis_dir, f"{tag}_input_hazy.jpg"))
                
        total_e2e_time = time.perf_counter() - t_e2e_start
        reader.close()
        if clean_reader:
            clean_reader.close()
            
        dt_arr = np.array(delta_ts)
        pos_pct = float(np.count_nonzero(dt_arr > 0.05) / dt_arr.size * 100.0)
        neg_pct = float(np.count_nonzero(dt_arr < -0.05) / dt_arr.size * 100.0)
        near_zero_pct = float(np.count_nonzero(np.abs(dt_arr) < 0.01) / dt_arr.size * 100.0)
        
        proc_fps = 1000.0 / float(np.mean(t_proc_list))
        e2e_fps = frames_proc / total_e2e_time
        
        res = {
            "video": os.path.basename(hazy_path),
            "frames": frames_proc,
            "fps_processing_only": round(proc_fps, 2),
            "fps_end_to_end": round(e2e_fps, 2),
            "latency_ms": {
                "decode": round(float(np.mean(t_decode_list)), 2),
                "prep": round(float(np.mean(t_prep_list)), 2),
                "physics_features": round(float(np.mean(t_physics_list)), 2),
                "cnn_inference": round(float(np.mean(t_cnn_list)), 2),
                "guided_smoothing": round(float(np.mean(t_guided_list)), 2),
                "temporal_coherence": round(float(np.mean(t_temp_list)), 2),
                "inversion": round(float(np.mean(t_inv_list)), 2),
                "total_proc": round(float(np.mean(t_proc_list)), 2)
            },
            "psnr": round(float(np.mean(psnrs)), 2) if psnrs else None,
            "ssim": round(float(np.mean(ssims)), 4) if ssims else None,
            "niqe": round(float(np.mean(niqes)), 2) if niqes else None,
            "temporal_flicker": round(float(np.mean(lum_diffs)), 3) if lum_diffs else None,
            "black_rate_pct": round(float(np.mean(black_rates)), 2),
            "white_rate_pct": round(float(np.mean(white_rates)), 2),
            "delta_t_stats": {
                "mean": round(float(np.mean(dt_arr)), 4),
                "std": round(float(np.std(dt_arr)), 4),
                "min": round(float(np.min(dt_arr)), 4),
                "max": round(float(np.max(dt_arr)), 4),
                "pct_above_pos_0_05": round(pos_pct, 2),
                "pct_below_neg_0_05": round(neg_pct, 2),
                "pct_near_zero_0_01": round(near_zero_pct, 2)
            }
        }
        results[tag] = res
        print(f"  FPS (Proc): {res['fps_processing_only']}, FPS (E2E): {res['fps_end_to_end']}")
        print(f"  CNN Latency: {res['latency_ms']['cnn_inference']} ms, Physics: {res['latency_ms']['physics_features']} ms")
        print(f"  Delta-t: Range=[{res['delta_t_stats']['min']}, {res['delta_t_stats']['max']}], Mean={res['delta_t_stats']['mean']}, Std={res['delta_t_stats']['std']}")
        print(f"  PSNR: {res['psnr']}, SSIM: {res['ssim']}, NIQE: {res['niqe']}, Flicker: {res['temporal_flicker']}")
        print(f"  Black: {res['black_rate_pct']}%, White: {res['white_rate_pct']}%")

    out_json = r"c:\Users\sakth\OneDrive\Desktop\Fog\fog-removal-project\scratch\session21_evaluation_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll evaluation metrics saved to {out_json}")
    return results

if __name__ == "__main__":
    run_evaluation()
