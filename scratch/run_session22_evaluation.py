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

def run_session22_benchmark(max_frames=50):
    print("=== Session 22: Controlled 4-Ablation Benchmark ===")
    base_tv = r"c:\Users\sakth\OneDrive\Desktop\Fog\fog-removal-project\test_videos"
    vis_dir = r"c:\Users\sakth\OneDrive\Desktop\Fog\fog-removal-project\scratch\session22_visuals"
    os.makedirs(vis_dir, exist_ok=True)

    videos = [
        ("dense_fog", os.path.join(base_tv, "foggy_video1.mp4"), None),
        ("dynamic_urban", os.path.join(base_tv, "foggy_video2.mp4"), None),
        ("synthetic_gt", os.path.join(base_tv, "car-detection_patchy_haze.mp4"), os.path.join(base_tv, "car-detection_clean.mp4")),
        ("difficult_night", os.path.join(base_tv, "foggy_video3.mp4"), None),
    ]

    modes = [
        ("A1_baseline", None, False, True),
        ("A2_no_temp_loss", "checkpoints/delta_t_tiny_unet_no_temp.pt", True, False),
        ("A3_temp_loss_no_inf", "checkpoints/delta_t_tiny_unet_best.pt", True, False),
        ("A4_proposed_full", "checkpoints/delta_t_tiny_unet_best.pt", True, True),
    ]

    all_results = {}

    for mode_id, ckpt_p, enable_learned, enable_inf_temp in modes:
        print(f"\n=================================================================")
        print(f"EVALUATING MODE: {mode_id} (ckpt={ckpt_p}, inf_temp={enable_inf_temp})")
        print(f"=================================================================")
        
        pipeline = LearnedResidualDehazingPipeline(
            checkpoint_path=ckpt_p,
            enable_learned_residual=enable_learned,
            enable_inference_temporal=enable_inf_temp,
        )

        mode_results = {}

        for tag, hazy_path, clean_path in videos:
            print(f"  --> Processing {tag} ({os.path.basename(hazy_path)})...")
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
            t_maes = []

            prev_lum = None
            saved_vis = False

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

                # Record delta_t
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

                # Clean GT metrics if available
                if clean_reader is not None:
                    try:
                        clean_f = clean_reader.get_data(i)
                        if clean_f.shape != out_f.shape:
                            clean_f = np.array(Image.fromarray(clean_f).resize((out_f.shape[1], out_f.shape[0]), Image.BILINEAR))
                        p = calculate_psnr(out_f, clean_f)
                        s = calculate_ssim(out_f, clean_f)
                        psnrs.append(float(p))
                        ssims.append(float(s))

                        # Transmission ground-truth calculation for car-detection
                        # Synthetic patchy haze uses A=255
                        A_true = 255.0
                        diff_h = np.abs(frame.astype(np.float32) - A_true)
                        diff_c = np.abs(clean_f.astype(np.float32) - A_true)
                        valid_mask = diff_c > 15.0
                        if np.any(valid_mask):
                            t_true_map = np.clip(diff_h / np.maximum(diff_c, 1e-6), 0.05, 1.0)
                            t_pred_full = np.array(Image.fromarray(diag["t_temporal"], mode="F").resize((frame.shape[1], frame.shape[0]), Image.BILINEAR))
                            t_mae = np.mean(np.abs(t_pred_full[valid_mask[:, :, 0]] - t_true_map[:, :, 0][valid_mask[:, :, 0]]))
                            t_maes.append(float(t_mae))
                    except Exception:
                        pass

                # Save visuals on frame 15
                if i == 15 and not saved_vis and mode_id == "A4_proposed_full":
                    saved_vis = True
                    dt_vis = np.clip((dt + 0.30) / 0.60 * 255.0, 0, 255).astype(np.uint8)
                    Image.fromarray(diag["t_fused"] * 255.0).convert("L").save(os.path.join(vis_dir, f"{tag}_t_fused.png"))
                    Image.fromarray(dt_vis).save(os.path.join(vis_dir, f"{tag}_delta_t.png"))
                    Image.fromarray(diag["t_constrained"] * 255.0).convert("L").save(os.path.join(vis_dir, f"{tag}_t_constrained.png"))
                    Image.fromarray(out_f).save(os.path.join(vis_dir, f"{tag}_dehazed_{mode_id}.jpg"))

            total_e2e_time = time.perf_counter() - t_e2e_start
            reader.close()
            if clean_reader:
                clean_reader.close()

            dt_arr = np.array(delta_ts)
            proc_fps = 1000.0 / float(np.mean(t_proc_list))
            e2e_fps = frames_proc / total_e2e_time

            res = {
                "video": os.path.basename(hazy_path),
                "mode": mode_id,
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
                    "total_proc": round(float(np.mean(t_proc_list)), 2),
                },
                "psnr": round(float(np.mean(psnrs)), 2) if psnrs else None,
                "ssim": round(float(np.mean(ssims)), 4) if ssims else None,
                "transmission_mae": round(float(np.mean(t_maes)), 4) if t_maes else None,
                "niqe": round(float(np.mean(niqes)), 2) if niqes else None,
                "temporal_flicker": round(float(np.mean(lum_diffs)), 3) if lum_diffs else None,
                "black_rate_pct": round(float(np.mean(black_rates)), 2),
                "white_rate_pct": round(float(np.mean(white_rates)), 2),
                "delta_t_stats": {
                    "mean": round(float(np.mean(dt_arr)), 4),
                    "std": round(float(np.std(dt_arr)), 4),
                    "min": round(float(np.min(dt_arr)), 4),
                    "max": round(float(np.max(dt_arr)), 4),
                    "pct_above_pos_0_05": round(float(np.count_nonzero(dt_arr > 0.05) / dt_arr.size * 100.0), 2),
                    "pct_below_neg_0_05": round(float(np.count_nonzero(dt_arr < -0.05) / dt_arr.size * 100.0), 2),
                    "pct_near_zero_0_01": round(float(np.count_nonzero(np.abs(dt_arr) < 0.01) / dt_arr.size * 100.0), 2),
                }
            }
            mode_results[tag] = res
            print(f"    FPS: {res['fps_processing_only']} (Proc), {res['fps_end_to_end']} (E2E) | PSNR: {res['psnr']}, SSIM: {res['ssim']}, NIQE: {res['niqe']}, Flicker: {res['temporal_flicker']}")
            print(f"    Black: {res['black_rate_pct']}%, White: {res['white_rate_pct']}%, Delta-t Mean: {res['delta_t_stats']['mean']}, Range: [{res['delta_t_stats']['min']}, {res['delta_t_stats']['max']}]")

        all_results[mode_id] = mode_results

    out_file = r"c:\Users\sakth\OneDrive\Desktop\Fog\fog-removal-project\scratch\session22_evaluation_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll 4 ablation benchmark results saved to {out_file}")
    return all_results

if __name__ == "__main__":
    run_session22_benchmark()
