import os
import sys
import time
import pandas as pd
import numpy as np
import imageio.v3 as iio
from PIL import Image

# Add algorithms to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "algorithms")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "novel_method")))

# Import algorithms
import dcp
import cap
import clahe_baseline
import gamma_baseline
import ceef
from light_dehazenet_wrapper import LightDehazeNetWrapper
from pipeline import RealTimeDehazingPipeline
import metrics

def evaluate_video(hazy_path, clean_path, algo_name, algo_func, max_frames=20):
    """
    Evaluates an algorithm on a video and returns averaged metrics.
    """
    try:
        reader_hazy = iio.imiter(hazy_path, plugin="pyav")
        reader_clean = iio.imiter(clean_path, plugin="pyav")
    except Exception as e:
        print(f"Failed to open {hazy_path} or {clean_path}: {e}")
        return None
        
    metrics_sum = {
        'Entropy': 0.0,
        'Colorfulness': 0.0,
        'Fog Density': 0.0,
        'PSNR': 0.0,
        'SSIM': 0.0,
        'FPS': 0.0
    }
    
    total_proc_time = 0.0
    evaluated_frames = 0
    frame_count = 0
    
    for idx, (frame_h, frame_c) in enumerate(zip(reader_hazy, reader_clean)):
        if frame_count >= max_frames:
            break
            
        start_t = time.perf_counter()
        dehazed = algo_func(frame_h)
        end_t = time.perf_counter()
        elapsed = end_t - start_t
        
        # Warmup first 2 frames to exclude JIT and module cold-start from FPS
        if idx >= 2:
            total_proc_time += elapsed
            evaluated_frames += 1
        
        # Calculate metrics
        metrics_sum['Entropy'] += metrics.calculate_entropy(dehazed)
        metrics_sum['Colorfulness'] += metrics.calculate_colorfulness(dehazed)
        metrics_sum['Fog Density'] += metrics.calculate_fog_density_proxy(dehazed)
        metrics_sum['PSNR'] += metrics.calculate_psnr(frame_c, dehazed)
        
        try:
            metrics_sum['SSIM'] += metrics.calculate_ssim(frame_c, dehazed)
        except Exception:
            metrics_sum['SSIM'] += 0.0 # Fallback
            
        frame_count += 1
    
    if frame_count == 0:
        return None
        
    # Averages for image quality metrics
    for k in ['Entropy', 'Colorfulness', 'Fog Density', 'PSNR', 'SSIM']:
        metrics_sum[k] /= frame_count

    # Robust Throughput FPS: evaluated_frames / total_proc_time
    if algo_name == 'Original Hazy':
        metrics_sum['FPS'] = np.nan
    else:
        metrics_sum['FPS'] = evaluated_frames / total_proc_time if total_proc_time > 0 else 0.0
        
    return metrics_sum

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate video dehazing algorithms")
    parser.add_argument("--max-frames", "-m", type=int, default=15, help="Number of frames to evaluate per video (default: 15)")
    args = parser.parse_args()

    test_videos_dir = "test_videos"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Define test pairs (hazy, clean)
    # Note: person-bicycle_patchy_haze.mp4 does not exist; use uniform_haze variant instead
    video_pairs = [
        ("car-detection_patchy_haze.mp4", "car-detection_clean.mp4"),
        ("person-bicycle_uniform_haze.mp4", "person-bicycle_clean.mp4")
    ]
    
    # Initialize stateful algorithms
    print("Loading models...")
    try:
        ldn_wrapper = LightDehazeNetWrapper()
        ldn_func = ldn_wrapper.process_frame
    except Exception as e:
        print(f"Warning: Light-DehazeNet failed to load: {e}")
        ldn_func = None
        
    novel_pipeline = RealTimeDehazingPipeline(
        downscale_res=(480, 270),
        omega=0.85,
        beta=1.0,
        kappa=1.20,
        enable_sky_model=True,
        preserve_resolution=True,
    )
    
    # Base baseline dummy function for original hazy frame
    def original_baseline(frame): return frame

    algorithms = {
        'Original Hazy': original_baseline,
        'CLAHE': clahe_baseline.process_frame,
        'Gamma (2.0)': lambda f: gamma_baseline.process_frame(f, 2.0),
        'CEEF': ceef.process_frame,
        'CAP': cap.process_frame,
        'DCP': dcp.process_frame,
        'Novel Method (Ours)': lambda f: novel_pipeline.process_frame(f)[0],
    }
    
    if ldn_func:
        _raw_ldn = ldn_func
        def _ldn_func(frame):
            result = _raw_ldn(frame)
            return result[0] if isinstance(result, tuple) else result
        algorithms['Light-DehazeNet'] = _ldn_func

    all_results = []
    
    for hazy_name, clean_name in video_pairs:
        hazy_path = os.path.join(test_videos_dir, hazy_name)
        clean_path = os.path.join(test_videos_dir, clean_name)
        
        if not os.path.exists(hazy_path) or not os.path.exists(clean_path):
            print(f"Skipping {hazy_name}, files not found.", flush=True)
            continue
            
        print(f"--- Evaluating video: {hazy_name} ---", flush=True)
        for algo_name, algo_func in algorithms.items():
            print(f"Running {algo_name}...", flush=True)
            if "Novel" in algo_name:
                novel_pipeline.reset()
            res = evaluate_video(hazy_path, clean_path, algo_name, algo_func, max_frames=args.max_frames)
            if res:
                res['Algorithm'] = algo_name
                res['Video'] = hazy_name
                all_results.append(res)
                print(f"  -> FPS: {res['FPS']:.1f}, PSNR: {res['PSNR']:.2f}, SSIM: {res['SSIM']:.3f}, Fog: {res['Fog Density']:.3f}", flush=True)

    if all_results:
        df = pd.DataFrame(all_results)
        
        # Average across videos
        avg_df = df.groupby('Algorithm').mean(numeric_only=True).reset_index()
        
        # Sort by PSNR descending
        avg_df = avg_df.sort_values('PSNR', ascending=False)
        
        csv_path = os.path.join(results_dir, "evaluation_results.csv")
        avg_df.to_csv(csv_path, index=False)
        print(f"\nSaved results to {csv_path}")
        print("\nConsolidated Results:")
        print(avg_df.to_string(index=False))
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()
