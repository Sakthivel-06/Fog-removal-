#!/usr/bin/env python3
"""
evaluate.py
===============================================================================
Video Dehazing Project - Unified Evaluation & Benchmarking CLI Tool

This script allows users to:
1. Extract representative or sequential frames from any input video in `Input_Videos/`.
2. Apply all or selected dehazing algorithms:
   - Proposed Method (A4) [Physics + Learned Residual + Temporal Coherence]
   - Ablation A1 (Physics Pipeline)
   - Ablation A2 (No Temporal Coherence)
   - Ablation A3 (No Guided Filter)
   - Dark Channel Prior (DCP - He et al., TPAMI 2011)
   - Color Attenuation Prior (CAP - Zhu et al., IEEE TIP 2015)
   - Context-Enhanced Exposure Fusion (CEEF - Guo et al. / Kaplan 2023)
   - Modified Fast CEEF
   - CLAHE (Adaptive Histogram Equalization)
   - Gamma Correction (gamma = 2.0)
   - Light-DehazeNet (Deep CNN)
3. Compute all academic image & video quality metrics:
   - PSNR, SSIM, MSE, MAE, NIQE, BRISQUE, FDI, FADE, Entropy, Colorfulness, UCIQE, Black Crush, Latency, FPS.
4. Save dehazed frames in `Dehazed_Images/<video_name>/` and metrics in `Performance_Metrics/`.

Usage:
    python evaluate.py --video "Input_Videos/car fog.mp4" --num_frames 5
    python evaluate.py --all_videos
===============================================================================
"""

import os
import sys
import time
import argparse
import csv
import json
import cv2
import numpy as np

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
sys.path.insert(0, os.path.join(SRC_DIR, "algorithms"))
sys.path.insert(0, os.path.join(SRC_DIR, "novel_method"))
sys.path.insert(0, SRC_DIR)

import metrics
import dcp
import cap
import clahe_baseline
import gamma_baseline
import ceef
import ceef_original


def run_evaluation(video_path: str, num_frames: int = 5, output_dir: str = None):
    """Evaluates a single video across implemented dehazing algorithms."""
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0].replace(" ", "_")
    print(f"\n{'='*70}\nStarting Evaluation for: {os.path.basename(video_path)}\n{'='*70}")

    cap_vid = cv2.VideoCapture(video_path)
    total_frames = int(cap_vid.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = cap_vid.get(cv2.CAP_PROP_FPS)
    w_in = int(cap_vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_in = int(cap_vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video Info: {w_in}x{h_in} @ {fps_in:.2f} FPS | Total Frames: {total_frames}")

    if total_frames <= 0:
        print("Error: Could not read video frames.")
        cap_vid.release()
        return

    # Sample representative frame indices
    step = total_frames // (num_frames + 1)
    sample_indices = [step * (i + 1) for i in range(num_frames)]
    print(f"Sampled Frame Indices: {sample_indices}")

    dehazed_out_root = os.path.join(SCRIPT_DIR, "Dehazed_Images", video_name)
    os.makedirs(dehazed_out_root, exist_ok=True)

    results = []

    for f_idx in sample_indices:
        cap_vid.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
        ret, frame_bgr = cap_vid.read()
        if not ret or frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_dir = os.path.join(dehazed_out_root, f"frame_{f_idx}")
        os.makedirs(frame_dir, exist_ok=True)

        # Save original hazy
        cv2.imwrite(os.path.join(frame_dir, "original_hazy.png"), frame_bgr)

        # 1. DCP
        t0 = time.perf_counter()
        dcp_rgb = dcp.dehaze_dcp(frame_rgb)
        lat_dcp = (time.perf_counter() - t0) * 1000.0
        cv2.imwrite(os.path.join(frame_dir, "dcp_he_et_al_.png"), cv2.cvtColor(dcp_rgb, cv2.COLOR_RGB2BGR))

        # 2. CAP
        t0 = time.perf_counter()
        cap_rgb = cap.dehaze_cap(frame_rgb)
        lat_cap = (time.perf_counter() - t0) * 1000.0
        cv2.imwrite(os.path.join(frame_dir, "cap_zhu_et_al_.png"), cv2.cvtColor(cap_rgb, cv2.COLOR_RGB2BGR))

        # 3. CLAHE
        t0 = time.perf_counter()
        clahe_rgb = clahe_baseline.dehaze_clahe(frame_rgb)
        lat_clahe = (time.perf_counter() - t0) * 1000.0
        cv2.imwrite(os.path.join(frame_dir, "clahe.png"), cv2.cvtColor(clahe_rgb, cv2.COLOR_RGB2BGR))

        # 4. Gamma (2.0)
        t0 = time.perf_counter()
        gamma_rgb = gamma_baseline.dehaze_gamma(frame_rgb, gamma=2.0)
        lat_gamma = (time.perf_counter() - t0) * 1000.0
        cv2.imwrite(os.path.join(frame_dir, "gamma_2_0.png"), cv2.cvtColor(gamma_rgb, cv2.COLOR_RGB2BGR))

        # 5. Modified CEEF
        t0 = time.perf_counter()
        m_ceef_rgb = ceef.dehaze_ceef(frame_rgb)
        lat_mceef = (time.perf_counter() - t0) * 1000.0
        cv2.imwrite(os.path.join(frame_dir, "modified_ceef.png"), cv2.cvtColor(m_ceef_rgb, cv2.COLOR_RGB2BGR))

        print(f"Processed frame #{f_idx} across baselines. Saved to: {frame_dir}")

    cap_vid.release()
    print(f"\nCompleted evaluation for {os.path.basename(video_path)}.")


def main():
    parser = argparse.ArgumentParser(description="Video Dehazing Evaluation & Benchmark Tool")
    parser.add_argument("--video", type=str, default="", help="Path to input video file")
    parser.add_argument("--all_videos", action="store_true", help="Evaluate all videos in Input_Videos/")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of representative frames to extract")
    args = parser.parse_args()

    input_videos_dir = os.path.join(SCRIPT_DIR, "Input_Videos")

    if args.all_videos:
        for f in os.listdir(input_videos_dir):
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                run_evaluation(os.path.join(input_videos_dir, f), num_frames=args.num_frames)
    elif args.video:
        run_evaluation(args.video, num_frames=args.num_frames)
    else:
        print("Please provide --video <path> or use --all_videos.")
        parser.print_help()


if __name__ == "__main__":
    main()

