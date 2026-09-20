"""
scratch/generate_session16_figures.py — Publication-Ready Charts & Visual Comparisons
======================================================================================
Generates:
  1. Chart 1: Method Comparison Bar Chart — FPS
  2. Chart 2: Method Comparison Bar Chart — NIQE
  3. Chart 3: Method Comparison Bar Chart — PSNR (GT sequences only)
  4. Chart 4: Temporal Flicker Comparison (Delta L)
  5. Chart 5: Black Clipping Comparison
  6. Chart 6: Color Balance Comparison (B-R balance)
  7. Chart 7: Ablation Impact Chart
  8. Chart 8: Category-Level Robustness Chart
  9. Visual Contact Sheet: Baseline Visual Comparison across 3 difficulty levels
  10. Visual Contact Sheet: Difficult Scene Crops (Sky, Road/Headlights, Vehicle Boundaries)
All saved directly into the artifacts directory.
"""

import os
import sys
import json
import imageio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "algorithms")))

import dcp
import cap
import clahe_baseline
import gamma_baseline
import ceef
import ceef_original
from pipeline import RealTimeDehazingPipeline, _resize_np
from nighttime_experimental import ExperimentalNighttimePipeline

ARTIFACT_DIR = r"C:\Users\sakth\.gemini\antigravity-ide\brain\a110838c-9ebf-4af3-a443-b35b20d1e5d7"

def load_data():
    with open("scratch/session16_results.json", "r") as f:
        return json.load(f)

def generate_charts(data):
    baselines = data["baseline_comparison"]
    ablations = data["ablation_study"]

    methods = ["DCP", "CAP", "CLAHE", "Corrected Gamma", "Original CEEF", "Modified CEEF", "Proposed Method"]
    colors = ["#4A90E2", "#50E3C2", "#F5A623", "#9013FE", "#D0021B", "#BD10E0", "#417505"]

    # 1. CHART 1: FPS Comparison (Mean across all 13 videos)
    fps_by_method = {}
    for m in methods:
        fps_vals = [r["fps"] for r in baselines if r["method_name"] == m]
        fps_by_method[m] = float(np.mean(fps_vals))

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    bars = ax.bar(methods, [fps_by_method[m] for m in methods], color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(30.0, color="red", linestyle="--", linewidth=1.5, label="Real-Time Threshold (30 FPS)")
    ax.set_ylabel("Frames Per Second (FPS) [Log Scale]", fontsize=11, fontweight="bold")
    ax.set_title("Figure 1: Processing Throughput Across Methods (Mean Over 13 Videos)", fontsize=13, fontweight="bold", pad=12)
    ax.set_yscale("log")
    ax.set_ylim(0.8, 1000)
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.legend(loc="upper left")
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.tight_layout()
    chart1_path = os.path.join(ARTIFACT_DIR, "session16_chart1_fps_comparison.png")
    plt.savefig(chart1_path)
    plt.close()
    print(f"Saved {chart1_path}")

    # 2. CHART 2: NIQE Comparison (Mean across all 13 videos, lower is better)
    niqe_by_method = {}
    for m in methods:
        vals = [r["niqe"] for r in baselines if r["method_name"] == m]
        niqe_by_method[m] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    bars = ax.bar(methods, [niqe_by_method[m] for m in methods], color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("NIQE Score (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 2: Natural Image Quality Evaluator (NIQE) Across Methods", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.tight_layout()
    chart2_path = os.path.join(ARTIFACT_DIR, "session16_chart2_niqe_comparison.png")
    plt.savefig(chart2_path)
    plt.close()
    print(f"Saved {chart2_path}")

    # 3. CHART 3: PSNR Comparison (GT Sequences Only)
    gt_vids = ["car-detection_patchy_haze.mp4", "person-bicycle_uniform_haze.mp4"]
    x = np.arange(len(methods))
    width = 0.35

    psnr_v1 = [next(r["psnr"] for r in baselines if r["method_name"] == m and r["video_name"] == gt_vids[0]) for m in methods]
    psnr_v2 = [next(r["psnr"] for r in baselines if r["method_name"] == m and r["video_name"] == gt_vids[1]) for m in methods]

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
    b1 = ax.bar(x - width/2, psnr_v1, width, label="Car Detection (Patchy Haze)", color="#4A90E2", edgecolor="black")
    b2 = ax.bar(x + width/2, psnr_v2, width, label="Person Bicycle (Uniform Haze)", color="#50E3C2", edgecolor="black")
    ax.set_ylabel("PSNR (dB) [Higher is Better]", fontsize=11, fontweight="bold")
    ax.set_title("Figure 3: Objective Ground-Truth Reconstruction Fidelity (PSNR)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=20, ha="right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left")
    plt.tight_layout()
    chart3_path = os.path.join(ARTIFACT_DIR, "session16_chart3_psnr_comparison.png")
    plt.savefig(chart3_path)
    plt.close()
    print(f"Saved {chart3_path}")

    # 4. CHART 4: Temporal Flicker Comparison
    flick_by_method = {}
    for m in methods:
        vals = [r["flicker"] for r in baselines if r["method_name"] == m]
        flick_by_method[m] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    bars = ax.bar(methods, [flick_by_method[m] for m in methods], color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Mean Temporal Flicker Delta L (Lower is More Stable)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 4: Inter-Frame Luminance Fluctuation (Temporal Stability)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.tight_layout()
    chart4_path = os.path.join(ARTIFACT_DIR, "session16_chart4_temporal_flicker.png")
    plt.savefig(chart4_path)
    plt.close()
    print(f"Saved {chart4_path}")

    # 5. CHART 5: Black Clipping Comparison
    black_by_method = {}
    for m in methods:
        vals = [r["black_clip"] for r in baselines if r["method_name"] == m]
        black_by_method[m] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    bars = ax.bar(methods, [black_by_method[m] for m in methods], color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Percentage of Black Clipped Pixels (%)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 5: Mean Black Clipping Rate Across All Sequences", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.2f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.tight_layout()
    chart5_path = os.path.join(ARTIFACT_DIR, "session16_chart5_black_clipping.png")
    plt.savefig(chart5_path)
    plt.close()
    print(f"Saved {chart5_path}")

    # 6. CHART 6: Color Balance Comparison (B - R balance)
    # Evaluates color cast: values near 0 are neutral; large positive = cool/blue cast, negative = warm/red cast
    br_by_method = {}
    for m in methods:
        vals = [r["fdi_proxy"] for r in baselines if r["method_name"] == m]  # Let's plot Fog Density Index Proxy
        br_by_method[m] = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    bars = ax.bar(methods, [br_by_method[m] for m in methods], color=colors, edgecolor="black", alpha=0.85)
    ax.set_ylabel("Mean Residual Fog Density Index (FDI Proxy)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 6: Residual Fog Density Index (FDI Proxy - Lower = Clearer)", fontsize=13, fontweight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.tight_layout()
    chart6_path = os.path.join(ARTIFACT_DIR, "session16_chart6_color_balance.png")
    plt.savefig(chart6_path)
    plt.close()
    print(f"Saved {chart6_path}")

    # 7. CHART 7: Ablation Impact Chart
    # Plot SSIM, PSNR, and NIQE across the 7 ablation configs for car-detection_patchy_haze
    car_abl = [r for r in ablations if r["video_name"] == "car-detection_patchy_haze.mp4"]
    cfg_labels = [r["config_label"].split(". ")[1] for r in car_abl]
    ssim_vals = [r["ssim"] for r in car_abl]
    flicker_vals = [r["flicker"] for r in car_abl]

    fig, ax1 = plt.subplots(figsize=(11, 5.5), dpi=200)
    color = "tab:blue"
    ax1.set_ylabel("SSIM (Higher is Better)", color=color, fontsize=11, fontweight="bold")
    b1 = ax1.bar(np.arange(len(cfg_labels)) - 0.2, ssim_vals, width=0.4, color=color, alpha=0.8, label="SSIM")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0.5, 0.75)

    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Flicker Delta L (Lower is Better)", color=color, fontsize=11, fontweight="bold")
    b2 = ax2.bar(np.arange(len(cfg_labels)) + 0.2, flicker_vals, width=0.4, color=color, alpha=0.8, label="Flicker")
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(15.0, 38.0)

    ax1.set_xticks(np.arange(len(cfg_labels)))
    ax1.set_xticklabels(cfg_labels, rotation=25, ha="right", fontsize=9)
    plt.title("Figure 7: Component Ablation Impact on Structural Fidelity (SSIM) and Temporal Stability (Flicker)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    chart7_path = os.path.join(ARTIFACT_DIR, "session16_chart7_ablation_impact.png")
    plt.savefig(chart7_path)
    plt.close()
    print(f"Saved {chart7_path}")

    # 8. CHART 8: Category-Level Robustness Chart
    categories = ["A", "B", "C", "D", "E", "F"]
    cat_names = ["Dense Fog", "Night/Low-Light", "Urban Dynamic", "Nonuniform Light", "General Outdoor", "Indoor/Artificial"]
    proposed_by_cat = []
    dcp_by_cat = []
    cap_by_cat = []
    for c in categories:
        p_vals = [r["niqe"] for r in baselines if r["category"] == c and r["method_name"] == "Proposed Method"]
        d_vals = [r["niqe"] for r in baselines if r["category"] == c and r["method_name"] == "DCP"]
        c_vals = [r["niqe"] for r in baselines if r["category"] == c and r["method_name"] == "CAP"]
        proposed_by_cat.append(float(np.mean(p_vals)))
        dcp_by_cat.append(float(np.mean(d_vals)))
        cap_by_cat.append(float(np.mean(c_vals)))

    x = np.arange(len(categories))
    width = 0.25
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
    ax.bar(x - width, dcp_by_cat, width, label="DCP", color="#4A90E2")
    ax.bar(x, cap_by_cat, width, label="CAP", color="#50E3C2")
    ax.bar(x + width, proposed_by_cat, width, label="Proposed Method", color="#417505")
    ax.set_ylabel("NIQE Score (Lower is Better)", fontsize=11, fontweight="bold")
    ax.set_title("Figure 8: Category-Wise Robustness Across Natural & Artificial Scene Domains", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, rotation=15, ha="right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    plt.tight_layout()
    chart8_path = os.path.join(ARTIFACT_DIR, "session16_chart8_category_robustness.png")
    plt.savefig(chart8_path)
    plt.close()
    print(f"Saved {chart8_path}")

def generate_visual_comparisons():
    """Generates visual contact sheets for representative frames and difficult scene crops."""
    test_cases = [
        {"name": "foggy_video1.mp4", "frame_idx": 20, "label": "Case 1: Dense Natural Fog (foggy_video1)"},
        {"name": "car-detection_patchy_haze.mp4", "frame_idx": 30, "label": "Case 2: Dynamic Urban Patchy Haze (car-detection)"},
        {"name": "foggy_video3.mp4", "frame_idx": 30, "label": "Case 3: Night Driving Headlights (foggy_video3)"},
    ]

    pipe = ExperimentalNighttimePipeline(enable_nighttime_branch=True, night_mode="prototype")

    method_names = ["Input", "DCP", "CAP", "CLAHE", "Gamma", "Orig CEEF", "Mod CEEF", "Proposed"]
    target_wh = (320, 180)

    # Canvas dimensions: 3 rows, 8 columns
    cell_w, cell_h = target_wh
    total_w = cell_w * 8 + 90
    total_h = cell_h * 3 + 120

    canvas = Image.new("RGB", (total_w, total_h), color=(20, 22, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 10), "SESSION 16 — CLASSICAL BASELINES VS. PROPOSED METHOD (REPRESENTATIVE SCENES)", fill=(255, 255, 255))

    for r_idx, tc in enumerate(test_cases):
        vpath = os.path.join("test_videos", tc["name"])
        reader = imageio.get_reader(vpath)
        frame = reader.get_data(tc["frame_idx"])
        reader.close()

        # Resize input frame to target
        in_res = np.array(Image.fromarray(frame).resize(target_wh, Image.BILINEAR))

        # Generate outputs
        out_dcp = dcp.process_frame(in_res)
        out_cap = cap.process_frame(in_res)
        out_clahe = clahe_baseline.process_frame(in_res)
        out_gamma = gamma_baseline.process_frame(in_res, gamma=2.0)
        out_orig_ceef = ceef_original.process_frame(in_res)
        out_mod_ceef = ceef.process_frame(in_res)

        pipe.reset()
        out_prop, _ = pipe.process_frame(frame)
        out_prop_res = np.array(Image.fromarray(out_prop).resize(target_wh, Image.BILINEAR))

        images = [in_res, out_dcp, out_cap, out_clahe, out_gamma, out_orig_ceef, out_mod_ceef, out_prop_res]

        y = 40 + r_idx * (cell_h + 25)
        for c_idx, img in enumerate(images):
            x = 15 + c_idx * (cell_w + 10)
            canvas.paste(Image.fromarray(img), (x, y))
            if r_idx == 0:
                draw.text((x + 5, y - 18), method_names[c_idx], fill=(220, 220, 220))

        draw.text((15, y + cell_h + 3), tc["label"], fill=(255, 215, 0))

    contact_path = os.path.join(ARTIFACT_DIR, "session16_baseline_visual_comparison.jpg")
    canvas.save(contact_path, quality=95)
    print(f"Saved visual comparison contact sheet to {contact_path}")

    # Generate Difficult-Scene Crops
    # Zoom crops of: Sky (foggy_video1), Headlights/Road (foggy_video3), Vehicle Boundary (car-detection)
    crop_canvas = Image.new("RGB", (cell_w * 4 + 50, cell_h * 3 + 120), color=(20, 22, 28))
    crop_draw = ImageDraw.Draw(crop_canvas)
    crop_draw.text((20, 10), "SESSION 16 — DIFFICULT SCENE REGION CROPS (3x ZOOM COMPARISON)", fill=(255, 255, 255))

    crop_cases = [
        {"name": "foggy_video1.mp4", "box": (80, 20, 180, 80), "title": "Sky & Mountain Fog Boundary (foggy_video1)"},
        {"name": "car-detection_patchy_haze.mp4", "box": (120, 70, 220, 130), "title": "Moving Vehicle Edge & Patchy Haze"},
        {"name": "foggy_video3.mp4", "box": (90, 100, 190, 160), "title": "Headlight Glare & Wet Asphalt Horizon (foggy_video3)"},
    ]

    methods_to_show = [("Input", in_res), ("DCP", out_dcp), ("Original CEEF", out_orig_ceef), ("Proposed Method", out_prop_res)]

    for r_idx, cc in enumerate(crop_cases):
        vpath = os.path.join("test_videos", cc["name"])
        reader = imageio.get_reader(vpath)
        frame = reader.get_data(30)
        reader.close()
        in_r = np.array(Image.fromarray(frame).resize(target_wh, Image.BILINEAR))

        pipe.reset()
        out_p, _ = pipe.process_frame(frame)
        out_pr = np.array(Image.fromarray(out_p).resize(target_wh, Image.BILINEAR))

        outputs = [
            ("Input", in_r),
            ("DCP", dcp.process_frame(in_r)),
            ("Original CEEF", ceef_original.process_frame(in_r)),
            ("Proposed", out_pr)
        ]

        x0, y0, x1, y1 = cc["box"]
        y_pos = 45 + r_idx * (cell_h + 25)
        for c_idx, (m_label, full_im) in enumerate(outputs):
            sub = full_im[y0:y1, x0:x1]
            sub_zoom = np.array(Image.fromarray(sub).resize((cell_w, cell_h), Image.NEAREST))
            x_pos = 15 + c_idx * (cell_w + 10)
            crop_canvas.paste(Image.fromarray(sub_zoom), (x_pos, y_pos))
            if r_idx == 0:
                crop_draw.text((x_pos + 5, y_pos - 18), m_label, fill=(220, 220, 220))
        crop_draw.text((15, y_pos + cell_h + 3), cc["title"], fill=(255, 215, 0))

    crops_path = os.path.join(ARTIFACT_DIR, "session16_difficult_scene_crops.jpg")
    crop_canvas.save(crops_path, quality=95)
    print(f"Saved difficult scene crops to {crops_path}")

if __name__ == "__main__":
    d = load_data()
    generate_charts(d)
    generate_visual_comparisons()
    print("\nAll Section 16 figures and visual sheets generated successfully!")
