import os
import sys
import json
import numpy as np
import imageio
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, ".")
from novel_method.pipeline import RealTimeDehazingPipeline
from novel_method.pipeline_learned_experimental import LearnedResidualDehazingPipeline

def generate_all_figures():
    print("=== Generating Publication-Quality Figures for Session 23 ===")
    fig_dir = "scratch/session23_figures"
    os.makedirs(fig_dir, exist_ok=True)
    
    # -------------------------------------------------------------
    # FIGURE 1: Final Architecture / Workflow Diagram
    # -------------------------------------------------------------
    print("Generating Figure 1: Final Architecture Workflow...")
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    
    # Define style
    box_blue = dict(boxstyle="round,pad=0.4", fc="#E3F2FD", ec="#1E88E5", lw=2)
    box_green = dict(boxstyle="round,pad=0.4", fc="#E8F5E9", ec="#43A047", lw=2)
    box_orange = dict(boxstyle="round,pad=0.4", fc="#FFF3E0", ec="#FB8C00", lw=2)
    box_red = dict(boxstyle="round,pad=0.4", fc="#FFEBEE", ec="#E53935", lw=2)
    box_gray = dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="#757575", lw=2)
    
    # Draw blocks
    ax.text(1.2, 5.5, "Input Hazy Video\n$I(x, y, t)$", ha="center", va="center", bbox=box_gray, fontsize=11, weight="bold")
    
    # Physics branch
    ax.text(3.8, 6.2, "CAP Depth Estimate\n$t_{\\mathrm{CAP}}(x,y)$", ha="center", va="center", bbox=box_blue, fontsize=10)
    ax.text(3.8, 4.8, "DCP Dark Channel\n$t_{\\mathrm{DCP}}(x,y)$", ha="center", va="center", bbox=box_blue, fontsize=10)
    ax.text(3.8, 3.4, "Atmospheric Light $A$\nQuadtree RGB Floor", ha="center", va="center", bbox=box_blue, fontsize=10)
    
    # Spatial fusion
    ax.text(6.4, 5.5, "Spatial Confidence Fusion\n$t_{\\mathrm{fused}} = w t_{\\mathrm{CAP}} + (1-w) t_{\\mathrm{DCP}}$", ha="center", va="center", bbox=box_blue, fontsize=10, weight="bold")
    
    # Learned residual
    ax.text(6.4, 3.4, "Tiny-UNet Residual Module\n(13,921 params, 8ch input)\n$\\Delta t = 0.30 \\cdot \\tanh(F_\\theta(X))$", ha="center", va="center", bbox=box_red, fontsize=10, weight="bold")
    
    # Coupled candidate
    ax.text(9.2, 4.5, "Coupled Candidate & Floor\n$t_{\\mathrm{candidate}} = t_{\\mathrm{fused}} + \\Delta t$\n$t_{\\mathrm{constrained}} = \\max(t_{\\mathrm{cand}}, t_{\\min})$", ha="center", va="center", bbox=box_orange, fontsize=10, weight="bold")
    
    # Refinement and Temporal
    ax.text(11.8, 5.8, "Guided Edge Smoothing\nBox Guided Filter ($r=5$)", ha="center", va="center", bbox=box_green, fontsize=10)
    ax.text(11.8, 4.2, "Motion-Aware Temporal Coherence\n$t_{\\mathrm{temp}} = (1-\\alpha) t_{\\mathrm{prev}} + \\alpha t_{\\mathrm{curr}}$", ha="center", va="center", bbox=box_green, fontsize=10)
    
    # Inversion
    ax.text(11.8, 2.2, "Physical Scene Inversion\n$J = (I - A)/t_{\\mathrm{temp}} + A$", ha="center", va="center", bbox=box_gray, fontsize=11, weight="bold")
    
    # Arrows
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.8, color="#37474F"))
                    
    draw_arrow(2.1, 5.5, 2.8, 6.2)
    draw_arrow(2.1, 5.5, 2.8, 4.8)
    draw_arrow(2.1, 5.5, 2.8, 3.4)
    
    draw_arrow(4.8, 6.2, 5.3, 5.7)
    draw_arrow(4.8, 4.8, 5.3, 5.3)
    
    # Features to CNN
    draw_arrow(2.1, 5.5, 5.1, 3.4)
    draw_arrow(4.8, 3.4, 5.1, 3.4)
    draw_arrow(6.4, 4.8, 6.4, 4.2)
    
    # Summation to candidate
    draw_arrow(7.6, 5.5, 8.1, 4.8)
    draw_arrow(7.6, 3.4, 8.1, 4.2)
    
    # Candidate to Guided
    draw_arrow(10.3, 4.5, 10.8, 5.8)
    draw_arrow(11.8, 5.2, 11.8, 4.8)
    draw_arrow(11.8, 3.6, 11.8, 2.8)
    
    plt.title("Figure 1: Complete System Architecture of the Proposed Physics-Subordinated Video Dehazing Framework", fontsize=13, weight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure1_system_architecture.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 2: Multi-Stage Transmission Decomposition
    # -------------------------------------------------------------
    print("Generating Figure 2: Multi-Stage Transmission Pipeline...")
    pipe_learned = LearnedResidualDehazingPipeline(
        checkpoint_path="checkpoints/delta_t_tiny_unet_best.pt",
        downscale_res=(240, 135),
        ablation_mode="A4"
    )
    
    # Run on sample frame from car-detection
    reader = imageio.get_reader("test_videos/car-detection_patchy_haze.mp4")
    f_car = reader.get_data(15)
    reader.close()
    
    dehazed_out, _, diag = pipe_learned.process_frame(f_car, return_diagnostics=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=300)
    
    axes[0, 0].imshow(f_car)
    axes[0, 0].set_title("(a) Input Hazy Frame", fontsize=11, weight="bold")
    axes[0, 0].axis("off")
    
    im1 = axes[0, 1].imshow(diag["t_fused"], cmap="viridis", vmin=0.1, vmax=0.9)
    axes[0, 1].set_title("(b) Physics Fused Transmission ($t_{\\mathrm{fused}}$)", fontsize=11, weight="bold")
    axes[0, 1].axis("off")
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    im2 = axes[0, 2].imshow(diag["delta_t"], cmap="bwr", vmin=-0.3, vmax=0.3)
    axes[0, 2].set_title("(c) Learned Residual ($\Delta t \in [-0.30, +0.30]$)", fontsize=11, weight="bold")
    axes[0, 2].axis("off")
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    im3 = axes[1, 0].imshow(diag["t_candidate"], cmap="viridis", vmin=0.1, vmax=0.9)
    axes[1, 0].set_title("(d) Candidate Transmission ($t_{\\mathrm{fused}} + \Delta t$)", fontsize=11, weight="bold")
    axes[1, 0].axis("off")
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    im4 = axes[1, 1].imshow(diag["t_temporal"], cmap="viridis", vmin=0.1, vmax=0.9)
    axes[1, 1].set_title("(e) Refined & Temporal Coherent ($t_{\\mathrm{final}}$)", fontsize=11, weight="bold")
    axes[1, 1].axis("off")
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    axes[1, 2].imshow(dehazed_out)
    axes[1, 2].set_title("(f) Final Dehazed Radiance ($J$)", fontsize=11, weight="bold")
    axes[1, 2].axis("off")
    
    plt.suptitle("Figure 2: Step-by-Step Multi-Stage Transmission Decomposition on `car-detection_patchy_haze.mp4`", fontsize=13, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure2_transmission_pipeline.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 3: Temporal Coherence Behavior (Luminance Traces)
    # -------------------------------------------------------------
    print("Generating Figure 3: Temporal Coherence Comparison Traces...")
    # Trace static background patch over 35 frames on foggy_video1
    r_fog = imageio.get_reader("test_videos/foggy_video1.mp4")
    raw_fog_frames = [r_fog.get_data(i) for i in range(35)]
    r_fog.close()
    
    pipe_a1 = RealTimeDehazingPipeline(downscale_res=(240, 135))
    pipe_a4 = LearnedResidualDehazingPipeline(
        checkpoint_path="checkpoints/delta_t_tiny_unet_best.pt",
        downscale_res=(240, 135),
        ablation_mode="A4"
    )
    
    a1_means = []
    a4_means = []
    
    # Monitor static road ROI: y: 200..230, x: 180..240
    for f in raw_fog_frames:
        out_a1, _ = pipe_a1.process_frame(f)
        out_a4, _ = pipe_a4.process_frame(f)
        
        a1_roi = np.mean(out_a1[160:200, 40:80])
        a4_roi = np.mean(out_a4[160:200, 40:80])
        a1_means.append(a1_roi)
        a4_means.append(a4_roi)
        
    frames_axis = np.arange(1, len(a1_means)+1)
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(frames_axis, a1_means, "r--o", label="Baseline A1 (Physics Only, Frame-by-Frame)", lw=1.8, markersize=4)
    ax.plot(frames_axis, a4_means, "b-s", label="Proposed A4 (Physics + $\Delta t$ + Temporal Coherence)", lw=2.2, markersize=4)
    
    ax.set_xlabel("Frame Index", fontsize=11, weight="bold")
    ax.set_ylabel("Mean ROI Luminance (Asphalt Patch)", fontsize=11, weight="bold")
    ax.set_title("Figure 3: Inter-Frame Temporal Luminance Stability in Static Road Region (`foggy_video1.mp4`)", fontsize=12, weight="bold", pad=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10, loc="upper right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure3_temporal_behavior.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 4: Representative Visual Comparisons
    # -------------------------------------------------------------
    print("Generating Figure 4: Representative Visual Comparisons...")
    # Load 1 frame from each category: dense_fog, dynamic_urban, synthetic_gt, difficult_night
    v_paths = [
        ("Dense Fog (`foggy_video1`)", "test_videos/foggy_video1.mp4", 15),
        ("Dynamic Urban (`foggy_video2`)", "test_videos/foggy_video2.mp4", 15),
        ("Synthetic GT (`car-detection`)", "test_videos/car-detection_patchy_haze.mp4", 15),
        ("Difficult Night (`foggy_video3`)", "test_videos/foggy_video3.mp4", 15)
    ]
    
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), dpi=300)
    
    for row_idx, (seq_name, v_path, f_idx) in enumerate(v_paths):
        r = imageio.get_reader(v_path)
        f_in = r.get_data(f_idx)
        r.close()
        
        p1 = RealTimeDehazingPipeline(downscale_res=(240, 135))
        p4 = LearnedResidualDehazingPipeline(
            checkpoint_path="checkpoints/delta_t_tiny_unet_best.pt",
            downscale_res=(240, 135),
            ablation_mode="A4"
        )
        # Warmup 2 frames for temporal coherence
        p1.process_frame(f_in)
        out_p1, _ = p1.process_frame(f_in)
        p4.process_frame(f_in)
        out_p4, _ = p4.process_frame(f_in)
        
        axes[row_idx, 0].imshow(f_in)
        axes[row_idx, 0].set_ylabel(seq_name, fontsize=10, weight="bold")
        if row_idx == 0:
            axes[row_idx, 0].set_title("Input Hazy Frame", fontsize=11, weight="bold")
        axes[row_idx, 0].set_xticks([])
        axes[row_idx, 0].set_yticks([])
        
        axes[row_idx, 1].imshow(out_p1)
        if row_idx == 0:
            axes[row_idx, 1].set_title("Physics Baseline (A1)", fontsize=11, weight="bold")
        axes[row_idx, 1].set_xticks([])
        axes[row_idx, 1].set_yticks([])
        
        axes[row_idx, 2].imshow(out_p4)
        if row_idx == 0:
            axes[row_idx, 2].set_title("Proposed Method (A4)", fontsize=11, weight="bold")
        axes[row_idx, 2].set_xticks([])
        axes[row_idx, 2].set_yticks([])
        
    plt.suptitle("Figure 4: Visual Dehazing Quality Across Challenging Real-World Driving and Synthetic Benchmark Sequences", fontsize=13, weight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure4_visual_comparisons.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 5: Regional Delta-t Behavior
    # -------------------------------------------------------------
    print("Generating Figure 5: Regional Delta-t Behavior...")
    regions = ["Overall", "Sky", "Road / Asphalt", "Midground / Vehicles"]
    fog1_means = [0.018, -0.110, +0.077, +0.069]
    car_means = [-0.076, -0.096, -0.021, -0.102]
    
    x = np.arange(len(regions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    rects1 = ax.bar(x - width/2, fog1_means, width, label="Real Dense Fog (`foggy_video1`)", color="#1E88E5", alpha=0.9)
    rects2 = ax.bar(x + width/2, car_means, width, label="Synthetic Patchy Haze (`car-detection`)", color="#FB8C00", alpha=0.9)
    
    ax.axhline(0, color="black", lw=1.0, linestyle="--")
    ax.set_ylabel("Mean Learned Residual $\Delta t$", fontsize=11, weight="bold")
    ax.set_title("Figure 5: Contextual Transmission Residual ($\Delta t$) by Semantic Region", fontsize=12, weight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(regions, fontsize=10, weight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(-0.16, 0.12)
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    
    # Annotate bars
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f"{h:+.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, weight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f"{h:+.3f}", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 3 if h >= 0 else -12), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, weight="bold")
                    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure5_regional_delta_t.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # FIGURE 6: Runtime Breakdown & FPS Comparison
    # -------------------------------------------------------------
    print("Generating Figure 6: CPU Runtime Breakdown & Throughput...")
    stages = ["Frame Prep", "Physics Features", "Tiny-UNet (CPU)", "Guided Filter", "Temporal Coherence", "Inversion"]
    a1_latencies = [2.89, 4.25, 0.00, 0.59, 0.22, 1.51]
    a4_latencies = [3.28, 4.43, 6.46, 0.78, 0.25, 1.77]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    y = np.arange(len(stages))
    ax1.barh(y - 0.2, a1_latencies, 0.4, label="Baseline A1 (Physics)", color="#757575", alpha=0.85)
    ax1.barh(y + 0.2, a4_latencies, 0.4, label="Proposed A4 (Physics + CNN)", color="#43A047", alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels(stages, fontsize=10, weight="bold")
    ax1.set_xlabel("Latency per Frame (ms on CPU)", fontsize=10, weight="bold")
    ax1.set_title("(a) Component Latency Breakdown", fontsize=11, weight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, axis="x", linestyle=":", alpha=0.6)
    
    # FPS Comparison across videos
    v_labels = ["Dense Fog\n(foggy1)", "Dynamic Urban\n(foggy2)", "Difficult Night\n(foggy3)", "Synthetic GT\n(car-detection)"]
    proc_fps_a1 = [104.7, 93.6, 104.7, 51.0]
    proc_fps_a4 = [58.7, 46.3, 51.4, 26.8]
    e2e_fps_a4 = [32.8, 27.4, 26.1, 10.7]
    
    xv = np.arange(len(v_labels))
    wv = 0.25
    ax2.bar(xv - wv, proc_fps_a1, wv, label="A1 Processing FPS", color="#757575", alpha=0.8)
    ax2.bar(xv, proc_fps_a4, wv, label="A4 Processing FPS", color="#43A047", alpha=0.8)
    ax2.bar(xv + wv, e2e_fps_a4, wv, label="A4 End-to-End FPS (incl I/O)", color="#1E88E5", alpha=0.8)
    ax2.axhline(30.0, color="red", linestyle="--", lw=1.5, label="30 FPS Real-Time Target")
    
    ax2.set_ylabel("Frames per Second (FPS)", fontsize=10, weight="bold")
    ax2.set_title("(b) Measured CPU Throughput", fontsize=11, weight="bold")
    ax2.set_xticks(xv)
    ax2.set_xticklabels(v_labels, fontsize=9, weight="bold")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    
    plt.suptitle("Figure 6: Computational Complexity and Real-Time Execution Profiling on CPU", fontsize=13, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "figure6_runtime_breakdown.png"), dpi=300, bbox_inches="tight")
    plt.close()
    
    print("All 6 publication figures successfully generated in scratch/session23_figures/")

if __name__ == "__main__":
    generate_all_figures()
