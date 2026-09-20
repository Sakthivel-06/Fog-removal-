"""
scratch/generate_session18_diagnostic_visuals.py — Phase 8 Diagnostic Visual Maps
"""
import os
import sys
import imageio
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
from pipeline import RealTimeDehazingPipeline, _resize_np

ARTIFACT_DIR = r"C:\Users\sakth\.gemini\antigravity-ide\brain\a110838c-9ebf-4af3-a443-b35b20d1e5d7"

def generate_visual_maps():
    rc = imageio.get_reader("test_videos/person-bicycle_clean.mp4")
    rh = imageio.get_reader("test_videos/person-bicycle_uniform_haze.mp4")
    fc = rc.get_data(10)
    fh = rh.get_data(10)
    rc.close()
    rh.close()

    # Reference transmission & airlight
    A_true = np.array([255.0, 255.0, 255.0], dtype=np.float32)
    denom = fc.astype(np.float32) - A_true
    numer = fh.astype(np.float32) - A_true
    valid_mask = np.all(np.abs(denom) >= 25.0, axis=-1)
    t_channels = numer / (denom + 1e-6)
    t_ref = np.median(t_channels, axis=-1)
    t_ref = np.clip(t_ref, 0.05, 1.0)

    # Proposed output
    pipe = RealTimeDehazingPipeline()
    for w in range(5):
        pipe.process_frame(fh)
    out_prop, _ = pipe.process_frame(fh)
    
    t_prop = pipe._cached_t
    if t_prop.shape[:2] != fh.shape[:2]:
        t_prop_full = np.array(Image.fromarray(t_prop).resize((fh.shape[1], fh.shape[0]), Image.BILINEAR))
    else:
        t_prop_full = t_prop

    # Error map of transmission
    t_err_map = np.abs(t_prop_full - t_ref)

    # Ideal physical reconstruction using true t_ref and true A
    t_3d = np.stack([t_ref]*3, axis=-1)
    J_ideal = (fh.astype(np.float32) - A_true) / (t_3d + 1e-4) + A_true
    J_ideal = np.clip(np.round(J_ideal), 0, 255).astype(np.uint8)

    if out_prop.shape[:2] != fc.shape[:2]:
        out_prop = np.array(Image.fromarray(out_prop).resize((fc.shape[1], fc.shape[0]), Image.BILINEAR))

    # Difference map (Proposed vs Clean GT)
    diff_map = np.abs(out_prop.astype(np.float32) - fc.astype(np.float32)) / 255.0

    # Create diagnostic contact figure
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=150)
    
    axes[0, 0].imshow(fc)
    axes[0, 0].set_title("(1) Clean Ground Truth", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(fh)
    axes[0, 1].set_title("(2) Synthetic Hazy Input", fontsize=10, fontweight="bold")
    axes[0, 1].axis("off")

    im_tref = axes[0, 2].imshow(t_ref, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, 2].set_title(f"(3) Reference Transmission t_ref (Mean={np.mean(t_ref):.2f})", fontsize=10, fontweight="bold")
    axes[0, 2].axis("off")
    plt.colorbar(im_tref, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im_tprop = axes[0, 3].imshow(t_prop_full, cmap="viridis", vmin=0.0, vmax=1.0)
    axes[0, 3].set_title(f"(4) Proposed Transmission t_est (Mean={np.mean(t_prop_full):.2f})", fontsize=10, fontweight="bold")
    axes[0, 3].axis("off")
    plt.colorbar(im_tprop, ax=axes[0, 3], fraction=0.046, pad=0.04)

    im_terr = axes[1, 0].imshow(t_err_map, cmap="inferno", vmin=0.0, vmax=0.5)
    axes[1, 0].set_title("(5) Transmission Error |t_est - t_ref|", fontsize=10, fontweight="bold")
    axes[1, 0].axis("off")
    plt.colorbar(im_terr, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(J_ideal)
    axes[1, 1].set_title("(6) Inversion with True t & True A (38.8 dB)", fontsize=10, fontweight="bold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(out_prop)
    axes[1, 2].set_title("(7) Proposed Dehazed Output (10.1 dB)", fontsize=10, fontweight="bold")
    axes[1, 2].axis("off")

    im_diff = axes[1, 3].imshow(np.mean(diff_map, axis=-1), cmap="coolwarm", vmin=0.0, vmax=0.6)
    axes[1, 3].set_title("(8) Normalized Lum Difference (|Pred - GT|)", fontsize=10, fontweight="bold")
    axes[1, 3].axis("off")
    plt.colorbar(im_diff, ax=axes[1, 3], fraction=0.046, pad=0.04)

    plt.suptitle("SESSION 18 — SYNTHETIC HAZE PHYSICS DIAGNOSTIC MAPS (Person-Bicycle)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    out_path = os.path.join(ARTIFACT_DIR, "session18_physics_diagnostic_maps.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved diagnostic maps to {out_path}")

if __name__ == "__main__":
    generate_visual_maps()
