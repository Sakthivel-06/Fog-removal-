import os
import sys
import numpy as np
from PIL import Image
import imageio.v3 as iio

sys.path.insert(0, os.path.abspath("."))
from novel_method.pipeline import RealTimeDehazingPipeline
from novel_method.pipeline_learned_experimental import LearnedResidualDehazingPipeline

r = iio.imiter("test_videos/foggy_video9.mp4")
frame = next(r)
print(f"Loaded foggy_video9.mp4 frame shape: {frame.shape}")

# Save original frame
os.makedirs("scratch/dark_bg_test", exist_ok=True)
Image.fromarray(frame).save("scratch/dark_bg_test/original.jpg")

# Test 1: Current settings (beta=1.4, omega=0.90)
p_curr = RealTimeDehazingPipeline(downscale_res=(512, 288), beta=1.4, omega=0.90, preserve_resolution=False)
out_curr, _, diag_curr = p_curr.process_frame(frame, return_diagnostics=True)
Image.fromarray(out_curr).save("scratch/dark_bg_test/out_beta1.4_omega0.90.jpg")
print("T1 (beta=1.4, omega=0.90) A:", p_curr.last_A)

# Test 2: Standard settings (beta=1.0, omega=0.85)
p_std = RealTimeDehazingPipeline(downscale_res=(512, 288), beta=1.0, omega=0.85, preserve_resolution=False)
out_std, _, diag_std = p_std.process_frame(frame, return_diagnostics=True)
Image.fromarray(out_std).save("scratch/dark_bg_test/out_beta1.0_omega0.85.jpg")
print("T2 (beta=1.0, omega=0.85) A:", p_std.last_A)

# Test 3: Mild settings (beta=0.8, omega=0.75)
p_mild = RealTimeDehazingPipeline(downscale_res=(512, 288), beta=0.8, omega=0.75, preserve_resolution=False)
out_mild, _, _ = p_mild.process_frame(frame, return_diagnostics=True)
Image.fromarray(out_mild).save("scratch/dark_bg_test/out_beta0.8_omega0.75.jpg")

# Test 4: Higher transmission floor t_min=0.25
p_floor = RealTimeDehazingPipeline(downscale_res=(512, 288), beta=1.0, omega=0.80, t_min=0.25, preserve_resolution=False)
out_floor, _, _ = p_floor.process_frame(frame, return_diagnostics=True)
Image.fromarray(out_floor).save("scratch/dark_bg_test/out_tmin0.25.jpg")

# Test 5: A4 Proposed Method with beta=1.0, omega=0.85
best_ckpt = "checkpoints/delta_t_tiny_unet_best.pt"
if os.path.exists(best_ckpt):
    p_a4 = LearnedResidualDehazingPipeline(checkpoint_path=best_ckpt, ablation_mode="A4", downscale_res=(512, 288), beta=1.0, omega=0.85, preserve_resolution=False)
    out_a4 = p_a4.process_frame(frame)[0]
    Image.fromarray(out_a4).save("scratch/dark_bg_test/out_a4_beta1.0.jpg")

# Measure mean brightness of background region (middle band: y in [0.35, 0.60], x across width)
h, w = frame.shape[:2]
bg_slice_orig = frame[int(h*0.35):int(h*0.55), :]
print(f"Original Background Luminance: {bg_slice_orig.mean():.1f}")

for name, im in [
    ("out_beta1.4_omega0.90", out_curr),
    ("out_beta1.0_omega0.85", out_std),
    ("out_beta0.8_omega0.75", out_mild),
    ("out_tmin0.25", out_floor)
]:
    bg_slice = im[int(im.shape[0]*0.35):int(im.shape[0]*0.55), :]
    print(f"{name} Background Luminance: {bg_slice.mean():.1f}")
