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
h, w = frame.shape[:2]

# 1. Base A4 output with standard beta=1.0, omega=0.85
p_a4 = LearnedResidualDehazingPipeline(
    checkpoint_path="checkpoints/delta_t_tiny_unet_best.pt",
    ablation_mode="A4",
    downscale_res=(512, 288),
    beta=1.0,
    omega=0.85,
    preserve_resolution=False
)
out_base = p_a4.process_frame(frame)[0]

# Approach A: Exposure Compensation via Adaptive Shadow Lift
# Lifts dark/shadow tones smoothly without over-exposing the sky
def apply_shadow_compensation(img_uint8, gain=0.25):
    img_f = img_uint8.astype(np.float32) / 255.0
    lum = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
    # Shadow weight: maximum in dark/mid-tones (0.1 - 0.4), decays to 0 in sky (>0.7)
    shadow_weight = np.clip((0.75 - lum) / 0.50, 0.0, 1.0)[:, :, None]
    # Smooth gamma/power adjustment on shadows
    lifted = img_f + gain * shadow_weight * (1.0 - img_f) * (0.5 - np.abs(img_f - 0.5))
    return np.clip(lifted * 255.0, 0, 255).astype(np.uint8)

# Approach B: Mild Tone Curve (Gamma = 0.85 on shadows)
def apply_adaptive_gamma(img_uint8, gamma=0.80):
    img_f = img_uint8.astype(np.float32) / 255.0
    lum = 0.299 * img_f[:, :, 0] + 0.587 * img_f[:, :, 1] + 0.114 * img_f[:, :, 2]
    sky_mask = np.clip((lum - 0.70) / 0.20, 0.0, 1.0)[:, :, None]
    # Apply gamma only to non-sky regions
    lifted = np.power(img_f, gamma)
    blended = sky_mask * img_f + (1.0 - sky_mask) * lifted
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)

# Approach C: Physics-based (beta=0.80, omega=0.75, t_min=0.20)
p_opt = LearnedResidualDehazingPipeline(
    checkpoint_path="checkpoints/delta_t_tiny_unet_best.pt",
    ablation_mode="A4",
    downscale_res=(512, 288),
    beta=0.80,
    omega=0.78,
    preserve_resolution=False
)
out_opt = p_opt.process_frame(frame)[0]

out_shadow_02 = apply_shadow_compensation(out_base, gain=0.25)
out_shadow_04 = apply_shadow_compensation(out_base, gain=0.45)
out_gamma = apply_adaptive_gamma(out_base, gamma=0.75)

# Save images to inspect
os.makedirs("scratch/bg_recovery", exist_ok=True)
Image.fromarray(out_base).save("scratch/bg_recovery/1_base_a4.jpg")
Image.fromarray(out_opt).save("scratch/bg_recovery/2_physics_opt.jpg")
Image.fromarray(out_shadow_02).save("scratch/bg_recovery/3_shadow_lift_mild.jpg")
Image.fromarray(out_shadow_04).save("scratch/bg_recovery/4_shadow_lift_strong.jpg")
Image.fromarray(out_gamma).save("scratch/bg_recovery/5_adaptive_gamma.jpg")

bg_slice_orig = frame[int(h*0.35):int(h*0.55), :].mean()
print(f"Original Background Lum:    {bg_slice_orig:.1f}")
print(f"1. Base A4 Background Lum:  {out_base[int(h*0.35):int(h*0.55), :].mean():.1f}")
print(f"2. Physics Opt Lum:         {out_opt[int(h*0.35):int(h*0.55), :].mean():.1f}")
print(f"3. Shadow Lift 0.25 Lum:    {out_shadow_02[int(h*0.35):int(h*0.55), :].mean():.1f}")
print(f"4. Shadow Lift 0.45 Lum:    {out_shadow_04[int(h*0.35):int(h*0.55), :].mean():.1f}")
print(f"5. Adaptive Gamma Lum:      {out_gamma[int(h*0.35):int(h*0.55), :].mean():.1f}")
