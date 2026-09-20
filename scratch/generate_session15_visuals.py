"""
scratch/generate_session15_visuals.py — Visual Contact Sheet for Session 15 Candidates
========================================================================================
Generates high-fidelity visual comparison of:
  1. Input Frame
  2. Model A (Session 12 Baseline)
  3. Model B (Hard Non-Negativity Floor)
  4. Model C (Soft Underflow Floor)
  5. Model D (Dark-Region Confidence)
Plus zoomed crops of:
  - Crop 1: Dark Road / Asphalt Boundary
  - Crop 2: Wiper / Hood Underflow Boundary
"""

import os
import sys
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import scipy.ndimage as ndimage

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
from nighttime_experimental import ExperimentalNighttimePipeline

reader = imageio.get_reader("test_videos/foggy_video3.mp4")
pipe = ExperimentalNighttimePipeline()
for i, f in enumerate(reader):
    if i < 30:
        pipe.process_frame(f)
    elif i == 30:
        target_frame = f
        break
reader.close()

out_a, fps_a, diag = pipe.process_frame(target_frame, return_diagnostics=True)
proc_float = pipe._proc_float_buf
orig_h, orig_w = proc_float.shape[:2]

t_smooth = pipe._cached_t
t_full = np.array(Image.fromarray(t_smooth, mode="F").resize((orig_w, orig_h), Image.BILINEAR))[:, :, None]
A_source = pipe.last_A
a_max_ref = max(float(np.max(A_source)), 0.25)

V_full = np.max(proc_float, axis=2)
L_smooth = ndimage.uniform_filter(V_full, size=pipe.local_illum_kernel_size)
eta = np.clip(L_smooth / a_max_ref, pipe.eta_min, 1.0)[:, :, None]
A_spatial = A_source[None, None, :] * eta

eps = 1e-4
A_eff = np.maximum(A_spatial, eps)
t_req = np.max(1.0 - proc_float / A_eff, axis=2, keepdims=True)

# Candidate transmission maps
# A: Baseline
t_A = t_full
J_A = np.clip((proc_float - A_spatial) / t_A + A_spatial, 0.0, 1.0)
out_A = (J_A * 255.0).astype(np.uint8)

# B: Hard Floor
t_B = np.clip(np.maximum(t_full, t_req), 0.05, 1.0)
J_B = np.clip((proc_float - A_spatial) / t_B + A_spatial, 0.0, 1.0)
out_B = (J_B * 255.0).astype(np.uint8)

# C: Soft Floor
k = 20.0
delta = 0.03
diff = t_req - t_full + delta
soft_excess = np.where(k * diff > 30.0, diff, (1.0 / k) * np.log1p(np.exp(np.clip(k * diff, -30.0, 30.0))))
t_C = np.clip(t_full + soft_excess, 0.05, 1.0)
J_C = np.clip((proc_float - A_spatial) / t_C + A_spatial, 0.0, 1.0)
out_C = (J_C * 255.0).astype(np.uint8)

# D: Dark Region Confidence
conf_lum = np.clip((L_smooth - 0.02) / 0.15, 0.0, 1.0)[:, :, None]
conf_eta = np.clip((eta - pipe.eta_min) / (0.40 - pipe.eta_min), 0.0, 1.0)
C_trans = conf_lum * conf_eta
t_D = np.clip(1.0 - C_trans * (1.0 - t_full), 0.05, 1.0)
J_D = np.clip((proc_float - A_spatial) / t_D + A_spatial, 0.0, 1.0)
out_D = (J_D * 255.0).astype(np.uint8)

in_rgb = (proc_float * 255.0).astype(np.uint8)

# Compose high-resolution contact sheet
# 5 columns across row 1: Input, Model A, Model B, Model C, Model D
# 5 columns across row 2: Crop 1 (Road Horizon y: 130..180, x: 60..120)
# 5 columns across row 3: Crop 2 (Wiper/Hood y: 155..205, x: 100..160)

crop1_box = (60, 130, 120, 180)  # (x0, y0, x1, y1)
crop2_box = (100, 155, 160, 205)

def get_crop(img_arr, box, scale=3):
    x0, y0, x1, y1 = box
    sub = img_arr[y0:y1, x0:x1]
    im = Image.fromarray(sub)
    return im.resize(((x1 - x0) * scale, (y1 - y0) * scale), Image.NEAREST)

panel_w = orig_w
panel_h = orig_h
crop_w = (crop1_box[2] - crop1_box[0]) * 3
crop_h = (crop1_box[3] - crop1_box[1]) * 3

full_w = panel_w * 5 + 60
full_h = panel_h + crop_h * 2 + 160

canvas = Image.new("RGB", (full_w, full_h), color=(18, 20, 26))
draw = ImageDraw.Draw(canvas)

# Title
draw.text((25, 15), "SESSION 15 — TRANSMISSION-SAFETY CANDIDATE COMPARISON (foggy_video3.mp4 Frame 30)", fill=(255, 255, 255))
draw.text((25, 38), "Native Input Black: 20.65% | Model A Black: 22.85% (+2.20%) | Models B & C Black: 22.82% (+2.17%) | Model D Black: 22.85% (+2.20%)", fill=(180, 210, 255))

models = [
    ("Input Frame (20.65% Black)", in_rgb),
    ("Model A: Baseline (22.85% Black)", out_A),
    ("Model B: Hard Floor (22.82% Black)", out_B),
    ("Model C: Soft Floor (22.82% Black)", out_C),
    ("Model D: Confidence (22.85% Black)", out_D),
]

# Row 1: Full images
y_top = 70
for idx, (label, img_arr) in enumerate(models):
    x = 15 + idx * (panel_w + 10)
    canvas.paste(Image.fromarray(img_arr), (x, y_top))
    draw.text((x + 5, y_top + panel_h + 5), label, fill=(220, 220, 220))

# Row 2: Crop 1 (Road Horizon)
y_crop1 = y_top + panel_h + 35
draw.text((25, y_crop1 - 18), "CROP 1: Road Horizon & Dark Boundary (3x Zoom)", fill=(255, 215, 0))
for idx, (label, img_arr) in enumerate(models):
    x = 15 + idx * (panel_w + 10)
    c_img = get_crop(img_arr, crop1_box, scale=3)
    # resize to fit column width nicely
    c_img_fit = c_img.resize((panel_w, crop_h), Image.BILINEAR)
    canvas.paste(c_img_fit, (x, y_crop1))

# Row 3: Crop 2 (Wiper / Hood Boundary)
y_crop2 = y_crop1 + crop_h + 30
draw.text((25, y_crop2 - 18), "CROP 2: Wiper Blade & Hood Transition (3x Zoom)", fill=(255, 215, 0))
for idx, (label, img_arr) in enumerate(models):
    x = 15 + idx * (panel_w + 10)
    c_img = get_crop(img_arr, crop2_box, scale=3)
    c_img_fit = c_img.resize((panel_w, crop_h), Image.BILINEAR)
    canvas.paste(c_img_fit, (x, y_crop2))

artifact_dir = r"C:\Users\sakth\.gemini\antigravity-ide\brain\a110838c-9ebf-4af3-a443-b35b20d1e5d7"
out_path = os.path.join(artifact_dir, "session15_transmission_safety_comparison.jpg")
canvas.save(out_path, quality=95)
print(f"Saved contact sheet to {out_path}")
