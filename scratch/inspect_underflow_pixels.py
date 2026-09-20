"""
scratch/inspect_underflow_pixels.py — Pixel-by-pixel inspection of underflow in foggy_video3.mp4
=============================================================================================
"""

import os
import sys
import imageio
import numpy as np
from PIL import Image
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

out, fps, diag = pipe.process_frame(target_frame, return_diagnostics=True)
proc_float = pipe._proc_float_buf
orig_h, orig_w = proc_float.shape[:2]

# Reconstruct exact intermediate terms
half_img = proc_float[::2, ::2]
gray_h = 0.299 * half_img[:, :, 0] + 0.587 * half_img[:, :, 1] + 0.114 * half_img[:, :, 2]
A = pipe.last_A

v = np.max(half_img, axis=2)
dc_img = np.min(half_img, axis=2)
s = (v - dc_img) / (v + 1e-6)
depth = ndimage.uniform_filter(0.121779 + 0.959710 * v - 0.780245 * s, size=3)
norm = half_img / (A[None, None, :] + 1e-6)
dc = np.min(norm, axis=2)
dc_min = ndimage.minimum_filter(dc, size=5)
h_local = ndimage.uniform_filter(dc, size=3)

omega_night = min(pipe.omega, 0.85)
t_dcp = 1.0 - omega_night * dc_min
t_cap = np.exp(-pipe.beta * np.clip(depth, 0.0, 2.5))

dark_penalty = np.clip((0.30 - gray_h) / 0.20, 0.0, 1.0)
w_raw = 0.20 * (1.0 - 0.70 * dark_penalty) + 0.30 * np.clip(h_local, 0.0, 1.0)
w_dcp = ndimage.gaussian_filter(np.clip(w_raw, 0.05, 0.60), sigma=1.5)
t_hybrid = w_dcp * t_dcp + (1.0 - w_dcp) * t_cap

t_base = max(pipe.t_min, 0.15)
floor_dark = 0.25 * dark_penalty
t_min_map = np.clip(t_base + floor_dark, t_base, 0.85)
t_hybrid = np.clip(t_hybrid, t_min_map, 1.0)

t_cached = pipe._cached_t
t_full = np.array(Image.fromarray(t_cached, mode="F").resize((orig_w, orig_h), Image.BILINEAR))[:, :, None]

V_full = np.max(proc_float, axis=2)
L_smooth = ndimage.uniform_filter(V_full, size=pipe.local_illum_kernel_size)
a_max_ref = max(float(np.max(A)), 0.25)
eta = np.clip(L_smooth / a_max_ref, pipe.eta_min, 1.0)[:, :, None]
A_spatial = A[None, None, :] * eta

J_raw = (proc_float - A_spatial) / t_full + A_spatial

# Identify where J_raw < 0
underflow_mask = np.any(J_raw < 0.0, axis=2)
in_lum = 0.299 * proc_float[:, :, 0] + 0.587 * proc_float[:, :, 1] + 0.114 * proc_float[:, :, 2]
out_lum = 0.299 * np.clip(J_raw[:, :, 0], 0, 1) + 0.587 * np.clip(J_raw[:, :, 1], 0, 1) + 0.114 * np.clip(J_raw[:, :, 2], 0, 1)

print(f"Total pixels: {in_lum.size}")
print(f"Underflow pixels (any channel J_raw < 0): {np.count_nonzero(underflow_mask)} ({np.count_nonzero(underflow_mask)/in_lum.size*100:.2f}%)")
print(f"Input pixels < 1/255 (pure black): {np.count_nonzero(in_lum < 1/255)} ({np.count_nonzero(in_lum < 1/255)/in_lum.size*100:.2f}%)")
print(f"Output pixels < 1/255: {np.count_nonzero(out_lum < 1/255)} ({np.count_nonzero(out_lum < 1/255)/in_lum.size*100:.2f}%)")

# Let's inspect the underflow pixels
u_I = proc_float[underflow_mask]
u_A = A_spatial[underflow_mask]
u_t = t_full[underflow_mask]
u_J = J_raw[underflow_mask]
u_eta = eta[underflow_mask]

print("\nUnderflow pixels distribution:")
print(f"  Input I:   min={np.min(u_I):.4f}, mean={np.mean(u_I):.4f}, max={np.max(u_I):.4f}")
print(f"  A(x):      min={np.min(u_A):.4f}, mean={np.mean(u_A):.4f}, max={np.max(u_A):.4f}")
print(f"  t(x):      min={np.min(u_t):.4f}, mean={np.mean(u_t):.4f}, max={np.max(u_t):.4f}")
print(f"  J_raw:     min={np.min(u_J):.4f}, mean={np.mean(u_J):.4f}, max={np.max(u_J):.4f}")
print(f"  eta(x):    min={np.min(u_eta):.4f}, mean={np.mean(u_eta):.4f}, max={np.max(u_eta):.4f}")

# Compare t vs required_t = 1 - I/A
req_t = 1.0 - u_I / np.maximum(u_A, 1e-4)
print(f"  Required t: min={np.min(req_t):.4f}, mean={np.mean(req_t):.4f}, max={np.max(req_t):.4f}")

# Where are these underflow pixels spatially?
y_coords, x_coords = np.nonzero(underflow_mask)
print(f"  Spatial Y span: min={np.min(y_coords)}, max={np.max(y_coords)} (out of {orig_h})")
print(f"  Spatial X span: min={np.min(x_coords)}, max={np.max(x_coords)} (out of {orig_w})")
print(f"  Y percentiles: 10%={np.percentile(y_coords, 10):.0f}, 50%={np.percentile(y_coords, 50):.0f}, 90%={np.percentile(y_coords, 90):.0f}")

# Break down by image region:
top_third = np.count_nonzero(y_coords < orig_h * 0.33)
mid_third = np.count_nonzero((y_coords >= orig_h * 0.33) & (y_coords < orig_h * 0.66))
bot_third = np.count_nonzero(y_coords >= orig_h * 0.66)
total_u = len(y_coords)
print(f"\nRegion breakdown of underflow pixels:")
print(f"  Top third (Sky / distant unlit trees): {top_third} ({top_third/total_u*100:.1f}%)")
print(f"  Middle third (Road horizon / roadside): {mid_third} ({mid_third/total_u*100:.1f}%)")
print(f"  Bottom third (Vehicle hood / near road): {bot_third} ({bot_third/total_u*100:.1f}%)")
