"""
scratch/diagnose_multi_source_videos.py — Phase 2 & Phase 3 Diagnostics
========================================================================
Analyzes candidate light sources and bright-light regions in:
  1. foggy_video3.mp4
  2. foggy_video4.mp4
  3. foggy_video5.mp4
  4. foggy_video14.mp4
  5. foggy_video2.mp4

Measures:
  - RGB mean, median, HSV hue, saturation, luminance
  - Spatial centroids, local radius, influence area %
  - Clusters candidate light regions by chromaticity
  - Tests false detections on colored objects (vehicle bodies, signs, vegetation)
"""

import os
import sys
import imageio
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage

TEST_VIDEOS = [
    {"name": "foggy_video3.mp4", "label": "Night Driving Headlights", "frame_idx": 25},
    {"name": "foggy_video4.mp4", "label": "4K Aerial Night City", "frame_idx": 15},
    {"name": "foggy_video5.mp4", "label": "4K Neon City Lights", "frame_idx": 15},
    {"name": "foggy_video14.mp4", "label": "Stadium Floodlight Glare", "frame_idx": 15},
    {"name": "foggy_video2.mp4", "label": "Urban Traffic Signals", "frame_idx": 20},
]

def rgb_to_hsv_np(rgb_float):
    """Dependency-free RGB float [0, 1] to HSV conversion."""
    r, g, b = rgb_float[:, :, 0], rgb_float[:, :, 1], rgb_float[:, :, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c
    
    # Hue
    h = np.zeros_like(max_c)
    mask = delta > 1e-6
    
    # r is max
    idx = mask & (max_c == r)
    h[idx] = 60.0 * (((g[idx] - b[idx]) / delta[idx]) % 6.0)
    # g is max
    idx = mask & (max_c == g)
    h[idx] = 60.0 * (((b[idx] - r[idx]) / delta[idx]) + 2.0)
    # b is max
    idx = mask & (max_c == b)
    h[idx] = 60.0 * (((r[idx] - g[idx]) / delta[idx]) + 4.0)
    
    # Saturation
    s = np.zeros_like(max_c)
    s[max_c > 1e-6] = delta[max_c > 1e-6] / max_c[max_c > 1e-6]
    
    # Value
    v = max_c
    return h, s, v

print("=" * 90)
print("PHASE 2 & PHASE 3: LIGHT SOURCE CHROMATICITY & OBJECT CONFUSION DIAGNOSTICS")
print("=" * 90)

for item in TEST_VIDEOS:
    vpath = os.path.join("test_videos", item["name"])
    if not os.path.exists(vpath):
        vpath = os.path.join(r"C:\Users\sakth\Desktop\Fog\fog-removal-project\test_videos", item["name"])
        
    reader = imageio.get_reader(vpath)
    frame = reader.get_data(item["frame_idx"])
    reader.close()
    
    # Resize to standard processing max res for fair analysis
    h_orig, w_orig = frame.shape[:2]
    scale = min(426.0 / w_orig, 240.0 / h_orig, 1.0)
    cur_w, cur_h = int(w_orig * scale), int(h_orig * scale)
    img_pil = Image.fromarray(frame).resize((cur_w, cur_h), Image.BILINEAR)
    img = np.array(img_pil).astype(np.float32) / 255.0
    
    lum = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    hue, sat, val = rgb_to_hsv_np(img)
    
    # 1. Naive Brightness Detector: Pixels with lum > 0.70
    naive_bright = (lum > 0.70)
    
    # 2. Local Contrast / Excess Brightness:
    # A genuine light emitter has strong local luminance excess: V(x) - local_mean(V) > 0.15
    local_mean_val = ndimage.uniform_filter(val, size=15)
    local_excess = val - local_mean_val
    
    # 3. Conservative Light Source Detector:
    # Criteria:
    # (a) Absolute brightness: val > 0.75 OR (val > 0.60 and local_excess > 0.18)
    # (b) NOT a flat reflective surface (requires either near-saturation core val > 0.88 or high local contrast)
    # (c) Minimum cluster size to reject single-pixel noise (min 5 connected pixels)
    core_light = (val > 0.85) | ((val > 0.65) & (local_excess > 0.20))
    
    # Label connected components
    labeled, num_features = ndimage.label(core_light)
    
    # Filter by component area (min 6 pixels, max 25% of image to avoid sky)
    valid_clusters = []
    total_pixels = cur_w * cur_h
    
    for c_id in range(1, num_features + 1):
        c_mask = (labeled == c_id)
        c_size = np.count_nonzero(c_mask)
        if c_size < 6 or c_size > (0.25 * total_pixels):
            continue
            
        # Centroid
        y_idx, x_idx = np.nonzero(c_mask)
        cy, cx = float(np.mean(y_idx)), float(np.mean(x_idx))
        
        # Color stats
        c_rgb = img[c_mask]
        rgb_mean = np.mean(c_rgb, axis=0)
        rgb_med = np.median(c_rgb, axis=0)
        mean_h = float(np.mean(hue[c_mask]))
        mean_s = float(np.mean(sat[c_mask]))
        mean_v = float(np.mean(val[c_mask]))
        
        # Radius estimate
        radius = np.sqrt(c_size / np.pi)
        
        # Surrounding glow: evaluate annular region around the source
        dilated = ndimage.binary_dilation(c_mask, iterations=int(max(radius * 0.8, 3)))
        halo_ring = dilated & (~c_mask)
        halo_lum = float(np.mean(lum[halo_ring])) if np.any(halo_ring) else 0.0
        
        valid_clusters.append({
            "id": c_id,
            "size": c_size,
            "area_pct": (c_size / total_pixels) * 100.0,
            "cx": cx,
            "cy": cy,
            "radius": radius,
            "rgb_mean": rgb_mean,
            "rgb_med": rgb_med,
            "hue": mean_h,
            "sat": mean_s,
            "val": mean_v,
            "halo_lum": halo_lum,
        })
        
    print(f"\nSequence: {item['label']} ({item['name']}) [Frame {item['frame_idx']}, Size: {cur_w}x{cur_h}]")
    print(f"  Naive Bright Pixels (>0.70): {np.count_nonzero(naive_bright)/total_pixels*100:.2f}%")
    print(f"  Conservative Light Candidates: {len(valid_clusters)} clusters detected")
    
    # Chromatic grouping of clusters
    chroma_groups = {"Amber/Warm": [], "Blue/White/Cool": [], "Red": [], "Green": [], "Neutral White": []}
    for cl in valid_clusters:
        r, g, b = cl["rgb_mean"]
        h, s, v = cl["hue"], cl["sat"], cl["val"]
        
        if s < 0.15:
            chroma_groups["Neutral White"].append(cl)
        elif (h >= 20 and h <= 55) or (r > g and g > b and (r - b) > 0.15):
            chroma_groups["Amber/Warm"].append(cl)
        elif (h >= 170 and h <= 260) or (b > r and (b - r) > 0.10):
            chroma_groups["Blue/White/Cool"].append(cl)
        elif (h >= 340 or h <= 15) and s > 0.30:
            chroma_groups["Red"].append(cl)
        elif (h >= 75 and h <= 165) and s > 0.25:
            chroma_groups["Green"].append(cl)
        else:
            chroma_groups["Amber/Warm"].append(cl)
            
    for grp_name, members in chroma_groups.items():
        if members:
            tot_area = sum(m["area_pct"] for m in members)
            avg_rgb = np.mean([m["rgb_mean"] for m in members], axis=0)
            avg_h = np.mean([m["hue"] for m in members])
            avg_s = np.mean([m["sat"] for m in members])
            print(f"    - {grp_name:<16}: {len(members):2d} sources | Area: {tot_area:4.2f}% | RGB: [{avg_rgb[0]:.2f}, {avg_rgb[1]:.2f}, {avg_rgb[2]:.2f}] | Hue: {avg_h:4.1f} deg, Sat: {avg_s:.2f}")

print("\nDiagnostics complete.")
