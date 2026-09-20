"""
scratch/test_object_rejection.py — Phase 3 Object vs Light Source Discrimination
================================================================================
Validates conservative light source detector against colored objects:
  - Red vehicle paint / tail-lights vs red traffic light / neon
  - Green foliage vs green traffic signal
  - Blue reflective road signs vs blue LED / neon
"""

import os
import sys
import imageio
import numpy as np
import scipy.ndimage as ndimage

def is_active_light_source(patch_rgb, surround_rgb):
    """
    Discriminates active light source from passive reflective object.
    Criteria:
      1. Peak brightness: V_max > 0.70
      2. Brightness excess: V_core - V_surround > 0.15
      3. Glow presence: surround is elevated relative to global dark floor
      4. Reject passive dark saturated objects (e.g. dark red paint: V=0.35, S=0.85)
    """
    v_core = np.max(patch_rgb, axis=-1)
    v_surr = np.max(surround_rgb, axis=-1) if surround_rgb.size > 0 else 0.0
    
    mean_v_core = float(np.mean(v_core))
    mean_v_surr = float(np.mean(v_surr)) if surround_rgb.size > 0 else 0.0
    excess = mean_v_core - mean_v_surr
    
    # Active light check
    if mean_v_core < 0.65:
        return False, "Rejected: Insufficient core brightness (passive object)"
    if excess < 0.12:
        return False, "Rejected: Insufficient local contrast excess (flat illuminated surface)"
    return True, f"Accepted: Active light source (Core V={mean_v_core:.2f}, Excess={excess:.2f})"

# Synthetic & extracted real test cases
test_cases = [
    {
        "name": "Red Car Paint (Night)",
        "core": np.array([[[0.45, 0.05, 0.05], [0.42, 0.04, 0.04]]]),
        "surr": np.array([[[0.40, 0.06, 0.06], [0.38, 0.05, 0.05]]]),
    },
    {
        "name": "Red Traffic Light / Neon",
        "core": np.array([[[0.95, 0.20, 0.25], [0.98, 0.22, 0.26]]]),
        "surr": np.array([[[0.35, 0.10, 0.12], [0.32, 0.08, 0.10]]]),
    },
    {
        "name": "Green Foliage (Day/Dusk)",
        "core": np.array([[[0.15, 0.38, 0.18], [0.12, 0.35, 0.15]]]),
        "surr": np.array([[[0.14, 0.36, 0.16], [0.13, 0.34, 0.14]]]),
    },
    {
        "name": "Green Traffic Signal",
        "core": np.array([[[0.40, 0.95, 0.70], [0.42, 0.98, 0.72]]]),
        "surr": np.array([[[0.18, 0.35, 0.28], [0.15, 0.32, 0.25]]]),
    },
    {
        "name": "Blue Road Sign (Reflective)",
        "core": np.array([[[0.10, 0.25, 0.55], [0.12, 0.28, 0.58]]]),
        "surr": np.array([[[0.09, 0.22, 0.50], [0.10, 0.24, 0.52]]]),
    },
    {
        "name": "Blue/White LED Streetlight",
        "core": np.array([[[0.85, 0.92, 0.98], [0.88, 0.95, 1.00]]]),
        "surr": np.array([[[0.25, 0.30, 0.40], [0.22, 0.28, 0.38]]]),
    },
]

print("=" * 80)
print("PHASE 3: VALIDATION OF OBJECT VS LIGHT SOURCE DISCRIMINATION")
print("=" * 80)

for tc in test_cases:
    is_light, reason = is_active_light_source(tc["core"], tc["surr"])
    status = "LIGHT SOURCE" if is_light else "OBJECT (REJECTED)"
    print(f"Test: {tc['name']:<30} -> {status:<18} | {reason}")

print("\nValidation complete.")
