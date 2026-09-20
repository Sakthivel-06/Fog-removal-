"""
scratch/verify_daytime_safety.py — Phase 10 Daytime Safety Verification
========================================================================
Runs across:
  - foggy_video1.mp4
  - foggy_video6.mp4
  - foggy_video7.mp4
  - foggy_video2.mp4
  - car-detection_patchy_haze.mp4

Compares:
  1. Authoritative Frozen Daytime Pipeline (novel_method/pipeline.py)
  2. Experimental Pipeline with Illumination Gate (novel_method/nighttime_experimental.py)
Verifies:
  max pixel delta == 0
  mean pixel delta == 0.0000
"""

import os
import sys
import imageio
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
from pipeline import RealTimeDehazingPipeline
from nighttime_experimental import ExperimentalNighttimePipeline

DAY_VIDEOS = [
    "foggy_video1.mp4",
    "foggy_video6.mp4",
    "foggy_video7.mp4",
    "foggy_video2.mp4",
    "car-detection_patchy_haze.mp4",
]

def verify_daytime_equivalence():
    print("=" * 80)
    print("SESSION 15 PHASE 10 — DAYTIME BIT-EXACT SAFETY VERIFICATION")
    print("=" * 80)

    all_passed = True
    results = {}

    for vname in DAY_VIDEOS:
        vpath = os.path.join("test_videos", vname)
        if not os.path.exists(vpath):
            print(f"Warning: {vpath} not found, skipping.")
            continue

        reader = imageio.get_reader(vpath)
        frames = []
        for idx, f in enumerate(reader):
            if idx >= 15:  # test 15 frames
                break
            frames.append(f)
        reader.close()

        pipe_frozen = RealTimeDehazingPipeline()
        pipe_exp = ExperimentalNighttimePipeline(enable_nighttime_branch=True, night_mode="prototype")

        max_diffs = []
        mean_diffs = []
        gate_decisions = []

        for f in frames:
            out_frozen, _ = pipe_frozen.process_frame(f)
            out_exp, _, diag = pipe_exp.process_frame(f, return_diagnostics=True)

            gate_decisions.append(diag.get("gate_decision", "UNKNOWN"))
            diff = np.abs(out_frozen.astype(int) - out_exp.astype(int))
            max_diffs.append(int(np.max(diff)))
            mean_diffs.append(float(np.mean(diff)))

        overall_max = max(max_diffs)
        overall_mean = float(np.mean(mean_diffs))
        is_exact = (overall_max == 0 and overall_mean == 0.0)

        results[vname] = {
            "max_pixel_delta": overall_max,
            "mean_pixel_delta": overall_mean,
            "gate_decisions": list(set(gate_decisions)),
            "bit_exact": is_exact,
        }

        status = "PASSED (BIT-EXACT 0.0000)" if is_exact else "FAILED"
        if not is_exact:
            all_passed = False

        print(f"{vname:30s} | Max Delta: {overall_max:d} | Mean Delta: {overall_mean:.6f} | Gate: {list(set(gate_decisions))} | {status}")

    print("-" * 80)
    if all_passed:
        print("PHASE 10 ALL DAYTIME TESTS PASSED: 100% BIT-EXACT IDENTICAL TO FROZEN DAYTIME PIPELINE.")
    else:
        print("PHASE 10 FAILED: Differences detected!")
    return all_passed, results

if __name__ == "__main__":
    passed, res = verify_daytime_equivalence()
    if not passed:
        sys.exit(1)
