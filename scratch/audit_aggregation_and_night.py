"""
scratch/audit_aggregation_and_night.py — Audit of Aggregation and Nighttime Metrics
"""
import json
import numpy as np

with open("scratch/session16_results.json", "r") as f:
    data = json.load(f)

baseline = data.get("baseline_comparison", [])

print("=" * 90)
print("SECTION 10 & 12 — AUDIT OF NIGHTTIME CLIPPING & CATEGORY AGGREGATION")
print("=" * 90)

# 1. Nighttime Videos
night_recs = [r for r in baseline if r["category"] == "B"]
print("\n--- NIGHTTIME PER-VIDEO BLACK CLIPPING ---")
for r in night_recs:
    print(f"{r['video_name']:18s} | Method: {r['method_name']:16s} | Black Clip: {r.get('black_clip', 0.0):5.2f}% | White Clip: {r.get('white_clip', 0.0):5.2f}% | Flicker: {r.get('flicker', 0.0):5.2f} | NIQE: {r.get('niqe', 0.0):5.2f}")

# 2. Check Averaging Methodology
# In Session 16, did each video receive EQUAL weight when calculating Category and Overall averages?
cats = sorted(list(set(r["category"] for r in baseline)))
methods = ["DCP", "CAP", "CLAHE", "Corrected Gamma", "Original CEEF", "Modified CEEF", "Proposed Method"]

print("\n--- EQUAL-WEIGHTED CATEGORY AVERAGES (NIQE, FLICKER, FPS) ---")
for cat in cats:
    cat_name = [r["category_name"] for r in baseline if r["category"] == cat][0]
    print(f"\n[Category {cat}: {cat_name}]")
    for m in methods:
        m_recs = [r for r in baseline if r["category"] == cat and r["method_name"] == m]
        # Equal video weighting
        avg_niqe = np.mean([r["niqe"] for r in m_recs])
        avg_flicker = np.mean([r["flicker"] for r in m_recs])
        avg_fps = np.mean([r["fps"] for r in m_recs])
        avg_black = np.mean([r["black_clip"] for r in m_recs])
        print(f"  {m:16s} | NIQE: {avg_niqe:5.2f} | Flicker: {avg_flicker:5.2f} | FPS: {avg_fps:5.1f} | Black: {avg_black:5.2f}%")

print("\n--- OVERALL EQUAL-WEIGHTED SUMMARY (13 VIDEOS) ---")
for m in methods:
    m_recs = [r for r in baseline if r["method_name"] == m]
    avg_niqe = np.mean([r["niqe"] for r in m_recs])
    avg_flicker = np.mean([r["flicker"] for r in m_recs])
    avg_fps = np.mean([r["fps"] for r in m_recs])
    avg_black = np.mean([r["black_clip"] for r in m_recs])
    print(f"  {m:16s} | Mean NIQE: {avg_niqe:5.2f} | Mean Flicker: {avg_flicker:5.2f} | Mean FPS: {avg_fps:5.1f} | Mean Black: {avg_black:5.2f}%")
