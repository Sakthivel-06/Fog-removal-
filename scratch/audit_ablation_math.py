"""
scratch/audit_ablation_math.py — Mathematical Verification of Ablation Study
"""
import json

with open("scratch/session16_results.json", "r") as f:
    data = json.load(f)

abl = data.get("ablation_study", [])
vids = sorted(list(set(a["video_name"] for a in abl)))

print("=" * 90)
print("SECTION 16 — COMPONENT ABLATION AUDIT")
print("=" * 90)

for v in vids:
    print(f"\n--- Video: {v} ---")
    v_recs = {a["config_key"]: a for a in abl if a["video_name"] == v}
    full = v_recs.get("A_Full")
    if not full:
        continue
    
    print(f"Full Config A: Flicker={full['flicker']:.2f}, NIQE={full['niqe']:.2f}, FPS={full['fps']:.1f}, PSNR={full['psnr']}, SSIM={full['ssim']}")
    
    for k in ["B_NoTmin", "C_NoFusion", "D_NoAirlight", "E_NoABound", "F_NoRefine", "G_NoTemporal"]:
        sub = v_recs.get(k)
        if not sub:
            continue
        # Check Flicker change
        f_diff = sub["flicker"] - full["flicker"]
        # Percent increase when removed: (sub - full) / full * 100
        pct_inc_when_removed = (sub["flicker"] - full["flicker"]) / (full["flicker"] + 1e-6) * 100.0
        # Percent reduction achieved by feature: (sub - full) / sub * 100
        pct_red_by_feature = (sub["flicker"] - full["flicker"]) / (sub["flicker"] + 1e-6) * 100.0
        
        # Check SSIM change
        if full["ssim"] != "Requires Ground Truth" and sub["ssim"] != "Requires Ground Truth":
            ssim_diff = sub["ssim"] - full["ssim"]
            ssim_str = f"SSIM diff: {ssim_diff:+.4f} (from {full['ssim']:.4f} to {sub['ssim']:.4f})"
        else:
            ssim_str = "No GT"
            
        print(f"  {k:14s}: Flicker {full['flicker']:.2f} -> {sub['flicker']:.2f} | Disabling increases flicker by {pct_inc_when_removed:+5.1f}% (Feature achieves {pct_red_by_feature:4.1f}% reduction) | {ssim_str} | FPS: {sub['fps']:.1f}")
