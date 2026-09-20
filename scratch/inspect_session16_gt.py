import json

with open("scratch/session16_results.json", "r") as f:
    data = json.load(f)

print("=== BASELINE ENTRIES COUNT ===")
print("Total baseline records:", len(data.get("baseline_comparison", [])))

gt_records = [r for r in data.get("baseline_comparison", []) if r.get("has_gt")]
print("\n=== GROUND TRUTH RECORDS ===")
for r in gt_records:
    print(f"Video: {r['video_name']} | Method: {r['method_name']:16s} | PSNR: {r['psnr']} | SSIM: {r['ssim']} | NIQE: {r['niqe']} | Flicker: {r['flicker']} | FPS: {r['fps']}")

print("\n=== ABLATION RECORDS ===")
for a in data.get("ablation_study", []):
    print(f"Config: {a['config']} ({a['config_name'][:25]:25s}) | Video: {a['video_name']} | FDI: {a['fdi_proxy']:.3f} | Flicker: {a['flicker']:.2f} | PSNR: {a['psnr']} | SSIM: {a['ssim']}")
