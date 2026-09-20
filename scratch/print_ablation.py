import json

with open("scratch/session16_results.json", "r") as f:
    data = json.load(f)

for a in data.get("ablation_study", []):
    print(f"{a['video_name']:35s} | {a['config_key']:12s} | Flicker: {a['flicker']:6.2f} | PSNR: {str(a['psnr'])[:6]} | SSIM: {str(a['ssim'])[:6]} | NIQE: {a['niqe']:5.2f} | FPS: {a['fps']:5.1f}")
