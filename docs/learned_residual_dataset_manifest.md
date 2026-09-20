# Learned Transmission Residual Dataset Manifest

## Overview
This manifest records the video-level splits, generation parameters, and isolation controls for training and validating the experimental learned transmission residual CNN ($\Delta t$).

## Data Leakage Prevention Rules
1. **Split by Source Video:** Consecutive frames from the same video are never partitioned across splits.
2. **Benchmark Quarantine:** The 4 official evaluation sequences (`car-detection_patchy_haze.mp4`, `foggy_video1.mp4`, `foggy_video2.mp4`, `foggy_video3.mp4`) are strictly quarantined and never used for training or validation.
3. **Storage Isolation:** All generated coherent training data is stored in `data/training_coherent_haze/` and kept completely separated from `test_videos/` and `results/`.

---

## Dataset Split Manifest

| Video ID | Source Video File | Split | Frames Allocated | Resolution | Generator Beta ($\beta_0, \beta_{\text{var}}$) | Wind Advection $(v_x, v_y)$ | Airlight Range ($A$) | Ground Truth |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TRAIN-01** | `person-bicycle_clean.mp4` | **TRAIN** | 120 frames (f0–f119) | $240 \times 135$ | $\beta_0=1.20, \beta_{\text{var}}=0.60$ | $(+0.80, +0.25)$ px/f | $A = [0.86, 0.88, 0.86]$ | Known analytical $t_{\text{gt}}$ |
| **TRAIN-02** | `WhatsApp Video 2026-09-10 at 10.00.16 PM.mp4` | **TRAIN** | 100 frames (f0–f99) | $240 \times 135$ | $\beta_0=1.40, \beta_{\text{var}}=0.70$ | $(+0.60, -0.20)$ px/f | $A = [0.84, 0.84, 0.86]$ | Known analytical $t_{\text{gt}}$ |
| **VAL-01** | `WhatsApp Video 2026-09-10 at 10.00.25 PM.mp4` | **VAL** | 50 frames (f0–f49) | $240 \times 135$ | $\beta_0=1.30, \beta_{\text{var}}=0.65$ | $(+0.75, +0.30)$ px/f | $A = [0.85, 0.87, 0.85]$ | Known analytical $t_{\text{gt}}$ |
| **TEST-01** | `WhatsApp Video 2026-09-10 at 9.43.25 PM.mp4` | **TEST** | 50 frames (f0–f49) | $240 \times 135$ | $\beta_0=1.25, \beta_{\text{var}}=0.50$ | $(+0.90, +0.15)$ px/f | $A = [0.88, 0.88, 0.90]$ | Known analytical $t_{\text{gt}}$ |

---

## Seed & Reproducibility Control
- Base seed for TRAIN: `20260914`
- Base seed for VAL: `20260915`
- Base seed for TEST: `20260916`
- Transmission constraint: $0 < t_{\text{gt}} \le 1.0$ guaranteed across all frames.
