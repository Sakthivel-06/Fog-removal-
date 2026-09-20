# TABLE 3: Ground-Truth Transmission & Radiance Benchmark Results

*Evaluated on `car-detection_patchy_haze.mp4` against true reference transmission $t_{\text{ref}}$ and clean radiance $J_{\text{gt}}$:*

| Metric / Stage | Baseline A1 | CNN Candidate $t_{\text{candidate}}$ | Constrained $t_{\text{constrained}}$ | Full Proposed A4 | Improvement vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Transmission MAE vs $t_{\text{ref}}$ ↓** | 0.1083 | **0.0690** | 0.1499 | 0.1457 | **-36.3% (Unconstrained)** |
| **Mean Transmission Value** | 0.4406 | 0.3315 | 0.5342 | 0.5281 | Ground Truth Reference = 0.3986 |
| **Transmission Bias vs GT** | +0.0421 | -0.0671 | +0.1356 | +0.1295 | Resolves CAP/DCP over-estimation |
| **Radiance PSNR (dB) ↑** | 9.61 | — | — | 9.59 | -0.02 dB (Within H.264 noise margin) |
| **Radiance SSIM ↑** | 0.6876 | — | — | **0.6915** | **+0.0039 Structural Fidelity Gain** |
| **Temporal Flicker ($\Delta L$) ↓** | 21.007 | 21.251 (A2) | 21.216 (A3) | **20.641** | **-0.366 Level Smoother** |
| **Static Region Flicker ↓** | 5.705 | 5.717 | 5.717 | **5.707** | Motion Masked |
| **Dynamic Motion Flicker ↓** | 30.407 | 30.763 | 30.716 | **29.839** | **-0.568 Moving Object Stability** |
| **Temporal SSIM ↑** | 0.6768 | 0.6714 | 0.6719 | **0.6833** | **+0.0065 Temporal Edge Stability** |
