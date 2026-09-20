# TABLE 2: Official Ablation Matrix Across Evaluation Sequences

*Configurations:*
- **A1:** Physics Baseline (Frozen Sessions 1–19, zero learned weights, SHA256 verified)
- **A2:** Physics + CNN Residual (Trained without temporal loss, no inference temporal coherence)
- **A3:** Physics + CNN Residual (Trained with temporal loss, no inference temporal coherence)
- **A4:** Full Proposed Method (Physics + CNN Residual with temporal loss + inference motion-aware temporal coherence)

| Sequence | Mode | PSNR (dB) ↑ | SSIM ↑ | Temporal Flicker ↓ | NIQE (Proxy) ↓ | Black Clip % | White Clip % | Processing FPS ↑ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Synthetic GT (`car-detection`) | **A1: Physics Baseline** | 9.61 | 0.6876 | 21.007 | 3.54 | 0.00% | 0.33% | 51.0 |
| Synthetic GT (`car-detection`) | **A2: CNN Residual (No Temp Loss, No Inf Filter)** | 9.64 | 0.6866 | 21.251 | 3.50 | 0.00% | 0.33% | 27.7 |
| Synthetic GT (`car-detection`) | **A3: CNN Residual (With Temp Loss, No Inf Filter)** | 9.64 | 0.6868 | 21.216 | 3.50 | 0.00% | 0.33% | 30.0 |
| Synthetic GT (`car-detection`) | **A4: Full Proposed (CNN Residual + Temp Loss + Inf Filter)** | 9.59 | 0.6915 | 20.641 | 3.53 | 0.00% | 0.33% | 26.8 |
| Dense Natural Fog (`foggy_video1`) | **A1: Physics Baseline** | Requires GT | Requires GT | 9.486 | 3.86 | 0.69% | 6.09% | 104.7 |
| Dense Natural Fog (`foggy_video1`) | **A2: CNN Residual (No Temp Loss, No Inf Filter)** | Requires GT | Requires GT | 8.433 | 4.12 | 0.07% | 7.08% | 50.7 |
| Dense Natural Fog (`foggy_video1`) | **A3: CNN Residual (With Temp Loss, No Inf Filter)** | Requires GT | Requires GT | 8.360 | 4.04 | 0.05% | 7.13% | 61.1 |
| Dense Natural Fog (`foggy_video1`) | **A4: Full Proposed (CNN Residual + Temp Loss + Inf Filter)** | Requires GT | Requires GT | 8.483 | 4.05 | 0.10% | 6.06% | 58.7 |
| Dynamic Urban Roadway (`foggy_video2`) | **A1: Physics Baseline** | Requires GT | Requires GT | 2.919 | 3.86 | 2.51% | 0.55% | 93.6 |
| Dynamic Urban Roadway (`foggy_video2`) | **A2: CNN Residual (No Temp Loss, No Inf Filter)** | Requires GT | Requires GT | 2.760 | 3.87 | 0.14% | 1.04% | 40.6 |
| Dynamic Urban Roadway (`foggy_video2`) | **A3: CNN Residual (With Temp Loss, No Inf Filter)** | Requires GT | Requires GT | 2.714 | 3.88 | 0.10% | 0.89% | 61.5 |
| Dynamic Urban Roadway (`foggy_video2`) | **A4: Full Proposed (CNN Residual + Temp Loss + Inf Filter)** | Requires GT | Requires GT | 2.708 | 3.92 | 0.23% | 0.88% | 46.3 |
| Difficult Night Lighting (`foggy_video3`) | **A1: Physics Baseline** | Requires GT | Requires GT | 8.060 | 5.75 | 34.89% | 2.46% | 104.7 |
| Difficult Night Lighting (`foggy_video3`) | **A2: CNN Residual (No Temp Loss, No Inf Filter)** | Requires GT | Requires GT | 8.161 | 5.52 | 33.32% | 2.31% | 54.2 |
| Difficult Night Lighting (`foggy_video3`) | **A3: CNN Residual (With Temp Loss, No Inf Filter)** | Requires GT | Requires GT | 8.126 | 5.52 | 33.33% | 2.29% | 54.9 |
| Difficult Night Lighting (`foggy_video3`) | **A4: Full Proposed (CNN Residual + Temp Loss + Inf Filter)** | Requires GT | Requires GT | 8.221 | 5.56 | 33.73% | 2.29% | 51.4 |
