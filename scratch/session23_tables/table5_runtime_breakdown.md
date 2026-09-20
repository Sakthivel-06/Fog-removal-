# TABLE 5: Component-Wise CPU Latency & Execution Breakdown

*Measured on standard CPU (Intel/AMD x86_64, Single-Thread Execution, Resolution $426 \times 240$ display, $240 \times 135$ guide):*

| Pipeline Stage | Baseline A1 Latency (ms) | Proposed A4 Latency (ms) | Proportion of Proc Time (%) | Algorithmic Implementation |
| :--- | :---: | :---: | :---: | :--- |
| **1. PyAV Video Decode (I/O)** | 12.02 ms | 10.85 ms | — (I/O Dependent) | PyAV / FFmpeg Demuxing & H.264 Decode |
| **2. Frame Prep & Float Normalization** | 2.89 ms | 3.28 ms | 19.2% | NumPy uint8 $\to$ float32, Pillow downscale |
| **3. CAP Depth & Atmospheric Light $A$** | 2.55 ms | 2.65 ms | 15.5% | Color Attenuation Prior + Quadtree Airlight |
| **4. DCP Dark Channel Map** | 1.70 ms | 1.78 ms | 10.4% | Min-filter over RGB channels |
| **5. Spatial Transmission Fusion ($t_{\text{fused}}$)** | 0.04 ms | 0.04 ms | 0.2% | Confidence-weighted spatial combination |
| **6. TinyUNet Residual Inference ($\Delta t$)** | — (N/A) | **6.46 ms** | **37.9%** | 13,921-param PyTorch CPU Forward Pass |
| **7. Adaptive Safety Floor ($t_{\min}(x,y)$)** | 0.02 ms | 0.02 ms | 0.1% | Clamping: $\max(t_{\text{candidate}}, t_{\min})$ |
| **8. Edge-Preserving Guided Smoothing** | 0.59 ms | 0.78 ms | 4.6% | Box-filter guided refinement ($r=5$) |
| **9. Motion-Aware Temporal Coherence** | 0.22 ms | 0.25 ms | 1.5% | Inter-frame motion mask & adaptive $\alpha(x)$ |
| **10. Physical Radiance Inversion** | 1.51 ms | 1.77 ms | 10.4% | Koschmieder Inversion: $J = (I - A)/t + A$ |
| **Total Pure Processing Latency** | **9.55 ms** | **17.04 ms** | **100.0%** | Excludes disk and PyAV video decode |
| **Total End-to-End Latency** | **22.93 ms** | **30.49 ms** | — | Fully inclusive of I/O and format conversion |
| **Processing-Only FPS ↑** | **104.7 FPS** | **58.7 FPS** | — | **Exceeds 30 FPS Real-Time Target** |
| **End-to-End System FPS ↑** | **43.6 FPS** | **32.8 FPS** | — | **Exceeds 30 FPS Real-Time Target** |
