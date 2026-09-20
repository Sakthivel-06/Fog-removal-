# TABLE 6: Literature Comparison and Algorithmic Characteristics

*Comparative positioning against recent published driving video dehazing literature (2024–2025) and classic priors:*

| Method | Publication | Core Mechanism | Trainable Parameters | OpenCV Dependency | GPU Required? | Temporal Coherence Mechanism | Target Scenario |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **DCP (He et al.)** | TPAMI 2011 | Dark Channel Prior heuristic | 0 | None (in our impl) | No | None (Frame-by-frame) | General Single Images |
| **CAP (Zhu et al.)** | TIP 2015 | Color Attenuation depth model | 0 | None (in our impl) | No | None (Frame-by-frame) | General Single Images |
| **CEEF (Kaplan)** | Access 2023 | Multi-Exposure Fusion | 0 | None (in our impl) | No | None (Frame-by-frame) | Daytime Roadways |
| **DVD (Fan et al.)** | CVPR 2024 | Non-Aligned Regularization | ~11.5M | Required (official) | Required (CUDA) | Video alignment loss | Driving Safety Videos |
| **DCL (Fan et al.)** | AAAI 2025 | Depth-Centric Scattering Model | ~14.2M | Required (official) | Required (CUDA) | Inter-frame depth consistency | Driving Depth + Dehaze |
| **Unpaired Recurrent (Pan)** | Pat. Rec. 2025 | Recurrent feature alignment | ~8.9M | Required (official) | Required (CUDA) | Hidden state recurrence | Unpaired Real Videos |
| **MSTF-Net (Zheng et al.)** | CVIU 2025 | Multiscale Spatio-Temporal Fusion | ~16.4M | Required (official) | Required (CUDA) | 3D conv / Attention blocks | Low-Light Dynamic Roads |
| **Proposed Method A4 (Ours)** | **Session 23** | **Physics-Subordinated Residual $\Delta t$** | **13,921 (13.9k)** | **ZERO (`cv2`-free)** | **NO (Pure CPU)** | **Motion-aware temporal coherence** | **Real-Time Edge Driving** |
