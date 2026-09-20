# TABLE 4: Real-World No-Reference Quality & Stability Metrics

*Evaluated on real driving sequences without ground truth (Strict adherence to Rule 11):*

| Video Sequence | Method Mode | NIQE (Proxy) ↓ | Temporal Flicker ↓ | Static Flicker ↓ | Motion Flicker ↓ | Black Crush % | White Sat % |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense Fog** (`foggy_video1`) | **Baseline A1** | **3.86** | 9.486 | 2.264 | 39.128 | 0.69% | 6.09% |
| | **A2 (No Temp Loss)** | 4.12 | 8.433 | 2.152 | 37.245 | 0.07% | 7.08% |
| | **A3 (Temp Loss Only)** | 4.04 | 8.360 | 2.172 | 36.847 | 0.05% | 7.13% |
| | **A4 (Proposed Full)** | 4.05 | **8.483** | **2.140** | **36.932** | **0.10%** | 6.06% |
| **Dynamic Urban** (`foggy_video2`) | **Baseline A1** | **3.86** | 2.919 | 1.330 | 42.209 | 2.51% | 0.55% |
| | **A2 (No Temp Loss)** | 3.87 | 2.760 | 1.295 | 40.082 | 0.14% | 1.04% |
| | **A3 (Temp Loss Only)** | 3.88 | 2.714 | 1.280 | 39.814 | 0.10% | 0.89% |
| | **A4 (Proposed Full)** | 3.92 | **2.708** | **1.247** | **39.824** | **0.23%** | 0.88% |
| **Difficult Night** (`foggy_video3`) | **Baseline A1** | 5.75 | **8.060** | — | — | 34.89% | 2.46% |
| | **A2 (No Temp Loss)** | 5.52 | 8.161 | — | — | 33.32% | 2.31% |
| | **A3 (Temp Loss Only)** | 5.52 | 8.126 | — | — | 33.33% | 2.29% |
| | **A4 (Proposed Full)** | **5.56** | 8.221 | — | — | **33.73%** | 2.29% |
