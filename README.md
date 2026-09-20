---
title: Aeroclear Dehazing
emoji: 🌫️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# AeroClear: Real-Time Fog Removal & Video Dehazing Suite

**Official Academic Title**: *Real-Time Fog Removal Using SSIF–CLAHE Temporal Coherence*

AeroClear is a high-throughput, physics-grounded, OpenCV-free real-time video dehazing engine and web suite. It delivers temporal consistency, robust atmospheric light estimation, chromaticity preservation, and dynamic baseline comparisons (DCP, CAP, CEEF, CLAHE, Light-DehazeNet) at 35–60+ FPS on CPU.

---

## 🌟 Key Capabilities & Architectural Highlights

1. **Decoupled Multi-Threaded Engine**:
   - Background worker processes video frames at maximum hardware speed (35–60+ FPS).
   - Condition-variable broadcasting prevents network lag from blocking the processing pipeline.

2. **100% OpenCV-Free Production Stack**:
   - Built purely on NumPy, SciPy, Pillow, scikit-image, and PyTorch CPU.
   - Zero `cv2` dependencies across all production algorithms.

3. **Physics-Dominant Processing**:
   - Adaptive transmission floor ($t_{\min}$) and chromaticity-preserving atmospheric light bounds.
   - Motion-aware temporal coherence that prevents flicker on static backgrounds while tracking moving objects without ghosting.

4. **Multi-Algorithm Evaluation Suite**:
   - **A4 (Novel Hybrid Mode)**: High-resolution detail preservation with zero color crush.
   - **A1, A2, A3 Modes**: High-efficiency and balanced operational modes.
   - **Academic Baselines**: Dark Channel Prior (DCP), Color Attenuation Prior (CAP), CLAHE, and Cost-Efficient Edge-Preserving Filter (CEEF).
   - **Light-DehazeNet**: Deep learning comparative baseline.

5. **Full Ground Truth & Evaluation Suite**:
   - Live real-time metric calculation: PSNR, SSIM, NIQE, LPIPS, Fog Density Index (FDI proxy), and Information Entropy.

---

## 🚀 Getting Started

### Running Locally (Native Python)

```bash
# 1. Clone the repository
git clone https://github.com/Sakthivel-06/Fog-removal-.git
cd Fog-removal-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch web app
python run_web_app.py --port 8000
```
Open your browser at `http://localhost:8000`.

### Running via Docker / Hugging Face Spaces

```bash
docker build -t aeroclear .
docker run -p 7860:7860 aeroclear
```
Open your browser at `http://localhost:7860`.

---

## 📁 Repository Structure

```
├── algorithms/              # Baseline implementations (DCP, CAP, CLAHE, CEEF, Light-DehazeNet)
├── checkpoints/             # Trained neural model weights
├── config/                  # Training and evaluation configuration YAMLs
├── data/                    # Coherent haze training and validation datasets
├── docs/                    # Research and dataset manifests
├── novel_method/            # Core physics-grounded pipeline and residual models
├── papers/                  # Academic reference papers and literature summaries
├── results/                 # Metric evaluations, visual comparisons, and step validations
├── scratch/                 # Experimental diagnostic scripts and research session reports
├── test_videos/             # Real-world foggy benchmark videos and test sequences
├── web_app/                 # High-throughput MJPEG server and frontend dashboard
├── Dockerfile               # Production container definition for Hugging Face Spaces
└── requirements.txt         # Core Python dependencies
```

---

## 📜 License & Attribution

This project is developed for research and academic evaluation under the Apache 2.0 / MIT guidelines.
