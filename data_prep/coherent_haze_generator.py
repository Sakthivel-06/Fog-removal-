"""
data_prep/coherent_haze_generator.py
====================================
Session 21: Temporally Coherent Synthetic Haze Generator with Analytical Ground-Truth Transmission.

Key Capabilities:
1. Spatially correlated and temporally continuous advecting haze field beta(x, y, k).
2. Physical optical model:
     t_gt(x, y, k) = exp(-beta(x, y, k) * d(x, y))
     I_hazy(x, y, k) = J_clean(x, y, k) * t_gt(x, y, k) + A * (1 - t_gt(x, y, k))
3. Realistic atmospheric light randomization A in [0.78, 0.96] with chromaticity jitter.
4. Guaranteed physical constraints: 0 < t_gt <= 1.0, 0 <= I_hazy <= 255.
5. 100% NumPy + SciPy + Pillow + imageio. STRICTLY ZERO OpenCV (cv2).
"""

import os
import sys
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
import imageio.v3 as iio


def generate_perlin_grid_2d(shape, res, seed=42):
    """
    Generates a 2D smooth Perlin noise field with a deterministic random seed.
    Zero OpenCV dependency.
    """
    rng = np.random.RandomState(seed)
    h = ((shape[0] + res[0] - 1) // res[0]) * res[0]
    w = ((shape[1] + res[1] - 1) // res[1]) * res[1]

    delta = (res[0] / h, res[1] / w)
    d = (h // res[0], w // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * np.pi * rng.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))

    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    noise = np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)

    return noise[:shape[0], :shape[1]]


def generate_large_coherent_canvas(h, w, extra_pad=(300, 400), seed=42):
    """
    Creates an oversized 2D multi-scale smooth canvas to allow continuous advection.
    """
    canvas_h = h + extra_pad[0]
    canvas_w = w + extra_pad[1]
    res1 = (max(1, canvas_h // 120), max(1, canvas_w // 120))
    res2 = (max(1, canvas_h // 60), max(1, canvas_w // 60))

    n1 = generate_perlin_grid_2d((canvas_h, canvas_w), res1, seed=seed)
    n2 = generate_perlin_grid_2d((canvas_h, canvas_w), res2, seed=seed + 100)

    canvas = 0.7 * n1 + 0.3 * n2
    canvas = (canvas - np.min(canvas)) / (np.ptp(canvas) + 1e-6)
    return canvas


class CoherentHazeGenerator:
    """
    Generates temporally coherent synthetic fog over clean video sequences.
    """
    def __init__(
        self,
        height: int,
        width: int,
        beta_base: float = 1.20,
        beta_var: float = 0.80,
        wind_vx: float = 0.80, # pixels per frame
        wind_vy: float = 0.25, # pixels per frame
        seed: int = 12345,
        a_mean: float = 0.88,
        a_jitter: float = 0.04,
    ):
        self.h = height
        self.w = width
        self.beta_base = beta_base
        self.beta_var = beta_var
        self.wind_vx = wind_vx
        self.wind_vy = wind_vy
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Generate smooth advection canvas
        self.canvas = generate_large_coherent_canvas(height, width, extra_pad=(350, 450), seed=seed)
        self.ch, self.cw = self.canvas.shape

        # Initialize physical atmospheric light vector in [0.78, 0.96]
        a_base = np.clip(self.rng.normal(a_mean, a_jitter), 0.78, 0.96)
        chroma = self.rng.normal(0.0, 0.012, size=3)
        self.A = np.clip(a_base + chroma, 0.75, 0.98).astype(np.float32)

    def estimate_depth_cap(self, clean_rgb_float: np.ndarray) -> np.ndarray:
        """
        Estimates smooth depth from clean image using Color Attenuation Prior (Zhu et al.).
        """
        ch_h, ch_w = clean_rgb_float.shape[:2]
        v = np.max(clean_rgb_float, axis=2)
        dc = np.min(clean_rgb_float, axis=2)
        s = (v - dc) / (v + 1e-6)
        raw_d = 0.121779 + 0.959710 * v - 0.780245 * s
        # Apply vertical ground gradient prior to ensure distant objects/sky have deeper depth
        y_coords = np.linspace(0.2, 1.0, ch_h, dtype=np.float32)[:, None]
        geom_depth = 1.2 - 0.8 * y_coords
        depth = 0.6 * raw_d + 0.4 * geom_depth
        depth = ndimage.uniform_filter(np.clip(depth, 0.15, 2.5), size=7)
        return depth.astype(np.float32)

    def get_haze_field(self, frame_idx: int) -> np.ndarray:
        """
        Extracts smoothly shifting 2D beta field at frame_idx via wind advection.
        """
        offset_x = (self.wind_vx * frame_idx) % (self.cw - self.w - 1)
        offset_y = (self.wind_vy * frame_idx) % (self.ch - self.h - 1)

        ix = int(offset_x)
        iy = int(offset_y)
        fx = offset_x - ix
        fy = offset_y - iy

        # Bilinear sampling from canvas for perfect sub-pixel continuity
        patch = (
            (1 - fx) * (1 - fy) * self.canvas[iy : iy + self.h, ix : ix + self.w]
            + fx * (1 - fy) * self.canvas[iy : iy + self.h, ix + 1 : ix + self.w + 1]
            + (1 - fx) * fy * self.canvas[iy + 1 : iy + self.h + 1, ix : ix + self.w]
            + fx * fy * self.canvas[iy + 1 : iy + self.h + 1, ix + 1 : ix + self.w + 1]
        )

        # Gradual temporal oscillation (condensation/evaporation cycle)
        temp_cycle = 0.08 * np.sin(2.0 * np.pi * frame_idx / 180.0)
        beta_field = self.beta_base + self.beta_var * patch + temp_cycle
        return np.clip(beta_field, 0.20, 3.50).astype(np.float32)

    def synthesize_frame(
        self,
        clean_rgb_uint8: np.ndarray,
        frame_idx: int,
        cached_depth: np.ndarray = None,
    ):
        """
        Synthesizes a hazy frame and returns (hazy_uint8, t_gt_float, A_vector, beta_field, depth).
        """
        ch, cw = clean_rgb_uint8.shape[:2]
        if (cw, ch) != (self.w, self.h):
            clean_rgb_uint8 = np.array(
                Image.fromarray(clean_rgb_uint8).resize((self.w, self.h), Image.BILINEAR)
            )

        clean_float = clean_rgb_uint8.astype(np.float32) / 255.0

        if cached_depth is None:
            depth = self.estimate_depth_cap(clean_float)
        else:
            depth = cached_depth

        beta = self.get_haze_field(frame_idx)
        t_gt = np.exp(-beta * depth)
        t_gt = np.clip(t_gt, 0.01, 1.0).astype(np.float32)

        # Physical Koschmieder scattering model:
        # I = J * t + A * (1 - t)
        t3 = t_gt[:, :, None]
        A3 = self.A[None, None, :]
        hazy_float = clean_float * t3 + A3 * (1.0 - t3)
        hazy_uint8 = np.clip(hazy_float * 255.0, 0.0, 255.0).astype(np.uint8)

        return hazy_uint8, t_gt, self.A.copy(), beta, depth


def test_coherent_generator():
    """
    Phase 1 Validation: Generates a 20-frame debug sequence and computes temporal continuity metrics.
    """
    print("=== Testing Coherent Haze Generator (Phase 1) ===")
    test_h, test_w = 240, 320
    gen = CoherentHazeGenerator(test_h, test_w, beta_base=1.10, beta_var=0.60, seed=777)

    # Dummy clean video with moving synthetic object
    frames = []
    for k in range(20):
        img = np.zeros((test_h, test_w, 3), dtype=np.uint8)
        img[:] = [60, 90, 60]
        # Road
        img[140:, :] = [40, 45, 50]
        # Moving box
        bx = int(30 + k * 8)
        img[120:170, bx : bx + 50] = [180, 50, 40]
        frames.append(img)

    betas = []
    t_gts = []
    hazies = []

    cached_d = None
    for k, f in enumerate(frames):
        hazy, tgt, A, beta, depth = gen.synthesize_frame(f, k, cached_depth=cached_d)
        if cached_d is None:
            cached_d = depth
        betas.append(beta)
        t_gts.append(tgt)
        hazies.append(hazy)

    # 1. Temporal change between adjacent frames
    beta_diffs = [np.mean(np.abs(betas[i + 1] - betas[i])) for i in range(len(betas) - 1)]
    tgt_diffs = [np.mean(np.abs(t_gts[i + 1] - t_gts[i])) for i in range(len(t_gts) - 1)]
    hazy_lum_diffs = [
        np.mean(np.abs(0.299 * hazies[i + 1][:, :, 0] - 0.299 * hazies[i][:, :, 0]))
        for i in range(len(hazies) - 1)
    ]

    all_tgt = np.array(t_gts)
    all_beta = np.array(betas)

    print(f"1. Mean absolute beta-field change/frame: {np.mean(beta_diffs):.5f} (Target: < 0.03)")
    print(f"2. Mean absolute t_gt change/frame:        {np.mean(tgt_diffs):.5f} (Target: < 0.02)")
    print(f"3. Mean hazy luminance change/frame:       {np.mean(hazy_lum_diffs):.5f}")
    print(f"4. t_gt range:                             [{np.min(all_tgt):.4f}, {np.max(all_tgt):.4f}] (Mean: {np.mean(all_tgt):.4f})")
    print(f"5. beta range:                             [{np.min(all_beta):.4f}, {np.max(all_beta):.4f}]")
    print(f"6. Atmospheric light A:                    {gen.A.tolist()}")
    print(f"7. NaN check:                              beta NaN: {np.isnan(all_beta).any()}, t_gt NaN: {np.isnan(all_tgt).any()}")

    assert np.mean(beta_diffs) < 0.05, "Beta field change too large (temporal discontinuity)!"
    assert np.all(all_tgt > 0.0) and np.all(all_tgt <= 1.0), "t_gt violates 0 < t_gt <= 1.0!"
    assert not np.isnan(all_tgt).any(), "NaN detected in t_gt!"

    # Save a small sample
    out_dir = r"c:\Users\sakth\OneDrive\Desktop\Fog\fog-removal-project\scratch\phase1_coherent_test"
    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(hazies[0]).save(os.path.join(out_dir, "hazy_f0.jpg"))
    Image.fromarray(hazies[10]).save(os.path.join(out_dir, "hazy_f10.jpg"))
    Image.fromarray((t_gts[0] * 255.0).astype(np.uint8)).save(os.path.join(out_dir, "tgt_f0.png"))
    Image.fromarray((t_gts[10] * 255.0).astype(np.uint8)).save(os.path.join(out_dir, "tgt_f10.png"))
    print(f"Sample debug frames saved to {out_dir}")
    print("=== Phase 1 Validation PASSED ===")


if __name__ == "__main__":
    test_coherent_generator()
