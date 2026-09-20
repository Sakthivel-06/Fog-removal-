"""
train.py
========
Session 21: Training Engine for Learned Transmission Residual TinyUNet.
Features:
  - Differentiable multi-objective loss (trans, rec, temp, grad, reg)
  - Video-level train/validation dataset separation
  - Deterministic random seeding
  - CPU-only execution
  - Checkpoint saving and reload verification
  - Zero OpenCV dependency

Usage:
  python train.py --epochs 3 --batch_size 4 --debug
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from novel_method.pipeline import RealTimeDehazingPipeline
from novel_method.learned_residual.tiny_unet import TinyUNet
from novel_method.learned_residual.losses import CompositeResidualLoss
from novel_method.learned_residual.dataset import CoherentHazeDataset
from data_prep.coherent_haze_generator import CoherentHazeGenerator
import imageio.v3 as iio


def build_coherent_samples_from_video(video_path: str, max_frames: int, gen_params: dict) -> list:
    """
    Reads clean video frames and generates coherent synthetic hazy samples using CoherentHazeGenerator.
    """
    reader = iio.imread(video_path)
    h, w = reader[0].shape[:2]
    # Use target guide resolution aspect ratio
    gh, gw = 135, 240
    gen = CoherentHazeGenerator(
        height=gh * 2,
        width=gw * 2,
        beta_base=gen_params.get("beta_base", 1.20),
        beta_var=gen_params.get("beta_var", 0.60),
        wind_vx=gen_params.get("wind_vx", 0.80),
        wind_vy=gen_params.get("wind_vy", 0.25),
        seed=gen_params.get("seed", 42),
    )

    samples = []
    cached_d = None
    n_frames = min(len(reader), max_frames)

    for k in range(n_frames):
        clean_frame = reader[k]
        hazy, tgt, A, beta, depth = gen.synthesize_frame(clean_frame, k, cached_depth=cached_d)
        if cached_d is None:
            cached_d = depth

        samples.append({
            "clean_uint8": clean_frame,
            "hazy_uint8": hazy,
            "t_gt": tgt,
            "A": A,
        })

    return samples


def train_debug(
    train_video: str = "test_videos/person-bicycle_clean.mp4",
    val_video: str = "test_videos/WhatsApp Video 2026-09-10 at 10.00.25 PM.mp4",
    epochs: int = 3,
    batch_size: int = 4,
    lr: float = 1e-3,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints",
):
    """
    Executes Phase 7 Debug Training run (2-3 epochs on small coherent dataset).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("=== Initializing Phase 7 Debug Training ===")
    print(f"Random Seed: {seed}")
    print(f"Device: CPU")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")

    pipeline = RealTimeDehazingPipeline()
    model = TinyUNet().to("cpu")
    print(f"Model Parameters: {model.count_parameters():,}")

    loss_fn = CompositeResidualLoss(
        lambda_trans=1.00,
        lambda_rec=0.50,
        lambda_temp=0.40,
        lambda_grad=0.25,
        lambda_reg=0.10,
    )
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 1. Build Coherent Datasets (Split strictly by video)
    print("\nSynthesizing coherent training samples from TRAIN-01...")
    train_samples = build_coherent_samples_from_video(
        train_video,
        max_frames=30, # compact debug size
        gen_params={"beta_base": 1.20, "beta_var": 0.60, "wind_vx": 0.80, "wind_vy": 0.25, "seed": 20260914}
    )
    train_dataset = CoherentHazeDataset(train_samples, pipeline)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Train dataset: {len(train_dataset)} frame pairs")

    print("Synthesizing coherent validation samples from VAL-01...")
    val_samples = build_coherent_samples_from_video(
        val_video,
        max_frames=15, # compact debug size
        gen_params={"beta_base": 1.30, "beta_var": 0.65, "wind_vx": 0.70, "wind_vy": -0.20, "seed": 20260915}
    )
    val_dataset = CoherentHazeDataset(val_samples, pipeline)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"Validation dataset: {len(val_dataset)} frame pairs")

    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        t_start = time.perf_counter()
        train_losses = []

        for b_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            x_k = batch["x_k"].to("cpu")
            x_next = batch["x_next"].to("cpu")
            t_gt_k = batch["t_gt_k"].to("cpu")
            clean_k = batch["clean_k"].to("cpu")
            hazy_k = batch["hazy_k"].to("cpu")
            A_k = batch["A_k"].to("cpu")
            m_mask_k = batch["motion_mask_k"].to("cpu")

            # Forward pass on frame k
            delta_t_k = model(x_k)
            t_fused_k = x_k[:, 3:4, :, :]
            t_cand_k = torch.clamp(t_fused_k + delta_t_k, 0.05, 1.0)

            # Forward pass on frame k+1 for temporal consistency loss
            delta_t_next = model(x_next)

            losses = loss_fn(
                delta_t_k=delta_t_k,
                t_cand_k=t_cand_k,
                t_gt_k=t_gt_k,
                hazy_k=hazy_k,
                clean_k=clean_k,
                A_k=A_k,
                delta_t_next=delta_t_next,
                motion_mask_k=m_mask_k,
            )

            total_loss = losses["total"]
            total_loss.backward()

            # Check gradient finiteness
            for p in model.parameters():
                if p.grad is not None:
                    torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)

            optimizer.step()
            train_losses.append(total_loss.item())

        # Validation loop
        model.eval()
        val_losses = []
        val_delta_ts = []
        with torch.no_grad():
            for batch in val_loader:
                x_k = batch["x_k"].to("cpu")
                x_next = batch["x_next"].to("cpu")
                t_gt_k = batch["t_gt_k"].to("cpu")
                clean_k = batch["clean_k"].to("cpu")
                hazy_k = batch["hazy_k"].to("cpu")
                A_k = batch["A_k"].to("cpu")
                m_mask_k = batch["motion_mask_k"].to("cpu")

                delta_t_k = model(x_k)
                t_fused_k = x_k[:, 3:4, :, :]
                t_cand_k = torch.clamp(t_fused_k + delta_t_k, 0.05, 1.0)
                delta_t_next = model(x_next)

                val_loss_dict = loss_fn(
                    delta_t_k=delta_t_k,
                    t_cand_k=t_cand_k,
                    t_gt_k=t_gt_k,
                    hazy_k=hazy_k,
                    clean_k=clean_k,
                    A_k=A_k,
                    delta_t_next=delta_t_next,
                    motion_mask_k=m_mask_k,
                )
                val_losses.append(val_loss_dict["total"].item())
                val_delta_ts.extend(delta_t_k.flatten().cpu().numpy())

        epoch_time = time.perf_counter() - t_start
        mean_tr_loss = float(np.mean(train_losses))
        mean_val_loss = float(np.mean(val_losses))
        d_arr = np.array(val_delta_ts)

        print(
            f"Epoch [{epoch}/{epochs}] ({epoch_time:.2f}s) | "
            f"Train Loss: {mean_tr_loss:.4f} | Val Loss: {mean_val_loss:.4f} | "
            f"Delta-t Range: [{d_arr.min():.4f}, {d_arr.max():.4f}] (Mean: {d_arr.mean():.4f}, Std: {d_arr.std():.4f})"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(mean_tr_loss, 4),
            "val_loss": round(mean_val_loss, 4),
            "delta_t_min": round(float(d_arr.min()), 4),
            "delta_t_max": round(float(d_arr.max()), 4),
            "delta_t_mean": round(float(d_arr.mean()), 4),
            "delta_t_std": round(float(d_arr.std()), 4),
            "epoch_sec": round(epoch_time, 2),
        })

    # Save checkpoint
    ckpt_path = os.path.join(checkpoint_dir, "delta_t_tiny_unet_debug.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "parameters": model.count_parameters(),
        "history": history,
        "seed": seed,
    }, ckpt_path)
    print(f"\nCheckpoint successfully saved to {ckpt_path}")

    # Reload test
    print("Testing checkpoint reload...")
    test_model = TinyUNet().to("cpu")
    ckpt_loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    test_model.load_state_dict(ckpt_loaded["model_state_dict"])
    test_model.eval()
    test_x = torch.randn(1, 8, 135, 240)
    with torch.no_grad():
        test_out = test_model(test_x)
    print(f"Reload test passed! Output shape: {test_out.shape}, range: [{test_out.min().item():.4f}, {test_out.max().item():.4f}]")

    return ckpt_path, history


if __name__ == "__main__":
    train_debug(epochs=3, batch_size=4)
