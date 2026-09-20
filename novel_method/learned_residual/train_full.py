"""
novel_method/learned_residual/train_full.py
===========================================
Session 22: Full 25-Epoch Training Engine for TinyUNet Transmission Residual.
Trains with pre-cached feature tensors for ultra-fast, CPU-compatible execution.
Saves:
  - checkpoints/delta_t_tiny_unet_best.pt (Full loss with lambda_temp=0.40)
  - checkpoints/delta_t_tiny_unet_no_temp.pt (Ablation model with lambda_temp=0.00)
"""

import os
import sys
import time
import yaml
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from novel_method.pipeline import RealTimeDehazingPipeline
from novel_method.learned_residual.tiny_unet import TinyUNet
from novel_method.learned_residual.losses import CompositeResidualLoss
from novel_method.learned_residual.feature_extractor import extract_physics_features
from novel_method.temporal_coherence import compute_motion_mask


class CachedFeatureDataset(Dataset):
    def __init__(self, data_dir: str, target_guide_res=(240, 135)):
        self.data_dir = data_dir
        self.gw, self.gh = target_guide_res
        with open(os.path.join(data_dir, "manifest.json"), "r") as f:
            self.manifest = json.load(f)

        self.pipeline = RealTimeDehazingPipeline()
        print(f"Pre-caching physics features for {len(self.manifest)} frames in {data_dir}...")
        self.cached_frames = []

        for item in self.manifest:
            hazy_p = os.path.join(data_dir, item["hazy_file"])
            clean_p = os.path.join(data_dir, item["clean_file"])
            tgt_p = os.path.join(data_dir, item["tgt_file"])

            hazy_img = np.array(Image.open(hazy_p))
            clean_img = np.array(Image.open(clean_p))
            tgt_raw = np.load(tgt_p)
            A = np.array(item["A"], dtype=np.float32)

            # Resize to guide resolution
            clean_guide = np.array(Image.fromarray(clean_img).resize((self.gw, self.gh), Image.BILINEAR)).astype(np.float32) / 255.0
            hazy_guide = np.array(Image.fromarray(hazy_img).resize((self.gw, self.gh), Image.BILINEAR)).astype(np.float32) / 255.0

            # Pipeline expects proc_float to be (gh*2, gw*2) = (270, 480)
            proc_float = hazy_img.astype(np.float32) / 255.0
            features, meta = extract_physics_features(proc_float, self.pipeline)

            if tgt_raw.shape != (self.gh, self.gw):
                t_gt_guide = np.array(Image.fromarray(tgt_raw).resize((self.gw, self.gh), Image.BILINEAR))
            else:
                t_gt_guide = tgt_raw

            self.cached_frames.append({
                "features": features, # (8, gh, gw)
                "t_gt": t_gt_guide[None, :, :].astype(np.float32), # (1, gh, gw)
                "clean": np.transpose(clean_guide, (2, 0, 1)).astype(np.float32), # (3, gh, gw)
                "hazy": np.transpose(hazy_guide, (2, 0, 1)).astype(np.float32), # (3, gh, gw)
                "gray_h": meta["gray_h"], # (gh, gw)
                "A": A,
                "video": item["source_video"],
                "frame_idx": item["frame_in_video"],
            })

        print(f"Pre-caching complete! Total frames: {len(self.cached_frames)}")

    def __len__(self):
        return max(0, len(self.cached_frames) - 1)

    def __getitem__(self, idx):
        curr = self.cached_frames[idx]
        nxt = self.cached_frames[idx + 1]

        # Check if frames cross video boundary
        if curr["video"] != nxt["video"]:
            # Wrap to previous frame to avoid cross-video contamination
            nxt = curr

        # Compute motion mask between adjacent frames
        m_mask, _, _ = compute_motion_mask(
            nxt["gray_h"],
            curr["gray_h"],
            motion_thresh_low=0.03,
            motion_thresh_high=0.15,
            filter_size=3,
        )

        return {
            "x_k": torch.from_numpy(curr["features"]),
            "x_next": torch.from_numpy(nxt["features"]),
            "t_gt_k": torch.from_numpy(curr["t_gt"]),
            "t_gt_next": torch.from_numpy(nxt["t_gt"]),
            "clean_k": torch.from_numpy(curr["clean"]),
            "hazy_k": torch.from_numpy(curr["hazy"]),
            "A_k": torch.from_numpy(curr["A"]),
            "motion_mask_k": torch.from_numpy(m_mask[None, :, :].astype(np.float32)),
        }


def run_training_experiment(
    config_path="config/session22_training_config.yaml",
    train_ablation_no_temp=True,
):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    epochs = cfg["training"]["epochs"]
    batch_size = cfg["training"]["batch_size"]
    lr = cfg["training"]["learning_rate"]
    patience = cfg["training"].get("early_stopping_patience", 7)

    # Prepare datasets
    train_dataset = CachedFeatureDataset("data/training_coherent_haze/train")
    val_dataset = CachedFeatureDataset("data/training_coherent_haze/val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    def train_single_model(experiment_label, lambda_temp_val, save_path):
        print(f"\n=======================================================")
        print(f"Starting Training: {experiment_label} (lambda_temp = {lambda_temp_val})")
        print(f"=======================================================")
        torch.manual_seed(seed)
        model = TinyUNet().to("cpu")
        print(f"Model Parameters: {model.count_parameters():,}")

        loss_fn = CompositeResidualLoss(
            lambda_trans=cfg["loss_weights"]["lambda_trans"],
            lambda_rec=cfg["loss_weights"]["lambda_rec"],
            lambda_temp=lambda_temp_val,
            lambda_grad=cfg["loss_weights"]["lambda_grad"],
            lambda_reg=cfg["loss_weights"]["lambda_reg"],
        )

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg["training"]["weight_decay"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=cfg["training"]["min_lr"])

        best_val_loss = float("inf")
        best_epoch = 0
        patience_counter = 0
        history = []

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            model.train()

            tr_total = []
            tr_trans = []
            tr_rec = []
            tr_temp = []
            tr_grad = []
            tr_reg = []

            for batch in train_loader:
                optimizer.zero_grad()
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

                losses["total"].backward()
                for p in model.parameters():
                    if p.grad is not None:
                        torch.nan_to_num(p.grad, nan=0.0, posinf=1.0, neginf=-1.0)

                optimizer.step()

                tr_total.append(losses["total"].item())
                tr_trans.append(losses["l_trans"].item())
                tr_rec.append(losses["l_rec"].item())
                tr_temp.append(losses["l_temp"].item())
                tr_grad.append(losses["l_grad"].item())
                tr_reg.append(losses["l_reg"].item())

            scheduler.step()

            # Validation Loop
            model.eval()
            val_total = []
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
                    val_total.append(losses["total"].item())
                    val_delta_ts.extend(delta_t_k.flatten().cpu().numpy())

            dt_arr = np.array(val_delta_ts)
            epoch_time = time.perf_counter() - t0
            mean_tr = float(np.mean(tr_total))
            mean_val = float(np.mean(val_total))
            cur_lr = scheduler.get_last_lr()[0]

            log_entry = {
                "epoch": epoch,
                "train_loss": round(mean_tr, 4),
                "val_loss": round(mean_val, 4),
                "loss_trans": round(float(np.mean(tr_trans)), 4),
                "loss_rec": round(float(np.mean(tr_rec)), 4),
                "loss_temp": round(float(np.mean(tr_temp)), 4),
                "loss_grad": round(float(np.mean(tr_grad)), 4),
                "loss_reg": round(float(np.mean(tr_reg)), 4),
                "delta_t_min": round(float(dt_arr.min()), 4),
                "delta_t_max": round(float(dt_arr.max()), 4),
                "delta_t_mean": round(float(dt_arr.mean()), 4),
                "delta_t_std": round(float(dt_arr.std()), 4),
                "lr": round(cur_lr, 6),
                "time_sec": round(epoch_time, 2),
            }
            history.append(log_entry)

            is_best = mean_val < best_val_loss
            if is_best:
                best_val_loss = mean_val
                best_epoch = epoch
                patience_counter = 0
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "parameters": model.count_parameters(),
                    "config": cfg,
                }, save_path)
            else:
                patience_counter += 1

            best_marker = "(*BEST*)" if is_best else ""
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.2f}s) | "
                f"Train: {mean_tr:.4f} | Val: {mean_val:.4f} {best_marker} | "
                f"Delta-t: [{dt_arr.min():.4f}, {dt_arr.max():.4f}] (Mean: {dt_arr.mean():.4f}) | "
                f"Loss components: [T:{log_entry['loss_trans']:.3f}, R:{log_entry['loss_rec']:.3f}, Temp:{log_entry['loss_temp']:.3f}]"
            )

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch} (no improvement for {patience} epochs).")
                break

        print(f"\nTraining Complete for {experiment_label}!")
        print(f"Best Epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")
        print(f"Saved best model to: {save_path}")
        return {
            "experiment": experiment_label,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "save_path": save_path,
            "history": history,
        }

    # 1. Main Training: Full composite loss (lambda_temp = 0.40)
    res_full = train_single_model(
        experiment_label="Proposed TinyUNet (with Temporal Loss)",
        lambda_temp_val=cfg["loss_weights"]["lambda_temp"],
        save_path=cfg["checkpoints"]["best_checkpoint"],
    )

    # 2. Ablation Training: Without temporal loss (lambda_temp = 0.00)
    res_no_temp = None
    if train_ablation_no_temp:
        res_no_temp = train_single_model(
            experiment_label="Ablation A2 (without Temporal Loss)",
            lambda_temp_val=0.00,
            save_path=cfg["checkpoints"]["ablation_no_temp_checkpoint"],
        )

    diagnostics = {
        "full_model": res_full,
        "no_temp_model": res_no_temp,
    }
    diag_path = "scratch/session22_training_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"\nSaved full training diagnostics to {diag_path}")
    return diagnostics


if __name__ == "__main__":
    run_training_experiment()
