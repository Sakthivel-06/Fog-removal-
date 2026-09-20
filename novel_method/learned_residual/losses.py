"""
losses.py
=========
Session 21: Configurable Multi-Objective Loss Functions for Learned Transmission Residual.
Includes:
  1. Transmission Fidelity Loss (L_trans)
  2. Physical Scene Reconstruction Loss (L_rec) with L1 + SSIM
  3. Motion-Masked Temporal Consistency Loss (L_temp)
  4. Spatial Edge Gradient Consistency Loss (L_grad)
  5. Residual L2 Regularization (L_reg)

Zero OpenCV dependency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian_window_2d(window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    gauss = torch.tensor(
        [
            -(x - window_size // 2) ** 2 / float(2 * sigma**2)
            for x in range(window_size)
        ]
    ).exp()
    gauss = gauss / gauss.sum()
    _1d_window = gauss.unsqueeze(1)
    _2d_window = _1d_window.mm(_1d_window.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d_window


def ssim_tensor(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    c1: float = 0.01**2,
    c2: float = 0.03**2,
) -> torch.Tensor:
    """
    Computes differentiable SSIM on 4D image tensors in [0, 1] on CPU.
    """
    channel = img1.size(1)
    window = gaussian_window_2d(window_size, sigma).to(img1.device)
    window = window.expand(channel, 1, window_size, window_size).contiguous()

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean()


def spatial_gradient(x: torch.Tensor) -> tuple:
    """Computes forward spatial finite differences."""
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    return dx, dy


class CompositeResidualLoss(nn.Module):
    def __init__(
        self,
        lambda_trans: float = 1.00,
        lambda_rec: float = 0.50,
        lambda_temp: float = 0.40,
        lambda_grad: float = 0.25,
        lambda_reg: float = 0.10,
    ):
        super().__init__()
        self.lambda_trans = lambda_trans
        self.lambda_rec = lambda_rec
        self.lambda_temp = lambda_temp
        self.lambda_grad = lambda_grad
        self.lambda_reg = lambda_reg

    def forward(
        self,
        delta_t_k: torch.Tensor,
        t_cand_k: torch.Tensor,
        t_gt_k: torch.Tensor,
        hazy_k: torch.Tensor = None,
        clean_k: torch.Tensor = None,
        A_k: torch.Tensor = None,
        delta_t_next: torch.Tensor = None,
        motion_mask_k: torch.Tensor = None,
    ) -> dict:
        """
        Computes composite multi-objective loss.
        """
        losses = {}

        # 1. Transmission Fidelity Loss (L1)
        l_trans = F.l1_loss(t_cand_k, t_gt_k)
        losses["l_trans"] = l_trans

        # 2. Physical Scene Reconstruction Loss (L1 + SSIM)
        if hazy_k is not None and clean_k is not None and A_k is not None:
            # Physical Inversion: J = (I - A) / t + A
            # Safe inverse with clamping
            t_inv = torch.clamp(t_cand_k, min=0.05, max=1.0)
            if A_k.dim() == 2:
                A_k = A_k.unsqueeze(-1).unsqueeze(-1)
            J_pred = (hazy_k - A_k) / t_inv + A_k
            J_pred = torch.clamp(J_pred, 0.0, 1.0)

            l1_rec = F.l1_loss(J_pred, clean_k)
            ssim_val = ssim_tensor(J_pred, clean_k)
            l_rec = l1_rec + (1.0 - ssim_val)
            losses["l_rec"] = l_rec
        else:
            l_rec = torch.tensor(0.0, device=t_cand_k.device)
            losses["l_rec"] = l_rec

        # 3. Motion-Masked Temporal Consistency Loss
        if delta_t_next is not None and motion_mask_k is not None:
            # static weight = (1 - motion_mask)
            diff = torch.abs(delta_t_next - delta_t_k)
            static_weight = torch.clamp(1.0 - motion_mask_k, 0.0, 1.0)
            l_temp = torch.mean(static_weight * diff)
            losses["l_temp"] = l_temp
        else:
            l_temp = torch.tensor(0.0, device=t_cand_k.device)
            losses["l_temp"] = l_temp

        # 4. Gradient Consistency Loss
        dx_cand, dy_cand = spatial_gradient(t_cand_k)
        dx_gt, dy_gt = spatial_gradient(t_gt_k)
        l_grad = F.l1_loss(dx_cand, dx_gt) + F.l1_loss(dy_cand, dy_gt)
        losses["l_grad"] = l_grad

        # 5. Regularization (L2 on delta_t)
        l_reg = torch.mean(delta_t_k**2)
        losses["l_reg"] = l_reg

        # Total Weighted Loss
        total_loss = (
            self.lambda_trans * l_trans
            + self.lambda_rec * l_rec
            + self.lambda_temp * l_temp
            + self.lambda_grad * l_grad
            + self.lambda_reg * l_reg
        )
        losses["total"] = total_loss
        return losses
