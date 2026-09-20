"""
tiny_unet.py
============
Session 21: Ultra-Lightweight Multi-Scale Residual CNN for Transmission Estimation.
Target Parameter Count: 13,921 trainable parameters.
Input: 8 channels at guide resolution (e.g. 240x135)
Output: 1 channel scalar residual Delta-t in [-0.30, +0.30]

Zero OpenCV dependency. Pure PyTorch CPU compatible.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int = 8, base_channels: int = 12, delta_scale: float = 0.30):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.delta_scale = delta_scale

        # Encoder: 8 -> 12 -> 12
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Downsampling: 2x
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Bottleneck: 12 -> 24 -> 24
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        # Decoder: (24 + 12 skip = 36) -> 12 -> 1
        self.dec_conv = nn.Sequential(
            nn.Conv2d(base_channels * 2 + base_channels, base_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, 1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 8, H, W)
        Returns:
            delta_t: Tensor of shape (B, 1, H, W) bounded in [-0.30, +0.30]
        """
        # Encoder
        e1 = self.enc1(x)

        # Bottleneck
        p = self.pool(e1)
        b = self.bottleneck(p)

        # Bilinear Upsampling
        u = F.interpolate(b, size=e1.shape[2:], mode="bilinear", align_corners=False)

        # Skip concatenation & decoding
        cat = torch.cat([u, e1], dim=1)
        delta_t_raw = self.dec_conv(cat)

        # Strict physical bounding
        delta_t = self.delta_scale * delta_t_raw
        return delta_t

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = TinyUNet()
    p_count = model.count_parameters()
    print(f"TinyUNet Programmatic Parameter Count: {p_count:,}")
    dummy_x = torch.randn(1, 8, 135, 240)
    out = model(dummy_x)
    print(f"Input shape:  {tuple(dummy_x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Output range: [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"Has NaN:      {torch.isnan(out).any().item()}")
    assert p_count == 13921, f"Parameter count mismatch! Expected 13,921, got {p_count}"
    print("Verification SUCCESSFUL!")
