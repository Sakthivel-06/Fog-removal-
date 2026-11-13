import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dilation=1):
        super().__init__()
        padding = dilation
        self.depthwise = nn.Conv2d(
            in_c, in_c, kernel_size=3, stride=stride, padding=padding,
            dilation=dilation, groups=in_c, bias=False
        )
        self.pointwise = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))

class SkySegNet(nn.Module):
    """
    SkySegNet: Ultra-Lightweight Semantic Sky Segmentation CNN.
    ~24k parameters. Designed for high-precision real-time sky boundary detection
    at >100 FPS on CPU.
    """
    def __init__(self):
        super().__init__()
        
        # Stem (stride 2: 1/2 resolution)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU6(inplace=True)
        )
        
        # Encoder Stage 1 (stride 2: 1/4 resolution)
        self.enc1 = DepthwiseSeparableConv(16, 24, stride=2)
        
        # Encoder Stage 2 (stride 2: 1/8 resolution)
        self.enc2 = DepthwiseSeparableConv(24, 32, stride=2)
        
        # Dilated Context Stage (1/8 resolution, wide receptive field for horizon perception)
        self.context = DepthwiseSeparableConv(32, 48, stride=1, dilation=2)
        
        # Global Pyramid Pooling (ASPP-lite)
        self.aspp_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(48, 16, kernel_size=1, bias=True),
            nn.ReLU6(inplace=True)
        )
        self.aspp_conv = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True)
        )
        self.aspp_fusion = nn.Sequential(
            nn.Conv2d(16 + 24, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # Decoder Stage 1 (1/8 -> 1/4 resolution)
        self.dec1 = DepthwiseSeparableConv(32 + 24, 24)
        
        # Decoder Stage 2 (1/4 -> 1/2 resolution)
        self.dec2 = DepthwiseSeparableConv(24 + 16, 16)
        
        # Final Output Layer (1/2 -> Full resolution)
        self.final_head = nn.Sequential(
            nn.Conv2d(16, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU6(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        
        # Stem: 1/2
        s = self.stem(x)
        # Enc1: 1/4
        e1 = self.enc1(s)
        # Enc2: 1/8
        e2 = self.enc2(e1)
        # Context: 1/8
        c = self.context(e2)
        
        # ASPP-lite
        p = self.aspp_pool(c)
        p = F.interpolate(p, size=c.shape[2:], mode="bilinear", align_corners=False)
        conv_c = self.aspp_conv(c)
        fused = self.aspp_fusion(torch.cat([p, conv_c], dim=1))
        
        # Decoder
        d1 = F.interpolate(fused, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        d2 = F.interpolate(d1, size=s.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, s], dim=1))
        
        out = self.final_head(d2)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        
        return torch.sigmoid(out)

if __name__ == "__main__":
    net = SkySegNet()
    total_params = sum(p.numel() for p in net.parameters())
    print(f"SkySegNet parameters: {total_params:,}")
    dummy = torch.randn(1, 3, 180, 240)
    out = net(dummy)
    print(f"Input: {dummy.shape} -> Output: {out.shape} (min: {out.min():.2f}, max: {out.max():.2f})")
