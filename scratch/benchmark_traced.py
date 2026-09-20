import torch
import torch.nn as nn
import time

class TinyUNet(nn.Module):
    def __init__(self, in_ch=8, base_ch=12):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.pool = nn.AvgPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(base_ch * 2, base_ch * 2, 3, padding=1),
            nn.LeakyReLU(0.1)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec = nn.Sequential(
            nn.Conv2d(base_ch * 2 + base_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(base_ch, 1, 1),
            nn.Tanh()
        )
    def forward(self, x):
        e1 = self.enc1(x)
        p = self.pool(e1)
        b = self.bottleneck(p)
        u = self.up(b)
        if u.shape[2:] != e1.shape[2:]:
            u = nn.functional.interpolate(u, size=e1.shape[2:], mode='bilinear', align_corners=False)
        cat = torch.cat([u, e1], dim=1)
        return self.dec(cat) * 0.30

# Tiny DW-ResNet with downsampling
class FastDWResNet(nn.Module):
    def __init__(self, in_ch=8, ch=24, num_blocks=3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, stride=2, padding=1), # stride 2 downsamples directly
            nn.LeakyReLU(0.1)
        )
        # DW blocks at half res
        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
                nn.Conv2d(ch, ch, 1, bias=True),
                nn.LeakyReLU(0.1),
                nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
                nn.Conv2d(ch, ch, 1, bias=True)
            ))
        self.blocks = nn.ModuleList(blocks)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.tail = nn.Sequential(
            nn.Conv2d(ch, 1, 3, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        h = self.head(x)
        for b in self.blocks:
            h = h + b(h)
        return self.tail(self.up(h)) * 0.30

x = torch.randn(1, 8, 135, 240)
for name, m in [('FastDWResNet (stride=2)', FastDWResNet()), ('TinyUNet (base_ch=12)', TinyUNet(base_ch=12))]:
    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    m.eval()
    traced = torch.jit.trace(m, x)
    # Warmup
    for _ in range(10):
        _ = traced(x)
    t0 = time.perf_counter()
    n = 100
    for _ in range(n):
        _ = traced(x)
    dt = (time.perf_counter() - t0) / n * 1000.0
    print(f"{name}: Params={params:,}, Traced CPU Latency={dt:.2f} ms ({1000.0/dt:.1f} FPS)")
