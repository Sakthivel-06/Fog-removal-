import torch
import torch.nn as nn
import time

# --- Candidate 1: Depthwise-Separable Residual CNN ---
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=True)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        return self.act(self.pw(self.dw(x)))

class DWResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(ch, ch)
        self.conv2 = DepthwiseSeparableConv(ch, ch)
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class DWResidualCNN(nn.Module):
    def __init__(self, in_ch=8, base_ch=40, num_blocks=3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.body = nn.Sequential(*[DWResBlock(base_ch) for _ in range(num_blocks)])
        self.tail = nn.Sequential(
            nn.Conv2d(base_ch, 1, 1),
            nn.Tanh() # scaled to [-scale, +scale]
        )
    def forward(self, x):
        return self.tail(self.body(self.head(x))) * 0.30

# --- Candidate 2: Dilated Residual CNN ---
class DilatedResBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=True)
    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))

class DilatedResidualCNN(nn.Module):
    def __init__(self, in_ch=8, base_ch=28):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.b1 = DilatedResBlock(base_ch, dilation=1)
        self.b2 = DilatedResBlock(base_ch, dilation=2)
        self.b3 = DilatedResBlock(base_ch, dilation=4)
        self.tail = nn.Sequential(
            nn.Conv2d(base_ch, 1, 3, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        feat = self.head(x)
        feat = self.b1(feat)
        feat = self.b2(feat)
        feat = self.b3(feat)
        return self.tail(feat) * 0.30

# --- Candidate 3: Tiny U-Net ---
class TinyUNet(nn.Module):
    def __init__(self, in_ch=8):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.pool = nn.AvgPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec = nn.Sequential(
            nn.Conv2d(32 + 16, 16, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Tanh()
        )
    def forward(self, x):
        h, w = x.shape[2:]
        e1 = self.enc1(x)
        p = self.pool(e1)
        b = self.bottleneck(p)
        u = self.up(b)
        if u.shape[2:] != e1.shape[2:]:
            u = nn.functional.interpolate(u, size=e1.shape[2:], mode='bilinear', align_corners=False)
        cat = torch.cat([u, e1], dim=1)
        return self.dec(cat) * 0.30

def benchmark_model(name, model, x):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(x)
        t0 = time.perf_counter()
        n_iters = 50
        for _ in range(n_iters):
            _ = model(x)
        dt = (time.perf_counter() - t0) / n_iters * 1000.0 # ms
    print(f"=== {name} ===")
    print(f"  Params: {params:,}")
    print(f"  CPU Latency (Guide 240x135): {dt:.2f} ms ({1000.0/dt:.1f} FPS)")
    return params, dt

if __name__ == '__main__':
    x_guide = torch.randn(1, 8, 135, 240)
    m1 = DWResidualCNN(in_ch=8, base_ch=40, num_blocks=3)
    m2 = DilatedResidualCNN(in_ch=8, base_ch=28)
    m3 = TinyUNet(in_ch=8)

    benchmark_model("Candidate 1: Depthwise-Separable ResNet (DW-ResNet)", m1, x_guide)
    benchmark_model("Candidate 2: Dilated Multi-Scale ResNet (DMS-ResNet)", m2, x_guide)
    benchmark_model("Candidate 3: Tiny U-Net (Encoder-Decoder)", m3, x_guide)
