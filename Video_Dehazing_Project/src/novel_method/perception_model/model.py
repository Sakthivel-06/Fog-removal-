import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=stride, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class TinyHazePerceptionModule(nn.Module):
    """
    A lightweight CNN (<50k parameters) to predict a local haze density map.
    Designed for real-time CPU inference.
    """
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2, bias=False), # Downsample
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.enc2 = DepthwiseSeparableConv(16, 32, stride=2) # Downsample
        self.enc3 = DepthwiseSeparableConv(32, 64, stride=1)
        
        # Decoder
        self.dec1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_dec1 = nn.BatchNorm2d(32)
        self.relu_dec1 = nn.ReLU(inplace=True)
        
        self.dec2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)
        self.bn_dec2 = nn.BatchNorm2d(16)
        self.relu_dec2 = nn.ReLU(inplace=True)
        
        # Output layer (1 channel density map, sigmoid activation [0,1])
        self.out = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        d1 = self.relu_dec1(self.bn_dec1(self.dec1(e3)))
        d2 = self.relu_dec2(self.bn_dec2(self.dec2(d1)))
        
        out = torch.sigmoid(self.out(d2))
        return out

if __name__ == "__main__":
    model = TinyHazePerceptionModule()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"TinyHazePerceptionModule Parameters: {total_params}")
    
    # Test forward pass
    dummy = torch.randn(1, 3, 128, 128)
    out = model(dummy)
    print(f"Input shape: {dummy.shape}, Output shape: {out.shape}")
