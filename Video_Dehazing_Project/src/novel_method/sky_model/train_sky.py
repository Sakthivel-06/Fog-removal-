import os
import sys
import time
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
import imageio.v3 as iio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Ensure sky_model directory is on sys.path
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
from model import SkySegNet

def generate_pseudo_sky_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Multi-cue spatial-gradient-color pseudo-label generator for outdoor driving scenes.
    Extracts high-fidelity ground truth sky masks based on physical horizon properties.
    """
    h, w = img_rgb.shape[:2]
    img_f = img_rgb.astype(np.float32) / 255.0
    gray = 0.299 * img_f[:,:,0] + 0.587 * img_f[:,:,1] + 0.114 * img_f[:,:,2]
    
    # 1. Gradient magnitude (Sky has almost zero internal texture)
    grad = ndimage.gaussian_gradient_magnitude(gray, sigma=2.0)
    smooth_score = np.exp(-grad * 12.0)
    
    # 2. Vertical altitude prior (Sky is in top 50% of the camera frame)
    y_coords = np.linspace(0.0, 1.0, h)[:, None]
    y_map = np.tile(y_coords, (1, w))
    alt_prior = np.clip(1.0 - y_map * 1.8, 0.0, 1.0)
    
    # 3. Brightness / blue component
    bright_score = np.clip((gray - 0.35) / 0.45, 0.0, 1.0)
    
    # Combined multi-cue map
    raw_sky = alt_prior * smooth_score * bright_score
    mask = (raw_sky > 0.30).astype(np.float32)
    
    # Clean up mask via morphological closing/opening
    mask = ndimage.binary_opening(mask, structure=np.ones((5, 5))).astype(np.float32)
    mask = ndimage.binary_closing(mask, structure=np.ones((9, 9))).astype(np.float32)
    mask = ndimage.gaussian_filter(mask, sigma=3.0)
    return np.clip(mask, 0.0, 1.0)

class RoadSkyDataset(Dataset):
    def __init__(self, frames, masks, target_size=(240, 180)):
        self.target_size = target_size # (W, H)
        self.samples = []
        for f, m in zip(frames, masks):
            # Resize
            img = Image.fromarray(f).resize(self.target_size, Image.BILINEAR)
            mask = Image.fromarray((m * 255).astype(np.uint8)).resize(self.target_size, Image.BILINEAR)
            
            img_t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).permute(2, 0, 1)
            mask_t = torch.from_numpy(np.array(mask).astype(np.float32) / 255.0).unsqueeze(0)
            self.samples.append((img_t, mask_t))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()

def compute_miou(pred, target, thresh=0.5):
    pred_bin = (pred > thresh).float()
    target_bin = (target > thresh).float()
    intersection = (pred_bin * target_bin).sum().item()
    union = (pred_bin + target_bin).clamp(0, 1).sum().item()
    return (intersection + 1e-6) / (union + 1e-6)

def train():
    print("=" * 60)
    print("SkySegNet: Dataset Extraction & Deep Learning Training")
    print("=" * 60)
    
    # 1. Gather frames across test videos
    video_dir = os.path.abspath(os.path.join(_DIR, "..", "..", "test_videos"))
    target_videos = [
        "foggy_video1.mp4",
        "foggy_video.mp4",
        "car-detection_clean.mp4",
        "person-bicycle_clean.mp4",
    ]
    
    collected_frames = []
    for v_name in target_videos:
        v_path = os.path.join(video_dir, v_name)
        if not os.path.exists(v_path):
            continue
        try:
            reader = iio.imiter(v_path, plugin="pyav")
            for i, f in enumerate(reader):
                if i >= 60:
                    break
                if i % 3 == 0:
                    collected_frames.append(f)
            reader.close()
        except Exception as e:
            print(f"Skipping {v_name}: {e}")
            
    print(f"Extracted {len(collected_frames)} real outdoor driving frames.")
    
    # 2. Generate ground-truth pseudo-masks
    masks = [generate_pseudo_sky_mask(f) for f in collected_frames]
    
    # 3. Train/Val Split (80% / 20%)
    np.random.seed(42)
    indices = np.random.permutation(len(collected_frames))
    split = int(0.8 * len(collected_frames))
    train_idx, val_idx = indices[:split], indices[split:]
    
    train_frames = [collected_frames[i] for i in train_idx]
    train_masks  = [masks[i] for i in train_idx]
    val_frames   = [collected_frames[i] for i in val_idx]
    val_masks    = [masks[i] for i in val_idx]
    
    print(f"Train samples: {len(train_frames)}, Validation samples: {len(val_frames)}")
    
    train_dataset = RoadSkyDataset(train_frames, train_masks, target_size=(240, 180))
    val_dataset   = RoadSkyDataset(val_frames, val_masks, target_size=(240, 180))
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 4. Model, Optimizer, Loss
    device = torch.device("cpu")
    model = SkySegNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    bce_loss = nn.BCELoss()
    
    epochs = 20
    best_miou = 0.0
    weights_dir = os.path.join(_DIR, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    best_path = os.path.join(weights_dir, "sky_seg_net.pth")
    
    print("\nStarting Training on CPU...")
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val Acc':>8} | {'Val mIoU':>8} | {'Time (s)':>8}")
    print("-" * 65)
    
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss_total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = bce_loss(pred, y) + 0.5 * dice_loss(pred, y)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * len(x)
            
        train_loss = train_loss_total / len(train_dataset)
        
        # Validation
        model.eval()
        val_loss_total = 0.0
        total_correct = 0
        total_pixels = 0
        val_miou_sum = 0.0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = bce_loss(pred, y) + 0.5 * dice_loss(pred, y)
                val_loss_total += loss.item() * len(x)
                
                # Accuracy & mIoU
                pred_bin = (pred > 0.5).float()
                target_bin = (y > 0.5).float()
                total_correct += (pred_bin == target_bin).sum().item()
                total_pixels += pred_bin.numel()
                val_miou_sum += compute_miou(pred, y) * len(x)
                
        val_loss = val_loss_total / len(val_dataset)
        val_acc = (total_correct / total_pixels) * 100.0
        val_miou = (val_miou_sum / len(val_dataset)) * 100.0
        
        elapsed = time.time() - t_start
        print(f"{epoch:5d} | {train_loss:10.4f} | {val_loss:10.4f} | {val_acc:7.2f}% | {val_miou:7.2f}% | {elapsed:8.1f}")
        
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), best_path)
            
    print("-" * 65)
    print(f"Training Complete in {time.time() - t_start:.1f}s.")
    print(f"Best Validation mIoU: {best_miou:.2f}%, Saved weights to {best_path}")

if __name__ == "__main__":
    train()
