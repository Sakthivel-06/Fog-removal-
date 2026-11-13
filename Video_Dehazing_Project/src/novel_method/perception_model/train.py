import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.ndimage as ndimage

# Add data_prep to path to import synthetic_haze
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data_prep")))
from synthetic_haze import add_synthetic_fog

from model import TinyHazePerceptionModule

class SyntheticHazeDataset(Dataset):
    def __init__(self, size=200, img_size=128):
        self.size = size
        self.img_size = img_size
        
        # Base clean image (using a simple gradient/noise mix for variety)
        # Ideally, we would load real clean images, but this ensures it runs anywhere
        self.base_images = []
        for _ in range(10):
            # Generate random clean-ish backgrounds
            bg = np.random.randint(50, 200, (img_size, img_size, 3), dtype=np.uint8)
            # Smooth each channel with scipy gaussian filter (replaces cv2.GaussianBlur)
            bg = np.stack([ndimage.gaussian_filter(bg[:,:,c], sigma=5) for c in range(3)], axis=2).astype(np.uint8)
            self.base_images.append(bg)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        base_img = self.base_images[idx % len(self.base_images)]
        
        # Add patchy synthetic fog (transmission map is ground truth)
        hazy, trans = add_synthetic_fog(base_img, beta=1.5, pattern='patchy')
        
        # Convert to tensor [0, 1]
        hazy = hazy.astype(np.float32) / 255.0
        hazy = torch.from_numpy(hazy).permute(2, 0, 1)
        
        # Ground truth density map is (1 - transmission) 
        # Haze density is high when transmission is low
        density = 1.0 - trans
        density = torch.from_numpy(density).unsqueeze(0).float()
        
        return hazy, density

def train_model():
    print("Initializing CPU Training for TinyHazePerceptionModule...")
    
    device = torch.device('cpu')
    model = TinyHazePerceptionModule().to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # Tiny dataset for CPU training
    train_dataset = SyntheticHazeDataset(size=500)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    
    val_dataset = SyntheticHazeDataset(size=50)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    epochs = 10
    start_time = time.time()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for hazy, target_density in train_loader:
            hazy = hazy.to(device)
            target_density = target_density.to(device)
            
            optimizer.zero_grad()
            output = model(hazy)
            
            loss = criterion(output, target_density)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for hazy, target_density in val_loader:
                hazy = hazy.to(device)
                target_density = target_density.to(device)
                output = model(hazy)
                loss = criterion(output, target_density)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f} seconds on CPU.")
    
    # Save PyTorch weights
    os.makedirs("weights", exist_ok=True)
    torch_path = "weights/perception_model.pth"
    torch.save(model.state_dict(), torch_path)
    print(f"Saved PyTorch model to {torch_path}")
    
    # Export to ONNX for faster CPU inference
    try:
        onnx_path = "weights/perception_model.onnx"
        dummy_input = torch.randn(1, 3, 128, 128)
        torch.onnx.export(model, dummy_input, onnx_path, 
                          export_params=True, 
                          opset_version=11, 
                          do_constant_folding=True, 
                          input_names=['input'], 
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                                        'output': {0: 'batch_size', 2: 'height', 3: 'width'}})
        print(f"Exported ONNX model to {onnx_path}")
    except Exception as e:
        print(f"Failed to export ONNX: {e}")

if __name__ == "__main__":
    train_model()
