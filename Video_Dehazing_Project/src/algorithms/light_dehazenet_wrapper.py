import sys
import os
import numpy as np
import time
import torch
import imageio.v3 as iio
from PIL import Image

# Add Light-DehazeNet to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Light-DehazeNet")))
try:
    from lightdehazeNet import LightDehaze_Net
except ImportError as e:
    print(f"Warning: Light-DehazeNet not found or missing dependencies: {e}")
    LightDehaze_Net = None

class LightDehazeNetWrapper:
    def __init__(self):
        if LightDehaze_Net is None:
            raise RuntimeError("Light-DehazeNet could not be imported. Ensure PyTorch is installed.")
            
        self.device = torch.device("cpu")
        self.model = LightDehaze_Net().to(self.device)
        
        # Load weights on CPU
        weight_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "Light-DehazeNet", "trained_weights", "trained_LDNet.pth"))
            
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weights not found at {weight_path}")
            
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.eval()

    def compute_k(self, img_float: np.ndarray) -> np.ndarray:
        """
        Computes the transmission/airlight parameter map K(x) from float32 RGB [0, 1].
        LightDehazeNet formulation: J(x) = K(x) * I(x) - K(x) + 1.
        """
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(self.device)
        m = self.model
        with torch.no_grad():
            c1 = m.relu(m.e_conv_layer1(img_tensor))
            c2 = m.relu(m.e_conv_layer2(c1))
            c3 = m.relu(m.e_conv_layer3(c2))
            cat1 = torch.cat((c1, c3), 1)
            c4 = m.relu(m.e_conv_layer4(cat1))
            c5 = m.relu(m.e_conv_layer5(c4))
            c6 = m.relu(m.e_conv_layer6(c5))
            cat2 = torch.cat((c4, c6), 1)
            c7 = m.relu(m.e_conv_layer7(cat2))
            cat3 = torch.cat((c2, c5, c7), 1)
            k_tensor = m.relu(m.e_conv_layer8(cat3))
        return k_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    def process_frame(self, frame: np.ndarray):
        start = time.perf_counter()
        # Preprocessing: Assumes frame is already RGB [0, 255] uint8
        img = frame.astype(np.float32) / 255.0
        
        # To Tensor: (1, 3, H, W)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            clean_image = self.model(img_tensor)
            
        # Postprocessing
        out = clean_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = np.clip(out * 255, 0, 255).astype(np.uint8)
        
        end = time.perf_counter()
        fps = 1.0 / (end - start + 1e-9)
        # We output RGB directly since the rest of the project is now RGB
        return out, fps


if __name__ == "__main__":
    wrapper = LightDehazeNetWrapper()
    if len(sys.argv) > 1:
        video_input = sys.argv[1]
        if not os.path.exists(video_input):
            test_path = os.path.join(os.path.dirname(__file__), "..", "test_videos", video_input)
            if os.path.exists(test_path):
                video_input = test_path
                
        reader = iio.imiter(video_input, plugin="pyav")
        for frame in reader:
            # Resize to fit screen (max height 480) while maintaining aspect ratio
            h, w = frame.shape[:2]
            if h > 480:
                scale = 480 / float(h)
                new_w, new_h = int(w * scale), 480
                pil_img = Image.fromarray(frame).resize((new_w, new_h))
                frame = np.array(pil_img)
            
            start = time.perf_counter()
            dehazed = wrapper.process_frame(frame)
            end = time.perf_counter()
            fps = 1.0 / (end - start + 1e-5)
            print(f"LightDehazeNet FPS: {fps:.1f}")
