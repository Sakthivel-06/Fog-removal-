import numpy as np
import scipy.ndimage as ndimage
from skimage.color import rgb2gray
import time
import os
import sys
import imageio.v3 as iio
from PIL import Image

try:
    from .filters import guided_filter
except ImportError:
    from filters import guided_filter

def get_dark_channel(img, size=15):
    """Calculate the dark channel of an image."""
    # img is assumed to be RGB
    dc = np.min(img, axis=2)
    dark = ndimage.minimum_filter(dc, size=size)
    return dark

def get_atmosphere(img, dark, top_percent=0.001):
    """Estimate the atmospheric light."""
    h, w = img.shape[:2]
    num_pixels = h * w
    num_top = int(max(num_pixels * top_percent, 1))
    
    # Flatten and find indices of top brightest pixels in dark channel
    dark_flat = dark.ravel()
    img_flat = img.reshape(num_pixels, 3)
    
    indices = np.argpartition(dark_flat, -num_top)[-num_top:]
    
    # Average the color of these pixels in the original image
    atm_light = np.mean(img_flat[indices], axis=0)
    return atm_light

def get_transmission(img, atm_light, omega=0.95, size=15):
    """Estimate the transmission map."""
    norm_img = img / atm_light
    dark_norm = get_dark_channel(norm_img, size)
    transmission = 1 - omega * dark_norm
    return transmission

def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Standalone Dark Channel Prior (DCP) baseline.
    With Guided Filter refinement.
    """
    # Convert to float32 for calculations [0, 1]
    img = frame.astype('float32') / 255.0
    
    # 1. Dark Channel
    dark = get_dark_channel(img, size=15)
    
    # 2. Atmospheric Light
    atm_light = get_atmosphere(img, dark)
    
    # 3. Transmission Map
    trans = get_transmission(img, atm_light, omega=0.95, size=15)
    
    # 4. Refine Transmission with Guided Filter
    # Convert image to grayscale for guidance
    gray = rgb2gray(img)
    
    # Radius = 40, eps = 0.001 are standard DCP parameters
    trans_refined = guided_filter(guide=gray, src=trans, radius=40, eps=1e-3)
        
    # 5. Recover the Scene
    t0 = 0.1
    trans_refined = np.maximum(trans_refined, t0)
    
    # Initialize output array
    J = np.empty_like(img)
    for i in range(3):
        J[:, :, i] = (img[:, :, i] - atm_light[i]) / trans_refined + atm_light[i]
        
    # Clip and convert back to uint8
    J = np.clip(J, 0, 1)
    return (J * 255).astype('uint8')

if __name__ == "__main__":
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
            dehazed = process_frame(frame)
            end = time.perf_counter()
            fps = 1.0 / (end - start + 1e-5)
            print(f"DCP FPS: {fps:.1f}")
