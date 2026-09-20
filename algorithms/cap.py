import numpy as np
from skimage.color import rgb2hsv
import time
import os
import sys
import imageio.v3 as iio
from PIL import Image

try:
    from .filters import guided_filter
except ImportError:
    from filters import guided_filter

def get_depth_map(img_hsv):
    """
    Estimate depth map using Color Attenuation Prior (CAP).
    d(x) = theta_0 + theta_1 * v(x) + theta_2 * s(x)
    """
    # Parameters from Zhu et al. (2015)
    theta_0 = 0.121779
    theta_1 = 0.959710
    theta_2 = -0.780245
    
    h = img_hsv[:,:,0]
    s = img_hsv[:,:,1]
    v = img_hsv[:,:,2]
    
    # Calculate depth
    depth = theta_0 + theta_1 * v + theta_2 * s
    
    # Refine depth using guided filter
    depth_refined = guided_filter(guide=v, src=depth, radius=15, eps=1e-3)
        
    return depth_refined

def get_atmosphere_cap(img, depth, top_percent=0.001):
    """Estimate atmospheric light using the depth map."""
    h, w = img.shape[:2]
    num_pixels = h * w
    num_top = int(max(num_pixels * top_percent, 1))
    
    depth_flat = depth.ravel()
    img_flat = img.reshape(num_pixels, 3)
    
    # Find indices of top deepest pixels (farthest away)
    indices = np.argpartition(depth_flat, -num_top)[-num_top:]
    
    # Average color of these pixels
    atm_light = np.mean(img_flat[indices], axis=0)
    return atm_light

def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Standalone Color Attenuation Prior (CAP) baseline.
    Assumes frame is RGB.
    """
    # Convert to float32 [0, 1]
    img = frame.astype('float32') / 255.0
    
    # skimage rgb2hsv returns H, S, V in [0, 1]
    hsv = rgb2hsv(img)
    
    # 1. Depth Map
    depth = get_depth_map(hsv)
    
    # 2. Atmospheric Light
    atm_light = get_atmosphere_cap(img, depth)
    
    # 3. Transmission Map (beta = 1.0 is a common assumption)
    beta = 1.0 
    trans = np.exp(-beta * depth)
    
    # 4. Recover Scene
    t0 = 0.1
    trans = np.maximum(trans, t0)
    
    J = np.empty_like(img)
    for i in range(3):
        J[:, :, i] = (img[:, :, i] - atm_light[i]) / trans + atm_light[i]
        
    J = np.clip(J, 0, 1)
    return (J * 255).astype('uint8')

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_input = sys.argv[1]
        if not os.path.exists(video_input):
            test_path = os.path.join(os.path.dirname(__file__), "..", "test_videos", video_input)
            if os.path.exists(test_path):
                video_input = test_path
        
        # Using imageio to read video
        reader = iio.imiter(video_input, plugin="pyav")
        for frame in reader:
            # Resize if too large
            h, w = frame.shape[:2]
            if h > 480:
                scale = 480 / float(h)
                new_w, new_h = int(w * scale), 480
                pil_img = Image.fromarray(frame).resize((new_w, new_h))
                frame = np.array(pil_img)
            
            start = time.perf_counter()
            dehazed = process_frame(frame)
            end = time.perf_counter()
            fps = 1.0 / (end - start)
            print(f"CAP FPS: {fps:.1f}")
            
            # Since no cv2, we cannot easily imshow interactively here in a simple loop without blocking matplotlib
            # So we will just print FPS for the standalone test. Streamlit handles the actual UI.

