import numpy as np
import time
from skimage.exposure import equalize_adapthist
from skimage.color import rgb2lab, lab2rgb
import sys
import os
import imageio.v3 as iio
from PIL import Image

def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Standalone CLAHE baseline.
    Applies Contrast Limited Adaptive Histogram Equalization to the L channel of LAB color space.
    Assumes frame is RGB uint8.
    """
    img_float = frame.astype('float32') / 255.0
    # Convert to LAB color space
    lab = rgb2lab(img_float)
    
    # Apply CLAHE to L-channel
    l_chan = lab[:, :, 0] / 100.0
    cl = equalize_adapthist(l_chan, clip_limit=0.02)
    lab[:, :, 0] = cl * 100.0
    
    # Convert back to RGB color space
    final = lab2rgb(lab)
    
    return np.clip(final * 255, 0, 255).astype('uint8')

if __name__ == "__main__":
    # Simple test if run directly
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
            print(f"CLAHE FPS: {fps:.1f}")
