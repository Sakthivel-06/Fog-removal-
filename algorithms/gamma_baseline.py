import numpy as np
import time
import os
import sys
import imageio.v3 as iio
from PIL import Image

def process_frame(frame: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """
    Standalone Gamma Correction baseline.
    Enhances contrast by applying a power-law transform.
    """
    # Corrected gamma formulation (Session 16): output = input ** gamma (darkens/cuts haze, NOT inverse-gamma brightening)
    table = np.array([((i / 255.0) ** gamma) * 255.0
                      for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction using numpy advanced indexing (equivalent to cv2.LUT)
    return table[frame]

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
            dehazed = process_frame(frame, gamma=2.0)
            end = time.perf_counter()
            fps = 1.0 / (end - start + 1e-5)
            print(f"Gamma FPS: {fps:.1f}")
