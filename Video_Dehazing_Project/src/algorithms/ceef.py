import numpy as np
from skimage.exposure import equalize_adapthist
from skimage.color import rgb2lab, lab2rgb
import sys
import os
import time
import imageio.v3 as iio
from PIL import Image

try:
    from .filters import exposure_fusion
except ImportError:
    from filters import exposure_fusion

def process_frame(frame: np.ndarray) -> np.ndarray:
    """
    Modified CEEF (Contrast Enhancement and Exposure Fusion).
    Follows Kaplan (2023) and base CEEF formulation:
    1. Underexposure sequences via power-law: I^g for g in [1.5, 2.2, 3.0] to cut haze.
    2. CLAHE applied to L-channel of LAB color space to recover local contrast.
    3. Multi-scale Mertens Fusion.
    """
    img_float = frame.astype('float32') / 255.0
    
    gammas = [1.5, 2.2, 3.0]
    exposures = []
    
    for g in gammas:
        # Paper Eq (5): I^g creates underexposed versions that eliminate white haze
        gc = np.power(img_float, g)
        exposures.append(gc)
    
    # Apply CLAHE to L channel of original image
    lab_orig = rgb2lab(img_float)
    l_orig = lab_orig[:, :, 0] / 100.0
    cl_orig = equalize_adapthist(l_orig, clip_limit=0.015)
    lab_orig[:, :, 0] = cl_orig * 100.0
    orig_clahe = lab2rgb(lab_orig)
    exposures.append(orig_clahe)
    
    # Include original as base
    exposures.append(img_float)
    
    # Mertens Fusion
    res = exposure_fusion(exposures)
    
    # Convert from float32 [0,1] to uint8 [0,255]
    res = np.clip(res * 255, 0, 255).astype('uint8')
    return res

if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_input = sys.argv[1]
        if not os.path.exists(video_input):
            test_path = os.path.join(os.path.dirname(__file__), "..", "test_videos", video_input)
            if os.path.exists(test_path):
                video_input = test_path
                
        reader = iio.imiter(video_input, plugin="pyav")
        for frame in reader:
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
            print(f"CEEF FPS: {fps:.1f}")
