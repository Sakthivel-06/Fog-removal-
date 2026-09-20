"""
scratch/deep_audit_gt_spatial_shift.py — Deep Alignment Audit of GT Pairs
"""
import imageio
import numpy as np
import scipy.ndimage as ndimage

def check_alignment(hazy_path, clean_path, name):
    print(f"\nChecking Alignment for: {name}")
    rh = imageio.get_reader(hazy_path)
    rc = imageio.get_reader(clean_path)
    
    fh = rh.get_data(0)
    fc = rc.get_data(0)
    rh.close()
    rc.close()
    
    # Extract luminance
    gh = 0.299 * fh[:, :, 0] + 0.587 * fh[:, :, 1] + 0.114 * fh[:, :, 2]
    gc = 0.299 * fc[:, :, 0] + 0.587 * fc[:, :, 1] + 0.114 * fc[:, :, 2]
    
    # Compute Sobel gradient magnitude
    sy_h = ndimage.sobel(gh, axis=0)
    sx_h = ndimage.sobel(gh, axis=1)
    grad_h = np.sqrt(sx_h**2 + sy_h**2)
    
    sy_c = ndimage.sobel(gc, axis=0)
    sx_c = ndimage.sobel(gc, axis=1)
    grad_c = np.sqrt(sx_c**2 + sy_c**2)
    
    # Test shifts in a grid of [-5, 5] x [-5, 5] on gradient correlation
    best_shift = (0, 0)
    best_corr = -1.0
    shifts = {}
    
    for dy in range(-5, 6):
        for dx in range(-5, 6):
            # Shift grad_c
            gc_shifted = np.roll(np.roll(grad_c, dy, axis=0), dx, axis=1)
            # compute correlation
            corr = float(np.corrcoef(grad_h.flatten(), gc_shifted.flatten())[0, 1])
            shifts[(dy, dx)] = corr
            if corr > best_corr:
                best_corr = corr
                best_shift = (dy, dx)
                
    print(f"Shift (0, 0) Edge Correlation:   {shifts[(0, 0)]:.6f}")
    print(f"Best Shift (dy, dx):            {best_shift} with Correlation: {best_corr:.6f}")
    
    # Check if (0, 0) is the strict peak
    if best_shift == (0, 0):
        print("Verdict: TRUE PIXEL-FOR-PIXEL SPATIAL ALIGNMENT (dx=0, dy=0) at edge level!")
    else:
        print(f"Verdict: SPATIAL MISMATCH DETECTED: {best_shift}")

check_alignment("test_videos/car-detection_patchy_haze.mp4", "test_videos/car-detection_clean.mp4", "Car Detection")
check_alignment("test_videos/person-bicycle_uniform_haze.mp4", "test_videos/person-bicycle_clean.mp4", "Person Bicycle")
