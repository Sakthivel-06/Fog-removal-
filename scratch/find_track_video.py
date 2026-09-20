import os
import glob
from PIL import Image
import imageio.v3 as iio

# Find the video corresponding to the sports ground track
found = []
for p in sorted(glob.glob("test_videos/*.mp4")):
    try:
        r = iio.imiter(p)
        f = next(r)
        h, w = f.shape[:2]
        # Check red track at the bottom
        bottom = f[int(h * 0.85):, :]
        r_mean = float(bottom[:, :, 0].mean())
        g_mean = float(bottom[:, :, 1].mean())
        b_mean = float(bottom[:, :, 2].mean())
        if r_mean > g_mean + 15 and r_mean > b_mean + 15:
            found.append((p, r_mean, g_mean, b_mean))
    except Exception:
        pass

print("Found matching videos:", found)
