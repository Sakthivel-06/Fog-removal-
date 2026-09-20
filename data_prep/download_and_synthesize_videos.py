import urllib.request
import os
import sys
import imageio.v3 as iio
import imageio
import numpy as np
from PIL import Image

# Ensure synthetic_haze is importable
sys.path.insert(0, os.path.dirname(__file__))
from synthetic_haze import add_synthetic_fog

def download_video(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print("Download complete.")
        return True
    except Exception as e:
        print(f"Failed to download: {e}")
        return False

def process_video(input_path, output_path, pattern):
    print(f"Applying {pattern} haze to {input_path}...")
    
    # Read video metadata
    props = iio.improps(input_path, plugin="pyav")
    fps = props.fps if hasattr(props, 'fps') else 25.0
    
    # We will process only the first 100 frames
    max_frames = 100
    
    frames_in = list(iio.imiter(input_path, plugin="pyav"))
    frames_out = []
    
    for i, frame in enumerate(frames_in):
        if i >= max_frames:
            break
        hazy_frame, _ = add_synthetic_fog(frame, beta=1.5, pattern=pattern)
        frames_out.append(hazy_frame)
        if (i + 1) % 20 == 0:
            print(f"Processed {i+1} frames...")
    
    # Write output video using imageio
    iio.imwrite(output_path, frames_out, fps=fps, plugin="pyav", codec="libx264")
    print(f"Saved {output_path}")

def main():
    videos = {
        "car-detection": "https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4",
        "person-bicycle": "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
    }
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "test_videos")
    os.makedirs(out_dir, exist_ok=True)
    
    for name, url in videos.items():
        clean_path = os.path.join(out_dir, f"{name}_clean.mp4")
        if not os.path.exists(clean_path):
            success = download_video(url, clean_path)
            if not success:
                continue
                
        # Generate Hazy versions
        process_video(clean_path, os.path.join(out_dir, f"{name}_uniform_haze.mp4"), 'uniform')
        process_video(clean_path, os.path.join(out_dir, f"{name}_patchy_haze.mp4"), 'patchy')

if __name__ == "__main__":
    main()
