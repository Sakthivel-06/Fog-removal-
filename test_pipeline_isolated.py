import time
import imageio
import traceback
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from novel_method.pipeline_highres import HighResLearnedResidualDehazingPipeline

def main():
    try:
        print("Instantiating pipeline...")
        p = HighResLearnedResidualDehazingPipeline(upsample_mode="guided")
        
        print("Loading frame...")
        cap = imageio.get_reader('data/foggy_video1.mp4')
        frame = cap.get_data(0)
        print(f"Loaded frame: {frame.shape}, {frame.dtype}")
        
        print("Processing frame...")
        out, fps, diag = p.process_frame(frame, return_diagnostics=True)
        
        with open('scratch/test_error.txt', 'w') as f:
            f.write(f"SUCCESS! Output shape: {out.shape}, type: {out.dtype}\n")
    except Exception as e:
        with open('scratch/test_error.txt', 'w') as f:
            f.write(traceback.format_exc())

if __name__ == '__main__':
    main()
