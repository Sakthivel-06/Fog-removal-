import time
import imageio
import traceback
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from app.main import process_frame_unified

def main():
    try:
        cap = imageio.get_reader('data/foggy_video1.mp4')
        frame = cap.get_data(0)
        print(f"Loaded frame: {frame.shape}, {frame.dtype}")
        
        out, ms, fps, diag = process_frame_unified(frame, 'Final Proposed (A4)', return_diag=True, output_mode='High-Resolution A4')
        
        with open('scratch/test_error.txt', 'w') as f:
            f.write(f"SUCCESS! Output shape: {out.shape}, type: {out.dtype}\n")
    except Exception as e:
        with open('scratch/test_error.txt', 'w') as f:
            f.write(traceback.format_exc())

if __name__ == '__main__':
    main()
