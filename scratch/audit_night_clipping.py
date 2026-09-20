"""
scratch/audit_night_clipping.py — Nighttime Black Clipping & Crush Audit
"""
import os
import sys
import imageio
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "novel_method")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "algorithms")))

import dcp
import cap
import clahe_baseline
import gamma_baseline
import ceef
import ceef_original
from pipeline import RealTimeDehazingPipeline, _resize_np
from nighttime_experimental import ExperimentalNighttimePipeline

NIGHT_VIDEOS = ["foggy_video3.mp4", "foggy_video4.mp4"]

METHODS = [
    ("DCP", lambda img, p: dcp.process_frame(img)),
    ("CAP", lambda img, p: cap.process_frame(img)),
    ("CLAHE", lambda img, p: clahe_baseline.process_frame(img)),
    ("Corrected Gamma", lambda img, p: gamma_baseline.process_frame(img, gamma=2.0)),
    ("Original CEEF", lambda img, p: ceef_original.process_frame(img)),
    ("Modified CEEF", lambda img, p: ceef.process_frame(img)),
    ("Proposed (Night Prototype)", lambda img, p: p.process_frame(img)[0]),
]

def audit_night():
    for vname in NIGHT_VIDEOS:
        vpath = os.path.join("test_videos", vname)
        reader = imageio.get_reader(vpath)
        frames = [reader.get_data(i) for i in range(15)]
        reader.close()
        
        # Calculate native input black
        in_blacks = []
        for f in frames[5:]:
            f_res = _resize_np(f, (426, 239))
            lum = 0.299 * f_res[:, :, 0] + 0.587 * f_res[:, :, 1] + 0.114 * f_res[:, :, 2]
            in_blacks.append(np.count_nonzero(lum < 1.0) / lum.size * 100.0)
        native_in_black = float(np.mean(in_blacks))
        
        print(f"\n=======================================================")
        print(f"NIGHT CLIPPING AUDIT: {vname}")
        print(f"Native Input Black (< 1/255): {native_in_black:.2f}%")
        print(f"=======================================================")
        
        for m_name, m_func in METHODS:
            pipe = ExperimentalNighttimePipeline(enable_nighttime_branch=True, night_mode="prototype")
            for w in range(5):
                m_func(_resize_np(frames[w], (426, 239)), pipe)
                
            out_blacks = []
            for i in range(5, 15):
                f_res = _resize_np(frames[i], (426, 239))
                out = m_func(f_res, pipe)
                if out.shape[:2] != (239, 426):
                    out = _resize_np(out, (426, 239))
                lum = 0.299 * out[:, :, 0] + 0.587 * out[:, :, 1] + 0.114 * out[:, :, 2]
                out_blacks.append(np.count_nonzero(lum < 1.0) / lum.size * 100.0)
            
            mean_out_black = float(np.mean(out_blacks))
            crush = mean_out_black - native_in_black
            print(f"{m_name:28s} | Out Black: {mean_out_black:5.2f}% | Model Crush: {crush:+5.2f}%")

if __name__ == "__main__":
    audit_night()
