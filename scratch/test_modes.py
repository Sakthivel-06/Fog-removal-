import sys
import imageio
sys.path.insert(0, 'novel_method')
from nighttime_experimental import ExperimentalNighttimePipeline

r = imageio.get_reader('test_videos/foggy_video3.mp4')
f = r.get_data(0)
r.close()

for mode in ['prototype', 'multi_source', 'multi_source_smooth', 'frozen_daytime']:
    p = ExperimentalNighttimePipeline(night_mode=mode)
    out, fps, diag = p.process_frame(f, return_diagnostics=True)
    n_cl = diag.get("num_clusters", 0)
    print(f"Mode: {mode:<22} | Out: {out.shape} | FPS: {fps:5.1f} | Clusters: {n_cl}")
