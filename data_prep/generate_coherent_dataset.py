"""
data_prep/generate_coherent_dataset.py
======================================
Pre-generates and caches the coherent training and validation datasets
in data/training_coherent_haze/ according to config/session22_training_config.yaml.

Zero OpenCV dependency.
"""

import os
import sys
import yaml
import json
import numpy as np
from PIL import Image
import imageio.v3 as iio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_prep.coherent_haze_generator import CoherentHazeGenerator


def generate_and_cache_dataset(config_path="config/session22_training_config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    out_base = "data/training_coherent_haze"
    os.makedirs(os.path.join(out_base, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_base, "val"), exist_ok=True)

    synth_w, synth_h = cfg["resolution"]["synthesis_resolution"]

    def process_sources(sources_list, split_name):
        print(f"\n--- Generating {split_name.upper()} Split ---")
        split_dir = os.path.join(out_base, split_name)
        meta_list = []
        global_idx = 0

        for s_idx, src in enumerate(sources_list):
            vpath = src["video"]
            max_f = src["max_frames"]
            print(f"Loading source video: {vpath} (max {max_f} frames)...")
            reader = iio.imread(vpath)
            n_frames = min(len(reader), max_f)

            gen = CoherentHazeGenerator(
                height=synth_h,
                width=synth_w,
                beta_base=src["beta_base"],
                beta_var=src["beta_var"],
                wind_vx=src["wind_vx"],
                wind_vy=src["wind_vy"],
                seed=src["seed"],
            )

            cached_d = None
            for k in range(n_frames):
                clean_f = reader[k]
                hazy_u8, t_gt, A, beta, depth = gen.synthesize_frame(
                    clean_f, k, cached_depth=cached_d
                )
                if cached_d is None:
                    cached_d = depth

                # Save frame files
                clean_resized = np.array(Image.fromarray(clean_f).resize((synth_w, synth_h), Image.BILINEAR))
                
                c_fname = f"clean_{global_idx:05d}.jpg"
                h_fname = f"hazy_{global_idx:05d}.jpg"
                t_fname = f"tgt_{global_idx:05d}.npy"

                Image.fromarray(clean_resized).save(os.path.join(split_dir, c_fname), quality=95)
                Image.fromarray(hazy_u8).save(os.path.join(split_dir, h_fname), quality=95)
                np.save(os.path.join(split_dir, t_fname), t_gt.astype(np.float32))

                meta_list.append({
                    "id": global_idx,
                    "clean_file": c_fname,
                    "hazy_file": h_fname,
                    "tgt_file": t_fname,
                    "A": A.tolist(),
                    "source_video": os.path.basename(vpath),
                    "frame_in_video": k,
                    "beta_mean": float(np.mean(beta)),
                    "tgt_mean": float(np.mean(t_gt)),
                    "tgt_min": float(np.min(t_gt)),
                    "tgt_max": float(np.max(t_gt)),
                })
                global_idx += 1

        manifest_path = os.path.join(split_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(meta_list, f, indent=2)
        print(f"Saved {global_idx} frames to {split_dir} with manifest.")
        return meta_list

    train_meta = process_sources(cfg["dataset"]["train_sources"], "train")
    val_meta = process_sources(cfg["dataset"]["val_sources"], "val")
    print(f"\nTotal Dataset: Train={len(train_meta)} frames, Val={len(val_meta)} frames.")


if __name__ == "__main__":
    generate_and_cache_dataset()
