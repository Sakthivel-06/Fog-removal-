"""
dataset.py
==========
Session 21: PyTorch Dataset for Temporally Coherent Training Sequences.
Pairs frame k and frame k+1 with precomputed 8-channel physics features,
known ground-truth transmission t_gt, clean RGB, and motion confidence mask.

Zero OpenCV dependency.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from novel_method.temporal_coherence import compute_motion_mask
from novel_method.learned_residual.feature_extractor import extract_physics_features


class CoherentHazeDataset(Dataset):
    def __init__(
        self,
        samples_list: list,
        pipeline_inst,
        target_guide_res: tuple = (240, 135),
    ):
        """
        samples_list: list of dicts containing:
          - 'clean_uint8': (H, W, 3)
          - 'hazy_uint8': (H, W, 3)
          - 't_gt': (H, W) in [0, 1]
          - 'A': (3,) in [0, 1]
        """
        self.samples = samples_list
        self.pipeline = pipeline_inst
        self.target_guide_res = target_guide_res # (W, H)
        self.gw, self.gh = target_guide_res

    def __len__(self) -> int:
        return max(0, len(self.samples) - 1)

    def _prepare_frame(self, sample: dict):
        clean_u8 = sample["clean_uint8"]
        hazy_u8 = sample["hazy_uint8"]
        t_gt = sample["t_gt"]
        A = sample["A"].astype(np.float32)

        # Resize to guide resolution if needed
        h, w = hazy_u8.shape[:2]
        if (w, h) != (self.gw * 2, self.gh * 2):
            # pipeline expects proc_float to be 2x guide res
            clean_u8 = np.array(Image.fromarray(clean_u8).resize((self.gw * 2, self.gh * 2), Image.BILINEAR))
            hazy_u8 = np.array(Image.fromarray(hazy_u8).resize((self.gw * 2, self.gh * 2), Image.BILINEAR))

        proc_float = hazy_u8.astype(np.float32) / 255.0
        clean_float = clean_u8.astype(np.float32) / 255.0

        # Extract 8-channel features at guide resolution (gh, gw)
        features, meta = extract_physics_features(proc_float, self.pipeline)

        # Resize t_gt to guide resolution
        if t_gt.shape != (self.gh, self.gw):
            t_gt_guide = np.array(Image.fromarray(t_gt).resize((self.gw, self.gh), Image.BILINEAR))
        else:
            t_gt_guide = t_gt

        # Resize clean and hazy to guide resolution for reconstruction loss
        clean_guide = np.array(Image.fromarray((clean_float * 255.0).astype(np.uint8)).resize((self.gw, self.gh), Image.BILINEAR)).astype(np.float32) / 255.0
        hazy_guide = np.array(Image.fromarray(hazy_u8).resize((self.gw, self.gh), Image.BILINEAR)).astype(np.float32) / 255.0

        return {
            "features": features, # (8, gh, gw)
            "t_gt": t_gt_guide[None, :, :].astype(np.float32), # (1, gh, gw)
            "clean": np.transpose(clean_guide, (2, 0, 1)).astype(np.float32), # (3, gh, gw)
            "hazy": np.transpose(hazy_guide, (2, 0, 1)).astype(np.float32), # (3, gh, gw)
            "gray_h": meta["gray_h"], # (gh, gw)
            "A": A, # (3,)
        }

    def __getitem__(self, idx: int) -> dict:
        curr = self._prepare_frame(self.samples[idx])
        nxt = self._prepare_frame(self.samples[idx + 1])

        # Compute motion mask between frame k and frame k+1
        m_mask, _, _ = compute_motion_mask(
            nxt["gray_h"],
            curr["gray_h"],
            motion_thresh_low=0.03,
            motion_thresh_high=0.15,
            filter_size=3,
        )

        return {
            "x_k": torch.from_numpy(curr["features"]),
            "x_next": torch.from_numpy(nxt["features"]),
            "t_gt_k": torch.from_numpy(curr["t_gt"]),
            "t_gt_next": torch.from_numpy(nxt["t_gt"]),
            "clean_k": torch.from_numpy(curr["clean"]),
            "hazy_k": torch.from_numpy(curr["hazy"]),
            "A_k": torch.from_numpy(curr["A"]),
            "motion_mask_k": torch.from_numpy(m_mask[None, :, :].astype(np.float32)),
        }
