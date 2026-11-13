"""
novel_method/learned_residual
=============================
Session 21: Experimental Learned Transmission Residual Module.
Predicts bounded Delta-t in [-0.30, +0.30] to augment the physics-based transmission estimate.
Zero OpenCV (cv2) dependency.
"""

from .tiny_unet import TinyUNet
from .losses import CompositeResidualLoss
from .dataset import CoherentHazeDataset

__all__ = ["TinyUNet", "CompositeResidualLoss", "CoherentHazeDataset"]
