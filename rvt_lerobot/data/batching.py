"""Cached sensor arrays -> the tensor each arm actually consumes.

Everything here runs on the GPU at batch time rather than at cache time. The
reason is not speed, it is hygiene: the point cloud, the world-XYZ channels and
the virtual views must all be built from *the calibration the policy was told*,
so that when the miscalibration axis makes that calibration wrong, it is wrong
everywhere downstream, exactly as it would be on a real robot. Baking them into
the cache would quietly use the true extrinsics in one place and the reported
ones in another.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from ..models.policy import ArmSpec
from ..render.rig import FAR, WORKSPACE_CENTRE, WORKSPACE_EXTENT, unproject_torch
from ..render.virtual import render_views

#: Depth beyond this counts as "no return": the renderer's far plane and the
#: value the noise model writes into sensor holes.
NO_RETURN = FAR - 1e-3


class ConditionCache:
    """One rendered condition held in RAM, served as batches."""

    #: Gathered as numpy, not torch: images are held in their compact dtypes to
    #: fit (8k samples of 4-view RGBD at 96px is 1.5 GB as uint8/uint16 and 6 GB
    #: as float32, on a machine with 15 GB shared between CPU and GPU), and
    #: torch cannot fancy-index a uint16 CPU tensor at all.
    KEYS = ("rgb", "depth_mm", "K", "T", "target_pos", "target_rot6", "target_grip",
            "proprio", "event")

    def __init__(self, arrays: dict[str, np.ndarray], device: str = "cuda") -> None:
        self.device = device
        self.n = len(arrays["rgb"])
        self.arrays = {k: arrays[k] for k in self.KEYS}

    def batch(self, idx) -> dict[str, Tensor]:
        i = idx.numpy() if isinstance(idx, Tensor) else np.asarray(idx)
        out = {}
        for k, a in self.arrays.items():
            v = a[i]
            if v.dtype == np.uint16:
                v = v.astype(np.int32)   # uint16 has no torch dtype
            out[k] = torch.from_numpy(np.ascontiguousarray(v)).to(self.device, non_blocking=True)
        return out


def make_images(spec: ArmSpec, batch: dict[str, Tensor], virtual_size: int = 96) -> Tensor | None:
    """Build the (B, V, C, H, W) input for one arm from a raw sensor batch."""
    if spec.source == "none":
        return None

    rgb = batch["rgb"].permute(0, 1, 4, 2, 3).float() / 255.0      # (B,V,3,H,W)
    depth = batch["depth_mm"].float() / 1000.0                     # (B,V,H,W)
    valid = (depth > 1e-3) & (depth < NO_RETURN)

    if spec.source == "real":
        if spec.channels == "rgb":
            return rgb
        if spec.channels == "rgbd":
            # Normalise depth to roughly [0, 1] over the working range and write
            # zero where there was no return, so "far away" and "no reading" are
            # not the same number.
            dn = ((depth - 0.3) / 0.9).clamp(0, 1) * valid
            return torch.cat([rgb, dn.unsqueeze(2)], dim=2)
        if spec.channels == "rgbxyz":
            pts = unproject_torch(depth, batch["K"], batch["T"])    # (B,V,H,W,3)
            centre = torch.tensor(WORKSPACE_CENTRE, device=pts.device, dtype=pts.dtype)
            nxyz = (pts - centre) / (WORKSPACE_EXTENT / 2)
            nxyz = nxyz.clamp(-2, 2) * valid.unsqueeze(-1)
            return torch.cat([rgb, nxyz.permute(0, 1, 4, 2, 3)], dim=2)
        raise ValueError(spec.channels)

    # virtual: fuse every real view into one world-frame cloud, then re-render
    b, v, _, h, w = rgb.shape
    pts = unproject_torch(depth, batch["K"], batch["T"]).reshape(b, v * h * w, 3)
    centre = torch.tensor(WORKSPACE_CENTRE, device=pts.device, dtype=pts.dtype)
    col = rgb.permute(0, 1, 3, 4, 2).reshape(b, v * h * w, 3)
    nxyz = ((pts - centre) / (WORKSPACE_EXTENT / 2)).clamp(-2, 2)
    feats = torch.cat([col, nxyz], dim=-1)
    return render_views(
        pts,
        feats,
        valid.reshape(b, v * h * w),
        centre=centre,
        extent=WORKSPACE_EXTENT,
        img_size=virtual_size,
    )
