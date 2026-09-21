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

#: Mean camera-to-workspace distance for the nominal rig, in metres. Used only
#: to centre CAMERA-frame coordinates; world-frame arms centre on the workspace
#: itself and need no such constant.
NOMINAL_CAMERA_DISTANCE = 0.67

#: Points per cloud for the DP3-style arms. The PR defaults to 1024; the paper
#: reports 512-1024 and notes the encoder max-pools so more buys little.
POINTCLOUD_POINTS = 1024


class ConditionCache:
    """One rendered condition held in RAM, served as batches."""

    #: Gathered as numpy, not torch: images are held in their compact dtypes to
    #: fit (8k samples of 4-view RGBD at 96px is 1.5 GB as uint8/uint16 and 6 GB
    #: as float32, on a machine with 15 GB shared between CPU and GPU), and
    #: torch cannot fancy-index a uint16 CPU tensor at all.
    KEYS = ("rgb", "depth_mm", "K", "T", "target_pos", "target_rot6", "target_grip",
            "proprio", "proprio_full", "event")

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

    if spec.source == "pointcloud":
        # Built with the PULL REQUEST's own unproject and sampler, so this arm
        # measures that code rather than a local reimplementation of it. The
        # camera-frame variant never touches an extrinsic, which is the whole
        # claim the PR's `frame` default rests on.
        from ..vendor.lerobot_pointcloud import sample_points, unproject

        b, v, h, w = depth.shape
        if spec.world_frame_cloud:
            # Several cameras may be fused, because a world frame is what makes
            # their clouds commensurable.
            pts = unproject(depth, batch["K"], batch["T"]).reshape(b, v * h * w, 3)
            pts = pts - pts.new_tensor(WORKSPACE_CENTRE)
            mask = valid.reshape(b, v * h * w)
        else:
            # ONE camera. Clouds from different cameras sit in different frames
            # and concatenating them without extrinsics is meaningless -- the
            # PR's own processor raises rather than do it, and this has to
            # match or the arm would not be benchmarking the documented API.
            # One RGBD camera is also the setup DP3 is designed around.
            d0, k0 = depth[:, :1], batch["K"][:, :1]
            pts = unproject(d0, k0).reshape(b, h * w, 3)
            pts = pts - pts.new_tensor([0.0, 0.0, NOMINAL_CAMERA_DISTANCE])
            mask = valid[:, :1].reshape(b, h * w)
        # Crop to the workspace BEFORE subsampling. Without it a uniform
        # subsample of a whole-scene cloud is mostly floor and far tabletop:
        # measured on this scene, the block covers 28 of 8256 valid pixels, so
        # 1024 uniform samples contain about 3.5 points of the object the
        # policy has to localise -- 0.34% of its representation. DP3 crops for
        # exactly this reason, and the PR exposes the same option as
        # `workspace_centre` / `workspace_extent`.
        half = WORKSPACE_EXTENT / 2
        inside = (pts.abs() <= half).all(dim=-1)
        cloud = sample_points(pts, mask & inside, POINTCLOUD_POINTS)
        return (cloud / half).clamp(-1.0, 1.0)

    if spec.source == "real":
        if spec.channels == "rgb":
            return rgb
        if spec.channels == "rgbd":
            # Normalise depth to roughly [0, 1] over the working range and write
            # zero where there was no return, so "far away" and "no reading" are
            # not the same number.
            dn = ((depth - 0.3) / 0.9).clamp(0, 1) * valid
            return torch.cat([rgb, dn.unsqueeze(2)], dim=2)
        if spec.channels == "rgbxyz_cam":
            # Points in each camera's OWN frame: the pinhole inverse and nothing
            # else. No extrinsic is ever applied, so no calibration error can
            # enter. The network has to learn the camera-to-robot relation from
            # the data, which is precisely the trade DP3-style policies make.
            k = batch["K"]
            b, v, h, w = depth.shape
            rows, cols = torch.meshgrid(
                torch.arange(h, device=depth.device, dtype=depth.dtype),
                torch.arange(w, device=depth.device, dtype=depth.dtype),
                indexing="ij",
            )
            x = (cols - k[..., 0, 2][..., None, None]) / k[..., 0, 0][..., None, None] * depth
            y = (rows - k[..., 1, 2][..., None, None]) / k[..., 1, 1][..., None, None] * depth
            cam = torch.stack([x, y, depth], dim=-1)
            # Camera-frame z is an ABSOLUTE distance (0.4-1.2 m here), not an
            # offset from the workspace centre the way the world-frame arms'
            # coordinates are. Dividing it by the same half-extent sent every
            # depth beyond 0.6 m into the clamp -- measured mean 1.38 against a
            # clamp at 2.0, i.e. most of the scene's depth information was being
            # destroyed before the network saw it. Subtract the nominal
            # camera-to-workspace distance first so the workspace sits near zero.
            origin = cam.new_tensor([0.0, 0.0, NOMINAL_CAMERA_DISTANCE])
            cam = ((cam - origin) / (WORKSPACE_EXTENT / 2)).clamp(-2, 2)
            cam = cam * valid.unsqueeze(-1)
            return torch.cat([rgb, cam.permute(0, 1, 4, 2, 3)], dim=2)
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
