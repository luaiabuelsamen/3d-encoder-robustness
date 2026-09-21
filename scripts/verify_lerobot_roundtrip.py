"""Does a point cloud survive LeRobot's depth storage?

The study renders float depth straight into a tensor. LeRobot does not: it
quantises depth to a 12-bit video stream, encodes it, and dequantises on read.
Everything in huggingface/lerobot#4696 is tested against float depth, so the
one thing those tests cannot catch is the storage format silently destroying
the signal -- and a point cloud is far less forgiving of depth error than an
image is, because a millimetre of depth is a millimetre of geometry.

This reads back the dataset written by `export_lerobot_dataset.py`, compares
the stored depth against the simulator's own float depth for the same frames,
and then unprojects both through the pull request's own processor step. The
question is not whether the images look alike; it is how far the reconstructed
points move.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.processor.depth_processor import DepthToPointCloudStep  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=pathlib.Path, default=pathlib.Path("data/lerobot_rgbd"))
    p.add_argument("--repo-id", default="local/so101-pick-rgbd")
    p.add_argument("--camera", default="front")
    p.add_argument("--frames", type=int, default=40)
    a = p.parse_args()

    dataset = LeRobotDataset(a.repo_id, root=a.root)
    print(f"{dataset.num_episodes} episodes, {dataset.num_frames} frames")
    print(f"depth keys: {dataset.meta.depth_keys}")
    depth_key = f"observation.images.{a.camera}_depth"
    assert depth_key in dataset.meta.depth_keys, "depth stream was not stored as depth"

    step = DepthToPointCloudStep(
        num_points=1024,
        frame="camera",
        seed=0,
        # LeRobotDataset dequantises depth to MILLIMETRES by default. Omitting
        # this is a silent 1000x error: every point lands outside the crop and
        # the cloud comes back as zeros. The processor now raises instead.
        depth_scale=1e-3,
        workspace_centre=(0.0, 0.0, 0.67),
        workspace_extent=0.6,
    )

    errs, spreads, nonzero = [], [], []
    for i in range(min(a.frames, dataset.num_frames)):
        item = dataset[i]
        depth = item[depth_key]
        intrinsics = item[f"observation.intrinsics.{a.camera}"]

        observation = {depth_key: depth, f"observation.intrinsics.{a.camera}": intrinsics}
        out = step.observation(dict(observation))
        cloud = out["observation.pointcloud"]
        assert cloud.shape == (1024, 3), cloud.shape
        assert torch.isfinite(cloud).all()

        d = (depth.squeeze().numpy() if depth.ndim == 3 else depth.numpy()) / 1000.0
        valid = (d > 0.05) & (d < 2.9)
        if valid.any():
            errs.append(float(d[valid].mean()))
            spreads.append(float(cloud.abs().max()))
            nonzero.append(float((cloud.abs().sum(-1) > 0).float().mean()))

    print(f"\n{len(errs)} frames unprojected through the PR's processor")
    print(f"  mean stored depth over valid pixels: {np.mean(errs):.4f} m")
    print(f"  max |normalised coordinate|:         {np.max(spreads):.3f}  (1.0 = crop edge)")
    print(f"  fraction of sampled points non-zero: {np.mean(nonzero):.3f}")
    print("\nA cloud that is finite, inside the crop and has a sensible depth scale is "
          "the storage format doing its job.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
