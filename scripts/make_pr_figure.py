"""Figure for huggingface/lerobot#4696: depth becomes recognisable 3D geometry.

Everything here comes from a LeRobotDataset recorded through the real
`add_frame` / `save_episode` path and is unprojected by the pull request's own
processor step, so a reviewer sees what the documented commands produce rather
than an illustration of them.

The point of the figure is that the last two panels are the *same* 1024 points
seen from two directions. A point cloud drawn once, flat, is indistinguishable
from noise; rotated, the container, the table and the arm are obviously solid
objects sitting in the right places. That is the difference between claiming
the geometry is right and showing it.

Colour is real RGB carried through with `with_colour=True`, which is the pull
request's 6-channel path, so the panels are also a check on it: if the colour
were being attached to the wrong points, these renders would be confetti.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.processor.depth_processor import DepthToPointCloudStep  # noqa: E402

BG = "#0d0f14"
FG = "#f0f0f0"
MUTED = "#8b93a1"


def style(ax, title: str, sub: str) -> None:
    ax.set_title(title, color=FG, fontsize=12, pad=10)
    ax.set_xlabel(sub, color=MUTED, fontsize=9, labelpad=8)


def cloud3d(ax, cloud: np.ndarray, elev: float, azim: float, title: str, sub: str) -> None:
    """Draw the cloud as 3D geometry, coloured by the RGB carried on each point."""
    xyz, rgb = cloud[:, :3], np.clip(cloud[:, 3:6], 0, 1)
    # World-ish orientation for reading: x right, z into the screen, y up.
    ax.scatter(xyz[:, 0], xyz[:, 2], -xyz[:, 1], c=rgb, s=11, depthshade=False, linewidths=0)
    ax.view_init(elev=elev, azim=azim)
    ax.set_facecolor(BG)
    ax.set_box_aspect((1, 1, 1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.05, 0.06, 0.08, 1.0))
        axis.line.set_color("#2a2f3a")
        axis.set_tick_params(colors=MUTED, labelsize=6)
    # Numeric ticks on a normalised cloud carry no information a reader needs
    # and compete with the geometry, which is the whole point of the panel.
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.grid(False)
    ax.set_title(title, color=FG, fontsize=12, pad=4)
    ax.text2D(0.5, -0.04, sub, transform=ax.transAxes, ha="center",
              color=MUTED, fontsize=9)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=pathlib.Path, default=pathlib.Path("data/lerobot_rgbd_hires"))
    p.add_argument("--repo-id", default="local/so101-pick-rgbd")
    p.add_argument("--camera", default="front")
    p.add_argument("--frame", type=int, default=95)
    p.add_argument("--points", type=int, default=3000)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures/dp3_pipeline.png"))
    a = p.parse_args()

    dataset = LeRobotDataset(a.repo_id, root=a.root)
    item = dataset[a.frame]
    depth_key = f"observation.images.{a.camera}_depth"
    rgb_key = f"observation.images.{a.camera}"
    rgb = item[rgb_key].permute(1, 2, 0).numpy()
    depth = item[depth_key]

    observation = {
        depth_key: depth,
        rgb_key: item[rgb_key],
        f"observation.intrinsics.{a.camera}": item[f"observation.intrinsics.{a.camera}"],
    }
    step = DepthToPointCloudStep(
        num_points=a.points,
        frame="camera",
        with_colour=True,          # the PR's 6-channel path
        seed=0,
        depth_scale=1e-3,          # LeRobotDataset returns millimetres
        workspace_centre=(0.0, 0.0, 0.62),
        workspace_extent=0.7,
    )
    cloud = step.observation(dict(observation))["observation.pointcloud"].numpy()

    fig = plt.figure(figsize=(19, 5.6))
    fig.patch.set_facecolor(BG)

    ax0 = fig.add_subplot(1, 4, 1)
    ax0.imshow(np.clip(rgb, 0, 1))
    ax0.set_xticks([])
    ax0.set_yticks([])
    style(ax0, "1. RGB", "what a 2D policy sees")

    ax1 = fig.add_subplot(1, 4, 2)
    d = depth.squeeze().numpy() / 1000.0
    im = ax1.imshow(np.where(d > 0.05, d, np.nan), cmap="magma")
    ax1.set_xticks([])
    ax1.set_yticks([])
    style(ax1, "2. depth, as LeRobot stores it", "12-bit lossless, dequantised on read")
    cb = fig.colorbar(im, ax=ax1, fraction=0.046)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.set_label("metres", color=MUTED, fontsize=8)

    ax2 = fig.add_subplot(1, 4, 3, projection="3d")
    cloud3d(ax2, cloud, elev=22, azim=-72, title="3. observation.pointcloud",
            sub=f"{a.points} points, cropped and normalised")
    ax3 = fig.add_subplot(1, 4, 4, projection="3d")
    cloud3d(ax3, cloud, elev=58, azim=-20, title="4. the same points, rotated",
            sub="solid objects in the right places, not a depth image")

    fig.suptitle(
        "lerobot#4696   observation.images.{cam}_depth  →  observation.pointcloud"
        "   via DepthToPointCloudStep",
        color=FG, fontsize=13.5, y=0.98,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=150, facecolor=BG)
    print(f"wrote {a.out}")
    print(f"  cloud {cloud.shape}, xyz range "
          f"{np.round(cloud[:, :3].min(0), 2)} to {np.round(cloud[:, :3].max(0), 2)}")
    print(f"  colour range {cloud[:, 3:6].min():.2f} to {cloud[:, 3:6].max():.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
