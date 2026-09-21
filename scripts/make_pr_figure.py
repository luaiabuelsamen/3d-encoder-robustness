"""Figure for huggingface/lerobot#4696: what the depth path actually produces.

Drawn from the recorded LeRobotDataset and the pull request's own processor
step, not from this study's internal renderer, so every panel is the data a
reviewer would get by running the documented commands.

Panel 4 is the failure this PR now raises on. `LeRobotDataset` dequantises
depth to millimetres by default while the step works in metres, so the obvious
wiring of the two is off by a thousand: every point lands far outside the
workspace crop, and a frame with no surviving points is returned as zeros by
design, so that a dropped depth frame cannot kill a training run. What reaches
the encoder is 1024 copies of a single location with zero extent on every axis.
The policy trains on that and no metric moves. It is reproduced here through the
public API, with `max_depth` raised so the new magnitude check does not fire.

An earlier draft of this figure compared a cropped and an uncropped cloud. That
was dropped: this camera sees almost nothing but the workspace, so the two were
the same picture at different axis scales, and the caption claimed a difference
the image did not show.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.processor.depth_processor import DepthToPointCloudStep  # noqa: E402

BG = "#111318"
FG = "#e8e8e8"
MUTED = "#8b93a1"


def cloud_panel(ax, cloud: np.ndarray, title: str, sub: str, *, lim: float | None = None,
                crop_box: bool = False):
    """Look down the camera's x-z plane: horizontal position against depth."""
    x, z = cloud[:, 0], cloud[:, 2]
    colour = cloud[:, 1]  # height, so the table reads as one band
    ax.scatter(x, z, c=colour, s=6, cmap="viridis", linewidths=0, alpha=0.95)
    if lim is not None:
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    if crop_box:
        # The workspace cube maps to [-1, 1] after normalisation, so its edge is
        # a unit square here. Drawing it makes "outside the crop" literal.
        ax.add_patch(
            plt.Rectangle((-1, -1), 2, 2, fill=False, ls="--", lw=1.0,
                          edgecolor="#5a6472")
        )
        ax.annotate("workspace crop", xy=(0.0, 1.0), xytext=(0.0, 1.12),
                    ha="center", color="#5a6472", fontsize=7)
    ax.set_title(title, color=FG, fontsize=10, pad=8)
    ax.set_xlabel(sub, color=MUTED, fontsize=8)
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#2a2f3a")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=pathlib.Path, default=pathlib.Path("data/lerobot_rgbd"))
    p.add_argument("--repo-id", default="local/so101-pick-rgbd")
    p.add_argument("--camera", default="front")
    p.add_argument("--frame", type=int, default=120)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures/dp3_pipeline.png"))
    a = p.parse_args()

    dataset = LeRobotDataset(a.repo_id, root=a.root)
    item = dataset[a.frame]
    depth_key = f"observation.images.{a.camera}_depth"
    rgb = item[f"observation.images.{a.camera}"].permute(1, 2, 0).numpy()
    depth = item[depth_key]
    intrinsics = item[f"observation.intrinsics.{a.camera}"]
    observation = {depth_key: depth, f"observation.intrinsics.{a.camera}": intrinsics}

    workspace = dict(workspace_centre=(0.0, 0.0, 0.67), workspace_extent=0.6)
    correct = DepthToPointCloudStep(
        num_points=1024, frame="camera", seed=0, depth_scale=1e-3, **workspace
    ).observation(dict(observation))
    # The historical silent failure: millimetres unprojected as metres. max_depth
    # is raised only so the guard added in this PR does not fire, which is the
    # whole point -- without the guard this is what a user would have trained on.
    mis_scaled = DepthToPointCloudStep(
        num_points=1024, frame="camera", seed=0, depth_scale=1.0, max_depth=3000.0, **workspace
    ).observation(dict(observation))

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.4))
    fig.patch.set_facecolor(BG)

    axes[0].imshow(np.clip(rgb, 0, 1))
    axes[0].set_title("1. RGB", color=FG, fontsize=10, pad=8)
    axes[0].set_xlabel("what a 2D policy sees", color=MUTED, fontsize=8)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    d = depth.squeeze().numpy() / 1000.0  # LeRobotDataset returns millimetres
    valid = d > 0.05
    shown = np.where(valid, d, np.nan)
    im = axes[1].imshow(shown, cmap="magma")
    axes[1].set_title("2. depth, as LeRobot stores it", color=FG, fontsize=10, pad=8)
    axes[1].set_xlabel("12-bit lossless, dequantised on read", color=MUTED, fontsize=8)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    cb.set_label("metres", color=MUTED, fontsize=8)

    good = correct["observation.pointcloud"].numpy()
    bad = mis_scaled["observation.pointcloud"].numpy()
    cloud_panel(axes[2], good, "3. point cloud  (what DP3 conditions on)",
                "1024 points, cropped and normalised", lim=1.3, crop_box=True)
    cloud_panel(axes[3], bad, "4. the same call, depth read as metres",
                "outside the crop \u2192 one degenerate point", lim=2.7, crop_box=True)
    # Mark it: a single dark dot on a dark ground is easy to miss, and the whole
    # panel is the claim that there is exactly one.
    axes[3].scatter(bad[0, 0], bad[0, 2], s=90, facecolors="none",
                    edgecolors="#ff6b6b", linewidths=1.4, zorder=5)
    unique = len(np.unique(np.round(bad, 6), axis=0))
    axes[3].annotate(
        f"all 1024 points collapse to {unique} location\nzero extent on every axis",
        xy=(0.5, 0.15), xycoords="axes fraction", ha="center",
        color="#ff6b6b", fontsize=9,
    )

    fig.suptitle(
        "lerobot#4696  —  observation.images.{cam}_depth  →  observation.pointcloud, "
        "via DepthToPointCloudStep",
        color=FG, fontsize=11.5,
    )
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=170, facecolor=BG)
    print(f"wrote {a.out}")
    print(f"  correct:    max |coord| {np.abs(good).max():.2f}  (1.0 = crop edge)")
    print(f"  mis-scaled: {len(np.unique(np.round(bad, 6), axis=0))} unique point(s), "
          f"extent {np.round(bad.max(0) - bad.min(0), 4)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
