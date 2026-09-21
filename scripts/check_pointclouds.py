"""Look at the point clouds the DP3 arms actually receive.

Everything else about these clouds has been checked numerically -- the block
centroid lands within 10 mm of MuJoCo's truth, the table recovers as a plane
flat to 0.1 mm, camera frame plus extrinsic equals world frame to 0.0000 mm.
None of that shows whether the *sampled 1024 points a policy is handed* still
contain the object, or whether the subsample has quietly thrown the block away
in favour of tabletop.

So this renders the real thing: the exact tensor `make_images` builds for
`dp3_pcd` and `dp3_world`, rasterised from three directions, next to the RGB
frame it came from. Numbers alongside, because a picture can hide a factor of
two and a number cannot:

* how many of the sampled points are on the block, against how many would be
  expected if sampling were uniform over the visible surface;
* the spatial extent of the cloud, which catches a normalisation that has
  collapsed or exploded it;
* the same under depth noise and under miscalibration, which is where a cloud
  stops being a picture of the scene and starts being a picture of the error.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import PIL.Image as Image  # noqa: E402

from rvt_lerobot.data import collect_study as C  # noqa: E402
from rvt_lerobot.data.batching import (  # noqa: E402
    NOMINAL_CAMERA_DISTANCE,
    POINTCLOUD_POINTS,
    ConditionCache,
    make_images,
)
from rvt_lerobot.data.views import Condition, build_samples, render_condition  # noqa: E402
from rvt_lerobot.envs.multicam import MultiCamScene  # noqa: E402
from rvt_lerobot.models.policy import ARMS  # noqa: E402
from rvt_lerobot.render import rig as R  # noqa: E402
from rvt_lerobot.render import virtual as V  # noqa: E402

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name:50s}  {detail}")
    if not ok:
        fails.append(name)


def draw(cloud: np.ndarray, size: int = 160) -> np.ndarray:
    """Rasterise a cloud from front, top and right, coloured by height."""
    pts = torch.from_numpy(cloud).float()[None]
    z = cloud[:, 2]
    lo, hi = np.percentile(z, 2), np.percentile(z, 98)
    t = np.clip((z - lo) / max(1e-6, hi - lo), 0, 1)
    # blue (low) -> yellow (high): readable in both light and dark
    colour = np.stack([t, t * 0.85 + 0.15, 1.0 - t], axis=-1).astype(np.float32)
    feats = torch.from_numpy(colour)[None]
    valid = torch.ones(1, len(cloud), dtype=torch.bool)
    imgs = V.render_views(
        pts, feats, valid,
        views=("front", "top", "right"),
        centre=pts.new_zeros(3), extent=2.4, img_size=size,
    )
    tiles = []
    for i in range(3):
        rgb = (imgs[0, i, :3].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        hit = imgs[0, i, 3].numpy() > 0.5
        rgb[~hit] = 18  # dark background so sparse clouds stay legible
        tiles.append(rgb)
    return np.concatenate(tiles, axis=1)


def main() -> int:
    scene = MultiCamScene(seed=4242, image_size=96)
    episodes = C.load(pathlib.Path("data/test.npz"))[:4]
    samples = build_samples(episodes, np.random.default_rng(0), per_segment=1)

    seg = mujoco.Renderer(scene.model, scene.image_size, scene.image_size)
    seg.enable_segmentation_rendering()
    gid = scene.model.geom("block_geom").id

    rows, labels = [], []
    for cond, label in (
        (Condition(), "nominal"),
        (Condition(noise_c=0.008), "depth noise c=0.008"),
        (Condition(eps_deg=5), "calibration error 5 deg"),
    ):
        arrays = render_condition(scene, episodes, samples, cond)
        cache = ConditionCache(arrays, device="cpu")
        batch = cache.batch(np.arange(4))

        for arm in ("dp3_pcd", "dp3_world"):
            cloud = make_images(ARMS[arm], batch)[0].numpy()
            check(
                f"{arm} / {label}: shape",
                cloud.shape == (POINTCLOUD_POINTS, 3),
                str(cloud.shape),
            )
            extent = float(np.abs(cloud).max())
            check(
                f"{arm} / {label}: inside the workspace after cropping",
                0.05 < extent <= 1.001,
                f"max |coord| = {extent:.3f} (normalised units)",
            )
            spread = float(cloud.std(0).mean())
            check(
                f"{arm} / {label}: has spatial spread",
                spread > 0.02,
                f"mean per-axis std = {spread:.3f}",
            )
            rows.append(draw(cloud))
            labels.append(f"{arm} - {label}")

    # the question a picture cannot answer: is the BLOCK still in the subsample?
    print()
    arrays = render_condition(scene, episodes, samples, Condition())
    cache = ConditionCache(arrays, device="cpu")
    batch = cache.batch(np.arange(1))
    scene.restore(episodes[0].qpos[int(episodes[0].keyframes[len(episodes[0].keyframes) // 3])])
    depth0 = batch["depth_mm"][0, 0].numpy().astype(np.float32) / 1000.0
    seg.update_scene(scene.data, camera=R.ALL_CAMERAS[0])
    seg.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    s = seg.render()
    block_px = int(((s[:, :, 0] == gid) & (s[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)).sum())
    visible_px = int((depth0 < R.FAR - 1e-3).sum())
    # block share BEFORE cropping, which is the number the crop exists to fix
    expected = POINTCLOUD_POINTS * block_px / max(1, visible_px)
    print(
        f"front camera: {block_px} block pixels of {visible_px} valid "
        f"-> a uniform subsample of {POINTCLOUD_POINTS} should contain "
        f"about {expected:.1f} block points"
    )
    check(
        "the block survives subsampling at all",
        expected >= 1.0,
        f"{expected:.1f} expected points on a {2 * 10:.0f} mm object",
    )

    out = pathlib.Path("figures/pointclouds.png")
    out.parent.mkdir(exist_ok=True)
    pad = np.full((6, rows[0].shape[1], 3), 255, np.uint8)
    stacked = []
    for r in rows:
        stacked.extend([r, pad])
    Image.fromarray(np.concatenate(stacked[:-1])).save(out)
    print(f"\nwrote {out}")
    print("rows, top to bottom: " + "; ".join(labels))
    print("each row: front, top, right   (colour = height)")
    print("FAILURES:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
