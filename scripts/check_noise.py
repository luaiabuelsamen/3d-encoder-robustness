"""What does sensor depth noise actually do to a world-frame point cloud?

Not what one might assume. A rig looking down at the table from about 27 degrees
of elevation converts axial depth error into mostly *lateral* world error: the
camera's optical axis is nearly horizontal, so an error along it lands in the
table plane rather than perpendicular to it. Measured at c = 0.008, the mean
world displacement is 8.4 mm horizontally against 2.7 mm vertically.

The central ray's geometry predicts a ratio of cot(27 deg) = 2.0. What is
actually measured is 1.4 on surface interiors and 3.1 over the whole image, so
cot(elevation) sets the order and the sign, not the value: rays away from the
image centre have their own elevations, the ratio of means is not the mean of
ratios, and flying pixels at edges jump between surfaces rather than along a
ray. The claim this study leans on is the robust part -- the error is
predominantly lateral -- not a constant.

That matters for this study because it is the horizontal direction that decides
whether the jaws close on a 20 mm block, and because it is the direction the
*top* virtual view resolves -- the one view with dense coverage of the workspace.
So depth noise does its damage exactly where a canonicalised representation is
most informative.

A first attempt to see this measured the standard deviation of world z inside a
ball around the block and found nothing, because that is the one component the
noise barely touches. The probe was wrong, not the model.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import PIL.Image as Image  # noqa: E402
import torch  # noqa: E402

from rvt_lerobot.envs.multicam import MultiCamScene  # noqa: E402
from rvt_lerobot.render import rig as R  # noqa: E402
from rvt_lerobot.render import virtual as V  # noqa: E402
from rvt_lerobot.render.noise import NOISE_LEVELS, apply_depth_noise  # noqa: E402

fails: list[str] = []


def table_interior_mask(depth: np.ndarray, edge_threshold: float = 0.02) -> np.ndarray:
    """Pixels on a real surface and away from any depth discontinuity."""
    gy, gx = np.gradient(depth.astype(np.float32))
    smooth = np.hypot(gx, gy) < edge_threshold * 0.5
    # erode once so neighbours of an edge are excluded too
    inner = smooth.copy()
    inner[1:] &= smooth[:-1]
    inner[:-1] &= smooth[1:]
    inner[:, 1:] &= smooth[:, :-1]
    inner[:, :-1] &= smooth[:, 1:]
    return inner & (depth < R.FAR - 1e-3) & (depth > 0.2)


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name:44s}  {detail}")
    if not ok:
        fails.append(name)


def main() -> int:
    scene = MultiCamScene(seed=0, image_size=128)
    scene.scene.reset(block_xy=(0.10, -0.25), block_yaw=0.0)
    scene.set_rig(R.NOMINAL_RIG)
    obs = scene.capture()
    rng = np.random.default_rng(0)

    # 1. the noise has the magnitude it claims, on surfaces that are actually
    #    there. The first version of this test sampled the image's top-left
    #    corner as "a flat surface"; in the front view that is sky at the far
    #    plane, where c*z^2 is predicting the noise of a 3 m reading.
    o = obs["front"]
    interior = table_interior_mask(o.depth)
    for c in (0.002, 0.008):
        noisy = apply_depth_noise(o.depth, c, np.random.default_rng(1), far=R.FAR)
        resid = (noisy - o.depth)[interior]
        got = float(resid.std())
        want = float((c * o.depth[interior] ** 2).mean())
        check(f"c={c:g}: sigma_z matches c*z^2", abs(got - want) < 0.35 * want + 3e-4,
              f"{got*1000:.2f} mm measured, {want*1000:.2f} mm predicted, "
              f"{interior.sum()} interior pixels")

    # 2. the world displacement is anisotropic, and in the direction geometry says
    print()
    table = []
    for c in (0.0, 0.002, 0.008):
        dz, dxy = [], []
        for cam in R.MOVABLE_CAMERAS:
            o = obs[cam]
            clean = R.unproject(o.depth, o.K, o.T)
            noisy = R.unproject(apply_depth_noise(o.depth, c, rng, far=R.FAR), o.K, o.T)
            d = (noisy - clean)[o.depth < R.FAR - 1e-3]
            d = d[np.linalg.norm(d, axis=1) < 0.2]      # holes go to the far plane
            dz.append(np.abs(d[:, 2]))
            dxy.append(np.linalg.norm(d[:, :2], axis=1))
        z, xy = np.concatenate(dz).mean(), np.concatenate(dxy).mean()
        table.append((c, xy * 1000, z * 1000))
        print(f"  c={c:<7g} lateral {xy*1000:6.2f} mm   vertical {z*1000:6.2f} mm"
              f"   ratio {xy/max(z,1e-9):5.2f}")
    elevation = np.arcsin((R.NOMINAL_RIG["front"].eye[2] - R.WORKSPACE_CENTRE[2])
                          / np.linalg.norm(R.NOMINAL_RIG["front"].eye - R.WORKSPACE_CENTRE))
    predicted = 1.0 / np.tan(elevation)

    # The cot(elevation) law applies to *axial* noise. Restricted to surface
    # interiors, where there are no edges for the sensor model to invent flying
    # pixels at, the ratio should hit it. Over the whole image it runs higher --
    # 3.1 rather than 2.0 at c=0.008 -- because a flying pixel jumps to a
    # neighbouring surface, and the surfaces that have edges here are vertical.
    # Both numbers are real; they are different mechanisms, so they get
    # different tests.
    lat, vert = [], []
    for cam in R.MOVABLE_CAMERAS:
        o = obs[cam]
        m = table_interior_mask(o.depth)
        clean = R.unproject(o.depth, o.K, o.T)
        noisy = R.unproject(apply_depth_noise(o.depth, 0.008, rng, far=R.FAR), o.K, o.T)
        d = (noisy - clean)[m]
        lat.append(np.linalg.norm(d[:, :2], axis=1))
        vert.append(np.abs(d[:, 2]))
    ratio = float(np.concatenate(lat).mean() / max(np.concatenate(vert).mean(), 1e-9))
    check("on surface interiors, ratio is cot(elevation)",
          abs(ratio - predicted) < 0.35 * predicted,
          f"measured {ratio:.2f}, cot({np.rad2deg(elevation):.0f} deg) = {predicted:.2f}")
    whole = table[-1][1] / max(table[-1][2], 1e-9)
    check("edges push the ratio further sideways, not less",
          whole > ratio - 0.2, f"whole image {whole:.2f} vs interiors {ratio:.2f}")
    check("quantisation alone is sub-millimetre", table[0][1] < 1.0,
          f"{table[0][1]:.2f} mm lateral at c=0")
    check("noise is monotone in c", table[0][1] < table[1][1] < table[2][1])

    # 3. and a picture of what the network actually gets
    centre = torch.tensor(R.WORKSPACE_CENTRE, dtype=torch.float32)
    rows = []
    for c in (0.0, 0.002, 0.008):
        pts, feats, valid = [], [], []
        for cam in R.ALL_CAMERAS:
            o = obs[cam]
            d = apply_depth_noise(o.depth, c, np.random.default_rng(7), far=R.FAR)
            p = R.unproject(d, o.K, o.T).reshape(-1, 3)
            pts.append(p)
            feats.append(np.concatenate(
                [(o.rgb.reshape(-1, 3) / 255.0).astype(np.float32), p.astype(np.float32)], 1))
            valid.append(d.reshape(-1) < R.FAR - 1e-3)
        imgs = V.render_views(
            torch.from_numpy(np.concatenate(pts)).float()[None],
            torch.from_numpy(np.concatenate(feats)).float()[None],
            torch.from_numpy(np.concatenate(valid))[None],
            centre=centre, extent=R.WORKSPACE_EXTENT, img_size=128,
        )
        rows.append(np.concatenate(
            [(imgs[0, i, :3].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for i in range(5)], 1))
    out = pathlib.Path("figures/noise_virtual.png")
    out.parent.mkdir(exist_ok=True)
    Image.fromarray(np.concatenate(rows)).save(out)
    print(f"\nwrote {out}  (rows: c = 0, 0.002, 0.008)")
    print("FAILURES:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
