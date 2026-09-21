"""Look before sweeping, part two: does the re-rendering actually re-render?

Checks the batched orthographic renderer against facts we know independently:

* the block's world position, projected into each virtual view by hand, must
  land on the pixel where the block's colour actually appears;
* the z-buffer must show the near surface, not the far one;
* the XYZ channels must decode back to the world coordinates they encode;
* and the whole image must be **unchanged** when the real cameras move, which
  is the property the study is about.

It also writes the figure a reader needs to believe any of it.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import PIL.Image as Image  # noqa: E402

from rvt_lerobot.envs.multicam import MultiCamScene  # noqa: E402
from rvt_lerobot.render import rig as R  # noqa: E402
from rvt_lerobot.render import virtual as V  # noqa: E402

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name:46s}  {detail}")
    if not ok:
        fails.append(name)


def cloud_from(obs, cams, device="cpu"):
    """Fuse a capture into (1, N, 3) points, (1, N, 6) feats, (1, N) valid."""
    pts, feats, valid = [], [], []
    for c in cams:
        o = obs[c]
        p = R.unproject(o.depth, o.K, o.T).reshape(-1, 3)
        rgb = (o.rgb.reshape(-1, 3) / 255.0).astype(np.float32)
        v = (o.depth.reshape(-1) < R.FAR - 1e-3)
        pts.append(p)
        feats.append(np.concatenate([rgb, p.astype(np.float32)], axis=1))
        valid.append(v)
    pts = torch.from_numpy(np.concatenate(pts)).float()[None].to(device)
    feats = torch.from_numpy(np.concatenate(feats)).float()[None].to(device)
    valid = torch.from_numpy(np.concatenate(valid))[None].to(device)
    return pts, feats, valid


def main() -> int:
    size = 128
    scene = MultiCamScene(seed=0, image_size=size)
    scene.scene.reset(block_xy=(0.10, -0.25), block_yaw=0.0)
    scene.set_rig(R.NOMINAL_RIG)
    obs = scene.capture()
    truth = scene.scene.block_pos()
    centre = torch.tensor(R.WORKSPACE_CENTRE, dtype=torch.float32)
    extent, S = R.WORKSPACE_EXTENT, 128

    pts, feats, valid = cloud_from(obs, R.ALL_CAMERAS)
    imgs = V.render_views(pts, feats, valid, centre=centre, extent=extent, img_size=S)
    check("render shape", tuple(imgs.shape) == (1, 5, 7, S, S), str(tuple(imgs.shape)))

    # Coverage is wildly uneven by construction and that is not a bug: the top
    # view looks down at a table that fills the cube, the side views look along
    # it and see a thin band. The test is that every view carries real content,
    # not that they carry equal amounts.
    coverage = imgs[0, :, 6].mean(dim=(1, 2))
    for view, cov in zip(V.VIRTUAL_VIEWS, coverage.tolist()):
        check(f"{view}: has content", cov > 0.04, f"{cov:.0%} of pixels hit")

    # the XYZ channels must decode to the world point that wrote them
    for vi, view in enumerate(V.VIRTUAL_VIEWS):
        hit = imgs[0, vi, 6] > 0.5
        rows, cols = torch.nonzero(hit, as_tuple=True)
        xyz = imgs[0, vi, 3:6][:, rows, cols].T                       # (M, 3)
        u, v, au, av = V.unproject_from_view(
            cols.float(), rows.float(), view, centre, extent, S
        )
        err = torch.stack([xyz[:, au] - u, xyz[:, av] - v]).abs().max().item()
        check(f"{view}: pixel<->world consistent", err < extent / (S - 1), f"max |err| = {err*1000:.2f} mm")

    # the block must land where geometry says, with the right colour
    for vi, view in enumerate(V.VIRTUAL_VIEWS):
        col, row, _, inside = V.project_to_view(
            torch.tensor(truth, dtype=torch.float32)[None], view, centre, extent, S
        )
        if not bool(inside[0]):
            check(f"{view}: block inside cube", False)
            continue
        r, c = int(row[0]), int(col[0])
        patch = imgs[0, vi, :3, max(0, r - 3): r + 4, max(0, c - 3): c + 4]
        m = patch.reshape(3, -1)
        redness = (m[0] - m[1:].max(0).values).max().item()
        check(f"{view}: block visible at its projection", redness > 0.20,
              f"max(R - max(G,B)) = {redness:.2f} in a 7x7 patch at ({r},{c})")

    # z-buffer: from the top view, the pixel over the block must carry the
    # block's own height, not the table's
    col, row, _, _ = V.project_to_view(
        torch.tensor(truth, dtype=torch.float32)[None], "top", centre, extent, S
    )
    z_at_block = imgs[0, V.VIRTUAL_VIEWS.index("top"), 5, int(row[0]), int(col[0])].item()
    block_top = truth[2] + float(scene.model.geom_size[scene.scene.ids.block_geom][2])
    check("top view shows the near surface", abs(z_at_block - block_top) < 0.006,
          f"z = {z_at_block*1000:.1f} mm, block top = {block_top*1000:.1f} mm")

    # THE property: move the real cameras, re-render, compare the virtual images
    moved = R.perturb_rig(R.NOMINAL_RIG, 15.0, np.random.default_rng(3))
    scene.set_rig(moved)
    obs2 = scene.capture()
    pts2, feats2, valid2 = cloud_from(obs2, R.ALL_CAMERAS)
    imgs2 = V.render_views(pts2, feats2, valid2, centre=centre, extent=extent, img_size=S)

    both = (imgs[0, :, 6] > 0.5) & (imgs2[0, :, 6] > 0.5)
    rgb_delta = (imgs[0, :, :3] - imgs2[0, :, :3]).abs().mean(1)[both].mean().item()
    xyz_delta = (imgs[0, :, 3:6] - imgs2[0, :, 3:6]).abs().mean(1)[both].mean().item()
    cov1 = (imgs[0, :, 6] > 0.5).float().mean().item()
    cov2 = (imgs2[0, :, 6] > 0.5).float().mean().item()
    check("virtual RGB stable under a 15 deg rig move", rgb_delta < 0.06,
          f"mean |dRGB| = {rgb_delta:.3f} on co-visible pixels")
    # The floor here is resampling, not error: moving the cameras changes WHICH
    # surface sample lands in a given virtual pixel, so the XYZ written there
    # shifts by about one virtual pixel. Quoting the result in pixel widths is
    # the honest unit -- re-rendering is invariant up to resampling, not exactly.
    pix = extent / (S - 1)
    check("virtual XYZ stable under a 15 deg rig move", xyz_delta < 2 * pix,
          f"mean |dXYZ| = {xyz_delta*1000:.2f} mm = {xyz_delta/pix:.2f} virtual pixels "
          f"(1 px = {pix*1000:.1f} mm)")
    check("coverage roughly preserved", abs(cov1 - cov2) < 0.10,
          f"{cov1:.0%} -> {cov2:.0%}")

    # the figure
    out = pathlib.Path("figures")
    out.mkdir(exist_ok=True)
    scene.set_rig(R.NOMINAL_RIG)
    obs = scene.capture()
    real = [obs[c].rgb for c in R.ALL_CAMERAS]
    pts, feats, valid = cloud_from(obs, R.ALL_CAMERAS)
    imgs = V.render_views(pts, feats, valid, centre=centre, extent=extent, img_size=S)
    virt = [(imgs[0, i, :3].permute(1, 2, 0).numpy() * 255).astype(np.uint8) for i in range(5)]
    pad = np.full((S, S, 3), 255, np.uint8)
    top = np.concatenate(real + [pad], axis=1)
    bot = np.concatenate(virt, axis=1)
    Image.fromarray(np.concatenate([top, bot])).save(out / "virtual_views.png")
    print(f"\nwrote {out/'virtual_views.png'}  (top: 4 real cameras, bottom: 5 virtual)")
    print("FAILURES:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
