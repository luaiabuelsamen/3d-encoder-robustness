"""Verify the LeRobot PR's geometry against this project's MuJoCo pipeline.

The PR (huggingface/lerobot#4696) ships `unproject`, a point-cloud sampler, a
`PointCloudEncoder` and a processor step. Its unit tests check those against
synthetic tensors -- a plane at a known depth, a pure translation. That is
necessary but thin: synthetic tensors cannot catch a convention mismatch with a
real renderer, because both sides of the test share the same assumption.

Here the same code is run against real MuJoCo renders, with the block's true
position taken from the segmentation buffer, so the check is against physics
rather than against arithmetic. It also re-derives the calibration crossover
quoted in the PR body using the PR's own `unproject`, so the number in the
description and the number the code produces cannot drift apart.

Run from the repository root with the LeRobot checkout importable.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path.home() / "projects" / "lerobot" / "src"))

import mujoco  # noqa: E402

from rvt_lerobot.envs.multicam import MultiCamScene  # noqa: E402
from rvt_lerobot.render import rig as R  # noqa: E402

# Imported directly from the file so this runs without LeRobot's package
# imports, which need Python 3.12. The point is to exercise the PR's code, not
# its import graph.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "pr_pointcloud",
    pathlib.Path.home() / "projects/lerobot/src/lerobot/policies/common/pointcloud.py",
)
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name:52s}  {detail}")
    if not ok:
        fails.append(name)


def main() -> int:
    scene = MultiCamScene(seed=0, image_size=128)
    scene.scene.reset(block_xy=(0.10, -0.25), block_yaw=0.0)
    scene.set_rig(R.NOMINAL_RIG)
    obs = scene.capture()
    truth = scene.scene.block_pos()

    seg = mujoco.Renderer(scene.model, scene.image_size, scene.image_size)
    seg.enable_segmentation_rendering()
    gid = scene.model.geom("block_geom").id

    def block_mask(cam: str, depth: np.ndarray) -> np.ndarray:
        seg.update_scene(scene.data, camera=cam)
        seg.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        s = seg.render()
        return (
            (s[:, :, 0] == gid)
            & (s[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
            & (depth < R.FAR - 1e-3)
        )

    print(f"block truth = {truth.round(4)}\n")

    # 1. the PR's unproject, in world frame, against physical ground truth
    for cam in R.MOVABLE_CAMERAS:
        o = obs[cam]
        pts = pc.unproject(
            torch.from_numpy(o.depth)[None].float(),
            torch.from_numpy(o.K)[None].float(),
            torch.from_numpy(o.T)[None].float(),
        )[0].numpy()
        centroid = pts[block_mask(cam, o.depth)].mean(0)
        err = float(np.linalg.norm(centroid[:2] - truth[:2]))
        check(f"world frame vs MuJoCo truth ({cam})", err < 0.015,
              f"|err| = {err * 1000:.1f} mm")

    # 2. camera frame must agree with world frame once the extrinsic is applied
    #    by hand. This is the convention check synthetic tests cannot make.
    for cam in R.MOVABLE_CAMERAS:
        o = obs[cam]
        d = torch.from_numpy(o.depth)[None].float()
        k = torch.from_numpy(o.K)[None].float()
        t = torch.from_numpy(o.T)[None].float()
        cam_pts = pc.unproject(d, k)[0]
        world_pts = pc.unproject(d, k, t)[0]
        by_hand = cam_pts @ t[0, :3, :3].T + t[0, :3, 3]
        check(f"camera frame + extrinsic == world frame ({cam})",
              torch.allclose(by_hand, world_pts, atol=1e-4),
              f"max delta {(by_hand - world_pts).abs().max() * 1000:.4f} mm")

    # 3. the processor's contract on real data: a fixed-size, all-valid cloud
    o = obs["front"]
    depth = torch.from_numpy(o.depth)[None].float()
    k = torch.from_numpy(o.K)[None].float()
    pts = pc.unproject(depth, k).reshape(1, -1, 3)
    valid = ((depth > 0.01) & (depth < R.FAR - 1e-3)).reshape(1, -1)
    cloud = pc.sample_points(pts, valid, 1024)
    check("sampler returns the requested size on real depth",
          tuple(cloud.shape) == (1, 1024, 3), str(tuple(cloud.shape)))
    check("no sampled point is a far-plane return",
          bool((cloud[..., 2] < R.FAR - 1e-3).all()),
          f"max z = {cloud[..., 2].max():.3f} m")
    check("no sampled point sits at the sensor origin",
          bool((cloud.norm(dim=-1) > 1e-6).all()))

    # 4. the encoder consumes it and produces gradients
    enc = pc.PointCloudEncoder(in_channels=3, out_features=256)
    feat = enc(cloud)
    feat.sum().backward()
    check("encoder runs on a real cloud", tuple(feat.shape) == (1, 256), str(tuple(feat.shape)))
    check("gradients reach the encoder", all(p.grad is not None for p in enc.parameters()))

    # 5. THE claim in the PR body: re-derive the calibration displacement with
    #    the PR's own unproject, on real geometry.
    print()
    rng = np.random.default_rng(11)
    d_nominal = float(np.mean([
        np.linalg.norm(p.eye - R.WORKSPACE_CENTRE) for p in R.NOMINAL_RIG.values()
    ]))
    print(f"{'eps':>5s} {'predicted':>10s} {'measured':>9s}   (mm, PR unproject on MuJoCo depth)")
    ratios = []
    for eps in (1.0, 2.0, 5.0):
        moved = []
        for cam in R.MOVABLE_CAMERAS:
            o = obs[cam]
            m = block_mask(cam, o.depth)
            bad = R.miscalibrate({cam: o.T}, eps, rng)[cam]
            good_pts = pc.unproject(
                torch.from_numpy(o.depth)[None].float(),
                torch.from_numpy(o.K)[None].float(),
                torch.from_numpy(o.T)[None].float(),
            )[0].numpy()[m].mean(0)
            bad_pts = pc.unproject(
                torch.from_numpy(o.depth)[None].float(),
                torch.from_numpy(o.K)[None].float(),
                torch.from_numpy(bad)[None].float(),
            )[0].numpy()[m].mean(0)
            moved.append(np.linalg.norm(bad_pts - good_pts))
        measured = float(np.mean(moved)) * 1000
        predicted = float(np.hypot((np.pi / 4) * d_nominal * np.sin(np.deg2rad(eps)) * 1000, eps))
        ratios.append(measured / predicted)
        print(f"{eps:5g} {predicted:10.1f} {measured:9.1f}")
    check("PR's 9 mm/deg claim reproduces on MuJoCo geometry",
          0.85 < float(np.mean(ratios)) < 1.15,
          f"measured/predicted = {np.mean(ratios):.2f}")

    print("\nFAILURES:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
