"""Does fusing four cameras put more of the object in the policy's input?

A reviewer of huggingface/lerobot#4696 read the calibration table and stopped on
one column: the camera-frame arm scored 8.1-8.3 mm at every calibration error,
and matched four fused cameras even at zero error. Flat is expected -- nothing
in that path consumes an extrinsic, so there is no wrong number to be wrong --
but *tied* is not. Four cameras see strictly more of the scene than one. If
fusion buys nothing at zero calibration error, the likeliest explanation is that
fusion is broken and the world-frame arm is quietly learning from one camera.

This settles it without training anything, by counting what actually reaches the
encoder. The two questions, asked separately:

  1. At a fixed point budget, does fusion put more points ON THE OBJECT?
  2. When the object is occluded from one camera, do the others recover it?

Ground truth is MuJoCo's segmentation buffer, so "on the object" is the
simulator's answer and not a colour threshold. Sampling is the pull request's
own `sample_points`, with a label channel riding along so each surviving point
can be traced back to what it hit.

Writes results/fusion.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402

from rvt_lerobot.envs.multicam import MultiCamScene  # noqa: E402
from rvt_lerobot.vendor.so101_expert import ScriptedExpert  # noqa: E402
from rvt_lerobot.render import rig as R  # noqa: E402
from rvt_lerobot.vendor.lerobot_pointcloud import sample_points  # noqa: E402

#: Below this many pixels the object is not localisable from that view -- the
#: same threshold the study's centroid check uses before it declines to answer.
VISIBLE_PIXELS = 15


def block_mask_factory(scene: MultiCamScene):
    """Exact block mask per camera, shadows off (a shadow carries the caster's id)."""
    seg = mujoco.Renderer(scene.model, scene.image_size, scene.image_size)
    seg.enable_segmentation_rendering()
    gid = scene.model.geom("block_geom").id

    def mask(camera: str, depth: np.ndarray) -> np.ndarray:
        seg.update_scene(scene.data, camera=camera)
        seg.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        s = seg.render()
        return (
            (s[:, :, 0] == gid)
            & (s[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
            & (depth < R.FAR - 1e-3)
        )

    return mask


def labelled_cloud(obs, cameras, mask_fn, world: bool):
    """Stack the chosen cameras into (N, 4): xyz plus 1.0 where the pixel is block.

    The label rides through sampling as a fourth channel because
    `sample_points` gathers whole rows. That is the only way to ask what
    SURVIVED sampling, as opposed to what was available to it -- and the gap
    between those two is the entire question here.
    """
    xyz, label, valid = [], [], []
    for cam in cameras:
        o = obs[cam]
        T = o.T if world else np.eye(4)
        points = R.unproject(o.depth, o.K, T)
        m = mask_fn(cam, o.depth)
        good = o.depth < R.FAR - 1e-3
        xyz.append(points.reshape(-1, 3))
        label.append(m.reshape(-1).astype(np.float32))
        valid.append(good.reshape(-1))
    xyz = np.concatenate(xyz).astype(np.float32)
    label = np.concatenate(label)
    valid = np.concatenate(valid)
    return np.concatenate([xyz, label[:, None]], 1), valid


def crop(cloud, valid, centre, extent):
    inside = (np.abs(cloud[:, :3] - centre) <= extent / 2).all(1)
    return valid & inside


def on_object(cloud, valid, num_points, seed) -> tuple[int, int, int]:
    """Sample, then count. Returns (points on object, points sampled, valid available)."""
    t_cloud = torch.from_numpy(cloud)[None]
    t_valid = torch.from_numpy(valid)[None]
    if not t_valid.any():
        return 0, num_points, 0
    generator = torch.Generator().manual_seed(seed)
    sampled = sample_points(t_cloud, t_valid, num_points, generator=generator)[0]
    return int((sampled[:, 3] > 0.5).sum()), num_points, int(t_valid.sum())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--per-episode", type=int, default=5)
    p.add_argument("--num-points", type=int, default=1024)
    p.add_argument("--image", type=int, default=96)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/fusion.json"))
    a = p.parse_args()

    scene = MultiCamScene(seed=0, image_size=a.image)
    scene.set_rig(R.NOMINAL_RIG)
    mask_fn = block_mask_factory(scene)
    cameras = list(R.MOVABLE_CAMERAS)
    single = cameras[0]

    # The camera-frame cube is centred in front of the camera; the world-frame
    # cube is centred on the workspace. Same physical volume, different frame.
    world_centre = R.WORKSPACE_CENTRE.astype(np.float32)
    cam_centre = np.array([0.0, 0.0, 0.67], np.float32)

    # Frames are drawn from real expert trajectories rather than from the home
    # pose, because the occlusion this scene actually has is the arm's own body
    # passing in front of a camera -- which only happens mid-reach. Sampling
    # poses from the trajectory is also what the policy's training distribution
    # looks like, so the counts below describe the input it really gets.
    expert = ScriptedExpert(scene.scene)
    rng = np.random.default_rng(0)
    states = []
    for _ in range(a.episodes):
        scene.scene.reset()
        qpos = []
        expert.run(hook=lambda s: qpos.append(s.data.qpos.copy()))
        if len(qpos) < 2:
            continue
        for idx in rng.choice(len(qpos), size=min(a.per_episode, len(qpos)), replace=False):
            states.append(qpos[int(idx)])

    rows = []
    for i, qpos in enumerate(states):
        scene.restore(qpos)
        obs = scene.capture()

        per_camera = {c: int(mask_fn(c, obs[c].depth).sum()) for c in cameras}
        union = sum(per_camera.values())

        cloud_1, valid_1 = labelled_cloud(obs, [single], mask_fn, world=False)
        valid_1 = crop(cloud_1, valid_1, cam_centre, R.WORKSPACE_EXTENT)
        hit_1, _, avail_1 = on_object(cloud_1, valid_1, a.num_points, i)

        cloud_4, valid_4 = labelled_cloud(obs, cameras, mask_fn, world=True)
        valid_4 = crop(cloud_4, valid_4, world_centre, R.WORKSPACE_EXTENT)
        hit_4, _, avail_4 = on_object(cloud_4, valid_4, a.num_points, i)

        # The like-for-like control: one camera, but unprojected to world and
        # cropped by the world cube, so the only difference from `hit_4` is the
        # number of cameras -- not the frame and not the crop.
        cloud_1w, valid_1w = labelled_cloud(obs, [single], mask_fn, world=True)
        valid_1w = crop(cloud_1w, valid_1w, world_centre, R.WORKSPACE_EXTENT)
        hit_1w, _, avail_1w = on_object(cloud_1w, valid_1w, a.num_points, i)

        rows.append({
            "frame": i,
            "block_pixels": per_camera,
            "block_pixels_union": union,
            "single_sees_block": per_camera[single] >= VISIBLE_PIXELS,
            "any_sees_block": union >= VISIBLE_PIXELS,
            "on_object_1cam": hit_1,
            "on_object_4cam": hit_4,
            "on_object_1cam_world": hit_1w,
            "valid_1cam": avail_1,
            "valid_4cam": avail_4,
            "valid_1cam_world": avail_1w,
        })

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    occluded = [r for r in rows if not r["single_sees_block"]]
    recovered = [r for r in occluded if r["any_sees_block"]]

    summary = {
        "frames": len(rows),
        "num_points": a.num_points,
        "on_object_1cam_mean": mean("on_object_1cam"),
        "on_object_1cam_world_mean": mean("on_object_1cam_world"),
        "on_object_4cam_mean": mean("on_object_4cam"),
        "valid_1cam_mean": mean("valid_1cam"),
        "valid_4cam_mean": mean("valid_4cam"),
        "block_pixels_single_mean": float(np.mean([r["block_pixels"][single] for r in rows])),
        "block_pixels_union_mean": mean("block_pixels_union"),
        "frames_block_occluded_from_single": len(occluded),
        "frames_recovered_by_fusion": len(recovered),
        "single_camera": single,
        "cameras": cameras,
    }

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))

    print(f"{len(rows)} frames from {a.episodes} expert episodes, "
          f"{a.num_points}-point budget, {a.image}px cameras\n")
    print(f"  block pixels, {single:15s}     {summary['block_pixels_single_mean']:8.1f}")
    print(f"  block pixels, all 4 cameras       {summary['block_pixels_union_mean']:8.1f}")
    print()
    print(f"  valid points available,  1 cam    {summary['valid_1cam_mean']:8.0f}")
    print(f"  valid points available,  4 cam    {summary['valid_4cam_mean']:8.0f}")
    print()
    print(f"  ON OBJECT after sampling, 1 cam (camera frame) {summary['on_object_1cam_mean']:6.1f}")
    print(f"  ON OBJECT after sampling, 1 cam (world frame)  {summary['on_object_1cam_world_mean']:6.1f}")
    print(f"  ON OBJECT after sampling, 4 cam (world frame)  {summary['on_object_4cam_mean']:6.1f}")
    print()
    print(f"  block hidden from {single}: {len(occluded)}/{len(rows)} frames; "
          f"fusion recovered it in {len(recovered)}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
