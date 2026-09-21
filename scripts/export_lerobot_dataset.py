"""Record scripted RGBD demonstrations into a real LeRobotDataset.

A reviewer of huggingface/lerobot#4696 asked for the policy to be trained
through the *actual* dataset path with `use_depth=True`, not through this
study's own loader. That is the right thing to ask: the loader here renders
float depth straight into a tensor, while LeRobot quantises depth to a 12-bit
video stream and dequantises it on read. If the quantiser loses what a point
cloud needs, no amount of testing the encoder in isolation would show it.

So this writes a genuine `LeRobotDataset` -- the same `add_frame` /
`save_episode` path a real robot recording uses, the same depth encoder, the
same parquet and video layout -- from the MuJoCo scene and scripted expert this
study already validates against. It records:

    observation.state                 (6,)     arm joint positions
    action                            (6,)     the expert's control vector
    observation.images.{cam}          (H,W,3)  RGB
    observation.images.{cam}_depth    (H,W,1)  metric depth, flagged is_depth_map
    observation.intrinsics.{cam}      (3,3)    what LeRobot used to discard

The depth key follows the convention the robots already emit (`{cam}_depth`),
so a policy trained on this cannot rely on anything a real recording lacks.

Dense actions, not keyposes: DP3 and Diffusion Policy predict controls at every
step, and a keypose dataset cannot train them.

Run it with the Python that has LeRobot (3.12 here), for example:

    PYTHONPATH=~/projects/lerobot/src MUJOCO_GL=egl \\
        python scripts/export_lerobot_dataset.py --episodes 50
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from lerobot.configs.video import DepthEncoderConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

from rvt_lerobot.envs.multicam import MultiCamScene  # noqa: E402
from rvt_lerobot.render import rig as R  # noqa: E402
from rvt_lerobot.vendor.so101_expert import ScriptedExpert  # noqa: E402

TASK = "Pick up the block and place it in the container."


def features(camera: str, size: int) -> dict:
    return {
        "observation.state": {
            "dtype": "float32", "shape": (6,), "names": [f"joint_{i}" for i in range(6)],
        },
        "action": {
            "dtype": "float32", "shape": (6,), "names": [f"joint_{i}" for i in range(6)],
        },
        f"observation.images.{camera}": {
            "dtype": "video", "shape": (size, size, 3),
            "names": ["height", "width", "channels"],
        },
        f"observation.images.{camera}_depth": {
            "dtype": "video", "shape": (size, size, 1),
            "names": ["height", "width", "channels"],
            # The marker that makes LeRobot route this stream through the depth
            # quantiser instead of an 8-bit RGB codec.
            "info": {"is_depth_map": True},
        },
        f"observation.intrinsics.{camera}": {
            "dtype": "float32", "shape": (3, 3), "names": ["row", "col"],
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--camera", default="front")
    p.add_argument("--image", type=int, default=96)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--repo-id", default="local/so101-pick-rgbd")
    p.add_argument("--root", type=pathlib.Path,
                   default=pathlib.Path("data/lerobot_rgbd"))
    p.add_argument("--stride", type=int, default=2,
                   help="keep every Nth simulator frame; the expert runs far above task bandwidth")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    if a.root.exists():
        if not a.overwrite:
            print(f"{a.root} exists; pass --overwrite to replace it")
            return 1
        shutil.rmtree(a.root)

    scene = MultiCamScene(seed=7, image_size=a.image, cameras=(a.camera,))
    scene.set_rig(R.NOMINAL_RIG)
    expert = ScriptedExpert(scene.scene)
    intrinsics = scene._K[a.camera].astype(np.float32)

    dataset = LeRobotDataset.create(
        repo_id=a.repo_id,
        fps=a.fps,
        features=features(a.camera, a.image),
        root=a.root,
        robot_type="so101_follower",
        use_videos=True,
        # Lossless 12-bit quantisation: depth is the signal here, and a lossy
        # codec would put its artefacts straight into the point cloud.
        depth_encoder=DepthEncoderConfig(depth_min=R.NEAR, depth_max=R.FAR),
    )

    kept = 0
    for episode in range(a.episodes):
        scene.scene.reset()
        frames: list[dict] = []

        def hook(s, _frames=frames) -> None:
            obs = scene.capture()[a.camera]
            _frames.append({
                "observation.state": s.data.qpos[:6].astype(np.float32).copy(),
                "action": s.data.ctrl[:6].astype(np.float32).copy(),
                f"observation.images.{a.camera}": obs.rgb.copy(),
                f"observation.images.{a.camera}_depth":
                    obs.depth.astype(np.float32)[..., None].copy(),
                f"observation.intrinsics.{a.camera}": intrinsics,
                "task": TASK,
            })

        result = expert.run(hook=hook)
        # A behaviour-cloning set of demonstrations that did not work teaches the
        # wrong thing; the expert succeeds about 90% of the time, so dropping is
        # cheaper than any alternative.
        if not result.placed:
            print(f"  episode {episode}: expert failed, dropped", flush=True)
            continue

        for frame in frames[:: a.stride]:
            dataset.add_frame(frame)
        dataset.save_episode()
        kept += 1
        print(f"  episode {episode}: {len(frames[:: a.stride])} frames kept "
              f"({kept} episodes so far)", flush=True)

    print(f"\nwrote {kept} episodes to {a.root}")
    print(f"  total frames: {dataset.meta.total_frames}")
    print(f"  depth keys:   {dataset.meta.depth_keys}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
