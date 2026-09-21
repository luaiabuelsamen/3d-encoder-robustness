"""Train DP3 and stock Diffusion Policy on the same recorded dataset.

The reviewer's request behind this: show that DP3 trains through the *actual*
`LeRobotDataset` path, on the same data and the same budget as the policy it
subclasses, and that the point-cloud branch carries signal rather than merely
failing to break anything.

Both arms read the identical dataset written by `export_lerobot_dataset.py`,
with identical horizon, observation steps, optimiser, schedule, seed and U-Net.
They differ in exactly one thing:

    dp     conditions on RGB    (observation.images.front)
    dp3    conditions on a point cloud, unprojected from the depth stream of
           THAT SAME camera by the pull request's processor step

So the comparison is what the depth is worth, holding the camera fixed.

**Scale.** This runs on a CPU with a reduced U-Net, because the machine it was
written on has CUDA torch pinned to Python 3.10 while LeRobot main needs 3.12,
and the two cannot meet here. That makes this a *training-dynamics* check, not
a benchmark: it shows both arms optimise the same objective on the same data
and where their losses sit relative to each other. It is not a success rate,
and no conclusion about DP3-versus-DP performance should be drawn from it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.dp3.configuration_dp3 import DP3Config
from lerobot.policies.dp3.modeling_dp3 import OBS_POINTCLOUD, DP3Policy
from lerobot.processor.depth_processor import DepthToPointCloudStep

CAMERA = "front"
DEPTH_KEY = f"observation.images.{CAMERA}_depth"
RGB_KEY = f"observation.images.{CAMERA}"
INTRINSICS_KEY = f"observation.intrinsics.{CAMERA}"
STATE_DIM = ACTION_DIM = 6
NORM = {k: NormalizationMode.MEAN_STD for k in ("STATE", "ACTION", "VISUAL")}


def make_config(arm: str, args) -> DiffusionConfig:
    common = dict(
        horizon=args.horizon,
        n_action_steps=args.horizon // 2,
        n_obs_steps=args.obs_steps,
        down_dims=(128, 256),
        num_inference_steps=10,
        crop_shape=None,
    )
    if arm == "dp3":
        config = DP3Config(pointcloud_num_points=args.points,
                           pointcloud_feature_dim=64, **common)
    else:
        config = DiffusionConfig(**common)

    config.input_features = {
        "observation.state": PolicyFeature(FeatureType.STATE, (STATE_DIM,))
    }
    if arm == "dp":
        config.input_features[RGB_KEY] = PolicyFeature(FeatureType.VISUAL, (3, 96, 96))
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (ACTION_DIM,))}
    config.normalization_mapping = NORM
    return config


def prepare(batch: dict, arm: str, step: DepthToPointCloudStep) -> dict:
    """Turn a LeRobotDataset batch into what the policy expects."""
    out = {
        "observation.state": batch["observation.state"],
        "action": batch["action"],
        "action_is_pad": batch.get(
            "action_is_pad",
            torch.zeros(batch["action"].shape[:2], dtype=torch.bool),
        ),
    }
    if arm == "dp":
        # The policy stacks its own image_features into observation.images, so
        # hand it the camera key rather than a pre-stacked tensor.
        out[RGB_KEY] = batch[RGB_KEY]
        return out

    # DP3: unproject each observation step through the PR's processor. The
    # dataset hands back MILLIMETRES, hence depth_scale on the step.
    b, t = batch[DEPTH_KEY].shape[:2]
    clouds = []
    for i in range(t):
        observation = {
            DEPTH_KEY: batch[DEPTH_KEY][:, i],
            INTRINSICS_KEY: batch[INTRINSICS_KEY][:, i],
        }
        clouds.append(step.observation(observation)["observation.pointcloud"])
    out[OBS_POINTCLOUD] = torch.stack(clouds, dim=1)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=pathlib.Path, default=pathlib.Path("data/lerobot_rgbd"))
    p.add_argument("--repo-id", default="local/so101-pick-rgbd")
    p.add_argument("--arms", default="dp,dp3")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--obs-steps", type=int, default=2)
    p.add_argument("--points", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", type=pathlib.Path,
                   default=pathlib.Path("results/dp3_vs_dp.json"))
    a = p.parse_args()

    curves = {}
    for arm in a.arms.split(","):
        torch.manual_seed(a.seed)
        config = make_config(arm, a)
        fps = 30
        delta = {
            "observation.state": [i / fps for i in config.observation_delta_indices],
            "action": [i / fps for i in config.action_delta_indices],
        }
        for key in ((RGB_KEY,) if arm == "dp" else (DEPTH_KEY, INTRINSICS_KEY)):
            delta[key] = [i / fps for i in config.observation_delta_indices]

        dataset = LeRobotDataset(a.repo_id, root=a.root, delta_timestamps=delta)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True
        )
        print(f"\n=== {arm} === {len(dataset)} frames", flush=True)

        stats = dataset.meta.stats
        policy = (DP3Policy(config, dataset_stats=stats) if arm == "dp3"
                  else DiffusionPolicy(config, dataset_stats=stats))
        policy.train()
        n_par = sum(q.numel() for q in policy.parameters())
        optimiser = torch.optim.AdamW(policy.parameters(), lr=a.lr)
        step_fn = DepthToPointCloudStep(
            num_points=a.points, frame="camera", seed=a.seed, depth_scale=1e-3,
            workspace_centre=(0.0, 0.0, 0.67), workspace_extent=0.6,
        )

        history, t0, done = [], time.time(), 0
        while done < a.steps:
            for batch in loader:
                if done >= a.steps:
                    break
                loss, _ = policy.forward(prepare(batch, arm, step_fn))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                done += 1
                if done % a.log_every == 0 or done == 1:
                    history.append({"step": done, "loss": float(loss.detach()),
                                    "seconds": time.time() - t0})
                    print(f"  step {done:5d}  loss {float(loss):7.4f}  "
                          f"[{time.time()-t0:.0f}s]", flush=True)
        curves[arm] = {"params": n_par, "history": history}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(curves, indent=2))

    print("\nfinal loss")
    for arm, c in curves.items():
        print(f"  {arm:5s} {c['history'][-1]['loss']:.4f}   ({c['params']/1e6:.1f} M params)")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
