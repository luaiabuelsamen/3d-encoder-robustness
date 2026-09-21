"""Collect pick-and-place demos in PerAct/RLBench format from SoArmRVTEnv.

Default "demo policy" is a hand-tuned joint-space trajectory good enough to
produce non-trivial multi-camera RGBD streams for testing the data pipeline.
Swap in your own teleop / IK / RL policy via `--policy <module.callable>`.
"""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
from typing import Callable

import numpy as np

from rvt_lerobot.envs.so_arm_rvt_env import SoArmRVTEnv, CAMERAS
from rvt_lerobot.data.rlbench_format import save_episode, extract_keyframes


def scripted_pick_and_place(env: SoArmRVTEnv) -> list[np.ndarray]:
    """Open-loop joint waypoints — illustrative, not optimized for success.

    Returns a list of actions (ctrl vectors) the env will execute in sequence.
    Six actuators on SO-ARM100: [Rotation, Pitch, Elbow, Wrist_Pitch, Wrist_Roll, Jaw].
    Last actuator is the gripper jaw; high ctrl = closed.
    """
    nu = env.model.nu
    home = env._home_ctrl.astype(np.float32).copy()

    block = env._block_pos()
    box = env._box_pos()
    block_yaw = float(np.arctan2(block[0], -block[1]))
    box_yaw = float(np.arctan2(box[0], -box[1]))

    def wp(rotation, pitch, elbow, wrist_p=0.0, wrist_r=0.0, jaw=-1.0):
        a = home.copy()
        if nu >= 1: a[0] = rotation
        if nu >= 2: a[1] = pitch
        if nu >= 3: a[2] = elbow
        if nu >= 4: a[3] = wrist_p
        if nu >= 5: a[4] = wrist_r
        if nu >= 6: a[5] = jaw
        return a

    waypoints = [
        wp(block_yaw, -0.4, 1.0, 0.5, 0.0, jaw=-1.0),   # above block, open
        wp(block_yaw, -0.9, 1.5, 0.8, 0.0, jaw=-1.0),   # descend
        wp(block_yaw, -0.9, 1.5, 0.8, 0.0, jaw=+1.0),   # close
        wp(block_yaw, -0.3, 1.0, 0.5, 0.0, jaw=+1.0),   # lift
        wp(box_yaw,   -0.3, 1.0, 0.5, 0.0, jaw=+1.0),   # rotate to box
        wp(box_yaw,   -0.7, 1.3, 0.7, 0.0, jaw=+1.0),   # descend over box
        wp(box_yaw,   -0.7, 1.3, 0.7, 0.0, jaw=-1.0),   # open / release
        wp(box_yaw,   -0.3, 1.0, 0.5, 0.0, jaw=-1.0),   # retreat
    ]
    steps_per_wp = 12
    actions: list[np.ndarray] = []
    prev = home.copy()
    for tgt in waypoints:
        for t in range(steps_per_wp):
            alpha = (t + 1) / steps_per_wp
            actions.append(prev * (1 - alpha) + tgt * alpha)
        prev = tgt
    return actions


def load_policy(spec: str | None) -> Callable[[SoArmRVTEnv], list[np.ndarray]]:
    if spec is None:
        return scripted_pick_and_place
    module_path, name = spec.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), name)


def collect(
    out_root: str | os.PathLike,
    num_episodes: int,
    language_goal: str = "put the block in the box",
    task: str = "put_block_in_box",
    policy_spec: str | None = None,
    seed: int = 0,
    keyframes_only: bool = False,
) -> None:
    out_root = Path(out_root).expanduser()
    eps_dir = out_root / "train" / task / "all_variations" / "episodes"
    eps_dir.mkdir(parents=True, exist_ok=True)

    env = SoArmRVTEnv()
    policy = load_policy(policy_spec)

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        obs_seq = [obs]
        actions = policy(env)
        for a in actions:
            obs, _, term, trunc, info = env.step(a)
            obs_seq.append(obs)
            if term or trunc:
                break

        if keyframes_only:
            kfs = extract_keyframes(obs_seq)
            obs_seq = [obs_seq[i] for i in kfs]

        ep_dir = eps_dir / f"episode{ep}"
        save_episode(ep_dir, obs_seq, CAMERAS, language_goal, variation_number=0)
        print(f"[ep {ep}] success={info.get('success', False)} frames={len(obs_seq)} -> {ep_dir}")

    env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/rvt_lerobot_demos", help="dataset root")
    p.add_argument("--num", type=int, default=3)
    p.add_argument("--task", default="put_block_in_box")
    p.add_argument("--lang", default="put the block in the box")
    p.add_argument("--policy", default=None, help="dotted path to policy callable; defaults to scripted")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keyframes-only", action="store_true")
    args = p.parse_args()
    collect(
        out_root=args.out,
        num_episodes=args.num,
        language_goal=args.lang,
        task=args.task,
        policy_spec=args.policy,
        seed=args.seed,
        keyframes_only=args.keyframes_only,
    )


if __name__ == "__main__":
    main()
