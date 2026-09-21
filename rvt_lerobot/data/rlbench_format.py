"""PerAct/RLBench on-disk episode writer.

Emits an `episodeN/` directory consumable by `peract_colab.rlbench.utils.get_stored_demo`:

    episodeN/
      low_dim_obs.pkl
      variation_number.pkl
      variation_descriptions.pkl
      front_rgb/{i}.png            front_depth/{i}.png
      left_shoulder_rgb/...        left_shoulder_depth/...
      right_shoulder_rgb/...       right_shoulder_depth/...
      wrist_rgb/...                wrist_depth/...

Depth is stored as a 24-bit RGB-encoded PNG (RLBench convention): a float in
[0, 1] is packed across R/G/B at 8 bits each, then the loader unpacks and
rescales by (far - near).
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

DEPTH_SCALE = 2**24 - 1


def float_array_to_rgb_image(depth_m: np.ndarray, near: float, far: float) -> Image.Image:
    """Pack float depth (meters) -> 24-bit RGB PNG (RLBench format)."""
    norm = np.clip((depth_m - near) / max(1e-9, (far - near)), 0.0, 1.0)
    coded = (norm * DEPTH_SCALE).astype(np.uint32)
    r = ((coded >> 16) & 0xFF).astype(np.uint8)
    g = ((coded >> 8) & 0xFF).astype(np.uint8)
    b = (coded & 0xFF).astype(np.uint8)
    return Image.fromarray(np.stack([r, g, b], axis=-1), mode="RGB")


@dataclass
class StoredObservation:
    """Mirrors the subset of rlbench Observation fields the RVT loader reads."""
    joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    gripper_open: float = 1.0
    gripper_pose: np.ndarray = field(default_factory=lambda: np.zeros(7, dtype=np.float32))
    gripper_joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    gripper_touch_forces: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float32))
    misc: dict[str, Any] = field(default_factory=dict)
    variation_number: int = 0
    # raw rgb/depth left out of the pickle (saved as PNGs)


def obs_to_stored(obs: dict, cameras: tuple[str, ...]) -> StoredObservation:
    """Convert env-dict observation to a StoredObservation (no images in pickle)."""
    misc: dict[str, Any] = {}
    for c in cameras:
        misc[f"{c}_camera_intrinsics"] = obs[f"{c}_intrinsics"].astype(np.float32)
        misc[f"{c}_camera_extrinsics"] = obs[f"{c}_extrinsics"].astype(np.float32)
        misc[f"{c}_camera_near"] = float(obs[f"{c}_near"])
        misc[f"{c}_camera_far"] = float(obs[f"{c}_far"])
    return StoredObservation(
        joint_positions=obs["joint_positions"].astype(np.float32),
        joint_velocities=np.zeros_like(obs["joint_positions"], dtype=np.float32),
        gripper_open=float(obs["gripper_open"]),
        gripper_pose=obs["gripper_pose"].astype(np.float32),
        gripper_joint_positions=np.array([0.0, 0.0], dtype=np.float32),
        misc=misc,
    )


def save_episode(
    episode_dir: str | os.PathLike,
    obs_seq: list[dict],
    cameras: tuple[str, ...],
    language_goal: str,
    variation_number: int = 0,
) -> None:
    """Write one episode in PerAct/RLBench format to `episode_dir`.

    obs_seq: list of per-step env observation dicts (output of SoArmRVTEnv._get_obs).
    """
    out = Path(episode_dir)
    out.mkdir(parents=True, exist_ok=True)
    for c in cameras:
        (out / f"{c}_rgb").mkdir(exist_ok=True)
        (out / f"{c}_depth").mkdir(exist_ok=True)

    stored = []
    for i, obs in enumerate(obs_seq):
        for c in cameras:
            rgb = obs[f"{c}_rgb"]
            depth = obs[f"{c}_depth"]
            near, far = float(obs[f"{c}_near"]), float(obs[f"{c}_far"])
            Image.fromarray(rgb).save(out / f"{c}_rgb" / f"{i}.png")
            float_array_to_rgb_image(depth, near, far).save(out / f"{c}_depth" / f"{i}.png")
        s = obs_to_stored(obs, cameras)
        s.variation_number = int(variation_number)
        stored.append(s)

    with open(out / "low_dim_obs.pkl", "wb") as f:
        pickle.dump(stored, f)
    with open(out / "variation_number.pkl", "wb") as f:
        pickle.dump(int(variation_number), f)
    with open(out / "variation_descriptions.pkl", "wb") as f:
        pickle.dump([language_goal], f)


def extract_keyframes(obs_seq: list[dict], stopped_buffer: int = 4, stop_eps: float = 5e-3) -> list[int]:
    """RLBench-style keyframe indices: gripper state changes + near-zero joint velocity points.

    Approximation: we don't have joint_velocities in the observation, so we use
    finite differences on `joint_positions` to estimate "stopped" frames.
    Always include the last frame.
    """
    if not obs_seq:
        return []
    qs = np.stack([o["joint_positions"] for o in obs_seq])
    grippers = np.array([o["gripper_open"] for o in obs_seq])
    dq = np.linalg.norm(np.diff(qs, axis=0, prepend=qs[:1]), axis=1)
    stopped = dq < stop_eps

    keyframes = []
    last_gripper = grippers[0]
    stop_count = 0
    for i in range(len(obs_seq)):
        if stopped[i]:
            stop_count += 1
        else:
            stop_count = 0
        gripper_changed = abs(grippers[i] - last_gripper) > 0.5
        long_stop = stop_count >= stopped_buffer
        is_last = i == len(obs_seq) - 1
        if gripper_changed or long_stop or is_last:
            if not keyframes or i - keyframes[-1] > 1:
                keyframes.append(i)
            last_gripper = grippers[i]
            stop_count = 0
    return keyframes
