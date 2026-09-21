"""Collect scripted pick-and-place episodes as *replayable sim states*.

The dataset deliberately does not store images. It stores the physics state at
every keyframe plus the per-episode randomised model parameters, which is enough
to reproduce any frame exactly. Images are rendered on demand, which is what
makes the camera rig a free experimental variable: the same episodes can be
re-photographed from a perturbed rig, at a different resolution, or with a
different depth-noise model, without recollecting anything.

It is also 300x smaller. A 2000-episode dataset is about 4 MB.

Keyframes follow the RLBench convention used by PerAct and RVT: a frame is a
keyframe if the gripper changes state, or the arm has come to rest, and the
target for any observation is the *next* keyframe's end-effector pose. That is
the output space RVT was designed for, so it is the output space every arm in
this study predicts.
"""

from __future__ import annotations

import argparse
import pathlib
import time
from dataclasses import dataclass, field

import numpy as np

from ..vendor.so101_expert import ScriptedExpert
from ..vendor.so101_scene import JAW_OPEN, JAW_SHUT

#: A jaw command above this counts as "open". Midway between JAW_SHUT (-0.2)
#: and JAW_OPEN (0.55), well clear of both the force regulator's working range
#: and the closed stop.
GRIP_OPEN_THRESHOLD = 0.5 * (JAW_SHUT + JAW_OPEN)

#: Joint speed below which the arm counts as stopped, in radians per control
#: step. The expert's moves interpolate over 45 frames across ~1 rad, so a
#: moving joint travels ~0.02 rad/step; resting is two orders below that.
STOPPED_SPEED = 2e-3

#: Minimum gap between keyframes, in control steps. Without it the settle frames
#: at the end of every waypoint each register as their own keyframe.
MIN_KEYFRAME_GAP = 5


@dataclass
class Episode:
    """One scripted demonstration, stored as states rather than pictures."""

    qpos: np.ndarray            # (T, nq) float32 -- full physics state
    jaw_cmd: np.ndarray         # (T,) float32 -- commanded jaw angle
    tcp: np.ndarray             # (T, 3) float32 -- tool centre point, world
    jaw_rot: np.ndarray         # (T, 3, 3) float32 -- Fixed_Jaw orientation
    keyframes: np.ndarray       # (K,) int32 -- indices into the above
    block_size: np.ndarray      # (3,) randomised half-extents
    block_mass: float
    block_friction: np.ndarray  # (3,)
    box_pos: np.ndarray         # (3,) randomised container position
    picked: bool
    placed: bool
    meta: dict = field(default_factory=dict)


def extract_keyframes(jaw_cmd: np.ndarray, qpos_arm: np.ndarray) -> np.ndarray:
    """RLBench-style keyframe indices.

    A frame is a keyframe when the gripper's open/closed state differs from the
    previous frame, or when the arm is at rest and was not at rest recently, or
    at the end of the episode. Unlike the finite-difference heuristic this
    replaces, the gripper signal is the *command*, which is exactly observable
    and does not depend on contact.
    """
    t = len(jaw_cmd)
    if t == 0:
        return np.zeros(0, dtype=np.int32)
    open_state = jaw_cmd > GRIP_OPEN_THRESHOLD
    speed = np.linalg.norm(np.diff(qpos_arm, axis=0, prepend=qpos_arm[:1]), axis=1)
    stopped = speed < STOPPED_SPEED

    keys: list[int] = []
    for i in range(1, t):
        changed = open_state[i] != open_state[i - 1]
        came_to_rest = stopped[i] and not stopped[i - 1]
        if changed or came_to_rest or i == t - 1:
            if not keys or i - keys[-1] >= MIN_KEYFRAME_GAP:
                keys.append(i)
            elif changed:
                keys[-1] = i  # a gripper event always wins the slot
    return np.asarray(keys, dtype=np.int32)


def run_episode(_unused, expert: ScriptedExpert) -> Episode:
    """Run one scripted episode, recording everything needed to replay it."""
    scene = expert.scene
    qpos: list[np.ndarray] = []
    jaw: list[float] = []
    tcp: list[np.ndarray] = []
    rot: list[np.ndarray] = []

    def hook(s) -> None:
        qpos.append(s.data.qpos.copy())
        jaw.append(float(s.data.ctrl[5]))
        tcp.append(s.tcp())
        rot.append(s.data.xmat[s.ids.jaw].reshape(3, 3).copy())

    result = expert.run(hook=hook)

    m = scene.model
    ids = scene.ids
    qpos_arr = np.asarray(qpos, dtype=np.float32)
    jaw_arr = np.asarray(jaw, dtype=np.float32)
    return Episode(
        qpos=qpos_arr,
        jaw_cmd=jaw_arr,
        tcp=np.asarray(tcp, dtype=np.float32),
        jaw_rot=np.asarray(rot, dtype=np.float32),
        keyframes=extract_keyframes(jaw_arr, qpos_arr[:, :6]),
        block_size=m.geom_size[ids.block_geom].copy(),
        block_mass=float(m.body_mass[ids.block]),
        block_friction=m.geom_friction[ids.block_geom].copy(),
        box_pos=m.body_pos[ids.box].copy(),
        picked=bool(result.picked),
        placed=bool(result.placed),
        meta={
            "block_x": result.block_x,
            "block_y": result.block_y,
            "block_yaw": result.block_yaw,
            "grasp_yaw": result.grasp_yaw,
        },
    )


def collect(out: pathlib.Path, num_episodes: int, seed: int, require_placed: bool = True):
    """Collect `num_episodes` successful demonstrations and save them to `out`.

    Failed episodes are dropped by default: a behaviour-cloning dataset of
    demonstrations that did not work teaches the wrong thing, and the yield is
    high enough (~90%) that discarding is cheaper than any alternative.
    """
    # PickScene directly, not MultiCamScene: collection never renders, and
    # allocating EGL renderers it will not use both wastes memory and, on this
    # host, competes for the GPU context with whatever else is training.
    from ..vendor.so101_scene import PickScene

    scene = PickScene(seed=seed, randomise=True)
    expert = ScriptedExpert(scene)

    episodes: list[Episode] = []
    attempts = 0
    t0 = time.time()
    while len(episodes) < num_episodes:
        attempts += 1
        ep = run_episode(scene, expert)
        if require_placed and not ep.placed:
            continue
        if len(ep.keyframes) < 3:
            continue
        episodes.append(ep)
        if len(episodes) % 50 == 0:
            rate = len(episodes) / max(1e-9, time.time() - t0)
            print(
                f"  {len(episodes):5d}/{num_episodes}  yield={len(episodes)/attempts:.0%}  "
                f"{rate:.1f} ep/s  mean_keyframes={np.mean([len(e.keyframes) for e in episodes]):.1f}",
                flush=True,
            )

    out.parent.mkdir(parents=True, exist_ok=True)
    save(out, episodes)
    print(
        f"wrote {len(episodes)} episodes to {out} "
        f"({out.stat().st_size / 1e6:.1f} MB, yield {len(episodes)/attempts:.0%}, "
        f"{time.time()-t0:.0f} s)"
    )
    return episodes


def save(path: pathlib.Path, episodes: list[Episode]) -> None:
    """Save as a single npz with ragged episodes flattened plus an index."""
    lengths = np.array([len(e.qpos) for e in episodes], dtype=np.int32)
    kf_lengths = np.array([len(e.keyframes) for e in episodes], dtype=np.int32)
    np.savez_compressed(
        path,
        lengths=lengths,
        kf_lengths=kf_lengths,
        qpos=np.concatenate([e.qpos for e in episodes]),
        jaw_cmd=np.concatenate([e.jaw_cmd for e in episodes]),
        tcp=np.concatenate([e.tcp for e in episodes]),
        jaw_rot=np.concatenate([e.jaw_rot for e in episodes]).reshape(-1, 9),
        keyframes=np.concatenate([e.keyframes for e in episodes]),
        block_size=np.stack([e.block_size for e in episodes]),
        block_mass=np.array([e.block_mass for e in episodes], dtype=np.float32),
        block_friction=np.stack([e.block_friction for e in episodes]),
        box_pos=np.stack([e.box_pos for e in episodes]),
        picked=np.array([e.picked for e in episodes]),
        placed=np.array([e.placed for e in episodes]),
    )


def load(path: pathlib.Path) -> list[Episode]:
    """Read a dataset back into Episodes.

    Every field is decompressed **once**, before the loop. Indexing an NpzFile
    returns a freshly decompressed array each time, and the per-episode slices
    are views that keep their base alive -- so doing `z["qpos"][a:b]` inside the
    loop retains one full copy of every array per episode. On a 500-episode set
    that is about 8 GB, which is what the OOM killer took the first time this
    run was launched, silently and with no traceback. Slicing a single resident
    array instead costs one copy in total.
    """
    z = np.load(path)
    lengths, kf_lengths = z["lengths"], z["kf_lengths"]
    qpos, jaw_cmd, tcp = z["qpos"], z["jaw_cmd"], z["tcp"]
    jaw_rot, keyframes = z["jaw_rot"], z["keyframes"]
    block_size, block_mass = z["block_size"], z["block_mass"]
    block_friction, box_pos = z["block_friction"], z["box_pos"]
    picked, placed = z["picked"], z["placed"]

    offs = np.concatenate([[0], np.cumsum(lengths)])
    koffs = np.concatenate([[0], np.cumsum(kf_lengths)])
    out = []
    for i in range(len(lengths)):
        a, b = offs[i], offs[i + 1]
        ka, kb = koffs[i], koffs[i + 1]
        out.append(
            Episode(
                qpos=qpos[a:b],
                jaw_cmd=jaw_cmd[a:b],
                tcp=tcp[a:b],
                jaw_rot=jaw_rot[a:b].reshape(-1, 3, 3),
                keyframes=keyframes[ka:kb],
                block_size=block_size[i],
                block_mass=float(block_mass[i]),
                block_friction=block_friction[i],
                box_pos=box_pos[i],
                picked=bool(picked[i]),
                placed=bool(placed[i]),
            )
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--num", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-failures", action="store_true")
    a = p.parse_args()
    collect(a.out, a.num, a.seed, require_placed=not a.keep_failures)


if __name__ == "__main__":
    main()
