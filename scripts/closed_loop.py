"""Run trained arms on the actual task, not just on the error metric.

A millimetre figure only matters if it converts into something a robot does or
fails to do. Here the policy drives the arm: at every step it observes the scene
through the rig under a stated condition, predicts the next keypose and gripper
state, and a shared low-level controller executes it. The policy chooses every
waypoint and decides when to close and when to open.

Two references run under the *identical* wrapper, because a policy's failure
means nothing without them:

* **oracle keyposes** -- the demonstration's own recorded keyposes, replayed
  through this executor on the same scene. If the oracle cannot place the block,
  the executor is broken and no policy number from it is worth reading.
  **Measured before any policy was run: oracle replay places 13/15 and picks
  15/15, against the expert's own 14/15.** The executor is therefore sound, and
  a policy that fails here fails on its predictions.
* **the scripted expert** -- its own full routine, which is the ceiling.

**This evaluation is not deterministic, and 30 episodes is not enough.** The
same checkpoint run twice in the same process, on scenes drawn from the same
seed, scored 0.47 and 0.53 picked; across four runs it scored 0.33 to 0.60.
Nothing in the harness is seeded wrongly -- the scene and its RNG are rebuilt
per arm -- but rendering and inference on the GPU are not bit-reproducible, and
this is a closed loop: a sub-millimetre difference in one predicted keypose
changes the contact that follows, which changes the next observation. Small
perturbations do not stay small.

The observed spread (sd 0.115 over four runs) is close to the binomial standard
error at n = 30 and p = 0.48, which is 0.091, so most of it is ordinary
sampling noise rather than drift. The consequence is the same either way: at
n = 30 a single number carries about +-9 points at one sigma, which is wider
than most differences worth reporting. **Use several hundred episodes, or
report a mean over repeats with an interval.** A single 30-episode run is a
diagnostic, not a result -- and it is easy to mistake a lucky draw for an
effect.

One thing the executor knows that the policy does not: the approach to a grasp
is made with the tool frame set to the midpoint of the OPEN jaw gap rather than
the TCP. Descending on the TCP drives the fixed pad down the side of the block
at tens of newtons before the jaws ever close. The offset uses the *nominal*
block half-width, not the episode's actual one, so it is information a deployed
system would have and is identical for every arm and both references.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rvt_lerobot.device import pick_device  # noqa: E402
from rvt_lerobot.data.batching import make_images  # noqa: E402
from rvt_lerobot.data.collect_study import GRIP_OPEN_THRESHOLD, run_episode  # noqa: E402
from rvt_lerobot.data.views import Condition  # noqa: E402
from rvt_lerobot.envs.multicam import MultiCamScene  # noqa: E402
from rvt_lerobot.evaluate import rot6d_to_matrix  # noqa: E402
from rvt_lerobot.models.policy import ARMS, MultiViewPolicy  # noqa: E402
from rvt_lerobot.render import rig as R  # noqa: E402
from rvt_lerobot.render.noise import apply_depth_noise  # noqa: E402
from rvt_lerobot.vendor.so101_expert import ScriptedExpert, _Waypoints  # noqa: E402
from rvt_lerobot.vendor.so101_scene import JAW_OPEN, JAW_SHUT, TABLE_TOP  # noqa: E402

#: Demonstrations run to 11 keyframes; headroom lets a policy waste a step
#: without letting one loop forever.
MAX_KEYPOSES = 14

#: Fraction of the episode elapsed at each keypose, averaged over the training
#: demonstrations. The policy's `low_dim_state` carries a progress term, and at
#: training time that term is the frame index over the episode length -- so the
#: values it saw at keypose 5 are near 0.67, not 5/14 = 0.36. Sending the naive
#: ratio puts every observation off-distribution on a feature the policy leans
#: on to tell which phase it is in, and it grasped nothing at all: 0/3 while the
#: oracle replaying the same executor scored 3/3.
KEYPOSE_PROGRESS = (
    0.147, 0.278, 0.331, 0.356, 0.512, 0.674, 0.792, 0.838, 0.881, 0.981, 0.997,
)

#: Mid-range of the domain randomisation's block half-width. Used for the
#: approach tool offset -- see the module docstring.
NOMINAL_HALF_WIDTH = 0.008

#: The fixed pad's inner face, and how far the pads reach below the gap
#: midpoint. Both are geometry of this gripper, measured in the source project.
FIXED_PAD_X = 0.0116
PAD_REACH = 0.0294


def approach_tool(scene) -> np.ndarray:
    """Tool frame for descending with the jaws open, biased toward the fixed pad."""
    gap = scene.open_gap_local.copy()
    gap[0] = FIXED_PAD_X - NOMINAL_HALF_WIDTH - 0.006
    return gap


class Executor:
    """Turns a stream of (position, yaw, gripper) keyposes into robot motion."""

    def __init__(self, scene) -> None:
        self.scene = scene
        self.peak = 0.0
        self._z0 = float(scene.block_pos()[2])
        self.way = _Waypoints(scene, self._track)
        self.jaw_open = True

    def _track(self, s) -> None:
        self.peak = max(self.peak, float(s.block_pos()[2]) - self._z0)

    def step(self, pos: np.ndarray, yaw: float, want_open: bool) -> None:
        s = self.scene
        if self.jaw_open and not want_open:
            grasp_z = max(TABLE_TOP + PAD_REACH + 0.002, float(pos[2]))
            target = np.array([pos[0], pos[1], grasp_z])
            self.way.move(target, jaw=JAW_OPEN, yaw=yaw, frames=50,
                          tool_local=approach_tool(s))
            self.way.set_jaw(JAW_SHUT, frames=30, settle=12)
            self.jaw_open = False
        elif not self.jaw_open and want_open:
            self.way.move(pos, jaw=JAW_SHUT, yaw=yaw, frames=45, settle=6)
            self.way.set_jaw(JAW_OPEN, frames=22, settle=10)
            self.jaw_open = True
        else:
            self.way.move(pos, jaw=JAW_OPEN if self.jaw_open else JAW_SHUT,
                          yaw=yaw, frames=45, settle=6)

    def outcome(self) -> dict:
        s = self.scene
        picked = self.peak > 0.03
        return {
            "picked": bool(picked),
            "placed": bool(picked and s.block_in_box() and not s.crushed),
            "peak_lift_mm": float(self.peak * 1000),
        }


def observe(scene: MultiCamScene, cond: Condition, rng, device: str) -> dict:
    """One batch-of-one observation under the stated condition."""
    obs = scene.capture()
    cams = R.ALL_CAMERAS
    reported = {c: obs[c].T for c in cams}
    if cond.eps_deg > 0:
        reported.update(
            R.miscalibrate({c: obs[c].T for c in R.MOVABLE_CAMERAS}, cond.eps_deg, rng)
        )
    depth = np.stack(
        [apply_depth_noise(obs[c].depth, cond.noise_c, rng, far=R.FAR) for c in cams]
    )[None]
    out = {
        "rgb": torch.from_numpy(np.stack([obs[c].rgb for c in cams])[None]),
        "depth_mm": torch.from_numpy(np.clip(depth * 1000, 0, 65535).astype(np.int32)),
        "K": torch.from_numpy(np.stack([obs[c].K for c in cams])[None].astype(np.float32)),
        "T": torch.from_numpy(np.stack([reported[c] for c in cams])[None].astype(np.float32)),
    }
    return {k: v.to(device) for k, v in out.items()}


def policy_rollout(scene, model, cond, rng, device, image, oracle_yaw: float | None = None) -> dict:
    """Drive one episode with the policy.

    `oracle_yaw` is a diagnostic, not a mode: substituting the grasp heading a
    perfect policy would use, while leaving translation to the policy, isolates
    how much of a closed-loop failure is rotation rather than position. This
    gripper is known to be unforgiving about heading -- its fixed fingertip sits
    11.9 mm off the tool centre across the closing axis, so a dozen degrees of
    yaw error lands the tip on top of the block instead of beside it.
    """
    s = scene.scene
    ex = Executor(s)
    for step in range(MAX_KEYPOSES):
        batch = observe(scene, cond, rng, device)
        # The same four numbers the dataset provides: jaw angle, jaw command,
        # gripper open, and progress through the episode.
        span = max(1e-6, JAW_OPEN - JAW_SHUT)
        jaw_q, jaw_cmd = float(s.data.qpos[5]), float(s.data.ctrl[5])
        proprio = torch.tensor([[
            (jaw_q - JAW_SHUT) / span,
            (jaw_cmd - JAW_SHUT) / span,
            float(jaw_cmd > GRIP_OPEN_THRESHOLD),
            KEYPOSE_PROGRESS[min(step, len(KEYPOSE_PROGRESS) - 1)],
        ]], dtype=torch.float32).to(device)
        with torch.no_grad():
            out = model(make_images(model.spec, batch, virtual_size=image), proprio, calib=batch)
        pos = out["pos"][0].double().cpu().numpy()
        rot = rot6d_to_matrix(out["rot6"])[0].double().cpu().numpy()
        # solve_ik aims the jaw's local X -- the closing axis -- along a heading,
        # which is exactly the first column of the predicted rotation
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
        if oracle_yaw is not None:
            yaw = oracle_yaw
        want_open = bool(out["grip_logit"][0].item() > 0)
        ex.step(pos, yaw, want_open)
        # Stop once the block is in the container and the jaws have been opened
        # there: the task is done and further keyposes can only knock it out.
        if s.block_in_box() and ex.jaw_open:
            break
    return ex.outcome()


def oracle_rollout(scene, episode, pos_noise_mm: float = 0.0, rng=None) -> dict:
    """Replay a demonstration's own keyposes through the same executor.

    With `pos_noise_mm` the keyposes are perturbed by isotropic Gaussian noise
    of that magnitude before execution. Sweeping it converts the study's
    millimetre metric into task terms: it answers how accurate a keypose
    prediction has to be before the jaws actually close on the block, using the
    same executor and the same scenes, with no policy in the loop to confound
    the answer.
    """
    ex = Executor(scene.scene)
    for k in episode.keyframes:
        rot = episode.jaw_rot[k]
        pos = episode.tcp[k].astype(float)
        if pos_noise_mm > 0:
            direction = rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            pos = pos + direction * (pos_noise_mm / 1000.0)
        ex.step(
            pos,
            float(np.arctan2(rot[1, 0], rot[0, 0])),
            bool(episode.jaw_cmd[k] > GRIP_OPEN_THRESHOLD),
        )
    return ex.outcome()


def snapshot_model(scene):
    m, ids = scene.model, scene.ids
    return (m.geom_size[ids.block_geom].copy(), float(m.body_mass[ids.block]),
            m.geom_friction[ids.block_geom].copy(), m.body_pos[ids.box].copy())


def restore_model(scene, snap):
    m, ids = scene.model, scene.ids
    m.geom_size[ids.block_geom], m.body_mass[ids.block] = snap[0], snap[1]
    m.geom_friction[ids.block_geom], m.body_pos[ids.box] = snap[2], snap[3]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=pathlib.Path, default=pathlib.Path("runs"))
    p.add_argument("--arms", default="rgb,xyz_real,rvt")
    p.add_argument("--seeds", default="0")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--image", type=int, default=96)
    p.add_argument("--conditions", default="0,0,0;20,0,0;0,5,0;0,0,0.008")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/closed_loop.json"))
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--oracle-yaw", action="store_true",
                   help="diagnostic: use the expert's grasp heading, policy translation")
    a = p.parse_args()

    device = pick_device(allow_cpu=a.allow_cpu)
    conds = []
    for triple in a.conditions.split(";"):
        t, e, c = (float(x) for x in triple.split(","))
        conds.append(Condition(theta_deg=t, eps_deg=e, noise_c=c, seed=31))

    rows: list[dict] = []
    a.out.parent.mkdir(parents=True, exist_ok=True)

    def record(arm, seed, cond, results):
        rows.append({
            "arm": arm, "seed": seed, "theta": cond.theta_deg, "eps": cond.eps_deg,
            "noise": cond.noise_c, "n": len(results),
            "picked": float(np.mean([r["picked"] for r in results])),
            "placed": float(np.mean([r["placed"] for r in results])),
        })
        print(f"  {arm:16s} s{seed:<2d} picked {rows[-1]['picked']:.2f} "
              f"placed {rows[-1]['placed']:.2f}", flush=True)
        a.out.write_text(json.dumps(rows, indent=1))

    # --- the two references, once: neither depends on the sensing condition ---
    scene = MultiCamScene(seed=900, image_size=a.image)
    expert = ScriptedExpert(scene.scene)
    print("references (condition-independent):", flush=True)
    exp_res, oracle_res = [], []
    for _ in range(a.episodes):
        ep = run_episode(None, expert)
        exp_res.append({"picked": ep.picked, "placed": ep.placed})
        snap = snapshot_model(scene.scene)
        scene.scene.reset(block_xy=(ep.meta["block_x"], ep.meta["block_y"]),
                          block_yaw=ep.meta["block_yaw"])
        restore_model(scene.scene, snap)
        oracle_res.append(oracle_rollout(scene, ep))
    base = Condition(seed=31)
    record("scripted_expert", -1, base, exp_res)
    record("oracle_keyposes", -1, base, oracle_res)

    for cond in conds:
        print(f"\n{cond.name}", flush=True)
        for arm in a.arms.split(","):
            for seed in (int(s) for s in a.seeds.split(",")):
                ckpt = a.runs / f"{arm}_s{seed}" / "model.pt"
                if not ckpt.is_file():
                    print(f"  missing {ckpt}")
                    continue
                blob = torch.load(ckpt, map_location=device, weights_only=False)
                model = MultiViewPolicy(
                    ARMS[arm], image_size=blob.get("image", a.image),
                    patch=blob.get("patch", 12),
                ).to(device)
                model.load_state_dict(blob["state_dict"])
                model.eval()
                scene = MultiCamScene(seed=900, image_size=a.image)
                rng = np.random.default_rng(31)
                res = []
                for _ in range(a.episodes):
                    scene.scene.reset()
                    scene.set_rig(R.perturb_rig(R.NOMINAL_RIG, cond.theta_deg, rng))
                    yaw = None
                    if a.oracle_yaw:
                        # the heading the expert would choose for this block
                        yaw = ScriptedExpert(scene.scene).choose_grasp_yaw(
                            scene.scene.block_pos(), scene.scene.block_yaw()
                        )[0]
                    res.append(policy_rollout(scene, model, cond, rng, device, a.image, yaw))
                record(arm + ("+oracle_yaw" if a.oracle_yaw else ""), seed, cond, res)

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
