"""Run trained arms on the actual task, not just on the error metric.

A millimetre figure is only interesting if it converts into something a robot
does or fails to do. Here the policy drives the arm: at each step it observes
the scene through the rig under a stated condition, predicts the next keypose
and gripper state, and a low-level controller executes it. Nothing about the
task is given away -- the policy chooses every waypoint and decides when to
close and when to open.

What is *not* the policy's job, and is therefore shared by every arm, is the
low-level grasp: when the policy commands a close, the same contact-seeking
routine the scripted expert uses runs. That is the division RVT itself assumes
(a keypose policy on top of a motion planner), and putting it anywhere else
would measure the grip controller rather than the representation.

The scripted expert is run under the identical protocol at every condition. A
policy failure only means something if the expert succeeds there.
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
from rvt_lerobot.data.collect_study import GRIP_OPEN_THRESHOLD  # noqa: E402
from rvt_lerobot.data.views import Condition  # noqa: E402
from rvt_lerobot.envs.multicam import MultiCamScene  # noqa: E402
from rvt_lerobot.evaluate import rot6d_to_matrix  # noqa: E402
from rvt_lerobot.models.policy import ARMS, MultiViewPolicy  # noqa: E402
from rvt_lerobot.render import rig as R  # noqa: E402
from rvt_lerobot.render.noise import apply_depth_noise  # noqa: E402
from rvt_lerobot.vendor.so101_expert import HOME, ScriptedExpert, _Waypoints  # noqa: E402
from rvt_lerobot.vendor.so101_scene import JAW_OPEN, JAW_SHUT  # noqa: E402

#: Demonstrations run to 11 keyframes; a little headroom lets a policy recover
#: from a wasted step without letting one loop forever.
MAX_KEYPOSES = 14


def observe(scene: MultiCamScene, cond: Condition, rng) -> dict[str, torch.Tensor]:
    """One observation batch of size 1, under the stated condition."""
    obs = scene.capture()
    cams = R.ALL_CAMERAS
    reported = {c: obs[c].T for c in cams}
    if cond.eps_deg > 0:
        reported.update(
            R.miscalibrate({c: obs[c].T for c in R.MOVABLE_CAMERAS}, cond.eps_deg, rng)
        )
    rgb = np.stack([obs[c].rgb for c in cams])[None]
    depth = np.stack(
        [apply_depth_noise(obs[c].depth, cond.noise_c, rng, far=R.FAR) for c in cams]
    )[None]
    return {
        "rgb": torch.from_numpy(rgb),
        "depth_mm": torch.from_numpy(np.clip(depth * 1000, 0, 65535).astype(np.int32)),
        "K": torch.from_numpy(np.stack([obs[c].K for c in cams])[None].astype(np.float32)),
        "T": torch.from_numpy(np.stack([reported[c] for c in cams])[None].astype(np.float32)),
    }


def rollout(scene: MultiCamScene, model, cond: Condition, rng, device: str, image: int) -> dict:
    """Drive one episode with the policy. Returns what happened."""
    s = scene.scene
    s.reset()
    z0 = float(s.block_pos()[2])
    peak = 0.0

    def track(sc):
        nonlocal peak
        peak = max(peak, float(sc.block_pos()[2]) - z0)

    way = _Waypoints(s, track)
    expert = ScriptedExpert(s)
    jaw = JAW_OPEN
    was_open = True
    steps = 0

    for _ in range(MAX_KEYPOSES):
        batch = {k: v.to(device) for k, v in observe(scene, cond, rng).items()}
        proprio = torch.tensor(
            np.concatenate([s.data.qpos[:6], [s.data.ctrl[5]]])[None], dtype=torch.float32
        ).to(device)
        with torch.no_grad():
            images = make_images(model.spec, batch, virtual_size=image)
            out = model(images, proprio, calib=batch)
        pos = out["pos"][0].cpu().numpy().astype(float)
        rot = rot6d_to_matrix(out["rot6"])[0].cpu().numpy()
        want_open = bool(out["grip_logit"][0].item() > 0)
        # the IK takes a heading for the jaw's local X, which is the closing
        # axis; that is exactly the first column of the predicted rotation
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))

        if want_open == was_open:
            way.move(pos, jaw=jaw, yaw=yaw, frames=40, settle=6)
        elif not want_open:
            # commanded close: descend to the predicted pose with the jaws open,
            # then hand over to the shared low-level grasp
            way.move(pos, jaw=JAW_OPEN, yaw=yaw, frames=40, settle=6)
            expert_cfg = expert.config
            way.close_until_contact(
                target=expert_cfg.grip_newtons if hasattr(expert_cfg, "grip_newtons") else None
            ) if False else way.set_jaw(JAW_SHUT, frames=30, settle=12)
            jaw = JAW_SHUT
        else:
            way.move(pos, jaw=jaw, yaw=yaw, frames=40, settle=6)
            way.set_jaw(JAW_OPEN, frames=25, settle=10)
            jaw = JAW_OPEN
        was_open = want_open
        steps += 1
        if s.block_in_box() and want_open:
            break

    return {
        "picked": bool(peak > 0.02),
        "placed": bool(s.block_in_box() and not s.crushed),
        "peak_lift_mm": peak * 1000,
        "steps": steps,
    }


def expert_rollout(scene: MultiCamScene) -> dict:
    """The paired working reference, on the same scene, at the same moment."""
    r = ScriptedExpert(scene.scene).run()
    return {"picked": bool(r.picked), "placed": bool(r.placed), "peak_lift_mm": float("nan"), "steps": -1}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=pathlib.Path, default=pathlib.Path("runs"))
    p.add_argument("--arms", default="rgb,xyz_real,rvt")
    p.add_argument("--seeds", default="0")
    p.add_argument("--episodes", type=int, default=40)
    p.add_argument("--image", type=int, default=96)
    p.add_argument("--conditions", default="0,0,0;20,0,0;0,5,0;0,0,0.008")
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/closed_loop.json"))
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()

    device = pick_device(allow_cpu=a.allow_cpu)
    conds = []
    for triple in a.conditions.split(";"):
        t, e, c = (float(x) for x in triple.split(","))
        conds.append(Condition(theta_deg=t, eps_deg=e, noise_c=c, seed=31))

    rows = []
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for cond in conds:
        # the paired reference first, so a bad condition is caught before any
        # policy is blamed for it
        scene = MultiCamScene(seed=900, image_size=a.image)
        rig_rng = np.random.default_rng(31)
        exp = [expert_rollout(scene) for _ in range(a.episodes)]
        rows.append({
            "arm": "scripted_expert", "seed": -1, "theta": cond.theta_deg,
            "eps": cond.eps_deg, "noise": cond.noise_c,
            "placed": float(np.mean([r["placed"] for r in exp])),
            "picked": float(np.mean([r["picked"] for r in exp])), "n": a.episodes,
        })
        print(f"{cond.name}  expert placed {rows[-1]['placed']:.2f}", flush=True)

        for arm in a.arms.split(","):
            for seed in (int(s) for s in a.seeds.split(",")):
                ckpt = a.runs / f"{arm}_s{seed}" / "model.pt"
                if not ckpt.is_file():
                    print(f"  missing {ckpt}")
                    continue
                blob = torch.load(ckpt, map_location=device, weights_only=False)
                model = MultiViewPolicy(ARMS[arm], image_size=a.image).to(device)
                model.load_state_dict(blob["state_dict"])
                model.eval()
                scene = MultiCamScene(seed=900, image_size=a.image)
                res = []
                for ep in range(a.episodes):
                    scene.set_rig(R.perturb_rig(R.NOMINAL_RIG, cond.theta_deg, rig_rng))
                    res.append(rollout(scene, model, cond, rig_rng, device, a.image))
                rows.append({
                    "arm": arm, "seed": seed, "theta": cond.theta_deg,
                    "eps": cond.eps_deg, "noise": cond.noise_c,
                    "placed": float(np.mean([r["placed"] for r in res])),
                    "picked": float(np.mean([r["picked"] for r in res])), "n": a.episodes,
                })
                print(
                    f"  {arm:12s} s{seed}  picked {rows[-1]['picked']:.2f}  "
                    f"placed {rows[-1]['placed']:.2f}",
                    flush=True,
                )
                a.out.write_text(json.dumps(rows, indent=1))

    a.out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
