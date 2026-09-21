"""Evaluate every trained arm across the stress grid.

Each condition is rendered once and then handed to every checkpoint, so all
arms are scored on pixel-identical observations. That matters more than it
sounds: the camera perturbation and the depth noise are stochastic, and
re-rolling them per arm would put a random-seed difference inside every
comparison the study makes.

Three axes, run independently rather than as a full product, plus one small
two-dimensional slice where the two 3D-relevant axes are expected to trade:

  1. extrinsic shift with recalibration -- the cameras move, the policy is told
     where they went. This is the axis RVT is supposed to win.
  2. calibration error -- the cameras have not moved, but the extrinsics the
     policy is handed are wrong. Nothing an RGB policy consumes changes here at
     all; everything a 3D policy consumes does.
  3. depth noise -- a stereo sensor's sigma_z = c z^2 with holes and flying
     pixels at edges.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rvt_lerobot.device import pick_device  # noqa: E402
from rvt_lerobot.data import collect_study as C  # noqa: E402
from rvt_lerobot.data.batching import ConditionCache  # noqa: E402
from rvt_lerobot.data.views import Condition, build_samples, render_condition  # noqa: E402
from rvt_lerobot.envs.multicam import MultiCamScene  # noqa: E402
from rvt_lerobot.evaluate import evaluate  # noqa: E402
from rvt_lerobot.models.policy import ARMS, MultiViewPolicy  # noqa: E402

THETAS = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0)
EPSILONS = (0.0, 1.0, 2.0, 5.0, 10.0)
NOISES = (0.0, 0.0005, 0.001, 0.002, 0.004, 0.008)

#: The two-dimensional slice. Coarser than the 1D sweeps because it is a
#: product, and the question it answers -- where does the ordering between an
#: RGB policy and a 3D policy change hands -- needs a boundary, not a surface.
CROSS_THETAS = (0.0, 10.0, 20.0, 30.0)
CROSS_NOISES = (0.0, 0.002, 0.008)


def grid(axes: str) -> list[Condition]:
    conds: list[Condition] = []
    if "theta" in axes:
        conds += [Condition(theta_deg=t, seed=7) for t in THETAS]
    if "eps" in axes:
        conds += [Condition(eps_deg=e, seed=7) for e in EPSILONS if e > 0]
    if "noise" in axes:
        conds += [Condition(noise_c=c, seed=7) for c in NOISES if c > 0]
    if "cross" in axes:
        conds += [
            Condition(theta_deg=t, noise_c=c, seed=7)
            for t in CROSS_THETAS
            for c in CROSS_NOISES
            if not (t == 0.0 and c == 0.0)
        ]
    seen, out = set(), []
    for c in conds:
        if c.name not in seen:
            seen.add(c.name)
            out.append(c)
    return out


def load_models(runs: pathlib.Path, device: str, image: int, require_complete: bool = True):
    """Load finished checkpoints only.

    Training checkpoints at every evaluation so a long run survives being
    interrupted, which means a directory can hold a half-trained model that
    looks exactly like a finished one. Scoring those alongside completed arms
    would silently compare 500 steps against 2500 and read as an architecture
    difference.
    """
    models = []
    for d in sorted(runs.iterdir()):
        ckpt = d / "model.pt"
        if not ckpt.is_file():
            continue
        hist = d / "history.json"
        if require_complete and hist.is_file():
            done = json.loads(hist.read_text())
            if done.get("steps_done", 0) < done.get("steps_planned", 1):
                print(f"  skipping {d.name}: only {done.get('steps_done')} of "
                      f"{done.get('steps_planned')} steps", flush=True)
                continue
        blob = torch.load(ckpt, map_location=device, weights_only=False)
        spec = ARMS[blob["arm"]]
        m = MultiViewPolicy(
            spec, image_size=blob.get("image", image), patch=blob.get("patch", 12)
        ).to(device)
        try:
            m.load_state_dict(blob["state_dict"])
        except RuntimeError as exc:
            # A checkpoint left over from an earlier configuration. Name it and
            # move on: losing one arm is recoverable, losing the whole grid an
            # hour in is not.
            print(f"  SKIPPING {d.name}: {str(exc).splitlines()[-1].strip()}", flush=True)
            continue
        m.eval()
        models.append((blob["arm"], blob["seed"], m))
    return models


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=pathlib.Path, required=True)
    p.add_argument("--runs", type=pathlib.Path, default=pathlib.Path("runs"))
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/grid.json"))
    p.add_argument("--image", type=int, default=96)
    p.add_argument("--axes", default="theta,eps,noise,cross")
    p.add_argument("--limit-episodes", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--include-partial", action="store_true",
                   help="also score checkpoints that have not finished training")
    a = p.parse_args()

    dev = pick_device(allow_cpu=a.allow_cpu)
    eps = C.load(a.episodes)
    if a.limit_episodes:
        eps = eps[: a.limit_episodes]
    samples = build_samples(eps, np.random.default_rng(2024), per_segment=1)
    scene = MultiCamScene(seed=4242, image_size=a.image)

    models = load_models(a.runs, dev, a.image, require_complete=not a.include_partial)
    if not models:
        sys.exit(f"no checkpoints under {a.runs}")
    print(f"{len(models)} checkpoints, {len(samples)} test samples from {len(eps)} episodes")

    conditions = grid(a.axes)
    rows = []
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for i, cond in enumerate(conditions, 1):
        t0 = time.time()
        cache = ConditionCache(render_condition(scene, eps, samples, cond), device=dev)
        trender = time.time() - t0
        for arm, seed, model in models:
            m = evaluate(model, cache, virtual_size=a.image)
            rows.append(
                {"arm": arm, "seed": seed, "theta": cond.theta_deg,
                 "eps": cond.eps_deg, "noise": cond.noise_c, **m}
            )
        best = sorted(
            {r["arm"] for r in rows if r["theta"] == cond.theta_deg
             and r["eps"] == cond.eps_deg and r["noise"] == cond.noise_c},
            key=lambda arm: np.mean([
                r["trans_mm_median"] for r in rows
                if r["arm"] == arm and r["theta"] == cond.theta_deg
                and r["eps"] == cond.eps_deg and r["noise"] == cond.noise_c]),
        )
        print(
            f"[{i:2d}/{len(conditions)}] {cond.name:28s} render {trender:4.0f}s  "
            f"best: {', '.join(best[:3])}",
            flush=True,
        )
        a.out.write_text(json.dumps(rows, indent=1))
        del cache

    print(f"wrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
