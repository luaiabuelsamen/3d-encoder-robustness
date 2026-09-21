"""Train one arm, under one training condition, with one seed.

Every arm gets the identical recipe -- same backbone size, optimiser, schedule,
batch size, and number of gradient steps. The only thing the command line
changes is which arm, which seed, and (for the augmented arms) whether the
training cameras move. If a result in this study depends on a hyperparameter,
it is not a result.
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
from rvt_lerobot.data.batching import ConditionCache, make_images  # noqa: E402
from rvt_lerobot.data.views import Condition, build_samples, render_condition  # noqa: E402
from rvt_lerobot.envs.multicam import MultiCamScene  # noqa: E402
from rvt_lerobot.evaluate import evaluate  # noqa: E402
from rvt_lerobot.models.policy import ARMS, MultiViewPolicy, policy_loss  # noqa: E402

#: Training-camera perturbation for the augmented arms: a fresh theta drawn
#: uniformly up to this, per episode. Chosen to cover the evaluation grid's
#: lower half, so the augmented arms are being asked to interpolate in the
#: regime they were trained for and extrapolate beyond it.
AUG_THETA_DEG = 15.0


def render_cache(scene, episodes, samples, condition, device, label):
    t0 = time.time()
    arrays = render_condition(scene, episodes, samples, condition)
    gb = sum(a.nbytes for a in arrays.values()) / 1e9
    print(f"  [{label}] {len(samples)} samples, {gb:.2f} GB, {time.time()-t0:.0f} s", flush=True)
    return ConditionCache(arrays, device=device)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True, choices=list(ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=pathlib.Path, required=True)
    p.add_argument("--val-episodes", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs"))
    p.add_argument("--steps", type=int, default=8000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--image", type=int, default=96)
    p.add_argument("--per-segment", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()

    spec = ARMS[a.arm]
    dev = pick_device(allow_cpu=a.allow_cpu)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    run = a.out / f"{a.arm}_s{a.seed}"
    run.mkdir(parents=True, exist_ok=True)
    print(f"=== {a.arm} seed {a.seed} -> {run} ===", flush=True)

    scene = MultiCamScene(seed=a.seed, image_size=a.image)
    train_eps = C.load(a.episodes)
    val_eps = C.load(a.val_episodes)
    rng = np.random.default_rng(a.seed)
    train_samples = build_samples(train_eps, rng, per_segment=a.per_segment)
    val_samples = build_samples(val_eps, np.random.default_rng(12345), per_segment=1)

    train_cond = Condition(
        theta_deg=AUG_THETA_DEG if spec.camera_aug else 0.0,
        theta_random=spec.camera_aug,
        seed=a.seed,
    )
    train = render_cache(scene, train_eps, train_samples, train_cond, dev, "train")
    val = render_cache(scene, val_eps, val_samples, Condition(seed=999), dev, "val")

    model = MultiViewPolicy(spec, image_size=a.image).to(dev)
    n_par = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05
    )
    print(f"  {n_par/1e6:.2f}M parameters, {train.n} train / {val.n} val samples", flush=True)

    history = []
    t0 = time.time()
    for step in range(1, a.steps + 1):
        idx = rng.integers(0, train.n, size=a.batch)
        batch = train.batch(idx)
        images = make_images(spec, batch, virtual_size=a.image)
        out = model(images, batch["proprio"], calib=batch)
        loss, parts = policy_loss(out, batch, spec, model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % a.eval_every == 0 or step == a.steps:
            m = evaluate(model, val, virtual_size=a.image)
            m.update(step=step, loss=float(loss), seconds=time.time() - t0)
            history.append(m)
            print(
                f"  step {step:6d}  loss {float(loss):7.4f}  "
                f"val trans {m['trans_mm_median']:6.1f} mm (median) "
                f"{m['trans_mm_mean']:6.1f} (mean)  rot {m['rot_deg_median']:5.1f} deg  "
                f"grip {m['grip_acc']:.3f}  s@10mm {m['success_10mm']:.3f}  "
                f"[{time.time()-t0:.0f}s]",
                flush=True,
            )

    torch.save(
        {"state_dict": model.state_dict(), "arm": a.arm, "seed": a.seed, "image": a.image},
        run / "model.pt",
    )
    (run / "history.json").write_text(json.dumps({"args": vars(a) | {
        "episodes": str(a.episodes), "val_episodes": str(a.val_episodes), "out": str(a.out)},
        "params": n_par, "history": history}, indent=2))
    print(f"  saved {run/'model.pt'}  ({time.time()-t0:.0f} s total)", flush=True)


if __name__ == "__main__":
    main()
