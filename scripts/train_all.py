"""Train every arm at every seed in one process, sharing the rendered data.

Rendering the training set is not free (about five minutes for ten thousand
four-view RGBD frames), and every arm consumes the *same* pixels, so rendering
it once per run would spend hours re-photographing identical scenes and would
also need two gigabytes of disk this machine does not have. One process renders
the nominal condition once, the augmented condition once, and then trains each
arm from RAM.

Sharing the cache also removes a confound for free: every arm and every seed
sees byte-identical observations, so a difference between arms cannot be a
difference in which camera jitter they happened to draw.

Everything runs in float32. Mixed precision would roughly double throughput,
but the quantity under study is a millimetre-scale position derived from
metre-scale world coordinates, and float16's mantissa gives about 3e-3 relative
precision there -- the same order as the effects being measured. A numerical
artefact that looked like a finding would be worse than a slow run.
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

AUG_THETA_DEG = 15.0


def train_one(arm, seed, cache, val, args, device):
    spec = ARMS[arm]
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    model = MultiViewPolicy(spec, image_size=args.image, patch=args.patch).to(device)
    n_par = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05
    )

    history, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        batch = cache.batch(rng.integers(0, cache.n, size=args.batch))
        images = make_images(spec, batch, virtual_size=args.image)
        out = model(images, batch["proprio"], calib=batch)
        loss, _ = policy_loss(out, batch, spec, model)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(model, val, virtual_size=args.image)
            m.update(step=step, loss=float(loss), seconds=time.time() - t0)
            history.append(m)
            print(
                f"    step {step:5d}  loss {float(loss):7.4f}  val {m['trans_mm_median']:6.1f} mm "
                f"(median) rot {m['rot_deg_median']:5.1f} deg  grip {m['grip_acc']:.3f}  "
                f"s@10 {m['success_10mm']:.2f}  [{time.time()-t0:.0f}s]",
                flush=True,
            )

    run = args.out / f"{arm}_s{seed}"
    run.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "arm": arm, "seed": seed,
         "image": args.image, "patch": args.patch},
        run / "model.pt",
    )
    (run / "history.json").write_text(
        json.dumps({"arm": arm, "seed": seed, "params": n_par,
                    "steps": args.steps, "history": history}, indent=2)
    )
    return history[-1] if history else {}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=pathlib.Path, default=pathlib.Path("data/train.npz"))
    p.add_argument("--val-episodes", type=pathlib.Path, default=pathlib.Path("data/val.npz"))
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs"))
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--image", type=int, default=96)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--per-segment", type=int, default=2)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()

    device = pick_device(allow_cpu=a.allow_cpu)
    arms = [x for x in a.arms.split(",") if x]
    seeds = [int(s) for s in a.seeds.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)

    scene = MultiCamScene(seed=0, image_size=a.image)
    train_eps, val_eps = C.load(a.episodes), C.load(a.val_episodes)
    train_samples = build_samples(train_eps, np.random.default_rng(0), per_segment=a.per_segment)
    val_samples = build_samples(val_eps, np.random.default_rng(12345), per_segment=1)

    def render(eps, samples, cond, label):
        t0 = time.time()
        arrays = render_condition(scene, eps, samples, cond)
        gb = sum(x.nbytes for x in arrays.values()) / 1e9
        print(f"[{label}] {len(samples)} samples, {gb:.2f} GB, {time.time()-t0:.0f}s", flush=True)
        return ConditionCache(arrays, device=device)

    val = render(val_eps, val_samples, Condition(seed=999), "val")
    nominal = render(train_eps, train_samples, Condition(seed=0), "train/nominal")
    augmented = None
    if any(ARMS[x].camera_aug for x in arms):
        augmented = render(
            train_eps, train_samples,
            Condition(theta_deg=AUG_THETA_DEG, theta_random=True, seed=0),
            "train/camera-aug",
        )

    summary = []
    for arm in arms:
        cache = augmented if ARMS[arm].camera_aug else nominal
        for seed in seeds:
            if (a.out / f"{arm}_s{seed}" / "model.pt").is_file():
                print(f"  skip {arm} seed {seed} (already trained)", flush=True)
                continue
            print(f"  === {arm} seed {seed} ===", flush=True)
            final = train_one(arm, seed, cache, val, a, device)
            summary.append({"arm": arm, "seed": seed, **final})
            (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("done")


if __name__ == "__main__":
    main()
