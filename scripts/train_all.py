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
from rvt_lerobot.models.policy import ARMS, MultiViewPolicy, policy_loss, proprio_for  # noqa: E402
from rvt_lerobot.render.rig import ALL_CAMERAS as R_ALL  # noqa: E402

AUG_THETA_DEG = 15.0


def fingerprint(arm: str, args) -> dict:
    """Everything that changes what a checkpoint MEANS, not just how good it is.

    A resume-by-skip mechanism is only safe while the definition of the data and
    the model is fixed. It was not: proprioception changed from joint angles to
    PerAct's low_dim_state partway through, and the old proprio checkpoint was
    silently kept because its history said 2500/2500. It then failed to load
    with a shape mismatch -- which was lucky, because a change that did NOT
    alter a tensor shape would have been kept and quietly averaged into the
    results.
    """
    return {
        "proprio_dim": 7 if ARMS[arm].full_proprio else 4,
        "image": args.image,
        "patch": args.patch,
        "per_segment": args.per_segment,
        "steps": args.steps,
        "batch": args.batch,
        # build_samples gained the home -> first-keypose transition partway
        # through. A checkpoint trained without it cannot be compared against
        # one trained with it, and the difference changes no tensor shape, so
        # nothing else would notice.
        "home_transition": True,
        "first_weight": args.first_weight,
    }


def save(run, model, arm, seed, args, n_par, history, step):
    run.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "arm": arm, "seed": seed,
         "image": args.image, "patch": args.patch, "step": step},
        run / "model.pt",
    )
    (run / "history.json").write_text(
        json.dumps({"arm": arm, "seed": seed, "params": n_par,
                    "steps_done": step, "steps_planned": args.steps,
                    "fingerprint": fingerprint(arm, args),
                    "history": history}, indent=2)
    )


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

    run = args.out / f"{arm}_s{seed}"
    history, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        batch = cache.batch(rng.integers(0, cache.n, size=args.batch))
        images = make_images(spec, batch, virtual_size=args.image)
        out = model(images, proprio_for(spec, batch), calib=batch)
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
                f"(median) grasp {m['trans_mm_median_grasp']:6.1f}  rot {m['rot_deg_median']:5.1f} deg  "
                f"grip {m['grip_acc']:.3f}  s@10 {m['success_10mm']:.2f}  [{time.time()-t0:.0f}s]",
                flush=True,
            )
            # Checkpoint at every evaluation, not only at the end. This machine
            # is shared and a run that is interrupted at step 4500 of 5000
            # should cost nothing.
            save(run, model, arm, seed, args, n_par, history, step)

    save(run, model, arm, seed, args, n_par, history, args.steps)
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
    p.add_argument("--first-weight", type=int, default=1,
                   help="repeat the home -> first-keypose transition this many times")
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()

    device = pick_device(allow_cpu=a.allow_cpu)
    arms = [x for x in a.arms.split(",") if x]
    seeds = [int(s) for s in a.seeds.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)

    scene = MultiCamScene(seed=0, image_size=a.image)
    train_eps, val_eps = C.load(a.episodes), C.load(a.val_episodes)
    train_samples = build_samples(train_eps, np.random.default_rng(0),
                                  per_segment=a.per_segment, first_weight=a.first_weight)
    val_samples = build_samples(val_eps, np.random.default_rng(12345), per_segment=1)

    def available_gb() -> float:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
        return float("nan")

    def render(eps, samples, cond, label):
        # Report memory around every render. The first attempt at this run was
        # killed by the OOM killer here, with no traceback -- the log simply
        # stopped -- because measurement scripts were running concurrently and
        # a 1.8 GB allocation on a 15.6 GB box with a unified-memory GPU does
        # not always get the page cache reclaimed in time. If it happens again,
        # these two numbers say so immediately.
        need = len(samples) * len(R_ALL) * a.image * a.image * 5 / 1e9
        before = available_gb()
        print(f"[{label}] rendering {len(samples)} samples, ~{need:.2f} GB needed, "
              f"{before:.2f} GB available", flush=True)
        if before < need * 1.6:
            print(f"  WARNING: only {before:.2f} GB available for a {need:.2f} GB "
                  f"cache; check nothing else is running", flush=True)
        t0 = time.time()
        arrays = render_condition(scene, eps, samples, cond)
        gb = sum(x.nbytes for x in arrays.values()) / 1e9
        print(f"[{label}] done: {gb:.2f} GB in {time.time()-t0:.0f}s, "
              f"{available_gb():.2f} GB still available", flush=True)
        return ConditionCache(arrays, device=device)

    val = render(val_eps, val_samples, Condition(seed=999), "val")

    # Two phases, nominal then augmented, so only one training cache is resident
    # at a time. Each is about 1.8 GB for ten thousand four-view RGBD frames and
    # this machine has roughly eight free, shared with the GPU; holding both
    # plus the model and its activations is how a ten-hour run dies at hour six.
    summary = []

    def phase(label, cond, phase_arms):
        nonlocal summary
        # Seed-major, not arm-major. If a long run has to be stopped, this
        # leaves every arm trained at the seeds that finished rather than some
        # arms at three seeds and others at none -- the first is a smaller
        # study, the second is not a study at all.
        todo = []
        for seed in seeds:
            for arm in phase_arms:
                hist = a.out / f"{arm}_s{seed}" / "history.json"
                if hist.is_file():
                    done = json.loads(hist.read_text())
                    finished = done.get("steps_done", 0) >= done.get("steps_planned", 0)
                    want = fingerprint(arm, a)
                    same = done.get("fingerprint") == want
                    if finished and same:
                        print(f"  skip {arm} seed {seed} (trained to {done['steps_done']})",
                              flush=True)
                        continue
                    if finished and not same:
                        print(f"  RETRAIN {arm} seed {seed}: it was trained under different "
                              f"settings ({done.get('fingerprint')} != {want})", flush=True)
                todo.append((arm, seed))
        if not todo:
            return
        cache = render(train_eps, train_samples, cond, label)
        for arm, seed in todo:
            print(f"  === {arm} seed {seed} ===", flush=True)
            final = train_one(arm, seed, cache, val, a, device)
            summary.append({"arm": arm, "seed": seed, **final})
            (a.out / "summary.json").write_text(json.dumps(summary, indent=2))
        del cache
        import gc

        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    phase("train/nominal", Condition(seed=0), [x for x in arms if not ARMS[x].camera_aug])
    phase(
        "train/camera-aug",
        Condition(theta_deg=AUG_THETA_DEG, theta_random=True, seed=0),
        [x for x in arms if ARMS[x].camera_aug],
    )
    print("done")


if __name__ == "__main__":
    main()
