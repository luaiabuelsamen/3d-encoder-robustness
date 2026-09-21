"""Where does a training step actually go, and what configuration is affordable?

The first measured step cost was 650 ms for a 2.9 M-parameter transformer on
96px images, about thirteen times what the arithmetic predicts. Twenty-four
training runs at that rate is a day and a half, so before shrinking anything on
a hunch this measures the split -- data, input construction, forward, backward --
and then sweeps the two knobs that change it.

Run it with nothing else on the GPU; this machine is shared.
"""

from __future__ import annotations

import argparse
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
from rvt_lerobot.models.policy import ARMS, MultiViewPolicy, policy_loss, proprio_for  # noqa: E402


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def time_step(model, cache, spec, batch_size, image, device, iters=12):
    """Return (total, data, images, forward, backward) milliseconds per step."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    rng = np.random.default_rng(0)
    parts = np.zeros(4)
    for i in range(iters + 3):
        t0 = time.time()
        batch = cache.batch(rng.integers(0, cache.n, size=batch_size))
        sync(device)
        t1 = time.time()
        images = make_images(spec, batch, virtual_size=image)
        sync(device)
        t2 = time.time()
        out = model(images, proprio_for(spec, batch), calib=batch)
        loss, _ = policy_loss(out, batch, spec, model)
        sync(device)
        t3 = time.time()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sync(device)
        t4 = time.time()
        if i >= 3:  # warmup
            parts += np.array([t1 - t0, t2 - t1, t3 - t2, t4 - t3])
    parts = parts / iters * 1000
    return parts.sum(), *parts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=pathlib.Path, default=pathlib.Path("data/val.npz"))
    p.add_argument("--arms", default="rgb,rvt")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--image", type=int, default=96)
    p.add_argument("--allow-cpu", action="store_true")
    a = p.parse_args()

    device = pick_device(allow_cpu=a.allow_cpu)
    torch.backends.cudnn.benchmark = True

    scene = MultiCamScene(seed=0, image_size=a.image)
    eps = C.load(a.episodes)[:12]
    samples = build_samples(eps, np.random.default_rng(0), per_segment=1)
    cache = ConditionCache(render_condition(scene, eps, samples, Condition()), device=device)
    print(f"{cache.n} samples cached at {a.image}px on {device}\n")

    print(f"{'arm':12s} {'patch':>5s} {'dim':>4s} {'dep':>4s} {'tok':>5s} {'bs':>4s} "
          f"{'total':>8s} {'data':>7s} {'images':>7s} {'fwd':>7s} {'bwd':>7s}")
    for arm in a.arms.split(","):
        spec = ARMS[arm]
        # Two knobs, and a batch sweep, because a step this small on this GPU is
        # more likely to be bound by kernel-launch latency than by arithmetic --
        # in which case a bigger batch is nearly free and is the whole answer.
        for patch, dim, depth, bs in (
            (12, 192, 6, a.batch), (16, 192, 6, a.batch), (16, 192, 4, a.batch),
            (16, 192, 6, a.batch * 2), (16, 192, 6, a.batch * 4),
        ):
            torch.manual_seed(0)
            model = MultiViewPolicy(
                spec, image_size=a.image, patch=patch, dim=dim, depth=depth,
                heads=8 if dim % 8 == 0 else 6,
            ).to(device)
            g = a.image // patch
            tok = spec.n_views * g * g + 2
            try:
                total, d, im, fw, bw = time_step(model, cache, spec, bs, a.image, device)
            except RuntimeError as e:
                print(f"{arm:12s} {patch:5d} {dim:4d} {depth:4d} {tok:5d} {bs:4d}  FAILED {e}")
                continue
            print(f"{arm:12s} {patch:5d} {dim:4d} {depth:4d} {tok:5d} {bs:4d} "
                  f"{total:7.1f}ms {d:6.1f}ms {im:6.1f}ms {fw:6.1f}ms {bw:6.1f}ms "
                  f"{total/bs:6.2f}ms/sample", flush=True)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
