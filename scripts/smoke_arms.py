"""Every arm must forward, backward, and overfit a handful of samples.

An arm that cannot drive its training loss to near zero on 32 examples has a
wiring bug, not a learning problem, and finding that out here costs a minute
instead of an afternoon. The geometric decoders are the ones at risk: they
combine views through calibration, and a transposed rotation or a swapped image
axis still produces plausible-looking numbers.
"""

from __future__ import annotations

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
from rvt_lerobot.models.policy import ARMS, MultiViewPolicy, policy_loss  # noqa: E402

IMAGE = 96
N_EPISODES = 4
STEPS = 120


def main() -> int:
    dev = pick_device(allow_cpu=bool(int(__import__("os").environ.get("RVT_ALLOW_CPU", "0"))))
    torch.manual_seed(0)

    scene = MultiCamScene(seed=0, image_size=IMAGE)
    from rvt_lerobot.vendor.so101_expert import ScriptedExpert

    expert = ScriptedExpert(scene.scene)
    eps = []
    while len(eps) < N_EPISODES:
        e = C.run_episode(scene, expert)
        if e.placed and len(e.keyframes) >= 3:
            eps.append(e)

    rng = np.random.default_rng(0)
    samples = build_samples(eps, rng, per_segment=1)[:32]
    arrays = render_condition(scene, eps, samples, Condition())
    cache = ConditionCache(arrays, device=dev)
    print(f"{len(samples)} samples, image {IMAGE}px, device {dev}\n")

    idx = torch.arange(len(samples))
    batch = cache.batch(idx)
    spread = batch["target_pos"].std(0).cpu().numpy()
    print(f"target spread (mm): {np.round(spread * 1000, 1)}\n")

    fails = []
    for name, spec in ARMS.items():
        torch.manual_seed(0)
        model = MultiViewPolicy(spec, image_size=IMAGE, patch=16).to(dev)
        n_par = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

        images = make_images(spec, batch, virtual_size=IMAGE)
        if images is not None and images.shape[1] != spec.n_views:
            fails.append(f"{name}: view count {images.shape[1]} != {spec.n_views}")

        t0 = time.time()
        first = last = None
        for step in range(STEPS):
            out = model(images, batch["proprio"], calib=batch)
            loss, parts = policy_loss(out, batch, spec, model)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step == 0:
                first = float(loss)
            last = float(loss)
        dt = time.time() - t0

        with torch.no_grad():
            out = model(images, batch["proprio"], calib=batch)
            err = (out["pos"] - batch["target_pos"]).norm(dim=-1).mean().item()
        shape = tuple(images.shape[1:]) if images is not None else None
        ok = err < 0.03
        print(
            f"{'ok ' if ok else 'BAD'} {name:12s} {n_par/1e6:4.1f}M  in={str(shape):18s} "
            f"loss {first:7.4f} -> {last:7.4f}   train |err| = {err*1000:6.1f} mm   "
            f"{dt/STEPS*1000:5.1f} ms/step"
        )
        if not ok:
            fails.append(f"{name}: cannot overfit 32 samples ({err*1000:.0f} mm)")

    print("\nFAILURES:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
