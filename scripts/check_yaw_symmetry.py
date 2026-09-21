"""How much of the heading error is a wrong branch rather than a wrong angle?

A two-jaw gripper closing along a line grasps a block at heading `theta` and at
`theta + pi` by swapping which jaw goes where. If the demonstrations contain
both and the inverse kinematics has no reason to prefer one, the network is
asked to fit a function with two equally correct answers, and least squares
will split the difference. That failure has a signature: raw heading error
shows a mode near 180 degrees while error taken modulo pi does not.

    raw >> mod pi   a branch problem; fix the target or the loss
    raw == mod pi   an ordinary angular error; the model is just imprecise

Run against the checkpoint whose rotation target was packed wrongly, this
printed a median of 1.6 degrees with 12.5% of predictions more than 90 degrees
out, and taking the error modulo pi removed that tail completely -- which looks
exactly like a branch problem and is not one. Both the prediction and the
target were being decoded through the same broken 6D packing, so the numbers
described a mirrored coordinate system rather than the robot's heading. The
apparent bimodality was the mirroring.

With the packing fixed the same measurement reads:

                       median   mean    p90   >45 deg  >90 deg
      heading error      11.9   21.5   51.5     11.3%     4.4%
        modulo pi        11.8   18.0   43.7      9.2%     0.0%

    off by roughly pi but otherwise correct: 1.2% (8 of 663)

So the branch ambiguity is real but small, and what remains is ordinary angular
imprecision -- the median is a true 11.9 degrees rather than a self-consistent
1.6. The lesson is that a diagnostic which decodes both sides of a comparison
the same way cannot see an error in the decoder, and will happily produce a
clean, confident, wrong story instead.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rvt_lerobot.device import pick_device  # noqa: E402
from rvt_lerobot.data import collect_study as C  # noqa: E402
from rvt_lerobot.data.batching import ConditionCache, make_images  # noqa: E402
from rvt_lerobot.data.views import Condition, build_samples, render_condition  # noqa: E402
from rvt_lerobot.envs.multicam import MultiCamScene  # noqa: E402
from rvt_lerobot.evaluate import rot6d_to_matrix  # noqa: E402
from rvt_lerobot.models.policy import ARMS, MultiViewPolicy, proprio_for  # noqa: E402


def heading(rot: np.ndarray) -> np.ndarray:
    """The jaw's closing-axis heading -- the only rotational quantity executed."""
    return np.arctan2(rot[:, 1, 0], rot[:, 0, 0])


def wrap(a: np.ndarray, period: float) -> np.ndarray:
    """Signed angular difference folded into (-period/2, period/2]."""
    return (a + period / 2) % period - period / 2


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=pathlib.Path, default=pathlib.Path("runs_12k/dp3_pcd_s0"))
    p.add_argument("--val", type=pathlib.Path, default=pathlib.Path("data/val.npz"))
    p.add_argument("--image", type=int, default=96)
    a = p.parse_args()

    device = pick_device()
    ckpt = torch.load(a.run / "model.pt", map_location=device, weights_only=False)
    spec = ARMS[ckpt["arm"]]
    model = MultiViewPolicy(spec, image_size=ckpt["image"], patch=ckpt["patch"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    scene = MultiCamScene(seed=0, image_size=a.image)
    episodes = C.load(a.val)
    samples = build_samples(episodes, np.random.default_rng(12345), per_segment=1)
    cache = ConditionCache(render_condition(scene, episodes, samples, Condition(seed=999)),
                           device=device)

    pred, true, is_grasp = [], [], []
    for start in range(0, cache.n, 64):
        batch = cache.batch(np.arange(start, min(start + 64, cache.n)))
        with torch.no_grad():
            out = model(make_images(spec, batch, virtual_size=a.image),
                        proprio_for(spec, batch), calib=batch)
        pred.append(rot6d_to_matrix(out["rot6"]).double().cpu().numpy())
        true.append(rot6d_to_matrix(batch["target_rot6"]).double().cpu().numpy())
        is_grasp.append(batch["event"].cpu().numpy() if "event" in batch else
                        np.zeros(len(pred[-1])))
    pred, true = np.concatenate(pred), np.concatenate(true)

    err_raw = np.abs(np.degrees(wrap(heading(pred) - heading(true), 2 * np.pi)))
    err_mod = np.abs(np.degrees(wrap(heading(pred) - heading(true), np.pi)))

    print(f"{len(err_raw)} validation keyposes, checkpoint {a.run}\n")
    print(f"{'':16s}{'median':>9s}{'mean':>9s}{'p90':>9s}{'>45 deg':>10s}{'>90 deg':>10s}")
    for name, e in (("heading error", err_raw), ("  modulo pi", err_mod)):
        print(f"{name:16s}{np.median(e):9.1f}{e.mean():9.1f}{np.percentile(e, 90):9.1f}"
              f"{100*(e > 45).mean():9.1f}%{100*(e > 90).mean():9.1f}%")

    flipped = (err_raw > 90) & (err_mod < 30)
    print(f"\npredictions off by roughly pi but otherwise correct: "
          f"{100*flipped.mean():.1f}% ({flipped.sum()} of {len(flipped)})")

    # Where does the true heading live? If the demonstrations only ever use half
    # the circle, the ambiguity exists in principle but is never exercised, and
    # the diagnosis above would be a false positive.
    h = np.degrees(heading(true))
    print(f"demonstrated headings span {h.min():.0f} to {h.max():.0f} deg; "
          f"{100*(h > 0).mean():.0f}% positive")
    return 0


if __name__ == "__main__":
    sys.exit(main())
