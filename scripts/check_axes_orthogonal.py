"""Each stress axis must change exactly what it claims to, and nothing else.

The three axes are only separable if they are actually separate in the data.
This asserts that directly, by rendering the same frames under each condition
and diffing the raw arrays:

  theta  cameras move and the calibration follows  -> RGB, depth AND extrinsics change
  eps    the calibration is wrong, cameras have not moved
                                                   -> ONLY the extrinsics change
  c      the depth sensor is noisy                 -> ONLY depth changes

If eps changed a pixel, it would be secretly moving the cameras and the
"invariance versus calibration" distinction -- the study's main claim --
would be measuring something else. If c changed the extrinsics, the depth-noise
axis would be contaminated. Neither is visible by reading the code; both are
one array subtraction away.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rvt_lerobot.data import collect_study as C  # noqa: E402
from rvt_lerobot.data.views import Condition, build_samples, render_condition  # noqa: E402
from rvt_lerobot.envs.multicam import MultiCamScene  # noqa: E402

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name:44s}  {detail}")
    if not ok:
        fails.append(name)


def main() -> int:
    scene = MultiCamScene(seed=4242, image_size=96)
    episodes = C.load(pathlib.Path("data/test.npz"))[:3]
    samples = build_samples(episodes, np.random.default_rng(2024), per_segment=1)
    print(f"{len(samples)} frames from {len(episodes)} episodes\n")

    ref = render_condition(scene, episodes, samples, Condition(seed=7))

    def diff(cond):
        a = render_condition(scene, episodes, samples, cond)
        return (
            float(np.abs(a["rgb"].astype(np.int16) - ref["rgb"].astype(np.int16)).mean()),
            float(np.abs(a["depth_mm"].astype(np.int32) - ref["depth_mm"].astype(np.int32)).mean()),
            float(np.abs(a["T"] - ref["T"]).max()),
        )

    print(f"{'condition':26s} {'|dRGB|':>8s} {'|dDepth| mm':>12s} {'max|dT|':>9s}")
    rows = {}
    for cond in (
        Condition(theta_deg=5, seed=7),
        Condition(theta_deg=30, seed=7),
        Condition(eps_deg=10, seed=7),
        Condition(noise_c=0.008, seed=7),
    ):
        rows[cond.name] = diff(cond)
        print(f"{cond.name:26s} {rows[cond.name][0]:8.2f} {rows[cond.name][1]:12.1f} "
              f"{rows[cond.name][2]:9.4f}")
    print()

    t5, t30 = rows["theta5_eps0_c0"], rows["theta30_eps0_c0"]
    e10 = rows["theta0_eps10_c0"]
    c8 = rows["theta0_eps0_c0.008"]

    check("theta moves the cameras (pixels change)", t5[0] > 1.0, f"|dRGB| = {t5[0]:.2f}")
    check("theta also updates the extrinsics", t5[2] > 1e-3, f"max|dT| = {t5[2]:.4f}")
    check("theta is monotone", t30[0] > t5[0] and t30[2] > t5[2],
          f"|dRGB| {t5[0]:.1f} -> {t30[0]:.1f}, max|dT| {t5[2]:.3f} -> {t30[2]:.3f}")

    check("eps leaves every pixel untouched", e10[0] == 0.0 and e10[1] == 0.0,
          f"|dRGB| = {e10[0]:.4f}, |dDepth| = {e10[1]:.4f} mm")
    check("eps corrupts only the reported extrinsics", e10[2] > 1e-3,
          f"max|dT| = {e10[2]:.4f}")

    check("depth noise leaves RGB untouched", c8[0] == 0.0, f"|dRGB| = {c8[0]:.4f}")
    check("depth noise leaves the extrinsics untouched", c8[2] == 0.0,
          f"max|dT| = {c8[2]:.6f}")
    check("depth noise actually corrupts depth", c8[1] > 1.0, f"|dDepth| = {c8[1]:.1f} mm")

    print("\nFAILURES:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
