"""How accurate does a keypose have to be before the jaws close on the block?

Every number in this study is a millimetre of translation error, and a
millimetre is only meaningful if it converts into something the robot does.
This measures the conversion directly: take each demonstration's own keyposes,
perturb them by a known amount, execute through the same controller, and count
the placements.

No policy is involved, so nothing about a particular architecture confounds the
answer. The result is a curve that reads every other table in the project.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

_spec = importlib.util.spec_from_file_location(
    "cl", pathlib.Path(__file__).resolve().parent / "closed_loop.py"
)
cl = importlib.util.module_from_spec(_spec)
sys.argv = [sys.argv[0]]
_spec.loader.exec_module(cl)

from rvt_lerobot.data.collect_study import run_episode  # noqa: E402
from rvt_lerobot.envs.multicam import MultiCamScene  # noqa: E402
from rvt_lerobot.vendor.so101_expert import ScriptedExpert  # noqa: E402

NOISE_MM = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0)
EPISODES = 25


def main() -> int:
    scene = MultiCamScene(seed=900, image_size=64)
    expert = ScriptedExpert(scene.scene)

    print("collecting demonstrations to replay...", flush=True)
    demos = []
    while len(demos) < EPISODES:
        ep = run_episode(None, expert)
        if ep.placed and len(ep.keyframes) >= 3:
            demos.append(ep)

    rows = []
    print(f"\n{'noise':>7s} {'picked':>8s} {'placed':>8s}   (n={EPISODES})")
    for noise in NOISE_MM:
        rng = np.random.default_rng(7)
        out = []
        for ep in demos:
            snap = cl.snapshot_model(scene.scene)
            scene.scene.reset(
                block_xy=(ep.meta["block_x"], ep.meta["block_y"]),
                block_yaw=ep.meta["block_yaw"],
            )
            cl.restore_model(scene.scene, snap)
            out.append(cl.oracle_rollout(scene, ep, pos_noise_mm=noise, rng=rng))
        picked = float(np.mean([r["picked"] for r in out]))
        placed = float(np.mean([r["placed"] for r in out]))
        rows.append({"noise_mm": noise, "picked": picked, "placed": placed, "n": EPISODES})
        print(f"{noise:7.1f} {picked:8.2f} {placed:8.2f}", flush=True)

    out_path = pathlib.Path("results/precision_cliff.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
