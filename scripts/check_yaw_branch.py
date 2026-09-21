"""Are the two grasp headings really interchangeable on THIS gripper?

`check_yaw_symmetry.py` found that 12.5% of predicted headings are more than 90
degrees wrong, and that taking the error modulo pi removes that tail entirely:
the demonstrations contain both `theta` and `theta + pi` for equivalent grasps,
so the target is ambiguous and least squares splits the difference.

The obvious fix is to canonicalise the target into a half-circle. That is only
correct if the two branches actually produce the same grasp, and on this gripper
that is not obvious: one jaw is fixed and one moves, and the fixed pad sits 11.9
mm off the tool centre across the closing axis. Flipping the heading by pi swaps
which side of the block the fixed pad approaches. A symmetric parallel gripper
would not care. This one might.

So: replay the demonstrations' own keyposes through the executor, once as
recorded and once with every heading rotated by pi, and compare. No policy
involved, so the difference is the gripper's asymmetry and nothing else.

    same success     the branches are interchangeable; canonicalise the target
    flipped is worse the asymmetry is real; fix the loss, not the target
"""

from __future__ import annotations

import argparse
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from rvt_lerobot.data.collect_study import GRIP_OPEN_THRESHOLD, run_episode  # noqa: E402
from rvt_lerobot.envs.multicam import MultiCamScene  # noqa: E402
from rvt_lerobot.vendor.so101_expert import ScriptedExpert  # noqa: E402

from closed_loop import Executor, restore_model, snapshot_model  # noqa: E402


def replay(scene, episode, flip: bool) -> dict:
    """Replay one demonstration's keyposes, optionally rotating every heading by pi."""
    ex = Executor(scene.scene)
    for k in episode.keyframes:
        rot = episode.jaw_rot[k]
        yaw = float(np.arctan2(rot[1, 0], rot[0, 0])) + (np.pi if flip else 0.0)
        ex.step(episode.tcp[k].astype(float), yaw,
                bool(episode.jaw_cmd[k] > GRIP_OPEN_THRESHOLD))
    return ex.outcome()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=25)
    a = p.parse_args()

    scene = MultiCamScene(seed=900, image_size=96)
    expert = ScriptedExpert(scene.scene)

    # Paired: both branches are replayed on the SAME episode and the same
    # randomised block, so the comparison is not confounded by scene draw.
    results = {False: [], True: []}
    for _ in range(a.episodes):
        ep = run_episode(None, expert)
        snap = snapshot_model(scene.scene)
        for flip in (False, True):
            scene.scene.reset(block_xy=(ep.meta["block_x"], ep.meta["block_y"]),
                              block_yaw=ep.meta["block_yaw"])
            restore_model(scene.scene, snap)
            results[flip].append(replay(scene, ep, flip))

    for flip in (False, True):
        label = "heading + pi" if flip else "as recorded "
        r = results[flip]
        print(f"  {label}  picked {np.mean([x['picked'] for x in r]):.2f}  "
              f"placed {np.mean([x['placed'] for x in r]):.2f}  (n={len(r)})")

    d = (np.mean([x["picked"] for x in results[True]])
         - np.mean([x["picked"] for x in results[False]]))
    print(f"\ndelta in pick rate from flipping every heading: {d:+.2f}")
    print("branches interchangeable -> canonicalise the target" if abs(d) < 0.1
          else "branches NOT interchangeable -> the gripper asymmetry is real")
    return 0


if __name__ == "__main__":
    sys.exit(main())
