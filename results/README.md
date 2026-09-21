# What is in here, and which files predate the rotation fix

On 2026-09-21 a bug was found in the 6D rotation encoding: `rot_to_6d` packed a
rotation's two columns row-major while `rot6d_to_matrix` read them
column-major, so a decoded rotation about z came back with its yaw negated and
the executor drove every grasp mirrored. See §8 of `paper/report.md`.

**Translation results are unaffected.** The bug lived entirely in the rotation
path, and the retrain confirms it: the corrected run matches the old one step
for step on translation (11.0 mm vs 11.0 mm at 6000 steps, 6.4 vs 6.8 at
12000). Every grid, sweep and calibration result below therefore stands.

**Closed-loop results are affected**, because the closed loop is the only place
a predicted rotation was executed. Files are marked accordingly.

| file | status |
|---|---|
| `grid_s01.json`, `grid_seed0.json`, `grid_cross.json` | valid — translation metric |
| `calibration_law.json` | valid — no policy involved |
| `precision_cliff.json` | valid — oracle keyposes, no predicted rotation |
| `fusion.json`, `fusion_4096.json` | valid — no policy involved |
| `closed_loop_fixed.json` | **current** — 0.60 picked / 0.50 placed |
| `closed_loop_12k.json` | superseded — 0.10 picked, mirrored heading |
| `closed_loop_12k_yaw.json` | superseded — the oracle-heading diagnostic that localised the bug (0.50 picked) |
| `closed_loop_6x.json`, `closed_loop_partial.json` | superseded — mirrored heading |

The superseded files are kept deliberately. They are the measurement that
found the bug, and a repository that quietly deletes its wrong numbers is
harder to trust than one that labels them.

`dp3_vs_dp_s*.json` are CPU training-dynamics curves for the LeRobot pull
request, not a benchmark — see the docstring of `scripts/train_dp3_vs_dp.py`.
