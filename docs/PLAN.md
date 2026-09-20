# What does re-rendering actually buy you?

## Thesis

RVT-style policies unproject RGBD into a world-frame point cloud and re-render it
from *fixed* virtual cameras. The field's shorthand for why this helps is "it uses
3D structure." That conflates three separable mechanisms:

1. **depth as extra signal** — a 4th input channel,
2. **explicit geometric decoding** — predicting a pixel and back-projecting it
   through known calibration, instead of regressing coordinates,
3. **input canonicalisation** — the network's input stops depending on where the
   real cameras are.

Nobody has separated them. This project does, on one task, with one backbone, one
training recipe, and three stress axes.

## The claim we expect to be able to make

Re-rendering does not give you viewpoint invariance for free. It **trades a
viewpoint-generalisation problem for a calibration-accuracy problem.** Under
cameras that move *and are recalibrated*, 3D arms are flat and RGB arms collapse.
Under cameras that move and are *not* recalibrated — the common real-world case —
the ordering can inverts, because a 3D arm's error is the calibration error
rigidly applied to every point, while an RGB arm degrades gracefully.

Second claim: most of the invariance usually attributed to re-rendering comes from
mechanism (2), not (3). Cheap to get; you do not need the renderer.

## Arms (one backbone, same capacity, same data, same schedule)

| arm | input | decode | input moves with cameras? |
|---|---|---|---|
| A `rgb_reg`    | 4x RGB real views       | regression to xyz     | yes |
| B `rgbd_reg`   | 4x RGBD real views      | regression to xyz     | yes |
| C `rgbd_unproj`| 4x RGBD real views      | heatmap -> unproject  | yes (features) / no (output) |
| D `rvt_canon`  | 5x orthographic virtual | heatmap -> 3D         | no |
| E `rgb_reg_aug`| A + camera-pose augmentation at train time | regression | yes |
| F `rvt_canon_aug` | D + camera-pose augmentation | heatmap -> 3D | no |

## Stress axes (test time only, unless noted)

1. **Extrinsic shift, recalibrated** theta in {0,5,10,15,20,30} deg — cameras move,
   policy is told where they are.
2. **Calibration error** eps in {0,1,2,5,10} deg — cameras are where they always were,
   the extrinsics handed to the policy are wrong.
3. **Depth noise** c in {0,.0005,.001,.002,.004,.008}; sigma_z = c*z^2 (RealSense-like),
   plus 1 mm quantisation, edge dropout and flying pixels.

## Task / metric

SO-101 pick-and-place, next-keypose prediction (the RVT output space):
translation error (mm), rotation geodesic error (deg), gripper-state accuracy,
success@10mm. Plus closed-loop place rate for the headline arms.

## Protocol rules (house rules, non-negotiable)

- Look before sweeping: render frames and dump contacts before any grid.
- A 0/N is a defect until a paired working reference fails the same way.
- Every arm gets the identical backbone, optimiser, schedule, and data.
- 3 seeds minimum on anything that becomes a claim.
