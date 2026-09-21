# What does re-rendering actually buy you?

**Separating the three mechanisms inside RVT-style 3D robot policies, and
measuring what each one costs.**

Jitendra Malik, [October 2025](https://x.com/JitendraMalikCV): *"Many robotics
papers in the learning era weren't exploiting 3D structure, which IMHO is just
wasting valuable signal."* He is right, and this repository takes the point
seriously enough to ask the follow-up question: **exploiting 3D structure how,
and at what price?**

[RVT](https://robotic-view-transformer.github.io/) (Goyal et al., CoRL 2023)
unprojects RGBD into a world-frame point cloud and re-renders it from *fixed*
virtual cameras before a transformer ever sees it. It works. The field's
shorthand for why is "it uses 3D structure" — but that phrase covers three
separable mechanisms, and nobody has reported which one is doing the work:

1. **depth as extra signal** — a fourth input channel;
2. **explicit geometric decoding** — predict a pixel, then push it through known
   calibration, instead of regressing a coordinate;
3. **input canonicalisation** — the network's input stops depending on where the
   real cameras are.

This study takes them apart on one task, with one backbone, one training recipe,
and one output space, and then stresses all of it along three axes that a real
deployment actually travels.

> **Status.** Apparatus complete and self-tested; training and the evaluation
> grid in progress. Numbers below are filled in from `results/` as they land,
> and `results/findings.md` is generated, not written — including the
> comparisons that fail to resolve.

---

## The arms

Identical transformer, identical capacity, identical optimiser and schedule,
identical rotation and gripper heads, identical data. Two things vary.

| arm | encoder input | translation decoder | input moves with the cameras? |
|---|---|---|---|
| `proprio` | nothing | regress | — |
| `rgb` | 4 real RGB views | regress | yes |
| `rgbd` | 4 real RGB+D views | regress | yes |
| `rgbd_unproj` | 4 real RGB+D views | heatmap → unproject | features yes, output no |
| `xyz_real` | 4 real RGB+world-XYZ views | heatmap → unproject | features yes, output no |
| `rvt` | 5 canonical orthographic views | heatmap → orthographic | **no** |
| `rgb_aug` | `rgb` + camera-pose augmentation | regress | yes |
| `rvt_aug` | `rvt` + camera-pose augmentation | heatmap → orthographic | no |

`xyz_real` is the arm that makes the comparison sharp. It receives *the same
world-frame coloured point cloud* as `rvt` and decodes it through *the same*
explicit geometry. The only difference is which viewpoints the cloud is
rasterised from — the real cameras' or the canonical ones'. Whatever separates
those two arms is canonicalisation proper, and nothing else.

`proprio` is not a strawman, it is the load-bearing control: a scripted expert
is nearly deterministic given the scene, so if joint angles alone predict the
next keypose well, the task does not test perception and no other row means
anything.

## The stresses

| axis | what moves | what a policy is told |
|---|---|---|
| **θ** — extrinsic shift, recalibrated | cameras move by θ | the true new extrinsics |
| **ε** — calibration error | nothing moves | extrinsics wrong by ε |
| **c** — depth noise | nothing moves | σ_z = c·z², plus 1 mm quantisation, edge holes and flying pixels |

θ and ε are usually conflated and they are not the same experiment. θ asks
whether a representation is *invariant*. ε asks what the representation costs
when the numbers it trusts are wrong — and it is the common case, because
extrinsics drift and nobody recalibrates a working cell.

The depth-noise model is not additive Gaussian, which flatters point-cloud
methods. Stereo depth fails at discontinuities, producing holes *and* flying
pixels: matches that interpolate between foreground and background and leave
points hanging in mid-air around every silhouette. Those are not zero-mean, and
the object here is a 20 mm block.

## Task and metric

SO-101 arm, MuJoCo, pick a randomised block and place it in a randomised
container. Policies predict the **next keypose** — the RVT output space — from a
single observation: end-effector translation, 6D rotation, gripper state.

Translation error is the headline because the block is 20 mm across: a
prediction 20 mm off closes the jaws on air. Errors are also reported split by
phase, because averaging the two keyposes that decide the task (closing on the
block, opening over the container) together with nine transit poses that have
centimetres of slack will hide exactly the effect being looked for.

`scripts/closed_loop.py` converts millimetres into place rates, with the
demonstration's own keyposes replayed through the identical executor as the
paired reference.

## Reproducing

```bash
make check     # geometry and renderer self-tests -- run these first, always
make data      # 500 train / 60 val / 120 test demonstrations (22 MB)
make train     # every arm, every seed, one shared rendered cache
make grid      # every checkpoint across the stress grid
make figures   # figures, table, and results/findings.md
```

`make check` is not a formality. It asserts the geometry against facts known
independently of the code — the table is a plane at z = 0.09, the block is where
MuJoCo's segmentation buffer says it is — and it caught three real bugs before
any model was trained, including one where MuJoCo's antialiasing blended
segmentation ids along silhouette edges and moved the block's *measured* centroid
100 mm while the actual block pixels were landing within 1 mm.

![virtual views](figures/virtual_views.png)

*Top: the four real cameras. Bottom: the five canonical orthographic views
re-rendered from their fused point cloud. The reprojection is geometric, not
learned. Under a 15° rig perturbation these images move by 1.4 virtual pixels —
the resampling floor.*

## Design notes worth knowing before you copy anything

**Episodes are stored as replayable sim states, not images.** 500 episodes are
22 MB, and the camera rig becomes a free experimental variable: any condition is
produced by re-photographing the same episodes, so nothing about a comparison
changes except the thing under test.

**One process renders each training condition once and trains every arm from
it.** Besides the hours saved, it removes a confound: every arm and seed sees
byte-identical observations, so a gap between arms cannot be a difference in
which camera jitter they happened to draw.

**Everything runs in float32.** Mixed precision would roughly double throughput,
but the measured quantity is a millimetre offset carried in metre-scale world
coordinates, and float16's mantissa gives about 3 × 10⁻³ relative precision
there — the same order as the effects. A numerical artefact that looked like a
finding would cost more than the time saved.

**No `PYTHONPATH`.** See `rvt_lerobot/device.py`. This host has two torch
installs and the shell profile's `PYTHONPATH` is what selects the working one;
replacing *or* unsetting it silently selects a build whose CUDA runtime is newer
than the driver, and training quietly completes on the CPU.

## Credits

- [NVlabs/RVT](https://github.com/NVlabs/RVT) — Goyal, Xu, Guo, Blukis, Chao,
  Fox. The method this study takes apart.
- [PerAct](https://peract.github.io/) — the keypose formulation and on-disk
  format.
- The physics scene, IK and scripted expert are vendored from a sibling project
  of the author's, where they were validated at 78% / 60% pick and place; they
  re-measure at 9/10 on the nominal scene here.
- Zhou et al., CVPR 2019, for the 6D rotation encoding.
- The framing question is Jitendra Malik's, and the 3D-reconstruction tradition
  he was pointing at is the work of human researchers over four decades —
  photogrammetry, multi-view geometry, Hartley & Zisserman, and everything
  built on them.
