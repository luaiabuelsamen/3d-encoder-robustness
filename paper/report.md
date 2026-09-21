# What re-rendering buys, and what it costs

**Separating the mechanisms inside 3D robot policies, and measuring what each
one is robust to.**

---

## Abstract

RVT-style policies unproject calibrated RGBD into a world-frame point cloud and
re-render it from fixed virtual cameras. The field's shorthand for why this
helps is "it uses 3D structure." That phrase covers at least three separable
mechanisms — depth as an extra signal, explicit geometric decoding, and input
canonicalisation — and it is not obvious which does the work.

We separate them on one manipulation task with one backbone, one training
recipe and one output space, and stress every arm along three axes that a real
deployment travels: cameras that move and are recalibrated, calibration that is
wrong, and depth that is noisy.

Three results. First, the **decoder** accounts for most of the benefit usually
attributed to 3D: two arms consuming identical pixels differ by 2.9× depending
only on whether translation is regressed or read off a heatmap and pushed
through known calibration. Depth added as a fourth input channel is *worse*
than no depth at all. Second, **no 3D encoder is robust to all three stresses,
and no two share a failure profile** — canonical re-rendering is the most
accurate at nominal and the most fragile to both miscalibration and depth
noise, while an unordered point set is the most viewpoint-invariant of all.
Third, the **cost of miscalibration is predictable from rig geometry before any
training**, which turns architecture choice into a measurement.

---

## 1. Why this question

Jitendra Malik, October 2025: *"Many robotics papers in the learning era
weren't exploiting 3D structure, which IMHO is just wasting valuable signal."*

He is right, and the interesting question is the follow-up: exploiting it *how*,
and at what price. A world-frame representation is not free. It is a claim about
knowing where your cameras are, and that claim can be wrong.

## 2. Setup

**Task.** SO-101 arm in MuJoCo, pick a randomised block and place it in a
randomised container. Demonstrations from a scripted expert validated at 9/10
pick and place. Policies predict the **next keypose** from a single observation
— the RVT output space — as translation, 6D rotation and gripper state.

**Metric.** Median translation error in mm. The block is 20 mm across, so a
prediction 20 mm off closes the jaws on air. Errors are also reported split by
phase, because averaging the two keyposes that decide the task together with
nine transit poses that have centimetres of slack hides exactly the effect
being looked for.

**Arms.** One transformer, same capacity, same optimiser and schedule, same
rotation and gripper heads, same data. Two things vary: what the encoder sees,
and how translation is decoded.

**Stresses.** θ: cameras move by θ and the policy is told where they went.
ε: cameras have not moved, the extrinsics the policy is handed are wrong by ε.
c: stereo depth noise, σ_z = c·z², with 1 mm quantisation, edge dropout and
flying pixels — the last because they are not zero-mean and they are what hurts
a point cloud most.

θ and ε are routinely conflated and they are not the same experiment. θ asks
whether a representation is invariant. ε asks what it costs when the numbers it
trusts are wrong, which is the common case, because extrinsics drift and nobody
recalibrates a working cell.

## 3. Result 1: the calibration cost is geometry, not learning

A rotational calibration error of ε displaces every reconstructed point by

    sqrt( (⟨|sin θ|⟩ · d · sin ε)² + translation² ),   ⟨|sin θ|⟩ = π/4

with d the camera-to-workspace distance. The π/4 is the mean sine of the angle
between a uniformly random error axis and the line of sight: only the
perpendicular component moves a point along that ray.

Measured on the reconstruction directly, with no policy involved, across 60
held-out frames:

| ε (deg) | predicted | per camera | after fusing | between-camera disagreement |
|---|---|---|---|---|
| 0 | 0.0 | 0.0 | 0.0 | 10.2 |
| 0.5 | 4.6 | 4.5 | 2.7 | 10.7 |
| 1 | 9.2 | 9.1 | 5.3 | 12.2 |
| 2 | 18.5 | 18.1 | 10.7 | 17.4 |
| 5 | 46.2 | 45.4 | 26.7 | 37.5 |
| 10 | 92.0 | 90.6 | 53.8 | 73.2 |

Measured over predicted is **0.98, constant to two decimal places across a
twentyfold range of ε, with zero fitted constants.**

The paired control matters as much: move the cameras and let the calibration
follow, and the same measurement is flat — 0.9, 1.2, 2.2, 2.9 mm at θ = 5, 10,
20, 30°, with between-camera disagreement pinned at its 10.2 mm floor. **A
world-frame representation is not robust to viewpoint. It is robust to
viewpoint conditional on calibration, and the condition does all the work.**
One degree of stale calibration costs three times what thirty degrees of camera
motion costs.

Two consequences. Fusing cameras buys only the usual √n and cannot touch the
disagreement term, which smears an object rather than displacing it. And the
calibration budget is computable from a tape measure before an architecture is
chosen.

## 4. Result 2: the decoder does the work

Nominal condition, median mm:

| arm | input | decoder | median | grasp |
|---|---|---|---|---|
| proprio (blind) | — | regress | 23.6 | ~49 |
| RGB | RGB | regress | 17.3 | 21.8 |
| RGB+D | RGB + depth channel | regress | 21.9 | 47.6 |
| RGB+D unproject | RGB + depth channel | heatmap → unproject | 7.5 | 8.0 |
| RGB+XYZ world | RGB + world XYZ | heatmap → unproject | 4.8 | 2.6 |
| RVT | canonical virtual views | heatmap → orthographic | 3.6 | 2.0 |

Rows 3 and 4 consume **identical pixels**. The only difference is whether the
translation is regressed from a pooled token or read off a per-view heatmap and
pushed through known calibration: 21.9 → 7.5 mm, and 47.6 → 8.0 on grasps.

And depth added as a fourth channel is *worse than no depth* — 21.9 against
17.3. A convolutional encoder reads it as texture. This is the cheapest thing
to reach for and on this task it costs.

## 5. Result 3: no encoder is robust to all three stresses

| encoder | nominal | θ=30° | ε=5° | c=0.008 |
|---|---|---|---|---|
| RGB, regress | 17.3 | 27.3 | **17.3** | **17.3** |
| RGB + depth channel | 21.9 | 26.9 | **21.9** | **21.7** |
| RGB+XYZ camera frame | 13.6 | 28.9 | **13.6** | 17.4 |
| Point cloud, 1 cam, camera frame | 8.3 | 37.0 | **8.3** | **8.9** |
| Point cloud, 4 cam, world frame | 8.8 | **9.1** | 19.9 | **9.6** |
| RGB+D → unproject | 7.5 | 70.1 | 35.5 | 9.4 |
| RGB+XYZ world → unproject | 4.8 | 45.6 | 34.8 | 6.4 |
| RVT, canonical views | **3.6** | 11.6 | 54.3 | 22.4 |

Bold marks an entry that barely moved from nominal. Read down the columns.

**An unordered point set is more viewpoint-invariant than canonical
re-rendering.** Under a 30° rig move with correct extrinsics the world-frame
cloud is flat (8.8 → 9.1) while RVT degrades (3.6 → 11.6) and the real-view
geometric decoders collapse (45.6, 70.1). Re-rendering is canonical in *pose*
but reintroduces a sampling grid, and that grid resamples differently as
coverage changes. A set has no grid to resample.

**RVT is the most depth-noise-sensitive encoder measured** — 3.6 → 22.4 mm,
sixfold, where the world cloud goes 8.8 → 9.6 and RGB+XYZ barely moves,
4.8 → 6.4. Rasterising a noisy cloud onto a fixed grid compounds the error;
consuming it as a set does not.

Together these are the trade canonical re-rendering makes: **viewpoint
invariance, bought with sensitivity to everything that corrupts the geometry
being rendered.** It is the most accurate choice when the geometry is clean and
the worst when it is not, and "clean" is a property of the deployment, not of
the method.

A practitioner's table falls out:

| situation | encoder |
|---|---|
| cameras fixed, calibration maintained | canonical views — best nominal by 2× |
| cameras move, calibration follows | world-frame point cloud — flat across 30° |
| calibration uncertain | camera-frame point cloud — flat across 10° |
| depth noisy or sensor cheap | anything but canonical re-rendering |
| no depth | RGB regression, and do not bolt depth on as a channel |

## 6. Limitations

One task, one arm, simulation only. Small models (≈3 M parameters) trained for
2500 steps under a fixed budget — absolute numbers would improve with more, but
every arm shares the budget, so the comparison holds and the ordering is what
is claimed.

Two seeds for most arms, three for the point-cloud arms. Differences of a few
millimetres between adjacent rows are not resolved; the effects called findings
here are 2–20×.

`xyz_cam` differs from `xyz_real` in both frame and decoder. That confound is
intrinsic rather than careless: a world-frame action cannot be decoded through
explicit geometry without extrinsics somewhere, so a camera-frame policy must
learn the camera-to-robot map implicitly.

**A benchmark warning that generalises.** A blind policy given joint angles
scores 6.2 mm — better than five of eight sighted encoders. Scripted
demonstrations are a deterministic function of the scene, so by the time the
arm reaches keyframe *k* its configuration already encodes where the object is.
Any 3D-versus-2D comparison run on scripted data with joint-angle
proprioception is measuring almost nothing. Every arm here is given PerAct's
four low-dimensional numbers instead.

## 7. What was built

A dependency-free implementation of the pieces, since the reference ones need
PyTorch3D or a custom CUDA extension that do not build on many machines:

- a batched orthographic point-cloud renderer with an exact z-buffer from one
  int64 key per point and a single per-pixel `amin` — no sort, no kernel;
- a depth-sensor noise model with the shape stereo depth actually has;
- episodes stored as replayable simulator states rather than images, 300×
  smaller, which makes the camera rig a free experimental variable.

Contributed upstream to LeRobot as
[#4696](https://github.com/huggingface/lerobot/pull/4696): point-cloud
observations, the DP3 encoder, and RealSense intrinsics — which LeRobot was
discarding at capture time, leaving recorded depth maps geometrically unusable.

## Credit

The method under study is RVT, Goyal, Xu, Guo, Blukis, Chao and Fox, CoRL 2023,
and the keypose formulation is PerAct's. The point-cloud encoder is 3D
Diffusion Policy, Ze, Zhang, Zhang, Hu, Wang and Xu, RSS 2024. The 6D rotation
encoding is Zhou et al., CVPR 2019.

The framing question is Jitendra Malik's, and the tradition he was pointing at
— photogrammetry, multi-view geometry, Hartley and Zisserman, and four decades
of work by human researchers — is what makes any of this measurable at all.
