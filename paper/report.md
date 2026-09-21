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

### The two axes together

Sweeping camera motion and depth noise as a product, median mm (the clean,
static corner is measured in the 1-D sweeps above):

| encoder | best cell | worst cell | spread |
|---|---|---|---|
| point cloud, 4 cam world | 9.5 | 11.0 | **1.5** |
| RVT canonical views | 7.3 | 37.0 | 29.7 |
| point cloud, 1 cam camera frame | 9.1 | 39.6 | 30.5 |
| RGB+XYZ world → unproject | 7.2 | 53.0 | 45.8 |
| RGB, regress | 18.8 | 29.2 | 10.4 |

**The world-frame point cloud is flat over the whole plane.** RVT is ahead only
in the clean, static corner: by c = 0.002 it has already been passed (9.6
against 9.5) and at c = 0.008 it is beaten more than twofold (24.8 against
10.5), at every θ.

This is the practical form of the finding. The encoder with the best nominal
number is not the encoder to deploy unless the deployment is as clean as the
benchmark, and a benchmark run only at nominal cannot tell you which is which.

A practitioner's table falls out:

| situation | encoder |
|---|---|
| cameras fixed, calibration maintained | canonical views — best nominal by 2× |
| cameras move, calibration follows | world-frame point cloud — flat across 30° |
| calibration uncertain | camera-frame point cloud — flat across 10° |
| depth noisy or sensor cheap | anything but canonical re-rendering |
| no depth | RGB regression, and do not bolt depth on as a channel |

## 6. Result 4: the metric hides the only step that needs vision

Millimetres are only meaningful once you know what they buy. Taking each
demonstration's own keyposes, perturbing them by a known amount and executing
through the same controller gives the conversion directly, with no policy in
the loop:

| injected keypose error | picked | placed |
|---|---|---|
| 0 mm | 0.88 | 0.88 |
| 2 mm | 0.88 | 0.88 |
| 5 mm | 0.88 | 0.88 |
| 10 mm | 0.84 | 0.80 |
| 20 mm | 0.72 | 0.68 |
| 40 mm | 0.40 | 0.32 |

There is no cliff. The task absorbs 5 mm without losing a single placement and
degrades gracefully after. **So most of the differences in this report are
below the task's tolerance**: RVT's 3.6 mm and a point cloud's 8.3 mm both
place at 88%, and that gap is not worth anything here. What matters is the
stressed regime — 17, 37, 70 mm — which sits on the slope.

That should have predicted decent closed-loop performance from policies with
5-10 mm error. It did not: they placed 0-7% against an oracle replay at 87%.

Most of that gap turned out to be the rotation bug in §8, not a property of the
task or the metric — with the 6D packing corrected and nothing else changed,
the same architecture on the same data places 0.50 and picks 0.60. What follows
is the part of the gap that survives the fix, and it is a smaller effect than
the numbers above suggested.

| arm | home → first keypose | every other step | ratio |
|---|---|---|---|
| point cloud, 1 cam | 48.0 mm | 9.7 mm | **5.0×** |
| RGB, regress | 53.3 mm | 21.3 mm | **2.5×** |

The first keypose is the only one where the arm has not already been carried
toward the object by a previous waypoint, so it is the only one whose answer is
not partly encoded in the robot's own configuration. It is also one step in
twelve, so it contributes about 8% of the reported average — and it is 2.5 to 5
times worse than that average. At 48 mm the table above gives roughly 32%
placement before any subsequent state has drifted off-distribution.

**The aggregate keypose metric is dominated by steps where proprioception
already constrains the answer, and it hides the single step that actually
requires perception.** It also hides rotation entirely: the headline number is
a translation, and on this gripper heading is what decides the grasp. A median
heading error of 11.9 degrees costs more than the difference between any two
encoders in this report. This is the same warning as the joint-angle control in
§7, arriving from the other direction: a blind policy scores 6.2 mm because
most steps do not need eyes, and the one that does is averaged away.

For anyone building on this: report the first keypose separately, report
rotation alongside translation, and report closed-loop success. The mean over a
trajectory is not a perception metric, and a translation is not a grasp.

## 7. Result 5: fusing cameras buys coverage, not density

The study's world-frame arm sees four cameras and its camera-frame arm sees
one, and at zero calibration error they tie. Four cameras see strictly more of
the scene than one, so a tie is suspicious — the obvious reading is that fusion
is broken and the world-frame arm is quietly learning from a single view.

Counting what actually reaches the encoder settles it, with no training
involved: 60 frames drawn from expert trajectories so the arm's own body
occludes cameras the way it does in a real cell, the block mask taken from
MuJoCo's segmentation buffer, and a label channel carried through the sampler
so each surviving point can be traced to what it hit.

| | 1 camera | 4 cameras fused |
|---|---:|---:|
| object pixels available | 30.4 | 83.6 |
| valid points in the workspace crop | 6418 | 18906 |
| **object's share of the cloud** | **0.473%** | **0.442%** |
| points on the object, 1024-point budget | 4.6 | 4.4 |
| points on the object, 4096-point budget | 19.1 | 17.8 |

Fusion is not broken: it nearly triples the object pixels available. But a
uniform fixed-budget subsample does not see the count, it sees the *share*, and
fusion adds object points and background points in the same proportion. The
ratio is identical at both budgets, so this is not a budget that happens to be
too small.

Where fusion earns its calibration cost is occlusion: in 7 of 60 frames the
block was invisible to the single camera, and fusion recovered it in 5.

**More cameras buy coverage, not density.** Density on the object comes from
the workspace crop instead. Cropping is also the cheaper of the two
interventions by a wide margin: on this scene an uncropped 1024-point sample
contained about 3.5 points of the 20 mm block, and cropping to the workspace
fixes that without adding a camera or a calibration procedure.

## 8. A note on what almost went unnoticed

Two bugs in this work were invisible to every metric it reports, and both share
a shape worth naming: a quantity was encoded one way and decoded another, and
every consumer of it applied the same wrong decode, so the error cancelled
everywhere except where the number met the physical world.

**The rotation encoding.** `rot_to_6d` packed a rotation's two columns
row-major; `rot6d_to_matrix` read them column-major. For a rotation about z the
round trip returns the yaw *negated*. Training converged — the network learns
whatever vector it is shown. The rotation metric read a healthy 13.3 degrees —
it decodes prediction and target the same wrong way. Only the executed grasp
heading was exposed, and there the robot approached every block mirrored:
closed-loop 0.10 picked, against an oracle at 0.97 through the same executor
and a validation translation error of 6.4 mm.

**The depth unit.** `LeRobotDataset` dequantises depth to millimetres by
default; the point-cloud processor's `depth_scale` defaults to metres. Points
land a thousand times too far out, the workspace crop rejects all of them, and
a frame with no surviving points is returned as zeros by design so a dropped
depth frame cannot kill a run. A policy trains on empty clouds and no metric
moves.

Fixing the one line moved the task result by six-fold, on the same
architecture, data, budget, executor and scenes:

| | picked | placed |
|---|---:|---:|
| scripted expert | 1.00 | 0.97 |
| oracle keyposes, replayed | 0.97 | 0.87 |
| DP3 point cloud, before | 0.10 | 0.10 |
| DP3 point cloud, after | **0.60** | **0.50** |

Validation barely moved: 6.4 → 6.8 mm median translation, 0.74 → 0.72 within
10 mm. That is the tell. Every translation result in this report stands
unchanged, because the bug lived entirely in the rotation path; only `rot_deg`
changes meaning, from 13.3 degrees in a mirrored frame to a true geodesic 11.6.

The common lesson is not "write more tests". It is that a round trip through an
encoder and *its own decoder* is the cheapest test that exists, and neither
encoding had one, because neither number ever leaves the model — which is
exactly why nothing caught them. Every other convention in this project was
checked against ground truth from the first day. Both round trips are asserted
now.

## 9. Limitations

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

**The benchmark warning, restated.** A blind policy given joint angles
scores 6.2 mm — better than five of eight sighted encoders. Scripted
demonstrations are a deterministic function of the scene, so by the time the
arm reaches keyframe *k* its configuration already encodes where the object is.
Any 3D-versus-2D comparison run on scripted data with joint-angle
proprioception is measuring almost nothing. Every arm here is given PerAct's
four low-dimensional numbers instead.

## 10. What was built

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
