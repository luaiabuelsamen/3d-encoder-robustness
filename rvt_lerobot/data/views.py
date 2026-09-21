"""Turn stored episodes into rendered observations under a stated condition.

A *condition* is the experiment's independent variable: where the cameras are,
how badly their calibration is wrong, and how noisy the depth is. Because the
dataset stores sim states rather than pictures, every condition is produced by
replaying the same episodes and re-photographing them, so nothing about the
comparison changes except the thing under test.

What gets cached is the raw sensor product only -- RGB, depth in millimetres,
and the calibration handed to the policy. Point clouds, world-XYZ channels and
virtual views are all derived from those on the GPU at batch time, because they
are cheap to recompute and expensive to store.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..render import rig as R
from ..render.noise import apply_depth_noise
from ..vendor.so101_scene import JAW_OPEN, JAW_SHUT
from .collect_study import GRIP_OPEN_THRESHOLD, Episode


@dataclass(frozen=True)
class Condition:
    """One point in the stress grid."""

    theta_deg: float = 0.0   # how far the cameras moved
    eps_deg: float = 0.0     # how wrong the reported calibration is
    noise_c: float = 0.0     # depth noise coefficient, sigma_z = c z^2
    seed: int = 0
    theta_random: bool = False
    """Draw a fresh theta ~ U(0, theta_deg) per episode instead of using it flat.

    This is how the camera-augmented training arms get their data. It is a
    training-time condition, never an evaluation one: evaluating at a spread of
    perturbations would average the very effect the grid is there to resolve."""

    @property
    def name(self) -> str:
        r = "rand" if self.theta_random else ""
        return f"theta{r}{self.theta_deg:g}_eps{self.eps_deg:g}_c{self.noise_c:g}"


@dataclass
class Sample:
    """One training example: observe episode `ep` at `frame`, predict `target`."""

    ep: int
    frame: int
    target: int


def build_samples(
    episodes: list[Episode],
    rng: np.random.Generator,
    per_segment: int = 2,
) -> list[Sample]:
    """Observation/target pairs following RVT's next-keyframe convention.

    For every consecutive pair of keyframes (k, k+1) the target is keyframe k+1's
    end-effector pose, and the observations are keyframe k itself plus
    `per_segment - 1` frames drawn uniformly from the segment between them. The
    keyframe states are the ones a keypose policy actually sees when it runs
    closed-loop; the interior frames are what stop it from only ever having seen
    a dozen stereotyped postures.
    """
    out: list[Sample] = []
    for i, ep in enumerate(episodes):
        kf = ep.keyframes
        if len(kf):
            # Home -> first keypose. Omitting this was a real hole: it is the
            # only transition where the arm has NOT already been steered toward
            # the object, so it is the most vision-dependent decision in the
            # episode, and a policy never trained on it is off-distribution at
            # the very first step of any rollout. Measured: a policy scoring
            # 3.6 mm on keyframe-to-keyframe prediction commanded a grasp at the
            # home pose, closed on air, and carried an empty gripper to the end.
            out.append(Sample(i, 0, int(kf[0])))
            for _ in range(per_segment - 1):
                if kf[0] > 2:
                    out.append(Sample(i, int(rng.integers(1, kf[0])), int(kf[0])))
        for a, b in zip(kf[:-1], kf[1:]):
            out.append(Sample(i, int(a), int(b)))
            for _ in range(per_segment - 1):
                if b - a > 2:
                    out.append(Sample(i, int(rng.integers(a + 1, b)), int(b)))
    return out


def rot_to_6d(rot: np.ndarray) -> np.ndarray:
    """First two columns of a rotation matrix -- the continuous 6D encoding.

    Rotation matrices and quaternions both have discontinuities as regression
    targets (Zhou et al., CVPR 2019); the first two columns do not, and the
    third is recovered by cross product.
    """
    return rot[..., :, :2].reshape(*rot.shape[:-2], 6)


def targets_for(episodes: list[Episode], samples: list[Sample]) -> dict[str, np.ndarray]:
    """Ground truth: next keyframe pose, plus the proprioception at observation."""
    pos, rot6, grip, prop, prop_full, event = [], [], [], [], [], []
    for s in samples:
        e = episodes[s.ep]
        pos.append(e.tcp[s.target])
        rot6.append(rot_to_6d(e.jaw_rot[s.target]))
        now_open = float(e.jaw_cmd[s.target] > GRIP_OPEN_THRESHOLD)
        grip.append(now_open)
        # proprioception the robot genuinely has: its six joint angles and the
        # jaw command it is currently holding.
        prop.append(low_dim_state(e, s.frame))
        prop_full.append(np.concatenate([e.qpos[s.frame, :6], [e.jaw_cmd[s.frame]]]))
        event.append(_event_at(e, s.target))
    return {
        "target_pos": np.asarray(pos, dtype=np.float32),
        "target_rot6": np.asarray(rot6, dtype=np.float32),
        "target_grip": np.asarray(grip, dtype=np.float32),
        "proprio": np.asarray(prop, dtype=np.float32),
        "proprio_full": np.asarray(prop_full, dtype=np.float32),
        "event": np.asarray(event, dtype=np.int64),
    }


#: What PerAct and RVT actually feed their policies as `low_dim_state`: the
#: gripper's own state and how far through the episode it is. **Not** the joint
#: angles.
#:
#: The distinction is not pedantry, it is the difference between a perception
#: benchmark and a no-op. The scripted expert is a deterministic function of the
#: block's pose, so by the time the arm reaches keyframe k its joint
#: configuration already encodes where the block is. Measured on this dataset: a
#: policy that knows only which keyframe transition it is on predicts the next
#: keypose to 24.4 mm, one that copies its nearest neighbour in joint space gets
#: 16.3 mm, and a *trained* network given qpos[:6] reaches 6.1 mm with no cameras
#: at all. Any 3D-versus-2D comparison run on stereotyped scripted demos with
#: joint-angle proprioception is measuring almost nothing.
PROPRIO_DIM = 4
PROPRIO_FULL_DIM = 7


def low_dim_state(episode: Episode, frame: int) -> np.ndarray:
    """(jaw angle, jaw command, gripper open, progress) -- PerAct's four numbers."""
    jaw_q = float(episode.qpos[frame, 5])
    jaw_cmd = float(episode.jaw_cmd[frame])
    span = max(1e-6, JAW_OPEN - JAW_SHUT)
    return np.array([
        (jaw_q - JAW_SHUT) / span,
        (jaw_cmd - JAW_SHUT) / span,
        float(jaw_cmd > GRIP_OPEN_THRESHOLD),
        frame / max(1, len(episode.jaw_cmd) - 1),
    ], dtype=np.float32)


#: Phase labels. Averaging translation error over a whole trajectory hides the
#: only two keyposes whose accuracy decides the task: the one where the jaws
#: close on a 20 mm block, and the one where they open over the container. The
#: rest are transit poses with centimetres of slack, and a method can look fine
#: on the average while missing every grasp.
EVENT_NONE, EVENT_CLOSE, EVENT_OPEN = 0, 1, 2
EVENT_NAMES = {EVENT_NONE: "transit", EVENT_CLOSE: "grasp", EVENT_OPEN: "release"}


def _event_at(episode: Episode, frame: int) -> int:
    """Whether the gripper changes state at this keyframe, and in which direction."""
    if frame == 0:
        return EVENT_NONE
    was = episode.jaw_cmd[frame - 1] > GRIP_OPEN_THRESHOLD
    now = episode.jaw_cmd[frame] > GRIP_OPEN_THRESHOLD
    if was and not now:
        return EVENT_CLOSE
    if now and not was:
        return EVENT_OPEN
    return EVENT_NONE


def render_condition(
    scene,
    episodes: list[Episode],
    samples: list[Sample],
    condition: Condition,
    *,
    progress: int = 0,
) -> dict[str, np.ndarray]:
    """Photograph every sample under one condition.

    The rig is re-sampled **per episode**, not per frame: a camera that jitters
    between two frames of the same episode is not a camera that was remounted,
    it is a camera falling off the wall, and it would hand every method a free
    averaging signal that does not exist in reality.
    """
    import mujoco

    rng = np.random.default_rng(
        abs(hash((condition.theta_deg, condition.eps_deg, condition.noise_c, condition.seed)))
        % (2**32)
    )
    cams = R.ALL_CAMERAS
    n, v, s = len(samples), len(cams), scene.image_size
    rgb = np.empty((n, v, s, s, 3), dtype=np.uint8)
    depth_mm = np.empty((n, v, s, s), dtype=np.uint16)
    Ks = np.empty((n, v, 3, 3), dtype=np.float32)
    Ts = np.empty((n, v, 4, 4), dtype=np.float32)

    model, data = scene.model, scene.data
    ids = scene.scene.ids
    order = np.argsort([sm.ep for sm in samples], kind="stable")
    current_ep, ep_rig = -1, None

    for count, idx in enumerate(order):
        sm = samples[idx]
        e = episodes[sm.ep]
        if sm.ep != current_ep:
            current_ep = sm.ep
            theta = (
                float(rng.uniform(0.0, condition.theta_deg))
                if condition.theta_random
                else condition.theta_deg
            )
            ep_rig = R.perturb_rig(R.NOMINAL_RIG, theta, rng)
            scene.set_rig(ep_rig)
            # per-episode randomised model parameters, restored exactly
            model.geom_size[ids.block_geom] = e.block_size
            model.body_mass[ids.block] = e.block_mass
            model.geom_friction[ids.block_geom] = e.block_friction
            model.body_pos[ids.box] = e.box_pos
        data.qpos[:] = e.qpos[sm.frame]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        obs = scene.capture()
        true_T = {c: obs[c].T for c in cams}
        reported = dict(true_T)
        if condition.eps_deg > 0:
            # only the movable cameras are calibrated externally; the wrist
            # camera's pose comes from the robot's own kinematics, which is a
            # different and much better-conditioned estimate.
            bad = R.miscalibrate({c: true_T[c] for c in R.MOVABLE_CAMERAS}, condition.eps_deg, rng)
            reported.update(bad)

        for j, c in enumerate(cams):
            o = obs[c]
            d = apply_depth_noise(o.depth, condition.noise_c, rng, far=R.FAR)
            rgb[idx, j] = o.rgb
            depth_mm[idx, j] = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
            Ks[idx, j] = o.K
            Ts[idx, j] = reported[c]

        if progress and count % progress == 0:
            print(f"    rendered {count}/{n}", flush=True)

    out = {"rgb": rgb, "depth_mm": depth_mm, "K": Ks, "T": Ts}
    out.update(targets_for(episodes, samples))
    return out
