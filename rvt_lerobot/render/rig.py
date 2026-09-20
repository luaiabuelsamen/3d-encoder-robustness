"""The movable RGBD camera rig: poses, calibration, unprojection.

Conventions, stated once because every sign bug in this project would live here.

* **MuJoCo camera frame**: -z forward, +y up, +x right.
* **OpenCV camera frame**: +z forward, +y down, +x right. Used for everything
  downstream (intrinsics, unprojection), because that is what the RVT / PerAct
  stack and every depth-camera SDK assume.
  ``R_cv = R_mujoco @ diag(1, -1, -1)``.
* **Extrinsics** here always mean **camera-to-world** 4x4. A point in camera
  coordinates maps to the world as ``p_world = T[:3,:3] @ p_cam + T[:3,3]``.
* MuJoCo's depth renderer returns distance **along the optical axis** (z-depth),
  in metres -- not ray length. ``test_conventions()`` checks that by unprojecting
  the table and confirming it comes back flat.

The three third-person cameras hang off mocap bodies (see
``assets/so_arm_scene/scene_study.xml``) so their pose is a runtime variable.
The wrist camera is welded to ``Fixed_Jaw`` and is never perturbed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np

#: Third-person cameras, in the order the models consume them. The wrist camera
#: is appended last and is not part of the perturbable rig.
MOVABLE_CAMERAS = ("front", "left_shoulder", "right_shoulder")
WRIST_CAMERA = "wrist"
ALL_CAMERAS = MOVABLE_CAMERAS + (WRIST_CAMERA,)

#: Mocap body carrying each movable camera.
CAMERA_MOUNTS = {
    "front": "cam_front_mount",
    "left_shoulder": "cam_left_mount",
    "right_shoulder": "cam_right_mount",
}

#: Centre of the workspace the rig is aimed at, and the half-extent of the cube
#: the virtual cameras render. The block is sampled in x in [0.02, 0.16],
#: y in [-0.30, -0.20]; the container sits near (-0.1, -0.35); the table top is
#: at z = 0.09. A 0.6 m cube centred here contains all of it plus the arm.
WORKSPACE_CENTRE = np.array([0.0, -0.25, 0.15])
WORKSPACE_EXTENT = 0.6

NEAR, FAR = 0.02, 3.0


@dataclass(frozen=True)
class CameraPose:
    """One camera placed by look-at: where it sits and what it aims at."""

    eye: np.ndarray
    target: np.ndarray

    def mujoco_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """(position, quaternion wxyz) for the mocap body carrying this camera."""
        return self.eye.copy(), _look_at_quat(self.eye, self.target)


#: Nominal rig. Three cameras on a ring around the workspace, all aimed at its
#: centre, roughly RLBench's front / left-shoulder / right-shoulder geometry.
NOMINAL_RIG: dict[str, CameraPose] = {
    "front": CameraPose(np.array([0.00, -0.85, 0.45]), WORKSPACE_CENTRE.copy()),
    "left_shoulder": CameraPose(np.array([-0.60, -0.25, 0.45]), WORKSPACE_CENTRE.copy()),
    "right_shoulder": CameraPose(np.array([0.60, -0.25, 0.45]), WORKSPACE_CENTRE.copy()),
}


# --------------------------------------------------------------------- look-at


def _look_at_rotation(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    """MuJoCo-convention 3x3 rotation for a camera at `eye` aimed at `target`.

    Columns are the camera's (x_right, y_up, z_back) axes in world coordinates.
    MuJoCo looks down -z, so the third column is the *backward* direction.
    """
    forward = target - eye
    n = np.linalg.norm(forward)
    if n < 1e-9:
        raise ValueError("camera eye coincides with its target")
    forward = forward / n
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ world_up)) > 0.999:  # looking straight down: pick a tie-break
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return np.stack([right, up, -forward], axis=1)


def _look_at_quat(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, _look_at_rotation(eye, target).reshape(9))
    return q


# ---------------------------------------------------------------- perturbation


def perturb_rig(
    rig: dict[str, CameraPose],
    theta_deg: float,
    rng: np.random.Generator,
    *,
    centre: np.ndarray = WORKSPACE_CENTRE,
) -> dict[str, CameraPose]:
    """Jitter every movable camera by a perturbation of scale `theta_deg`.

    One scalar controls the whole rig so the stress axis is a single number.
    At level theta each camera independently gets

    * azimuth and elevation about `centre` jittered by U(-theta, +theta) degrees,
    * its radius scaled by U(1 - theta/200, 1 + theta/200)  (5% at theta=10),
    * its aim point moved by U(-theta/500, theta/500) metres per axis (2 cm at
      theta=10), which tilts the camera without moving it.

    theta = 0 returns the rig unchanged, exactly. This models the realistic
    failure: someone unbolted the cameras and put them back approximately.
    """
    if theta_deg <= 0:
        return {k: replace(v) for k, v in rig.items()}
    t = np.deg2rad(theta_deg)
    out: dict[str, CameraPose] = {}
    for name, pose in rig.items():
        d = pose.eye - centre
        r = float(np.linalg.norm(d))
        az = np.arctan2(d[1], d[0])
        el = np.arcsin(np.clip(d[2] / r, -1, 1))
        az += rng.uniform(-t, t)
        el = np.clip(el + rng.uniform(-t, t), np.deg2rad(-80), np.deg2rad(80))
        r *= rng.uniform(1 - theta_deg / 200.0, 1 + theta_deg / 200.0)
        eye = centre + r * np.array(
            [np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)]
        )
        target = pose.target + rng.uniform(-theta_deg / 500.0, theta_deg / 500.0, size=3)
        out[name] = CameraPose(eye, target)
    return out


def miscalibrate(
    extrinsics: dict[str, np.ndarray],
    eps_deg: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Corrupt the extrinsics *reported* to a policy, leaving the cameras put.

    This is the axis that separates "the cameras moved" from "we think we know
    where the cameras are". A 3D policy consumes these numbers as truth; an
    RGB policy never sees them at all. Each camera gets an independent rotation
    of exactly `eps_deg` about a uniformly random axis applied in the camera
    frame, plus a translation of ``eps_deg * 1 mm/deg`` in a random direction --
    roughly the residual of a hand-eye calibration that was done once and then
    drifted.
    """
    if eps_deg <= 0:
        return {k: v.copy() for k, v in extrinsics.items()}
    out = {}
    for name, T in extrinsics.items():
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        dR = _axis_angle(axis, np.deg2rad(eps_deg))
        dt = rng.normal(size=3)
        dt = dt / np.linalg.norm(dt) * (eps_deg * 1e-3)
        bad = T.copy()
        bad[:3, :3] = T[:3, :3] @ dR
        bad[:3, 3] = T[:3, 3] + dt
        out[name] = bad
    return out


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


# ------------------------------------------------------------------ calibration


def intrinsics(model, cam_name: str, height: int, width: int) -> np.ndarray:
    """Pinhole K from MuJoCo's vertical field of view.

    MuJoCo's `fovy` is the *vertical* FOV in degrees and the render is
    stretched to the requested aspect, so fx == fy == 0.5 * height / tan(fovy/2).
    """
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise KeyError(f"no camera named {cam_name!r}")
    f = 0.5 * height / np.tan(0.5 * np.deg2rad(float(model.cam_fovy[cam_id])))
    return np.array(
        [[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]]
    )


def extrinsics(model, data, cam_name: str) -> np.ndarray:
    """Camera-to-world 4x4 in OpenCV convention for a live camera."""
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise KeyError(f"no camera named {cam_name!r}")
    T = np.eye(4)
    T[:3, :3] = data.cam_xmat[cam_id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = data.cam_xpos[cam_id]
    return T


def apply_rig(model, data, rig: dict[str, CameraPose]) -> None:
    """Write a rig onto the scene's mocap bodies. Caller runs mj_forward after."""
    for name, pose in rig.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CAMERA_MOUNTS[name])
        if bid < 0:
            raise KeyError(f"no mocap mount for camera {name!r}")
        mid = int(model.body_mocapid[bid])
        if mid < 0:
            raise ValueError(f"body {CAMERA_MOUNTS[name]!r} is not a mocap body")
        pos, quat = pose.mujoco_pose()
        data.mocap_pos[mid] = pos
        data.mocap_quat[mid] = quat


# ---------------------------------------------------------------- unprojection


def unproject(depth: np.ndarray, K: np.ndarray, T_cam2world: np.ndarray) -> np.ndarray:
    """Depth image -> (H, W, 3) world-frame points.

    `depth` is z-depth in metres (distance along the optical axis), which is
    what MuJoCo's depth renderer returns. Pixels are treated as centres, matching
    the principal point convention in `intrinsics`.
    """
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    cam = np.stack([x, y, depth], axis=-1)
    return cam @ T_cam2world[:3, :3].T + T_cam2world[:3, 3]


def unproject_torch(depth, K, T_cam2world):
    """Batched torch twin of `unproject`.

    Shapes: depth (B, V, H, W); K (B, V, 3, 3); T (B, V, 4, 4).
    Returns (B, V, H, W, 3) world points. Differentiable in depth, K and T,
    which is what lets the miscalibration axis be evaluated without re-rendering.
    """
    import torch

    b, v, h, w = depth.shape
    vv, uu = torch.meshgrid(
        torch.arange(h, device=depth.device, dtype=depth.dtype),
        torch.arange(w, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    uu = uu.expand(b, v, h, w)
    vv = vv.expand(b, v, h, w)
    fx = K[..., 0, 0][..., None, None]
    fy = K[..., 1, 1][..., None, None]
    cx = K[..., 0, 2][..., None, None]
    cy = K[..., 1, 2][..., None, None]
    x = (uu - cx) / fx * depth
    y = (vv - cy) / fy * depth
    cam = torch.stack([x, y, depth], dim=-1)                       # (B,V,H,W,3)
    R = T_cam2world[..., :3, :3][:, :, None, None]                 # (B,V,1,1,3,3)
    t = T_cam2world[..., :3, 3][:, :, None, None]                  # (B,V,1,1,3)
    return (R @ cam[..., None]).squeeze(-1) + t
