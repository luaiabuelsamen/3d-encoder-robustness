"""PickScene plus a movable four-camera RGBD rig.

Thin wrapper: the physics, IK, randomisation and the scripted expert all come
from the vendored so101-bench scene, which is already validated (9/10 pick and
place on the nominal scene). This file adds only what the 3D study needs --
rendering four RGBD views from a rig whose pose is a runtime variable, and
exporting the calibration that goes with them.

Rendering is the bottleneck for dataset builds, so both renderers are allocated
once and reused, and depth is returned as float32 metres clipped to [NEAR, FAR].
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

from ..render import rig as R  # noqa: E402
from ..vendor.so101_scene import PickScene  # noqa: E402


@dataclass
class CamObs:
    """One camera's contribution to an observation."""

    rgb: np.ndarray        # (H, W, 3) uint8
    depth: np.ndarray      # (H, W) float32, metres, z-depth along optical axis
    K: np.ndarray          # (3, 3) float64
    T: np.ndarray          # (4, 4) float64 camera-to-world, OpenCV convention


class MultiCamScene:
    """A PickScene that can be photographed from an arbitrary rig."""

    def __init__(
        self,
        seed: int = 0,
        image_size: int = 128,
        cameras: tuple[str, ...] = R.ALL_CAMERAS,
        randomise: bool = True,
        **scene_kwargs,
    ) -> None:
        self.scene = PickScene(seed=seed, randomise=randomise, **scene_kwargs)
        self.image_size = int(image_size)
        self.cameras = tuple(cameras)
        self.model, self.data = self.scene.model, self.scene.data

        self._rgb = mujoco.Renderer(self.model, self.image_size, self.image_size)
        self._depth = mujoco.Renderer(self.model, self.image_size, self.image_size)
        self._depth.enable_depth_rendering()

        self._K = {
            c: R.intrinsics(self.model, c, self.image_size, self.image_size)
            for c in self.cameras
        }
        self.set_rig(R.NOMINAL_RIG)

    # ------------------------------------------------------------------- rig

    def set_rig(self, rig: dict[str, R.CameraPose]) -> None:
        """Place the movable cameras. Kinematics are refreshed immediately."""
        R.apply_rig(self.model, self.data, rig)
        mujoco.mj_forward(self.model, self.data)
        self.rig = rig

    # --------------------------------------------------------------- capture

    def capture(self) -> dict[str, CamObs]:
        """Render every camera at the current physics state."""
        out: dict[str, CamObs] = {}
        for cam in self.cameras:
            self._rgb.update_scene(self.data, camera=cam)
            rgb = self._rgb.render().copy()
            self._depth.update_scene(self.data, camera=cam)
            depth = self._depth.render().astype(np.float32)
            # MuJoCo returns the far-plane distance for pixels that hit nothing.
            depth = np.clip(depth, R.NEAR, R.FAR)
            out[cam] = CamObs(rgb, depth, self._K[cam], R.extrinsics(self.model, self.data, cam))
        return out

    def snapshot(self) -> np.ndarray:
        """Full physics state (qpos) -- enough to reproduce this frame exactly."""
        return self.data.qpos.copy()

    def restore(self, qpos: np.ndarray) -> None:
        """Put the scene back at a snapshotted state and refresh kinematics."""
        self.data.qpos[:] = qpos
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------ properties

    @property
    def tcp(self) -> np.ndarray:
        return self.scene.tcp()

    def jaw_frame(self) -> tuple[np.ndarray, np.ndarray]:
        """(position, 3x3 rotation) of the Fixed_Jaw body -- the keypose frame."""
        jid = self.scene.ids.jaw
        return self.data.xpos[jid].copy(), self.data.xmat[jid].reshape(3, 3).copy()

    def close(self) -> None:
        self.scene.close()
