"""VENDORED from ~/projects/so101-bench (src/tracking_error_il/scene.py).

Unmodified except: (1) SCENE_PATH points at this repo's study scene, and
(2) PickScene.__init__ takes an optional `scene_path`. Kept as a vendored copy
so this repo runs standalone. Original authorship and validation (78%/60% pick
and place, later 9/10 on the nominal scene) belong to that project.
"""


from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")  # headless hosts have no GLX
import mujoco  # noqa: E402

__all__ = [
    "DomainRandomization",
    "PickScene",
    "JAW_OPEN",
    "JAW_SHUT",
    "TABLE_TOP",
    "TCP_LOCAL",
    "RAD_PER_TICK",
]

SCENE_PATH = Path(__file__).resolve().parents[2] / "assets" / "so_arm_scene" / "scene_study.xml"

#: Grasp point expressed in the ``Fixed_Jaw`` body frame.
#:
#: Recovered from teleoperated demonstrations rather than read off the model: for
#: each successful demonstration the block's pose at the moment it left the table
#: was expressed in this frame, and across eight episodes the result agrees to
#: 0.8 mm and 1.8 mm in the two constrained axes. The third axis lies along the
#: jaw opening, where the block may sit anywhere, and scatters by 10 mm.
TCP_LOCAL = np.array([-0.003, -0.0926, 0.0])

#: Distance from the tool frame down to the fixed jaw's fingertip pad
#: (``fixed_jaw_pad_1``). Limits how deep an object can be seated in the jaw
#: before the tip strikes the table.
FINGERTIP_BELOW_TCP = 0.0088

#: Offset of that same fingertip from the tool centre *across* the closing axis.
#: An object wider than this in that direction is struck from above rather than
#: enclosed, so it bounds the graspable width.
FINGERTIP_LATERAL = 0.0119

#: Jaw joint limits in use. ``-0.2`` is fully closed; larger values open. Pad
#: separation is 25.6 mm at 0.0 and 76.9 mm at 1.0.
JAW_SHUT: float = -0.2
JAW_OPEN: float = 0.55

TABLE_TOP: float = 0.09

#: STS3215 encoder resolution. ``Goal_Position`` is an integer register, so both
#: observations and commands are rounded to counts; see :meth:`PickScene.delta`.
TICKS_PER_REV = 4096
RAD_PER_TICK = 2.0 * np.pi / TICKS_PER_REV

_CONTROL_HZ = 30.0


@dataclass(frozen=True)
class DomainRandomization:
    """Per-episode randomisation ranges.

    ``max_half_width`` is a physical limit, not a taste parameter: the fixed
    jaw's fingertip sits ``FINGERTIP_LATERAL`` (11.9 mm) from the tool centre
    across the closing axis, and a block wider than that is struck from above
    instead of enclosed. The cap leaves real margin rather than sitting on that
    limit -- at 11.0 mm the largest blocks had 0.9 mm of clearance, which an
    open-loop clamp forces through but a force-regulated grip cannot. Loosening
    the margin to 9.5 mm raised the clamp expert's place rate from 10/20 to
    15/20 and the force-regulated expert's pick rate from 8/20 to 15/20.
    """

    enabled: bool = True
    block_x: tuple[float, float] = (0.02, 0.16)
    block_y: tuple[float, float] = (-0.30, -0.20)
    block_yaw: tuple[float, float] = (-np.pi, np.pi)
    block_scale: tuple[float, float] = (0.8, 1.25)
    block_mass_scale: tuple[float, float] = (0.5, 2.0)
    block_friction: tuple[float, float] = (0.6, 1.4)
    box_jitter: float = 0.03
    max_half_width: float = 0.0095

    @classmethod
    def off(cls) -> DomainRandomization:
        return cls(enabled=False)


@dataclass
class _Ids:
    """Cached MuJoCo name lookups."""

    block: int
    box: int
    jaw: int
    block_geom: int
    block_qpos: int
    jaw_pads: frozenset[int]


class PickScene:
    """The pick-and-place scene, its kinematics and its randomisation.

    Args:
        seed: seed for the randomisation stream.
        randomise: ranges to sample per episode, or ``None`` for the defaults.
        render: allocate an offscreen renderer for :meth:`image`.
        width, height: render size in pixels.
        quantise: round commands and observations to whole encoder counts.
        obs_quantum: scales the *observation* quantum only; ``0.25`` models an
            encoder four times finer. This is the resolution knob for force
            ablations.
        crush_newtons: grip force above which the block is destroyed and the
            episode is lost. ``None`` keeps the block rigid.
    """

    def __init__(
        self,
        seed: int = 0,
        randomise: DomainRandomization | bool | None = None,
        render: bool = False,
        width: int = 480,
        height: int = 360,
        quantise: bool = True,
        obs_quantum: float = 1.0,
        crush_newtons: float | None = None,
        scene_path: str | Path | None = None,
    ) -> None:
        if isinstance(randomise, bool):
            randomise = DomainRandomization() if randomise else DomainRandomization.off()
        self.dr = randomise if randomise is not None else DomainRandomization()

        self.model = mujoco.MjModel.from_xml_path(str(scene_path or SCENE_PATH))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.quantise = quantise
        self.obs_quantum = obs_quantum
        #: Grip force above which the block counts as destroyed. ``None`` leaves
        #: the block rigid, which is the force-insensitive control condition.
        self.crush_newtons = crush_newtons
        self.crushed = False
        self.peak_grip_n = 0.0
        self.last_action = np.zeros(self.model.nu)
        self.recorder = None
        # Optional runtime jaw guard applied inside hold(): set a factory and
        # every reset() arms a fresh single-shot instance, so collection and
        # evaluation both run "through the guard" with the recorded action
        # being the APPLIED (guarded) one -- bus semantics.
        self.jaw_guard_factory = None
        self.jaw_guard = None

        self._renderer = mujoco.Renderer(self.model, height, width) if render else None
        m = self.model
        self.ids = _Ids(
            block=m.body("block").id,
            box=m.body("box").id,
            jaw=m.body("Fixed_Jaw").id,
            block_geom=m.geom("block_geom").id,
            block_qpos=int(m.jnt_qposadr[m.body_jntadr[m.body("block").id]]),
            jaw_pads=frozenset(
                m.geom(f"{side}_jaw_pad_{i}").id
                for side in ("fixed", "moving")
                for i in range(1, 5)
            ),
        )
        self._nominal = {
            "box_pos": m.body_pos[self.ids.box].copy(),
            "block_size": m.geom_size[self.ids.block_geom].copy(),
            "block_mass": float(m.body_mass[self.ids.block]),
            "block_friction": m.geom_friction[self.ids.block_geom].copy(),
        }
        self.substeps = int(round((1.0 / _CONTROL_HZ) / m.opt.timestep))
        self.open_gap_local = self._measure_open_gap()

    def _measure_open_gap(self) -> np.ndarray:
        """Midpoint of the jaw pads AT JAW_OPEN, in ``Fixed_Jaw`` coordinates.

        Distinct from ``TCP_LOCAL`` by 12.1 mm along the closing axis, and the
        difference is not pedantry. ``TCP_LOCAL`` was recovered from
        demonstrations as the block's position at the moment of lift -- that is,
        after the moving jaw has swept it against the FIXED jaw. Descending with
        that point aimed at the block therefore parks the block against the
        fixed pad's side while the jaws are still open: traced, the descent
        alone ramped the pad contact from 0 to 44 N before any close began, and
        no grip controller can remove force that the arm itself is applying.
        The descent must target the point midway between the OPEN pads; the
        close then sweeps the block to ``TCP_LOCAL`` where the demos say it
        belongs.
        """
        d = self.data
        saved = d.qpos.copy()
        mujoco.mj_resetDataKeyframe(self.model, d, self.model.key("home").id)
        d.qpos[5] = JAW_OPEN
        mujoco.mj_kinematics(self.model, d)
        R = d.xmat[self.ids.jaw].reshape(3, 3)
        origin = d.xpos[self.ids.jaw].copy()
        fixed = d.geom_xpos[self.model.geom("fixed_jaw_pad_3").id]
        moving = d.geom_xpos[self.model.geom("moving_jaw_pad_3").id]
        out = R.T @ (0.5 * (fixed + moving) - origin)
        d.qpos[:] = saved
        mujoco.mj_forward(self.model, d)
        return out

    # ------------------------------------------------------------------ scene

    def reset(
        self,
        block_xy: tuple[float, float] | None = None,
        block_yaw: float | None = None,
    ) -> np.ndarray:
        """Reset to the home keyframe and apply randomisation.

        Explicit ``block_xy`` / ``block_yaw`` override the sampled values, which
        makes individual episodes reproducible without disabling randomisation.

        Returns:
            The block's world position after the reset.
        """
        m, d, dr = self.model, self.data, self.dr
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        self._apply_randomisation()
        self.jaw_guard = (
            self.jaw_guard_factory() if self.jaw_guard_factory else None
        )

        if block_xy is None:
            block_xy = (
                (self.rng.uniform(*dr.block_x), self.rng.uniform(*dr.block_y))
                if dr.enabled
                else (0.10, -0.25)
            )
        if block_yaw is None:
            block_yaw = self.rng.uniform(*dr.block_yaw) if dr.enabled else 0.0

        half_height = m.geom_size[self.ids.block_geom][2]
        a = self.ids.block_qpos
        d.qpos[a : a + 3] = [block_xy[0], block_xy[1], TABLE_TOP + half_height + 0.001]
        d.qpos[a + 3 : a + 7] = [np.cos(block_yaw / 2), 0.0, 0.0, np.sin(block_yaw / 2)]
        d.qvel[:] = 0.0
        self.crushed = False
        self.peak_grip_n = 0.0
        mujoco.mj_forward(m, d)
        return self.block_pos()

    def _apply_randomisation(self) -> None:
        m, dr, rng = self.model, self.dr, self.rng
        nom = self._nominal
        if not dr.enabled:
            m.geom_size[self.ids.block_geom] = nom["block_size"]
            m.body_mass[self.ids.block] = nom["block_mass"]
            m.geom_friction[self.ids.block_geom] = nom["block_friction"]
            m.body_pos[self.ids.box] = nom["box_pos"]
            return

        size = nom["block_size"] * rng.uniform(*dr.block_scale, size=3)
        size[0] = min(size[0], dr.max_half_width)
        m.geom_size[self.ids.block_geom] = size
        m.body_mass[self.ids.block] = nom["block_mass"] * rng.uniform(*dr.block_mass_scale)
        friction = nom["block_friction"].copy()
        friction[0] = rng.uniform(*dr.block_friction)
        m.geom_friction[self.ids.block_geom] = friction
        box = nom["box_pos"].copy()
        box[:2] += rng.uniform(-dr.box_jitter, dr.box_jitter, size=2)
        m.body_pos[self.ids.box] = box

    # ------------------------------------------------------------------ state

    def block_pos(self) -> np.ndarray:
        return self.data.xpos[self.ids.block].copy()

    def block_yaw(self) -> float:
        R = self.data.xmat[self.ids.block].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def box_pos(self) -> np.ndarray:
        return self.data.xpos[self.ids.box].copy()

    def block_in_box(self) -> bool:
        """True when the block rests inside the container walls."""
        block, box = self.block_pos(), self.box_pos()
        return bool(
            abs(block[0] - box[0]) < 0.055
            and abs(block[1] - box[1]) < 0.055
            and block[2] < box[2] + 0.06
        )

    # ------------------------------------------------------------- fragility

    def update_crush(self) -> bool:
        """Latch whether the block has ever been gripped past ``crush_newtons``.

        This is what turns pick-and-place into a task that *needs* force. A rigid
        block tolerates any grip -- measured, an open-loop clamp at 301 N places
        as reliably as a 37 N regulated grip -- so nothing about the task rewards
        controlling the squeeze. A crush limit supplies the missing upper bound:
        below the limit the object may still slip, above it the episode is lost.
        Success then requires landing inside a *window*, and the width of that
        window in encoder counts is what the channel's resolution has to resolve.

        The limit is read from the simulator's contact solver, never observed.
        Call once per control frame; :meth:`hold` does this automatically.
        """
        force = self.grip_force()
        self.peak_grip_n = max(self.peak_grip_n, force)
        if self.crush_newtons is not None and force > self.crush_newtons:
            self.crushed = True
        return self.crushed

    # ------------------------------------------------------------- kinematics

    def tcp(self) -> np.ndarray:
        """World position of the jaw-gap centre at the current configuration."""
        R = self.data.xmat[self.ids.jaw].reshape(3, 3)
        return self.data.xpos[self.ids.jaw] + R @ TCP_LOCAL

    def tcp_at(self, q: np.ndarray, tool_local: np.ndarray | None = None) -> np.ndarray:
        """Tool position for arm configuration ``q``, leaving the state intact."""
        d = self.data
        saved = d.qpos.copy()
        d.qpos[:5] = q
        mujoco.mj_kinematics(self.model, d)
        if tool_local is None:
            out = self.tcp()
        else:
            R = d.xmat[self.ids.jaw].reshape(3, 3)
            out = d.xpos[self.ids.jaw] + R @ tool_local
        d.qpos[:] = saved
        mujoco.mj_kinematics(self.model, d)
        return out

    def solve_ik(
        self,
        target: np.ndarray,
        q0: np.ndarray,
        yaw: float | None = None,
        iters: int = 150,
        damping: float = 0.05,
        tol: float = 1.2e-3,
        tool_local: np.ndarray | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Damped least squares placing the tool on ``target``, jaws level.

        Five joints are solved against three position constraints plus an axis
        alignment holding the jaw's local +Y along world +Z. ``yaw`` additionally
        aims the *closing* axis -- the jaw's local X, not local Z -- along the
        requested heading, which is how the jaws are made to close across an
        object's short side.

        Returns:
            ``(q, converged)``. Note that convergence says nothing about whether
            the arm can *track* the solution; check contacts for that.
        """
        m, d = self.model, self.data
        tool = TCP_LOCAL if tool_local is None else np.asarray(tool_local, float)
        q = np.asarray(q0, dtype=float).copy()
        saved = d.qpos.copy()
        lo, hi = m.jnt_range[:5, 0], m.jnt_range[:5, 1]
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        up = np.array([0.0, 0.0, 1.0])
        converged = False

        for _ in range(iters):
            d.qpos[:5] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            R = d.xmat[self.ids.jaw].reshape(3, 3)
            pos = d.xpos[self.ids.jaw] + R @ tool

            err_pos = target - pos
            err_rot = np.cross(R[:, 1], up)
            if yaw is not None:
                heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
                err_rot = err_rot + 0.6 * np.cross(R[:, 0], heading)

            if np.linalg.norm(err_pos) < tol and np.linalg.norm(err_rot) < 0.04:
                converged = True
                break

            mujoco.mj_jac(m, d, jacp, jacr, pos, self.ids.jaw)
            J = np.vstack([jacp[:, :5], 0.30 * jacr[:, :5]])
            e = np.concatenate([err_pos, 0.30 * err_rot])
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), lo, hi)

        d.qpos[:] = saved
        mujoco.mj_forward(m, d)
        return q, converged

    # ----------------------------------------------------------------- drive

    def hold(self, ctrl: np.ndarray, frames: int = 1, hook=None) -> None:
        """Apply ``ctrl`` for ``frames`` control steps.

        Commands are rounded to encoder counts when ``quantise`` is set, because
        ``Goal_Position`` is an integer register. The recorder, if attached, sees
        the state as it was *before* the command was applied, paired with that
        command -- the ordering the real bus writes, which is what makes
        ``action[t-1] - state[t]`` mean the same thing here as in recorded data.
        """
        for _ in range(frames):
            c = np.asarray(ctrl, dtype=float)
            if self.quantise:
                c = np.round(c / RAD_PER_TICK) * RAD_PER_TICK
            if self.jaw_guard is not None:
                jaw = self.jaw_guard(
                    float(self.state_ticks()[5]), float(c[5] / RAD_PER_TICK)
                )
                c[5] = (round(jaw) if self.quantise else jaw) * RAD_PER_TICK
            if self.recorder is not None:
                self.recorder.append(
                    state=self.state_ticks(), action=c / RAD_PER_TICK, scene=self
                )
            self.last_action = c.copy()
            self.data.ctrl[:] = c
            for _ in range(self.substeps):
                mujoco.mj_step(self.model, self.data)
            self.update_crush()
            if hook is not None:
                hook(self)

    # ----------------------------------------------------------------- force

    def state_ticks(self) -> np.ndarray:
        """Joint positions as the bus reports them, in encoder counts."""
        q = self.data.qpos[: self.model.nu] / RAD_PER_TICK
        if not self.quantise:
            return q
        return np.round(q / self.obs_quantum) * self.obs_quantum

    def action_ticks(self) -> np.ndarray:
        """The most recent command, in encoder counts."""
        return self.last_action / RAD_PER_TICK

    def delta(self) -> np.ndarray:
        """Servo tracking error, ``action - state``, in encoder counts.

        A proportionally controlled servo holding against a load sits at an
        offset from its goal proportional to that load, so this is a force
        signal. It is the only force information the SO-101 exposes -- there is
        no torque sensor -- and it is present in every recorded dataset.
        """
        return self.action_ticks() - self.state_ticks()

    def grip_force(self) -> float:
        """True contact force between the jaw pads and the block, in newtons.

        Ground truth, available only in simulation. Provided so that
        :meth:`delta` can be validated against the quantity it proxies; it must
        never be given to a policy.
        """
        m, d = self.model, self.data
        total = 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            pair = {c.geom1, c.geom2}
            if (pair & self.ids.jaw_pads) and self.ids.block_geom in pair:
                wrench = np.zeros(6)
                mujoco.mj_contactForce(m, d, i, wrench)
                total += float(np.linalg.norm(wrench[:3]))
        return total

    # ---------------------------------------------------------------- render

    def image(self, camera: str | int = -1) -> np.ndarray:
        """Render one RGB frame. Requires ``render=True`` at construction."""
        if self._renderer is None:
            raise RuntimeError("PickScene was constructed with render=False")
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
