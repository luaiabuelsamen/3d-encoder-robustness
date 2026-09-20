"""VENDORED from ~/projects/so101-bench (src/tracking_error_il/expert.py), unmodified
except for the import path. See so101_scene.py header.
"""


from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .so101_scene import (
    JAW_OPEN,
    JAW_SHUT,
    RAD_PER_TICK,
    TABLE_TOP,
    PickScene,
)

__all__ = ["ExpertConfig", "EpisodeResult", "ScriptedExpert", "HOME"]

#: Home configuration, matching the scene's ``home`` keyframe.
HOME = np.array([0.0, -1.57079, 1.57079, 1.57079, -1.57079])

Hook = Callable[[PickScene], None]

#: Counts to retreat after the coarse contact find, before the fine re-approach.
#: Must exceed the coarse pass's worst overshoot (~6 frames x 3.9 counts).
_BACKOFF_COUNTS = 26.0


@dataclass(frozen=True)
class ExpertConfig:
    """Tunable parameters of the scripted policy.

    ``grasp_dz`` is the tool height at the grasp relative to the block centre.
    Deeper grasps trade a little pick rate for much better retention: at +4 mm
    the expert picks 15/20 and places 9; at 0 mm it picks 14/20 and places 11.

    ``grip_mode`` selects how the jaw closes. ``"clamp"`` drives it to a fixed
    angle and is the more reliable demonstrator (14/20 against 8/20), but the
    servo saturates and the force channel stops responding to the object.
    ``"force"`` regulates the grip through the quantised channel and is what a
    force-sensitive task needs.
    """

    approach_height: float = 0.075
    lift_height: float = 0.13
    grasp_dz: float = 0.0
    transport_frames: int = 55
    grip_mode: Literal["clamp", "force", "oracle"] = "clamp"
    #: Setpoint for ``force`` mode: the tracking error to regulate to, in
    #: encoder counts. Force is ~2.4 N per count at the pads.
    grip_target_counts: float = 20.0
    #: Retained for the open-loop overshoot grasp used in earlier measurements.
    grip_overshoot_counts: float = 8.0
    #: Force command for ``oracle`` mode, in newtons of true contact force.
    grip_target_newtons: float = 45.0
    #: Jaw angle for ``clamp`` mode. Full close by default; a tuned clamp may
    #: stop short of it. This is the open-loop controller's one knob, and it is
    #: swept in the tournament like every other setpoint -- comparing a tuned
    #: regulator against an untuned clamp would measure tuning, not information.
    clamp_jaw_angle: float = JAW_SHUT
    grip_step_rad: float = 0.006
    pick_threshold: float = 0.03
    #: Accept the aligned grasp heading when its IK residual is below this
    #: (metres); only beyond it is the 180-degree flip considered.
    yaw_fallback_tolerance: float = 0.004


@dataclass
class EpisodeResult:
    """Outcome and per-episode conditions, captured during the episode."""

    picked: bool = False
    placed: bool = False
    peak_rise_m: float = 0.0
    grip_counts: float = 0.0
    grip_force_n: float = 0.0
    block_x: float = 0.0
    block_y: float = 0.0
    block_yaw: float = 0.0
    grasp_yaw: float = 0.0
    ik_error_mm: float = 0.0
    block_half_width: float = 0.0
    block_half_length: float = 0.0
    block_mass: float = 0.0
    block_friction: float = 0.0
    crushed: bool = False
    peak_grip_n: float = 0.0


class _Waypoints:
    """Interpolating waypoint follower emitting joint-target commands.

    ``grip_regulator``, when set, is called every control frame of every arm
    move and returns the next jaw angle. Regulation that runs only during the
    grasp and then freezes the jaw is not force control: with the jaw frozen,
    lift and transport raised the held force by a consistent 15-30 N above the
    setpoint (median peak 57 N at a 25 N setpoint), which crushed blocks the
    grasp itself had gripped correctly. The loop has to stay closed for as long
    as the object is held.
    """

    def __init__(self, scene: PickScene, hook: Hook | None = None) -> None:
        self.scene = scene
        self.q = HOME.copy()
        self.jaw = JAW_OPEN
        self.hook = hook
        self.grip_regulator: Callable[[float], float] | None = None

    def move(
        self,
        target: np.ndarray,
        jaw: float | None = None,
        yaw: float | None = None,
        frames: int = 45,
        settle: int = 8,
        tool_local: np.ndarray | None = None,
    ) -> bool:
        scene = self.scene
        q1, converged = scene.solve_ik(target, self.q, yaw=yaw, tool_local=tool_local)
        j0 = self.jaw
        j1 = self.jaw if jaw is None else jaw
        q0 = self.q.copy()
        for k in range(frames):
            # Cosine easing: linear interpolation has velocity steps at both
            # segment ends, and those impulses showed up as 20-30 N transient
            # spikes in the held grip during transport.
            f = 0.5 * (1.0 - np.cos(np.pi * (k + 1) / frames))
            if jaw is None and self.grip_regulator is not None:
                self.jaw = self.grip_regulator(self.jaw)
                cmd_jaw = self.jaw
            else:
                cmd_jaw = j0 + (j1 - j0) * f
            scene.hold(
                np.concatenate([q0 + (q1 - q0) * f, [cmd_jaw]]),
                hook=self.hook,
            )
        self.q = q1
        if jaw is not None or self.grip_regulator is None:
            self.jaw = j1
        for _ in range(settle):
            if jaw is None and self.grip_regulator is not None:
                self.jaw = self.grip_regulator(self.jaw)
            scene.hold(np.concatenate([self.q, [self.jaw]]), hook=self.hook)
        return converged

    def set_jaw(self, jaw: float, frames: int = 25, settle: int = 10) -> None:
        """Open-loop jaw move, used for releasing and for the clamp grasp."""
        scene, j0 = self.scene, self.jaw
        for k in range(frames):
            scene.hold(
                np.concatenate([self.q, [j0 + (jaw - j0) * (k + 1) / frames]]),
                hook=self.hook,
            )
        self.jaw = jaw
        scene.hold(np.concatenate([self.q, [self.jaw]]), settle, hook=self.hook)

    def close_until_contact(
        self,
        step_rad: float,
        stall_fraction: float = 0.3,
        max_frames: int = 200,
        warmup: int = 4,
        abort: Callable[[], bool] | None = None,
    ) -> tuple[float, bool]:
        """Close the jaw until the servo stalls against the object.

        Contact is detected from the *measured* jaw motion, not from the
        tracking error: while the command advances ``step_rad`` per frame, a
        free jaw follows at very nearly that rate and a blocked one stops dead.
        Measured over a full close, the actual jaw moves 0.0100 rad/frame in
        free space against a 0.0100 rad/frame command, and 0.0001 rad/frame once
        the block stops it -- a hundredfold separation, with no calibration.

        Thresholding the tracking error instead is what failed. Its free-space
        value is not zero but a steady lag (-17 counts here), and it is not
        steady at the start of the move: over the first eight frames the servo is
        still accelerating and reads -6 rising to -17, so a baseline measured
        there underestimates the lag and any small margin above it triggers
        immediately. Brushing the block without gripping it moves the error only
        to -19..-21 at under 6 N, well inside that band, while a real stall runs
        to -38 and beyond. Both quantities are available on hardware; the
        measured velocity is simply the better-conditioned one.

        Returns:
            ``(jaw_angle, contact_found)``.
        """
        # Read the QUANTISED joint reading, not `data.qpos`. The raw simulator
        # state is not available to a real robot, and using it here would make
        # the expert immune to `obs_quantum` -- silently defeating the one
        # ablation this environment exists to run.
        #
        # Motion is then integrated over a WINDOW rather than compared frame to
        # frame, and the window widens with the quantum. At 0.006 rad the jaw
        # advances 3.9 counts per frame, so a quantum of 8 counts makes most
        # single-frame differences read zero and a frame-to-frame test would
        # declare a stall immediately. That would manufacture the very effect
        # this environment is meant to measure. Integrating until the expected
        # travel is a small multiple of the quantum gives the detector the best
        # it can do at each resolution, so whatever degradation survives is a
        # property of the channel and not of a naive controller.
        found = self._stall_close(step_rad, stall_fraction, max_frames, warmup,
                                  abort=abort)
        if not found:
            return self.jaw, False
        # Two-stage entry. Detection lags the stall by a few frames, and at
        # 0.006 rad/frame the jaw gains 9.4 N of grip per frame of lag -- the
        # measured entry force was 41 N median, 56-57 N peak, and that spike,
        # not anything later, was what tripped the crush latch: phase-tagged
        # traces put the episode's maximum force inside this function on every
        # episode. So having found the contact coarsely, back off clear of it
        # and re-approach at quarter speed; the same latency then costs ~2.4 N
        # per frame instead.
        self.jaw = float(min(JAW_OPEN, self.jaw + _BACKOFF_COUNTS * RAD_PER_TICK))
        self.scene.hold(np.concatenate([self.q, [self.jaw]]), 6, hook=self.hook)
        # The fine pass creeps at an eighth of the coarse speed. The pinch
        # between two rigid pads develops at ~220 N/mm of closure, so at the
        # coarse 0.31 mm/frame it built 69 N inside a single frame -- faster
        # than any per-frame abort can act. At 0.04 mm/frame the same stiffness
        # yields under ~9 N/frame, one detection frame's worth.
        fine = self._stall_close(step_rad / 8.0, stall_fraction, max_frames, warmup,
                                 abort=abort)
        return self.jaw, fine

    def _stall_close(
        self,
        step_rad: float,
        stall_fraction: float,
        max_frames: int,
        warmup: int,
        abort: Callable[[], bool] | None = None,
    ) -> bool:
        """One closing pass at fixed speed; True when contact is found.

        Two detectors run together. The windowed stall test is the reliable
        slow path. ``abort`` is the fast path: it is checked every frame, and a
        single-frame load signature stops the close immediately -- the stall
        window needs ~5 frames to confirm, and at 0.006 rad/frame those five
        frames are 47 N of extra squeeze, which was the entire entry spike.
        """
        scene = self.scene
        step_counts = step_rad / RAD_PER_TICK
        window = max(3, int(np.ceil(2.0 * scene.obs_quantum / max(step_counts, 1e-9))))
        history: deque[float] = deque(maxlen=window + 1)
        history.append(float(scene.state_ticks()[5]))
        for k in range(max_frames):
            if self.jaw <= JAW_SHUT:
                return False
            self.jaw = max(JAW_SHUT, self.jaw - step_rad)
            scene.hold(np.concatenate([self.q, [self.jaw]]), hook=self.hook)
            if abort is not None and k >= 1 and abort():
                return True
            history.append(float(scene.state_ticks()[5]))
            if k >= warmup and len(history) == window + 1:
                moved = abs(history[-1] - history[0])
                if moved < stall_fraction * window * step_counts:
                    return True
        return False

    def grip_by_overshoot(
        self,
        overshoot_counts: float,
        step_rad: float,
        settle: int = 15,
        ramp: int = 12,
    ) -> float:
        """Find contact, then squeeze a fixed number of encoder counts past it.

        This is the formulation that works, and it is the one available on
        hardware. Grip force is commanded as an integer count of jaw travel
        beyond the contact point, so it is set by a quantity the bus can actually
        represent: ``Goal_Position`` is an integer register, and the resolution
        of the commanded force is exactly the encoder quantum.

        The earlier formulation -- ramp at a fixed rate and stop when the
        tracking error reaches a target -- fails for a reason worth recording.
        Closing from ``JAW_OPEN`` to ``JAW_SHUT`` is 0.75 rad, and a 0.006 rad
        step over 120 frames covers only 0.72, so some episodes exhausted their
        travel before the threshold could fire while others stopped on a
        transient having touched nothing. Measured 8 picks in 20, with grip
        force scattered from 0 to 306 N. Detecting contact first and then
        commanding a displacement decouples "where is the object" from "how hard
        to squeeze", and only the second needs to be precise.

        Returns:
            Tracking error after the squeeze, in counts.
        """
        scene = self.scene
        _, found = self.close_until_contact(step_rad)
        if not found:
            return float(scene.delta()[5])
        return self._squeeze(self.jaw - overshoot_counts * RAD_PER_TICK, settle, ramp)

    def regulate_grip(
        self,
        target: float,
        measure: Callable[[], float],
        gain_rad: float,
        tolerance: float,
        max_frames: int = 90,
        settle: int = 10,
        hold_frames: int = 2,
    ) -> float:
        """Drive ``measure()`` to ``target`` by moving the jaw either way.

        A proportional regulator, which is what the grip needs and what the
        earlier one-directional search was not: it only ever closed, so if
        contact detection landed above the setpoint the loop exited immediately
        and left the entry force untouched. Every oracle target from 5 N to 40 N
        consequently produced the same 36.7 N, and the resulting "force floor"
        was an artifact of the controller rather than a property of the gripper.
        Force is 2.4 N per encoder count here; 15 N is six counts of penetration
        and there is nothing physical stopping it.

        The jaw is held still for ``hold_frames`` between adjustments so the
        reading is quasi-static. That matters for the ``delta`` variant: while
        the command is moving, tracking error is dominated by following lag, and
        only once stationary does it report load alone.
        """
        # One-count steps, deliberately. The obvious proportional law
        # (step = gain * error) needs the plant stiffness to choose its gain,
        # and the 2.4 N/count used for that was measured at ONE pose with ONE
        # block: wherever the contact is stiffer, the proportional step
        # overshoots and rings, and peak forces ran 2-5x the setpoint however
        # the gain was shaded. With a position interface and unknown, varying
        # stiffness, a fixed one-count step is the robust law -- worst-case
        # overshoot is bounded by whatever one count costs, with no calibration
        # to be wrong about. `gain_rad` is kept as the step size ceiling.
        scene = self.scene
        step = min(RAD_PER_TICK, abs(gain_rad)) if gain_rad else RAD_PER_TICK
        for _ in range(max_frames):
            error = target - measure()
            if abs(error) <= tolerance:
                break
            self.jaw = float(
                np.clip(self.jaw - np.sign(error) * step, JAW_SHUT, JAW_OPEN)
            )
            scene.hold(
                np.concatenate([self.q, [self.jaw]]), hold_frames, hook=self.hook
            )
        scene.hold(np.concatenate([self.q, [self.jaw]]), settle, hook=self.hook)
        return float(scene.delta()[5])

    def make_regulator(
        self,
        target: float,
        measure: Callable[[], float],
        gain_rad: float = RAD_PER_TICK,
        deadband: float = 3.0,
        jaw_ceiling: float = JAW_OPEN,
    ) -> Callable[[float], float]:
        """A per-frame grip law for :attr:`grip_regulator`.

        Proportional on the measurement error, slew-limited so a transient
        reading cannot fling the jaw open and drop the object mid-transport.
        Closing direction is negative jaw.
        """

        def law(jaw: float) -> float:
            error = target - measure()
            if abs(error) <= deadband:
                return jaw
            if error < 0.0:
                # Force above target. Relieve FAST: transport was measured
                # spiking to 2-5x the setpoint (35-121 N at a 25 N target) from
                # geometric load shifts the one-count law could not outrun, and
                # opening three counts a frame cannot lose a seated block.
                step = 3.0 * RAD_PER_TICK
                # ...but never past the ceiling. The regulator measures TOTAL
                # pad force while actuating only the MOVING jaw: when arm
                # motion presses the block against the FIXED pad above the
                # setpoint, opening cannot reduce that force, and an unclamped
                # relief law walked the jaw fully open while the reading stayed
                # high -- the block was then resting on the fixed pad's
                # friction alone and fell at a measured 17-24 N. Relief may
                # loosen the pinch; it must never surrender it.
                return float(min(jaw_ceiling, jaw + step))
            else:
                # Force below target. Squeeze SLOWLY: overshooting toward the
                # object is the direction that crushes, so it keeps the
                # one-count discipline.
                step = RAD_PER_TICK
            return float(np.clip(jaw - np.sign(error) * step, JAW_SHUT, JAW_OPEN))

        return law

    def seat_and_grip(
        self,
        rise: Callable[[], float],
        target: float,
        tolerance: float,
        step_rad: float,
        relief_counts: float = 4.0,
        max_sweep: int = 220,
        max_regulate: int = 160,
        sweep: Callable[[], bool] | None = None,
    ) -> float:
        """Sweep the object against the fixed jaw, then regulate the grip.

        The grasp has three regimes with different right speeds, and collapsing
        them into one loop is what kept failing:

        1. **Sweep.** After the descent the block sits mid-gap, ~15 mm (290
           counts) from the fixed pad. The moving pad must push it across the
           table -- fast, because nothing builds force here: sliding contact
           flickers at 0-4 N however quickly it is pushed.
        2. **Seat.** The pinch between the pads is stiff (~220 N/mm), so force
           rises the moment the block meets the fixed pad. ``rise()`` measures
           that excess over the sliding tap level; one coarse frame past the
           seat is ~10-15 N, which is why the sweep must stop on the first
           frame of genuine rise, not on an absolute threshold the taps flirt
           with.
        3. **Regulate.** From a seated, known contact, one-count steps walk the
           force to the setpoint. One count per iteration regardless of any
           stiffness calibration: the proportional version needed the plant
           gain, and the 2.4 N/count it assumed is a one-pose measurement that
           the randomised blocks do not honour.

        Returns the final tracking error in counts.
        """
        scene = self.scene
        # 1+2: sweep until seated. `sweep` returns True when seating was
        # detected; the default watches rise(), the delta variant substitutes
        # the windowed stall test because its rise signal is polluted by
        # following lag while the jaw is moving.
        if sweep is None:
            seated = self._sweep_until(rise, step_rad, max_sweep)
        else:
            seated = sweep()
        stalled = not seated
        # relieve the seating transient but keep the block against the pad
        self.jaw = float(min(JAW_OPEN, self.jaw + relief_counts * RAD_PER_TICK))
        scene.hold(np.concatenate([self.q, [self.jaw]]), 4, hook=self.hook)
        if stalled and rise() <= 0.0:
            return float(scene.delta()[5])   # nothing in the jaw; grasp failed
        # 3: one-count regulation to the setpoint.
        for _ in range(max_regulate):
            error = target - rise()
            if abs(error) <= tolerance:
                break
            self.jaw = float(
                np.clip(self.jaw - np.sign(error) * RAD_PER_TICK, JAW_SHUT, JAW_OPEN)
            )
            scene.hold(np.concatenate([self.q, [self.jaw]]), 2, hook=self.hook)
        scene.hold(np.concatenate([self.q, [self.jaw]]), 8, hook=self.hook)
        return float(scene.delta()[5])

    def _sweep_until(
        self, rise: Callable[[], float], step_rad: float, max_frames: int
    ) -> bool:
        """Close at full speed until ``rise()`` goes positive. True if it did."""
        scene = self.scene
        for _ in range(max_frames):
            if self.jaw <= JAW_SHUT:
                return False
            self.jaw = max(JAW_SHUT, self.jaw - step_rad)
            scene.hold(np.concatenate([self.q, [self.jaw]]), hook=self.hook)
            if rise() > 0.0:
                return True
        return False

    def grip_to_true_force(
        self,
        target_newtons: float,
        step_rad: float,
        settle: int = 10,
    ) -> float:
        """Seat and regulate on the TRUE contact force.

        Privileged: reads the simulator's contact solver, which no real robot
        can. It is the upper bound any force controller could reach on this
        task, so the gap to :meth:`grip_to_delta` is the price of having only a
        tracking-error proxy.
        """
        scene = self.scene

        def rise() -> float:
            # sliding taps reach 3.6 N; genuine seating rises past them at once
            return scene.grip_force() - 5.0

        out = self.seat_and_grip(
            rise,
            target=target_newtons - 5.0,
            tolerance=1.5,
            step_rad=step_rad / 2.0,
        )
        self.grip_regulator = self.make_regulator(
            target_newtons,
            scene.grip_force,
            deadband=3.0,
            jaw_ceiling=self.jaw + 10 * RAD_PER_TICK,
        )
        scene.hold(np.concatenate([self.q, [self.jaw]]), settle, hook=self.hook)
        return out

    def grip_to_delta(
        self,
        target_counts: float,
        step_rad: float,
        settle: int = 10,
    ) -> float:
        """Seat and regulate using only the quantised tracking error.

        The realisable counterpart of :meth:`grip_to_true_force`: identical
        structure, but every measurement is ``|delta|`` in encoder counts as
        the real bus reports it. Both its seat detection and its steady-state
        regulation error are bounded by the observation quantum -- the two
        places resolution genuinely enters, with no artificial threshold to
        manufacture either.
        """
        scene = self.scene

        # Seating is detected by the windowed STALL test, not by the tracking
        # error: while the jaw sweeps, |delta| carries ~17 counts of following
        # lag, so any threshold referenced to the stationary value fires in
        # free space -- measured, it aborted every sweep at the first frame and
        # placed 0/15. The stall test reads position history and is immune.
        # |delta| is only trusted as a load signal once the jaw is stationary,
        # which is exactly when the regulation phase runs.
        def sweep() -> bool:
            return self._stall_close(
                step_rad / 2.0, stall_fraction=0.3, max_frames=300, warmup=4
            )

        def rise() -> float:
            return abs(float(scene.delta()[5]))

        out = self.seat_and_grip(
            rise,
            target=target_counts,
            tolerance=1.0,
            step_rad=step_rad / 2.0,
            sweep=sweep,
        )
        self.grip_regulator = self.make_regulator(
            target_counts,
            lambda: abs(float(scene.delta()[5])),
            deadband=max(2.0, scene.obs_quantum),
            jaw_ceiling=self.jaw + 10 * RAD_PER_TICK,
        )
        scene.hold(np.concatenate([self.q, [self.jaw]]), settle, hook=self.hook)
        return out

    def _squeeze(self, target: float, settle: int, ramp: int) -> float:
        """Ramp the jaw to ``target`` and hold, returning the resulting error."""
        scene = self.scene
        target = max(JAW_SHUT, target)
        start = self.jaw
        for k in range(ramp):
            self.jaw = start + (target - start) * (k + 1) / ramp
            scene.hold(np.concatenate([self.q, [self.jaw]]), hook=self.hook)
        self.jaw = target
        scene.hold(np.concatenate([self.q, [self.jaw]]), settle, hook=self.hook)
        return float(scene.delta()[5])


class ScriptedExpert:
    """Runs scripted pick-and-place episodes on a :class:`PickScene`."""

    def __init__(self, scene: PickScene, config: ExpertConfig | None = None) -> None:
        self.scene = scene
        self.config = config or ExpertConfig()

    def choose_grasp_yaw(self, block: np.ndarray, yaw: float) -> tuple[float, float]:
        """Select the heading for closing across the block's short side.

        ``yaw + pi`` is used only as a FALLBACK, never as an equally-ranked
        alternative. A rectangular block's short axis is indeed unchanged by a
        180 degree rotation, but this gripper's tool frame is not symmetric about
        it: the fixed jaw's fingertip sits 11.9 mm to one side of the tool centre
        along the closing axis, so flipping by pi mirrors the pads about the tool
        and puts the block on the wrong side of the jaw. Ranking the two purely
        on IK residual therefore picks the flipped heading whenever it is
        marginally easier to reach and then fails to grasp -- on the nominal
        scene it won with a 0.0009 mm residual against 0.88 mm and lifted the
        block 1.3 mm.

        Near-miss headings are not offered at all. The same fingertip clears the
        block only while its half-width across the closing axis stays under
        11.9 mm, and 14 degrees of misalignment on a 25 mm long axis raises the
        effective half-width to 16.5 mm; the tip then lands on the block and
        presses it into the table. IK residual cannot see any of that.

        Returns:
            ``(heading, ik_residual_metres)``.
        """
        target = block + [0, 0, 0.004]

        def residual(candidate: float) -> float:
            q, _ = self.scene.solve_ik(block + [0, 0, 0.075], HOME, yaw=candidate)
            q, _ = self.scene.solve_ik(target, q, yaw=candidate)
            return float(np.linalg.norm(self.scene.tcp_at(q) - target))

        aligned = residual(yaw)
        if aligned <= self.config.yaw_fallback_tolerance:
            return yaw, aligned
        flipped = residual(yaw + np.pi)
        if flipped < aligned:
            return yaw + np.pi, flipped
        return yaw, aligned

    def run(self, hook: Hook | None = None) -> EpisodeResult:
        """Reset the scene and run one full episode."""
        scene, cfg = self.scene, self.config
        scene.reset()
        block = scene.block_pos()
        z0 = float(block[2])
        grasp_yaw, ik_err = self.choose_grasp_yaw(block, scene.block_yaw())

        model = scene.model
        result = EpisodeResult(
            block_x=float(block[0]),
            block_y=float(block[1]),
            block_yaw=scene.block_yaw(),
            grasp_yaw=grasp_yaw,
            ik_error_mm=1000.0 * ik_err,
            block_half_width=float(model.geom_size[scene.ids.block_geom][0]),
            block_half_length=float(model.geom_size[scene.ids.block_geom][1]),
            block_mass=float(model.body_mass[scene.ids.block]),
            block_friction=float(model.geom_friction[scene.ids.block_geom][0]),
        )

        peak = 0.0

        def track(s: PickScene) -> None:
            nonlocal peak
            peak = max(peak, float(s.block_pos()[2]) - z0)
            if hook is not None:
                hook(s)

        way = _Waypoints(scene, track)

        # Descend as deep as the fingertip allows rather than to a fixed offset
        # from the block centre: the tip is 8.8 mm below the tool, and keeping it
        # 2 mm clear of the table seats the block as far into the jaw as geometry
        # permits. Grasps that survived transport held the block within 3-12 mm
        # of the tool; every one that failed sat at 14-23 mm and was shed.
        # The descent aims the OPEN-jaw gap midpoint at the block, not the
        # demo-derived TCP: the TCP is where the block sits pressed against the
        # FIXED jaw after closing, and descending on it scraped the fixed pad
        # down the block's side at up to 44 N before any grasp began (see
        # PickScene.open_gap_local). The pads reach ~29 mm below the gap
        # midpoint, so the gap is held high enough that the fingertip clears
        # the table, and low enough that the pads straddle the block.
        # ...and within the gap, bias the target so the block arrives 3 mm from
        # the FIXED pad rather than mid-gap. Centring leaves ~15 mm between the
        # block and the fixed pad, and the closing sweep must then shove the
        # block that whole distance across the table -- during which a few
        # degrees of yaw error squirts it out of the jaw sideways: with the
        # grasp and transport controllers already clean (zero crushes), the
        # residual failures were all drops of exactly this kind. Starting the
        # block next to the fixed pad turns the sweep into a short nudge. The
        # clearance is 6 mm: at 3 mm the descent's own lateral noise (IK
        # residual plus eased-path wobble, ~2 mm) had the fixed pad clipping
        # the block on the way down, and the scrape came back at 300-400 N.
        # The fixed pad's inner face sits at local x = +0.0116.
        half_w = float(scene.model.geom_size[scene.ids.block_geom][0])
        gap_tool = scene.open_gap_local.copy()
        gap_tool[0] = 0.0116 - half_w - 0.006
        pad_reach = 0.0294
        grasp_z = max(TABLE_TOP + pad_reach + 0.002, block[2] + cfg.grasp_dz)
        way.move(block + [0, 0, cfg.approach_height], jaw=JAW_OPEN, yaw=grasp_yaw,
                 frames=55, tool_local=gap_tool)
        way.move(
            np.array([block[0], block[1], grasp_z]),
            jaw=JAW_OPEN,
            yaw=grasp_yaw,
            frames=45,
            tool_local=gap_tool,
        )

        # The three grips differ ONLY in what signal sets the squeeze, which is
        # what makes them a measurement rather than three implementations:
        #   clamp  -- no force feedback at all
        #   force  -- the quantised channel the real robot has
        #   oracle -- true contact force, privileged, an upper bound
        if cfg.grip_mode == "force":
            result.grip_counts = way.grip_to_delta(
                cfg.grip_target_counts, cfg.grip_step_rad
            )
        elif cfg.grip_mode == "oracle":
            result.grip_counts = way.grip_to_true_force(
                cfg.grip_target_newtons, cfg.grip_step_rad
            )
        else:
            way.set_jaw(cfg.clamp_jaw_angle)
            result.grip_counts = float(scene.delta()[5])
        result.grip_force_n = scene.grip_force()

        way.move(
            scene.block_pos() * [1, 1, 0] + [0, 0, z0 + cfg.lift_height],
            yaw=grasp_yaw,
            frames=45,
        )
        result.picked = peak > cfg.pick_threshold

        # The container walls top out at z = 0.145. Releasing with the tool at
        # box + 0.085 puts the block's underside above them, falling 6 cm, and it
        # bounces back out; drop it from just inside the rim instead.
        # Transport holds the grasp heading. With yaw unconstrained the IK is
        # free to rotate the closing axis mid-carry, and twisting the pinch
        # around a held block walks it off the 1 mm pad faces.
        box = scene.box_pos()
        way.move(box + [0, 0, 0.16], frames=cfg.transport_frames, yaw=grasp_yaw)
        way.move(box + [0, 0, 0.05], frames=40, yaw=grasp_yaw)
        way.grip_regulator = None          # regulation ends where holding ends
        way.set_jaw(JAW_OPEN, frames=20)
        way.move(box + [0, 0, 0.18], frames=35)

        # A crushed block is a lost episode however tidily it was delivered.
        result.crushed = scene.crushed
        result.peak_grip_n = scene.peak_grip_n
        result.placed = result.picked and scene.block_in_box() and not scene.crushed
        result.peak_rise_m = peak
        return result
