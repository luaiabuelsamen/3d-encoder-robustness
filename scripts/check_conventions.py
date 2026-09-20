"""Look before sweeping: render the rig, unproject it, check the geometry.

Every sign and frame convention in this project is tested here against a fact we
know independently -- the table top is a plane at z = 0.09, the block is where
MuJoCo's segmentation buffer says it is -- rather than assumed. Run this before
trusting any dataset.

The block mask comes from segmentation rendering, not from colour. A colour
heuristic was tried first and is not worth the argument: the arm's orange
(163, 55, 0) and the block's red (192, 89, 58) overlap under specular
highlights, and every threshold that separated them on one camera leaked on
another. Segmentation gives the geom id per pixel, so the mask is ground truth
and the test measures unprojection rather than colour.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import PIL.Image as Image  # noqa: E402

from rvt_lerobot.envs.multicam import MultiCamScene  # noqa: E402
from rvt_lerobot.render import rig as R  # noqa: E402

TABLE_TOP = 0.09

fails: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name:42s}  {detail}")
    if not ok:
        fails.append(name)


def block_mask_factory(scene: MultiCamScene):
    """Exact block mask from MuJoCo's segmentation buffer, shadows disabled.

    Two traps, both found by printing the masked pixels rather than trusting the
    buffer:

    * channel 0 is the object *id* and channel 1 the object *type*; ids are only
      unique within a type, so the type must be checked too;
    * with shadows on, MuJoCo writes the **caster's id into its shadow**. The
      block's shadow fell on the table 15 cm away, and those pixels dragged the
      "block centroid" 100 mm off while the real block pixels were landing
      within 1 mm. Shadows stay on for the RGB renders -- they are real image
      content the policies should see -- and off only here.
    """
    seg = mujoco.Renderer(scene.model, scene.image_size, scene.image_size)
    seg.enable_segmentation_rendering()
    gid = scene.model.geom("block_geom").id

    def mask(camera: str, depth: np.ndarray) -> np.ndarray:
        seg.update_scene(scene.data, camera=camera)
        seg.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        s = seg.render()
        return (
            (s[:, :, 0] == gid)
            & (s[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM)
            & (depth < R.FAR - 1e-3)
        )

    return mask


def block_centroid(scene, block_mask, obs, cam) -> np.ndarray | None:
    o = obs[cam]
    m = block_mask(cam, o.depth)
    if m.sum() < 15:
        return None
    return R.unproject(o.depth, o.K, o.T)[m].mean(0)


def main() -> int:
    scene = MultiCamScene(seed=0, image_size=128)
    scene.scene.reset(block_xy=(0.10, -0.25), block_yaw=0.0)
    scene.set_rig(R.NOMINAL_RIG)
    block_mask = block_mask_factory(scene)
    obs = scene.capture()
    truth = scene.scene.block_pos()

    print(f"\ncontacts at reset: ncon={scene.data.ncon}")
    for i in range(scene.data.ncon):
        c = scene.data.contact[i]
        print(
            f"   {scene.model.geom(c.geom1).name:24s} <-> "
            f"{scene.model.geom(c.geom2).name:24s} dist={c.dist:+.5f}"
        )
    print(f"block truth = {truth.round(4)}\n")

    # 1. every camera sees something that is not the far plane
    for cam, o in obs.items():
        frac = float((o.depth < R.FAR - 1e-3).mean())
        check(f"{cam}: sees geometry", frac > 0.5, f"{frac:.0%} of pixels hit something")

    # 2. the table unprojects to a flat plane at the right height
    for cam in R.MOVABLE_CAMERAS:
        o = obs[cam]
        pts = R.unproject(o.depth, o.K, o.T).reshape(-1, 3)
        pts = pts[o.depth.reshape(-1) < R.FAR - 1e-3]
        m = (
            (np.abs(pts[:, 0]) < 0.25)
            & (pts[:, 1] > -0.5)
            & (pts[:, 1] < 0.2)
            & (pts[:, 2] > 0.05)
        )
        z = pts[m][:, 2]
        hist, edges = np.histogram(z, bins=200, range=(0.05, 0.2))
        modal = 0.5 * (edges[hist.argmax()] + edges[hist.argmax() + 1])
        check(
            f"{cam}: table plane height",
            abs(modal - TABLE_TOP) < 0.004,
            f"modal z = {modal * 1000:.1f} mm (truth {TABLE_TOP * 1000:.0f} mm)",
        )
        flat = z[np.abs(z - TABLE_TOP) < 0.003]
        check(
            f"{cam}: table is flat",
            flat.std() < 0.002,
            f"std = {flat.std() * 1000:.2f} mm over {len(flat)} pts",
        )

    # 3. the block unprojects to where MuJoCo says the block is
    for cam in R.MOVABLE_CAMERAS:
        c = block_centroid(scene, block_mask, obs, cam)
        if c is None:
            check(f"{cam}: block visible", False, "fewer than 15 block pixels")
            continue
        # the centroid of the VISIBLE surface sits about a half-block in front of
        # the centre, so compare in xy and allow one block half-width
        err = float(np.linalg.norm(c[:2] - truth[:2]))
        check(
            f"{cam}: block xy from pixels",
            err < 0.015,
            f"|err| = {err * 1000:.1f} mm  seen={c.round(3)}",
        )

    # 4. the cameras agree with EACH OTHER. (A nearest-neighbour distance between
    #    two clouds would mostly measure point spacing, not calibration.)
    cents = [
        c
        for c in (block_centroid(scene, block_mask, obs, cam) for cam in R.MOVABLE_CAMERAS)
        if c is not None
    ]
    if len(cents) >= 2:
        arr = np.stack(cents)
        spread = float(np.linalg.norm(arr - arr.mean(0), axis=1).max())
        check("cameras agree on block centroid", spread < 0.012, f"max spread = {spread * 1000:.1f} mm")

    # 5. perturbation behaves
    r0 = R.perturb_rig(R.NOMINAL_RIG, 0.0, np.random.default_rng(0))
    check("theta=0 is identity", all(np.allclose(r0[k].eye, R.NOMINAL_RIG[k].eye) for k in r0))
    moved = R.perturb_rig(R.NOMINAL_RIG, 15.0, np.random.default_rng(1))
    shift = float(np.mean([np.linalg.norm(moved[k].eye - R.NOMINAL_RIG[k].eye) for k in moved]))
    check("theta=15 moves cameras", 0.02 < shift < 0.40, f"mean |dx| = {shift * 100:.1f} cm")

    # 6. THE load-bearing property: world-frame geometry is invariant to the rig,
    #    provided the calibration follows the camera.
    scene.set_rig(moved)
    obs2 = scene.capture()
    for cam in R.MOVABLE_CAMERAS:
        c = block_centroid(scene, block_mask, obs2, cam)
        if c is None:
            check(f"{cam}: block visible after perturb", False, "occluded or out of frame")
            continue
        err = float(np.linalg.norm(c[:2] - truth[:2]))
        check(f"{cam}: block xy after perturb", err < 0.015, f"|err| = {err * 1000:.1f} mm")

    # 7. miscalibration must MOVE the reconstruction -- it is the stress axis, so
    #    a no-op would silently null the whole experiment.
    scene.set_rig(R.NOMINAL_RIG)
    obs3 = scene.capture()
    bad = R.miscalibrate({c: obs3[c].T for c in R.MOVABLE_CAMERAS}, 5.0, np.random.default_rng(2))
    shifts = []
    for cam in R.MOVABLE_CAMERAS:
        o = obs3[cam]
        m = block_mask(cam, o.depth)
        good_c = R.unproject(o.depth, o.K, o.T)[m].mean(0)
        bad_c = R.unproject(o.depth, o.K, bad[cam])[m].mean(0)
        shifts.append(float(np.linalg.norm(good_c - bad_c)))
    check(
        "miscalibration displaces the cloud",
        0.01 < np.mean(shifts) < 0.25,
        f"mean |dx| at eps=5 deg = {np.mean(shifts) * 1000:.1f} mm",
    )

    # 8. save what we rendered so a human can look at it
    out = pathlib.Path("figures/conventions")
    out.mkdir(parents=True, exist_ok=True)
    for cam, o in obs3.items():
        Image.fromarray(o.rgb).save(out / f"{cam}_rgb.png")
        d = np.clip((o.depth - 0.3) / 1.0, 0, 1)
        Image.fromarray((255 * (1 - d)).astype(np.uint8)).save(out / f"{cam}_depth.png")
    print(f"\nwrote frames to {out}/")
    print("FAILURES:", fails if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
