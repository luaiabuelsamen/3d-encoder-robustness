"""LIBERO on a Modal GPU, for benchmarking DP3 from huggingface/lerobot#4696.

The PR adds a policy that consumes depth, and LeRobot's LIBERO wrapper does not
produce any. So before a benchmark number can exist at all, two things have to
be true, and this file establishes them in order:

    smoke   LIBERO runs headless on a GPU, robosuite will render depth, and the
            camera intrinsics needed to unproject it can be recovered.
    ...     everything after that depends on the answer.

This machine is a Jetson: the CUDA build of torch here is pinned to Python 3.10
while LeRobot main requires 3.12, so the two cannot meet locally. That is the
only reason this runs remotely.

    modal run cloud/modal_libero.py::smoke
"""

import modal

BRANCH = "3d-pointcloud-observations"
FORK = f"git+https://github.com/luaiabuelsamen/lerobot@{BRANCH}"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "git",
        # MuJoCo offscreen rendering. EGL rather than GLX because there is no
        # display server; osmesa is kept as a fallback for debugging.
        "libegl1",
        "libgl1",
        "libgles2",
        "libglib2.0-0",
        "libosmesa6",
        "libglew2.2",
        "patchelf",
        "ffmpeg",
        # egl_probe (a robosuite dependency, pulled in by hf-libero) builds a
        # native extension from source and shells out to cmake, so the wheel
        # build fails without these.
        "cmake",
        "build-essential",
    )
    .pip_install("torch", "torchvision")
    .run_commands(f'pip install "lerobot[libero] @ {FORK}"')
    # LIBERO prompts on stdin for its asset paths the first time
    # `libero.libero` is imported, and a container has no stdin, so the import
    # dies with EOFError. Write the config at build time from the installed
    # package layout instead. `find_spec` is used rather than an import because
    # importing the subpackage is what triggers the prompt.
    .run_commands(
        "python -c \""
        "import os, yaml, importlib.util; "
        "base = os.path.dirname(importlib.util.find_spec('libero.libero').origin); "
        "cfg = {"
        "'benchmark_root': base, "
        "'bddl_files': os.path.join(base, 'bddl_files'), "
        "'init_states': os.path.join(base, 'init_files'), "
        "'datasets': '/root/libero_datasets', "
        "'assets': os.path.join(base, 'assets')}; "
        "os.makedirs(os.path.expanduser('~/.libero'), exist_ok=True); "
        "yaml.safe_dump(cfg, open(os.path.expanduser('~/.libero/config.yaml'), 'w')); "
        "print('libero config:', cfg); "
        "print('exists:', {k: os.path.isdir(v) for k, v in cfg.items()})"
        "\""
    )
    .env({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("dp3-libero")


@app.function(image=image, gpu="A10G", timeout=1800)
def smoke() -> dict:
    """Can LIBERO render depth, and can we recover the intrinsics to unproject it?"""
    import numpy as np
    import torch

    out: dict = {"cuda": torch.cuda.is_available(), "torch": torch.__version__}
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  "
          f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")

    import lerobot
    print("lerobot", getattr(lerobot, "__version__", "?"))

    # The PR's own code must be importable in this image, or nothing downstream
    # is testing the PR.
    from lerobot.policies.dp3.modeling_dp3 import DP3Policy  # noqa: F401
    from lerobot.processor.depth_processor import DepthToPointCloudStep  # noqa: F401
    print("DP3 and the depth processor import from the installed branch")

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
    print(f"task 0: {task.language}")

    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=128,
        camera_widths=128,
        camera_depths=True,          # the question
        camera_names=["agentview", "robot0_eye_in_hand"],
    )
    env.seed(0)
    raw = env.reset()
    out["obs_keys"] = sorted(raw.keys())
    print("observation keys:", out["obs_keys"])

    depth_keys = [k for k in raw if "depth" in k]
    out["depth_keys"] = depth_keys
    if not depth_keys:
        out["ok"] = False
        print("NO DEPTH KEYS -- camera_depths did not take effect")
        env.close()
        return out

    d = np.asarray(raw[depth_keys[0]]).squeeze()
    out["depth_shape"] = list(d.shape)
    out["depth_range"] = [float(d.min()), float(d.max())]
    print(f"{depth_keys[0]}: shape {d.shape}, range {d.min():.4f} to {d.max():.4f}")

    # robosuite returns NORMALISED depth in [0, 1]; metric depth needs the near
    # and far planes. Getting this wrong is a scale error of the exact kind the
    # PR's guard exists to catch, so it is established here rather than assumed.
    sim = env.env.sim
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    metric = near / (1.0 - d * (1.0 - near / far))
    out["near_far"] = [float(near), float(far)]
    out["metric_depth_range"] = [float(metric.min()), float(metric.max())]
    print(f"near={near:.4f} far={far:.4f} -> metric depth {metric.min():.3f} to {metric.max():.3f} m")

    # Intrinsics. Without these the depth cannot be unprojected, which is the
    # whole reason the PR added get_intrinsics() for RealSense.
    from robosuite.utils.camera_utils import get_camera_intrinsic_matrix

    K = get_camera_intrinsic_matrix(sim, "agentview", 128, 128)
    out["intrinsics"] = np.asarray(K).tolist()
    print("agentview intrinsics:\n", np.round(K, 2))

    env.close()
    out["ok"] = True
    return out


@app.local_entrypoint()
def main():
    result = smoke.remote()
    print("\n=== smoke result ===")
    for k, v in result.items():
        print(f"  {k}: {v}")


@app.function(image=image, gpu="A10G", timeout=1800)
def probe_geometry() -> dict:
    """Which orientation must LIBERO depth be unprojected in?

    robosuite's image buffers are not in standard top-left origin: LeRobot's own
    `LiberoEnv.render` flips both axes before display. Unprojecting with the
    wrong convention mirrors the reconstruction, and nothing downstream would
    report it, so this is settled against ground truth rather than reasoned
    about.

    The observation dict carries true world positions for every object in the
    task (`akita_black_bowl_1_pos` and friends). So: unproject the depth with
    the intrinsics, put it in world coordinates with the extrinsics, and see
    which of the four candidate orientations puts reconstructed surface points
    nearest the objects the simulator says are there. Only the correct one can
    land close to all of them at once.
    """
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=128, camera_widths=128,
        camera_depths=True, camera_names=["agentview"],
    )
    env.seed(0)
    raw = env.reset()
    sim = env.env.sim

    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    d_norm = np.asarray(raw["agentview_depth"]).squeeze()
    depth = near / (1.0 - d_norm * (1.0 - near / far))

    K = np.asarray(get_camera_intrinsic_matrix(sim, "agentview", 128, 128))
    T = np.asarray(get_camera_extrinsic_matrix(sim, "agentview"))

    objects = {k[: -len("_pos")]: np.asarray(v)
               for k, v in raw.items() if k.endswith("_pos") and "robot0" not in k}
    print("ground-truth object positions:")
    for name, pos in objects.items():
        print(f"  {name:45s} {np.round(pos, 3)}")

    h, w = depth.shape
    vv, uu = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    def unproject(dep, flip_v: bool, flip_u: bool) -> np.ndarray:
        u = (w - 1 - uu) if flip_u else uu
        v = (h - 1 - vv) if flip_v else vv
        z = dep
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        cam = np.stack([x, y, z], -1).reshape(-1, 3)
        # robosuite's camera frame looks down -z with +y up (OpenGL), while the
        # pinhole model above assumes +z forward and +y down, so two axes flip.
        cam = cam * np.array([1.0, -1.0, -1.0])
        return cam @ T[:3, :3].T + T[:3, 3]

    results = {}
    for flip_v in (False, True):
        for flip_u in (False, True):
            world = unproject(depth, flip_v, flip_u)
            keep = (depth.reshape(-1) > near * 1.01) & (depth.reshape(-1) < 5.0)
            pts = world[keep]
            dists = [float(np.linalg.norm(pts - p, axis=1).min()) for p in objects.values()]
            key = f"flip_v={flip_v},flip_u={flip_u}"
            results[key] = {"worst_mm": max(dists) * 1000, "mean_mm": float(np.mean(dists)) * 1000}
            print(f"  {key:26s} nearest-surface-point to each object: "
                  f"worst {max(dists)*1000:7.1f} mm, mean {np.mean(dists)*1000:7.1f} mm")

    best = min(results, key=lambda k: results[k]["worst_mm"])
    print(f"\nbest orientation: {best}  (worst {results[best]['worst_mm']:.1f} mm)")
    env.close()
    return {"results": results, "best": best,
            "near_far": [float(near), float(far)],
            "intrinsics": K.tolist(), "extrinsics": T.tolist()}


@app.local_entrypoint()
def geometry():
    r = probe_geometry.remote()
    print("\n=== best orientation ===", r["best"])


@app.function(image=image, gpu="A10G", timeout=1800)
def probe_frames() -> dict:
    """Where does the 1.7 m error in probe_geometry come from?

    All four image orientations landed within 4% of each other at about 1.7 m,
    so the fault is not orientation. Something more basic is wrong, and the
    candidates are: the extrinsic is world-to-camera rather than camera-to-world,
    or robosuite's camera axes are not what the pinhole model assumes. This
    prints the intermediate quantities instead of scoring end results, so the
    step that is wrong is visible rather than inferred.
    """
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128,
                             camera_depths=True, camera_names=["agentview"])
    env.seed(0)
    raw = env.reset()
    sim = env.env.sim

    K = np.asarray(get_camera_intrinsic_matrix(sim, "agentview", 128, 128))
    T = np.asarray(get_camera_extrinsic_matrix(sim, "agentview"))
    cam_id = sim.model.camera_name2id("agentview")
    print("extrinsic matrix T:\n", np.round(T, 4))
    print("T[:3,3]                  =", np.round(T[:3, 3], 4))
    print("sim.data.cam_xpos        =", np.round(sim.data.cam_xpos[cam_id], 4))
    print("  -> if these agree, T is camera-to-world; if not, it is world-to-camera")

    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    d = np.asarray(raw["agentview_depth"]).squeeze()
    depth = near / (1.0 - d * (1.0 - near / far))
    print(f"depth: {depth.min():.3f} to {depth.max():.3f} m")

    objects = {k[:-4]: np.asarray(v) for k, v in raw.items()
               if k.endswith("_pos") and "robot0" not in k}
    gt = np.stack(list(objects.values()))
    print("object positions: x", np.round([gt[:,0].min(), gt[:,0].max()], 3),
          " y", np.round([gt[:,1].min(), gt[:,1].max()], 3),
          " z", np.round([gt[:,2].min(), gt[:,2].max()], 3))

    h, w = depth.shape
    vv, uu = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    x = (uu - K[0, 2]) * depth / K[0, 0]
    y = (vv - K[1, 2]) * depth / K[1, 1]
    cam_raw = np.stack([x, y, depth], -1).reshape(-1, 3)

    out = {}
    for axes_name, axes in (("+z fwd,+y down", (1, 1, 1)), ("OpenGL: -z fwd,+y up", (1, -1, -1))):
        c = cam_raw * np.array(axes, float)
        for t_name, world in (
            ("T as cam2world", c @ T[:3, :3].T + T[:3, 3]),
            ("T as world2cam", (c - T[:3, 3]) @ T[:3, :3]),
        ):
            keep = (depth.reshape(-1) > near * 1.01) & (depth.reshape(-1) < 5.0)
            pts = world[keep]
            dists = [float(np.linalg.norm(pts - p, axis=1).min()) for p in objects.values()]
            key = f"{axes_name} | {t_name}"
            out[key] = max(dists) * 1000
            print(f"  {key:42s} bounds x{np.round([pts[:,0].min(),pts[:,0].max()],2)} "
                  f"z{np.round([pts[:,2].min(),pts[:,2].max()],2)}  worst {max(dists)*1000:8.1f} mm")

    best = min(out, key=out.get)
    print(f"\nbest: {best}  ({out[best]:.1f} mm)")
    env.close()
    return {"scores_mm": out, "best": best}


@app.function(image=image, gpu="A10G", timeout=1800)
def probe_orient() -> dict:
    """Settle the image orientation, now that the axis convention is known.

    `probe_frames` established two things: the extrinsic from
    `get_camera_extrinsic_matrix` is camera-to-world, and the camera axes are
    plain pinhole (+z forward, +y down) rather than the OpenGL convention I had
    assumed, which was the 1.7 m error. With those fixed the worst object
    distance was still 153 mm, which is too large for a surface point.

    Two candidate explanations, and they are distinguished by looking at the
    objects individually rather than at the worst of them:

      * the image is stored bottom-up, so the reconstruction is mirrored;
      * nothing is wrong, and the worst object is simply occluded from this
        camera, so there is no surface point near it to find.

    Per-object distances separate those immediately. A mirrored cloud is wrong
    for every object at once. Occlusion is wrong for one.
    """
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128,
                             camera_depths=True, camera_names=["agentview"])
    env.seed(0)
    raw = env.reset()
    sim = env.env.sim

    K = np.asarray(get_camera_intrinsic_matrix(sim, "agentview", 128, 128))
    T = np.asarray(get_camera_extrinsic_matrix(sim, "agentview"))
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    d = np.asarray(raw["agentview_depth"]).squeeze()
    depth = near / (1.0 - d * (1.0 - near / far))

    objects = {k[:-4]: np.asarray(v) for k, v in raw.items()
               if k.endswith("_pos") and "robot0" not in k}
    h, w = depth.shape
    vv, uu = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    out = {}
    for flip_v in (False, True):
        for flip_u in (False, True):
            u = (w - 1 - uu) if flip_u else uu
            v = (h - 1 - vv) if flip_v else vv
            dep = depth[::-1] if flip_v else depth
            dep = dep[:, ::-1] if flip_u else dep
            x = (u - K[0, 2]) * dep / K[0, 0]
            y = (v - K[1, 2]) * dep / K[1, 1]
            cam = np.stack([x, y, dep], -1).reshape(-1, 3)
            world = cam @ T[:3, :3].T + T[:3, 3]
            keep = (dep.reshape(-1) > near * 1.01) & (dep.reshape(-1) < 5.0)
            pts = world[keep]
            per = {n: float(np.linalg.norm(pts - p, axis=1).min()) * 1000
                   for n, p in objects.items()}
            key = f"flip_v={flip_v},flip_u={flip_u}"
            out[key] = per
            vals = np.array(list(per.values()))
            print(f"{key}:  median {np.median(vals):6.1f} mm   worst {vals.max():7.1f} mm")
            for n, mm in sorted(per.items(), key=lambda kv: kv[1]):
                print(f"     {n:48s} {mm:8.1f} mm")

    best = min(out, key=lambda k: float(np.median(list(out[k].values()))))
    print(f"\nbest by median: {best}")
    env.close()
    return {"per_object_mm": out, "best": best}


@app.function(image=image, gpu="A10G", timeout=1800)
def probe_segmentation() -> dict:
    """Verify the unprojection against per-object segmentation, not nearest points.

    Two corrections got here. The extrinsic is camera-to-world, and the camera
    axes are plain pinhole rather than OpenGL (that assumption was a 1.7 m
    error). A third probe appeared to test image orientation and did not: it
    flipped the pixel grid and the depth array together, which cancels, so all
    four "orientations" were the same computation and returned identical
    numbers. Worth stating because identical results across variants is the
    signature of a control that is not controlling anything.

    Nearest-surface-point to an object centre is a weak test anyway: it is
    bounded below by the object's own radius and it cannot distinguish a
    correct cloud from one shifted along the line of sight. Segmentation is
    exact. Mask the points belonging to one object, take their centroid, and
    compare it to the position the simulator reports. If the convention is
    wrong the centroid lands somewhere else entirely; if it is right this
    agrees to within the visible-surface offset, which is one-sided and small.
    """
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
    env = OffScreenRenderEnv(
        bddl_file_name=bddl, camera_heights=128, camera_widths=128,
        camera_depths=True, camera_segmentations="instance", camera_names=["agentview"],
    )
    env.seed(0)
    raw = env.reset()
    sim = env.env.sim
    print("segmentation keys:", [k for k in raw if "segmentation" in k])

    K = np.asarray(get_camera_intrinsic_matrix(sim, "agentview", 128, 128))
    T = np.asarray(get_camera_extrinsic_matrix(sim, "agentview"))
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    depth = near / (1.0 - np.asarray(raw["agentview_depth"]).squeeze() * (1.0 - near / far))

    h, w = depth.shape
    vv, uu = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    def world_points(flip_v: bool) -> np.ndarray:
        dep = depth[::-1] if flip_v else depth
        x = (uu - K[0, 2]) * dep / K[0, 0]
        y = (vv - K[1, 2]) * dep / K[1, 1]
        cam = np.stack([x, y, dep], -1).reshape(-1, 3)
        return cam @ T[:3, :3].T + T[:3, 3]

    seg_key = next(k for k in raw if "segmentation" in k)
    seg = np.asarray(raw[seg_key]).squeeze()
    objects = {k[:-4]: np.asarray(v) for k, v in raw.items()
               if k.endswith("_pos") and "robot0" not in k}

    out = {}
    for flip_v in (False, True):
        pts = world_points(flip_v)
        best_per_object = {}
        for ident in np.unique(seg):
            mask = (seg == ident).reshape(-1)
            if mask.sum() < 20:
                continue
            centroid = pts[mask & (depth.reshape(-1) < 5.0)].mean(0)
            for name, truth in objects.items():
                dist = float(np.linalg.norm(centroid - truth)) * 1000
                if dist < best_per_object.get(name, (1e9, None))[0]:
                    best_per_object[name] = (dist, int(ident))
        med = float(np.median([v[0] for v in best_per_object.values()]))
        out[f"flip_v={flip_v}"] = {n: round(v[0], 1) for n, v in best_per_object.items()}
        print(f"\nflip_v={flip_v}: median best-matching-segment centroid error {med:.1f} mm")
        for n, (mm, ident) in sorted(best_per_object.items(), key=lambda kv: kv[1][0]):
            print(f"   {n:48s} {mm:8.1f} mm  (segment {ident})")

    env.close()
    return out


@app.function(image=image, gpu="A10G", timeout=1800)
def probe_segmentation_mj() -> dict:
    """Same check, but driving MuJoCo's segmentation renderer directly.

    robosuite's own `camera_segmentations="instance"` raises
    `OverflowError: Python integer 256 out of bounds for uint8` under numpy 2,
    because its segmentation sensor writes 256 into a uint8 buffer. That is
    robosuite's bug and there is no reason to route ground truth through it:
    the underlying MuJoCo model is reachable and its segmentation buffer gives
    exact per-geom membership.

    Channel 0 is the object id and channel 1 the object type, and ids are only
    unique within a type, so both are needed. Shadows are disabled because
    MuJoCo writes a caster's id into its own shadow, which in an earlier
    project dragged a measured centroid 100 mm off the object.
    """
    import mujoco
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128,
                             camera_depths=True, camera_names=["agentview"])
    env.seed(0)
    raw = env.reset()
    sim = env.env.sim
    m = sim.model._model
    d = sim.data._data

    K = np.asarray(get_camera_intrinsic_matrix(sim, "agentview", 128, 128))
    T = np.asarray(get_camera_extrinsic_matrix(sim, "agentview"))
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    depth = near / (1.0 - np.asarray(raw["agentview_depth"]).squeeze() * (1.0 - near / far))

    renderer = mujoco.Renderer(m, 128, 128)
    renderer.enable_segmentation_rendering()
    renderer.update_scene(d, camera="agentview")
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    seg = renderer.render()
    geom_ids = seg[:, :, 0]
    is_geom = seg[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
    # The id channel is not guaranteed to be a model geom index for every pixel
    # whose type channel reads geom: a first attempt indexed geom_bodyid with a
    # value of 330 against ngeom = 273. Rather than guess what those are,
    # restrict to ids that are valid model geoms and report how many were not.
    valid = is_geom & (geom_ids >= 0) & (geom_ids < m.ngeom)
    dropped = int((is_geom & ~valid).sum())
    print(f"segmentation {seg.shape}: ngeom={m.ngeom}, "
          f"id range {int(geom_ids[is_geom].min())}..{int(geom_ids[is_geom].max())}, "
          f"{len(np.unique(geom_ids[valid]))} valid geoms, {dropped} px dropped as out-of-range")
    is_geom = valid

    h, w = depth.shape
    vv, uu = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    out = {}
    for flip_v in (False, True):
        dep = depth[::-1] if flip_v else depth
        x = (uu - K[0, 2]) * dep / K[0, 0]
        y = (vv - K[1, 2]) * dep / K[1, 1]
        cam = np.stack([x, y, dep], -1)
        world = (cam.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]).reshape(h, w, 3)

        errs = []
        for gid in np.unique(geom_ids[is_geom]):
            mask = (geom_ids == gid) & is_geom & (dep < 5.0)
            if mask.sum() < 40:
                continue
            body = m.geom_bodyid[gid]
            truth = d.xpos[body]
            centroid = world[mask].mean(0)
            err = float(np.linalg.norm(centroid - truth)) * 1000
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"geom{gid}"
            errs.append((err, name, int(mask.sum())))
        errs.sort()
        med = float(np.median([e[0] for e in errs])) if errs else float("nan")
        out[f"flip_v={flip_v}"] = med
        print(f"\nflip_v={flip_v}: median geom-centroid error {med:.1f} mm over {len(errs)} geoms")
        for err, name, n in errs[:10]:
            print(f"   {name:42s} {err:8.1f} mm   ({n} px)")

    env.close()
    return out


@app.function(image=image, gpu="A10G", timeout=1800)
def probe_project() -> dict:
    """Settle the convention by projecting known points IN, not unprojecting out.

    Segmentation is unavailable here from either direction: robosuite's sensor
    overflows a uint8 under numpy 2, and a second `mujoco.Renderer` on LIBERO's
    existing EGL context returns out-of-range segids and dies inside MuJoCo's
    own render(). Neither is worth fighting, because the question does not
    actually need segmentation.

    Run the camera model forwards instead. Take an object's known world
    position, project it to a pixel with the intrinsics and the inverse
    extrinsic, and read the stored depth at that pixel. For a visible object the
    depth there must be slightly LESS than the distance to its centre, because
    what the camera sees is the near surface. A vertically flipped image sends
    the sample to a different part of the scene entirely, so the two hypotheses
    separate cleanly and the test can fail, which the earlier ones could not.
    """
    import numpy as np
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import (
        get_camera_extrinsic_matrix,
        get_camera_intrinsic_matrix,
    )

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=128, camera_widths=128,
                             camera_depths=True, camera_names=["agentview"])
    env.seed(0)
    raw = env.reset()
    sim = env.env.sim

    K = np.asarray(get_camera_intrinsic_matrix(sim, "agentview", 128, 128))
    T = np.asarray(get_camera_extrinsic_matrix(sim, "agentview"))
    extent = sim.model.stat.extent
    near = sim.model.vis.map.znear * extent
    far = sim.model.vis.map.zfar * extent
    depth = near / (1.0 - np.asarray(raw["agentview_depth"]).squeeze() * (1.0 - near / far))
    h, w = depth.shape

    R, t = T[:3, :3], T[:3, 3]
    objects = {k[:-4]: np.asarray(v) for k, v in raw.items()
               if k.endswith("_pos") and "robot0" not in k}

    rows = []
    for name, P in objects.items():
        cam = R.T @ (P - t)                       # world -> camera
        z = float(cam[2])
        if z <= 0:
            continue
        u = float(K[0, 0] * cam[0] / z + K[0, 2])
        v = float(K[1, 1] * cam[1] / z + K[1, 2])
        if not (0 <= u < w and 0 <= v < h):
            print(f"  {name:44s} projects outside the image ({u:.0f},{v:.0f}) -- skipped")
            continue
        iu, iv = int(round(u)), int(round(v))
        rows.append((name, z, {
            "as stored": float(depth[iv, iu]),
            "flip v": float(depth[h - 1 - iv, iu]),
            "flip u": float(depth[iv, w - 1 - iu]),
            "flip both (180 deg)": float(depth[h - 1 - iv, w - 1 - iu]),
        }))

    variants = list(rows[0][2]) if rows else []
    header = "".join(f"{v:>21s}" for v in variants)
    print(f"\n{'object':40s} {'true z':>8s}{header}")
    for name, z, samples in rows:
        cells = "".join(f"{samples[v]:21.3f}" for v in variants)
        print(f"{name[:40]:40s} {z:8.3f}{cells}")

    scores = {v: float(np.median([abs(s[v] - z) for _, z, s in rows])) * 1000 for v in variants}
    print("\nmedian |sampled depth - true z| (lower is right; residual is the"
          " surface-to-centre offset):")
    for v, mm in sorted(scores.items(), key=lambda kv: kv[1]):
        print(f"   {v:22s} {mm:8.1f} mm")
    best = min(scores, key=scores.get)
    print(f"\nverdict: {best}")
    env.close()
    return {"scores_mm": scores, "verdict": best}


@app.function(image=image, gpu="A10G", timeout=1800)
def verify_env_depth(branch: str = "libero-depth-observations") -> dict:
    """End-to-end check of the LiberoEnv depth path against ground truth.

    Installs the env branch over the image's LeRobot, builds a `LiberoEnv` with
    `use_depth=True`, and asks the same question that settled the convention:
    project each object's known world position to a pixel, read the depth the
    ENV emitted at that pixel, and check it lands just in front of the object.

    This is deliberately the same test, run through the public wrapper instead
    of through my own arithmetic. If the wrapper drops a conversion or an
    orientation, the number moves.
    """
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall",
         f"git+https://github.com/luaiabuelsamen/lerobot@{branch}"],
        check=True,
    )

    import numpy as np
    from robosuite.utils.camera_utils import get_camera_extrinsic_matrix

    from lerobot.envs.libero import DEPTH_IMAGE_ORIGIN, LiberoEnv
    from libero.libero import benchmark

    print("DEPTH_IMAGE_ORIGIN =", DEPTH_IMAGE_ORIGIN)
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    env = LiberoEnv(
        task_suite=suite, task_id=0, task_suite_name="libero_spatial",
        camera_name="agentview_image", obs_type="pixels",
        observation_height=128, observation_width=128, use_depth=True,
    )
    obs, _ = env.reset(seed=0)

    pixels = obs["pixels"]
    print("pixel keys:", sorted(pixels))
    assert "image_depth" in pixels, "env did not emit depth"
    depth = np.asarray(pixels["image_depth"]).squeeze()
    print(f"depth shape {depth.shape}, {depth.min():.3f} to {depth.max():.3f} m")
    K = np.asarray(obs["intrinsics"]["image"])
    print("intrinsics from env:\n", np.round(K, 2))

    sim = env._env.sim
    T = np.asarray(get_camera_extrinsic_matrix(sim, "agentview"))
    raw = sim._get_observations() if hasattr(sim, "_get_observations") else env._env.env._get_observations()
    objects = {k[:-4]: np.asarray(v) for k, v in raw.items()
               if k.endswith("_pos") and "robot0" not in k}

    h, w = depth.shape
    R, t = T[:3, :3], T[:3, 3]
    errs, signs = [], []
    for name, P in objects.items():
        cam = R.T @ (P - t)
        z = float(cam[2])
        if z <= 0:
            continue
        u = int(round(K[0, 0] * cam[0] / z + K[0, 2]))
        v = int(round(K[1, 1] * cam[1] / z + K[1, 2]))
        if not (0 <= u < w and 0 <= v < h):
            continue
        measured = float(depth[v, u])
        errs.append(abs(measured - z) * 1000)
        signs.append(measured <= z + 1e-6)
        print(f"  {name[:40]:40s} true {z:.3f} m   env depth {measured:.3f} m   "
              f"{'in front' if measured <= z else 'BEHIND centre'}")

    med = float(np.median(errs)) if errs else float("nan")
    ok = med < 80.0 and all(signs)
    print(f"\nmedian |env depth - true z| = {med:.1f} mm over {len(errs)} objects; "
          f"all in front of centre: {all(signs)}")
    print("PASS" if ok else "FAIL")
    env.close()
    return {"median_mm": med, "all_in_front": bool(all(signs)), "ok": bool(ok), "n": len(errs)}


@app.function(image=image, gpu="A10G", timeout=1800)
def run_libero_tests(branch: str = "libero-depth-observations") -> dict:
    """Run the depth unit tests where the libero extra actually exists.

    They are guarded by `pytest.importorskip("libero")`, which means they skip
    silently on a machine without it, including the one they were written on.
    A test that has only ever skipped is not a passing test, so it gets run
    somewhere it can fail.
    """
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
         f"git+https://github.com/luaiabuelsamen/lerobot@{branch}"],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)

    # The installed wheel has the source but not tests/, so fetch the file.
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch,
         "https://github.com/luaiabuelsamen/lerobot", "/tmp/lr"],
        check=True,
    )
    r = subprocess.run(
        [sys.executable, "-m", "pytest", "/tmp/lr/tests/envs/test_libero_depth.py",
         "-v", "--no-header", "-p", "no:cacheprovider"],
        capture_output=True, text=True,
    )
    print(r.stdout[-4000:])
    if r.returncode != 0:
        print("STDERR:", r.stderr[-2000:])
    return {"returncode": r.returncode, "tail": r.stdout[-1500:]}


@app.function(image=image, gpu="A10G", timeout=1800)
def probe_demos() -> dict:
    """Where do LIBERO training demonstrations with depth come from?

    DP3 needs depth at training time, and no existing LIBERO dataset has any.
    There are two ways to manufacture it and they are not equally sound:

      replay actions   open-loop replay of recorded actions in a fresh env.
                       Contact-rich manipulation diverges, so the rendered
                       depth would not correspond to the recorded actions.
      restore states   LIBERO's original HDF5 demos carry the full MuJoCo
                       state per step. Restoring each state and re-rendering
                       is exact, with no divergence to worry about.

    The second is obviously right if the states are actually available, so this
    checks what is on disk and what the HDF5 files contain rather than assuming.
    """
    import os

    from libero.libero import get_libero_path

    out = {}
    datasets = get_libero_path("datasets")
    out["datasets_path"] = datasets
    out["exists"] = os.path.isdir(datasets)
    print(f"datasets path: {datasets} (exists={out['exists']})")
    if out["exists"]:
        listing = sorted(os.listdir(datasets))[:10]
        out["listing"] = listing
        print("contents:", listing)

    # Is the LeRobot-format LIBERO dataset reachable? It has no depth, but it
    # would tell us the episode and task structure to mirror.
    try:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info("lerobot/libero")
        out["lerobot_libero_hub"] = True
        print(f"lerobot/libero on the hub: {len(info.siblings)} files")
    except Exception as e:  # noqa: BLE001
        out["lerobot_libero_hub"] = f"{type(e).__name__}: {e}"
        print("lerobot/libero:", out["lerobot_libero_hub"])

    # Are the original demos downloadable, and do they carry states?
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id="yifengzhu-hf/LIBERO-datasets", repo_type="dataset",
            allow_patterns=["libero_spatial/*demo.hdf5"], max_workers=4,
        )
        files = []
        for root, _, names in os.walk(path):
            files += [os.path.join(root, n) for n in names if n.endswith(".hdf5")]
        out["demo_files"] = len(files)
        print(f"downloaded {len(files)} demo files")
        if files:
            import h5py

            with h5py.File(files[0], "r") as f:
                demos = list(f["data"].keys())
                keys = list(f[f"data/{demos[0]}"].keys())
                out["demo_keys"] = keys
                out["n_demos_first_file"] = len(demos)
                has_states = "states" in keys
                out["has_states"] = has_states
                print(f"{os.path.basename(files[0])}: {len(demos)} demos, keys={keys}")
                print(f"  full MuJoCo states present: {has_states}")
                if has_states:
                    print(f"  states shape: {f[f'data/{demos[0]}/states'].shape}")
    except Exception as e:  # noqa: BLE001
        out["demos"] = f"{type(e).__name__}: {e}"
        print("demo download failed:", out["demos"])

    return out


VOL = modal.Volume.from_name("dp3-libero-data", create_if_missing=True)


@app.function(image=image, gpu="A10G", timeout=7200, volumes={"/data": VOL})
def build_dataset(tasks: int = 1, demos: int = 50, size: int = 128,
                  branch: str = "benchmark-dp3-libero") -> dict:
    """Re-render LIBERO demonstrations with depth into a LeRobotDataset.

    LIBERO's HDF5 demos carry the full MuJoCo state at every step, so each
    frame is restored exactly and re-rendered. That matters: the alternative,
    replaying the recorded actions open loop, diverges in contact-rich
    manipulation, and the resulting depth would not correspond to the actions
    stored beside it. The dataset would look entirely normal and be wrong.

    Depth is converted and oriented by the functions under review in #4709, not
    by a copy, so this dataset exercises the code it is meant to validate.
    """
    import os
    import subprocess
    import sys
    import time

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
         f"git+https://github.com/luaiabuelsamen/lerobot@{branch}"],
        check=True,
    )

    import h5py
    import numpy as np
    from huggingface_hub import snapshot_download
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from robosuite.utils.camera_utils import get_camera_intrinsic_matrix

    from lerobot.configs.video import DepthEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.envs.libero import depth_to_metres, orient_depth

    root = f"/data/libero_spatial_depth_{size}"
    if os.path.exists(root):
        import shutil

        shutil.rmtree(root)

    demo_root = snapshot_download(
        repo_id="yifengzhu-hf/LIBERO-datasets", repo_type="dataset",
        allow_patterns=["libero_spatial/*demo.hdf5"], max_workers=4,
    )
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()

    features = {
        "observation.state": {"dtype": "float32", "shape": (9,),
                              "names": [f"s{i}" for i in range(9)]},
        "action": {"dtype": "float32", "shape": (7,), "names": [f"a{i}" for i in range(7)]},
        "observation.images.agentview": {"dtype": "video", "shape": (size, size, 3),
                                         "names": ["height", "width", "channels"]},
        "observation.images.agentview_depth": {"dtype": "video", "shape": (size, size, 1),
                                               "names": ["height", "width", "channels"],
                                               "info": {"is_depth_map": True}},
        "observation.intrinsics.agentview": {"dtype": "float32", "shape": (3, 3),
                                             "names": ["row", "col"]},
    }
    dataset = LeRobotDataset.create(
        repo_id="local/libero-spatial-depth", fps=20, features=features, root=root,
        robot_type="panda", use_videos=True,
        depth_encoder=DepthEncoderConfig(depth_min=0.01, depth_max=5.0),
    )

    t0, n_frames, n_eps = time.time(), 0, 0
    for task_id in range(min(tasks, suite.n_tasks)):
        task = suite.get_task(task_id)
        bddl = f"{get_libero_path('bddl_files')}/{task.problem_folder}/{task.bddl_file}"
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=size, camera_widths=size,
                                 camera_depths=True, camera_names=["agentview"])
        env.reset()
        sim = env.env.sim
        K = np.asarray(get_camera_intrinsic_matrix(sim, "agentview", size, size), dtype=np.float32)

        # LIBERO's Task exposes no demo-file attribute, and the naming has
        # changed between releases, so resolve it by name with a listing-based
        # fallback rather than trusting one spelling.
        candidates = [
            os.path.join(demo_root, "libero_spatial", f"{task.name}_demo.hdf5"),
            os.path.join(demo_root, "libero_spatial",
                         f"{os.path.splitext(task.bddl_file)[0]}_demo.hdf5"),
        ]
        path = next((c for c in candidates if os.path.isfile(c)), None)
        if path is None:
            available = sorted(os.listdir(os.path.join(demo_root, "libero_spatial")))
            raise FileNotFoundError(
                f"no demo file for task {task.name!r}; tried {candidates}; "
                f"available: {available[:4]}..."
            )
        with h5py.File(path, "r") as f:
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:demos]
            for dk in keys:
                grp = f[f"data/{dk}"]
                states, actions = grp["states"][:], grp["actions"][:]
                for i in range(len(actions)):
                    sim.set_state_from_flattened(states[i])
                    sim.forward()
                    obs = env.env._get_observations()
                    depth = depth_to_metres(
                        np.asarray(obs["agentview_depth"], dtype=np.float32), sim)
                    dataset.add_frame({
                        "observation.state": np.asarray(
                            grp["robot_states"][i][:9], dtype=np.float32),
                        "action": np.asarray(actions[i], dtype=np.float32),
                        "observation.images.agentview": np.asarray(obs["agentview_image"]),
                        "observation.images.agentview_depth": orient_depth(depth)[..., None],
                        "observation.intrinsics.agentview": K,
                        "task": task.language,
                    })
                    n_frames += 1
                dataset.save_episode()
                n_eps += 1
        env.close()
        print(f"task {task_id}: {n_eps} episodes, {n_frames} frames, "
              f"{time.time() - t0:.0f}s elapsed", flush=True)

    dataset.finalize()
    VOL.commit()
    elapsed = time.time() - t0
    per_task = elapsed / max(1, min(tasks, suite.n_tasks))
    print(f"\n{n_eps} episodes, {n_frames} frames in {elapsed:.0f}s "
          f"({per_task:.0f}s per task; 10 tasks would be {per_task * 10 / 60:.1f} min)")
    return {"episodes": n_eps, "frames": n_frames, "seconds": elapsed,
            "per_task_s": per_task, "root": root}


@app.function(image=image, gpu="A10G", timeout=1800, volumes={"/data": VOL})
def verify_dataset(size: int = 128, branch: str = "benchmark-dp3-libero") -> dict:
    """Check the generated dataset survives the round trip into point clouds.

    Generating thirty minutes of data and discovering afterwards that the depth
    was stored in the wrong unit, or that every cloud collapses, is exactly the
    failure this project keeps hitting. So task 0 is checked before the other
    nine are built.
    """
    import subprocess
    import sys

    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--force-reinstall",
         f"git+https://github.com/luaiabuelsamen/lerobot@{branch}"],
        check=True,
    )

    import numpy as np
    import torch

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.processor.depth_processor import DepthToPointCloudStep

    root = f"/data/libero_spatial_depth_{size}"
    ds = LeRobotDataset("local/libero-spatial-depth", root=root)
    print(f"{ds.num_episodes} episodes, {ds.num_frames} frames")
    print("depth keys:", ds.meta.depth_keys)

    dkey = "observation.images.agentview_depth"
    step = DepthToPointCloudStep(
        num_points=1024, frame="camera", seed=0, depth_scale=1e-3,
        workspace_centre=(0.0, 0.0, 1.1), workspace_extent=1.6,
    )

    depths, uniques, extents = [], [], []
    for i in range(0, min(400, ds.num_frames), 40):
        item = ds[i]
        d = item[dkey].squeeze().numpy() / 1000.0
        depths.append((float(d.min()), float(d.max())))
        obs = {dkey: item[dkey], "observation.intrinsics.agentview":
               item["observation.intrinsics.agentview"]}
        cloud = step.observation(obs)["observation.pointcloud"].numpy()
        assert np.isfinite(cloud).all()
        uniques.append(len(np.unique(np.round(cloud, 6), axis=0)))
        extents.append(float(np.ptp(cloud, axis=0).min()))

    lo = float(np.min([d[0] for d in depths]))
    hi = float(np.max([d[1] for d in depths]))
    min_unique = int(min(uniques))
    min_extent = float(min(extents))
    print(f"depth range over sampled frames: {lo:.3f} to {hi:.3f} m")
    print(f"unique points per cloud: min {min_unique} of 1024")
    print(f"smallest per-axis extent: {min_extent:.3f} (0 means collapsed)")

    ok = 0.3 < lo < 2.0 and 1.0 < hi < 6.0 and min_unique > 500 and min_extent > 0.05
    print("PASS" if ok else "FAIL")
    return {"depth_lo": lo, "depth_hi": hi, "min_unique": min_unique,
            "min_extent": min_extent, "ok": bool(ok),
            "episodes": ds.num_episodes, "frames": ds.num_frames}
