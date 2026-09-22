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
