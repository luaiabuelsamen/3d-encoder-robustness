"""Draw the pipeline: what each encoder turns a scene into before the network.

The tables in this project compare encoders by a millimetre number, which says
nothing about what they are actually doing to the image. This renders the same
frame through every representation in the study, side by side, so the
comparisons have something to stand on.

Each column is one encoder's view of one instant:

  RGB            what a 2D policy sees
  depth          the fourth channel, and why it is not enough
  camera cloud   points in the camera's own frame, no calibration used
  world cloud    the same points after an extrinsic, fusable across cameras
  virtual views  RVT's canonical re-rendering of that cloud

Rendered from the study's own code paths, not redrawn, so what is shown is what
the policies were trained on.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import PIL.Image as Image  # noqa: E402
import PIL.ImageDraw as ImageDraw  # noqa: E402

from rvt_lerobot.data import collect_study as C  # noqa: E402
from rvt_lerobot.data.batching import ConditionCache, make_images  # noqa: E402
from rvt_lerobot.data.views import Condition, build_samples, render_condition  # noqa: E402
from rvt_lerobot.envs.multicam import MultiCamScene  # noqa: E402
from rvt_lerobot.models.policy import ARMS  # noqa: E402
from rvt_lerobot.render import rig as R  # noqa: E402
from rvt_lerobot.render import virtual as V  # noqa: E402

TILE = 190
BG = 16


def label(img: np.ndarray, text: str, sub: str = "") -> np.ndarray:
    """Caption a tile. Captions go under the picture, never over it."""
    h = 30 if sub else 18
    out = np.full((img.shape[0] + h, img.shape[1], 3), BG, np.uint8)
    out[: img.shape[0]] = img
    pil = Image.fromarray(out)
    d = ImageDraw.Draw(pil)
    d.text((4, img.shape[0] + 2), text, fill=(235, 235, 235))
    if sub:
        d.text((4, img.shape[0] + 15), sub, fill=(150, 150, 150))
    return np.asarray(pil)


def scatter_tile(
    points: np.ndarray,
    ax_u: int,
    ax_v: int,
    colour_axis: int,
    size: int = TILE,
    flip_v: bool = True,
    dot: int = 3,
) -> np.ndarray:
    """Scatter a cloud with EXPLICIT axes, coloured by an explicit axis.

    The first version of this reused the virtual renderer's view names, which
    are defined in world coordinates. Applied to a camera-frame cloud, "front"
    means horizontal-position against DEPTH, and the colour was depth too -- so
    the vertical axis and the colour axis were the same quantity and the panel
    came out a smooth gradient wash with no scene in it. Axes are named here
    rather than inherited.

    Points are drawn as `dot`-pixel squares. A 1024-point cloud at one pixel per
    point reads as static whatever the axes are.
    """
    u, v, c = points[:, ax_u], points[:, ax_v], points[:, colour_axis]
    lo, hi = np.percentile(c, 2), np.percentile(c, 98)
    t = np.clip((c - lo) / max(1e-6, hi - lo), 0, 1)
    colour = np.stack([t, t * 0.8 + 0.2, 1.0 - t], -1)

    span = max(np.ptp(u), np.ptp(v)) or 1.0
    cu, cv = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2
    margin = dot + 2
    scale = (size - 2 * margin) / span
    px = np.clip(((u - cu) * scale + size / 2).astype(int), margin, size - margin - 1)
    py = ((v - cv) * scale + size / 2).astype(int)
    if flip_v:
        py = size - 1 - py
    py = np.clip(py, margin, size - margin - 1)

    img = np.full((size, size, 3), BG, np.uint8)
    order = np.argsort(c)  # near/low drawn first so high sits on top
    for i in order:
        img[py[i]: py[i] + dot, px[i]: px[i] + dot] = (colour[i] * 255).astype(np.uint8)
    return img


def main() -> int:
    scene = MultiCamScene(seed=4242, image_size=96)
    scene.set_rig(R.NOMINAL_RIG)
    episodes = C.load(pathlib.Path("data/test.npz"))[:2]
    samples = build_samples(episodes, np.random.default_rng(0), per_segment=1)[:4]
    cache = ConditionCache(render_condition(scene, episodes, samples, Condition()), device="cpu")
    batch = cache.batch(np.arange(1))

    def up(a, s=TILE):
        return np.asarray(Image.fromarray(a).resize((s, s), Image.NEAREST))

    tiles = []

    rgb = batch["rgb"][0, 0].numpy().astype(np.uint8)
    tiles.append(label(up(rgb), "1. RGB", "what a 2D policy sees"))

    depth = batch["depth_mm"][0, 0].numpy().astype(np.float32) / 1000.0
    valid = (depth > 0.01) & (depth < R.FAR - 1e-3)
    d = np.zeros_like(depth)
    d[valid] = 1.0 - np.clip((depth[valid] - 0.4) / 0.8, 0, 1)
    dep = (np.stack([d * 0.2, d * 0.8, d], -1) * 255).astype(np.uint8)
    tiles.append(label(up(dep), "2. depth", "a 4th channel: worse than none"))

    cam = make_images(ARMS["dp3_pcd"], batch)[0].numpy()
    tiles.append(label(
        scatter_tile(cam, ax_u=0, ax_v=1, colour_axis=2, flip_v=False),
        "3. camera-frame cloud", "1 camera, colour = depth"))

    world = make_images(ARMS["dp3_world"], batch)[0].numpy()
    tiles.append(label(
        scatter_tile(world, ax_u=0, ax_v=2, colour_axis=2),
        "4. world-frame cloud", "4 cams fused, colour = height"))

    virt = make_images(ARMS["rvt"], batch, virtual_size=96)[0]
    v0 = (virt[0, :3].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    v1 = (virt[1, :3].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    pair = np.concatenate([up(v0, TILE // 2), up(v1, TILE // 2)], axis=1)
    pair = np.concatenate([pair, np.full((TILE - TILE // 2, TILE, 3), BG, np.uint8)])
    tiles.append(label(pair, "5. virtual views (RVT)", "canonical re-render of (4)"))

    strip = np.concatenate(
        [np.concatenate([t, np.full((t.shape[0], 8, 3), BG, np.uint8)], 1) for t in tiles], 1
    )
    out = pathlib.Path("figures/pipeline.png")
    out.parent.mkdir(exist_ok=True)
    Image.fromarray(strip).save(out)
    print(f"wrote {out}  ({strip.shape[1]}x{strip.shape[0]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
