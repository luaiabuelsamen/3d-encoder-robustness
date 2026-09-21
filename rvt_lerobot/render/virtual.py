"""Batched orthographic re-rendering of a point cloud into canonical views.

This is RVT's load-bearing operation and the thing the study is about: take the
fused world-frame point cloud and rasterise it from K virtual cameras whose
poses are **fixed in the world**, not attached to any real camera. The network
downstream therefore sees the same viewpoints on every frame of every episode,
whatever the real rig is doing.

The projection is geometric, not learned. Nothing here has parameters.

Implementation note -- the z-buffer. A scatter with "last write wins" gives an
arbitrary point per pixel, and sorting by depth per pixel is expensive. Instead
each point gets a single int64 key,

    key = round(depth_micrometres) * 2^20 + point_index

and a per-pixel `amin` over keys recovers the index of the nearest point exactly,
in one pass. Depth is offset so it is non-negative, and the 2^20 field holds any
point index below a million; both are asserted.

This replaces RVT-2's custom CUDA point renderer and RVT-1's PyTorch3D path,
neither of which builds on Jetson aarch64, and it is a few lines rather than a
build system.
"""

from __future__ import annotations

import torch
from torch import Tensor

#: Virtual camera set: the five faces RVT uses (no bottom -- the table is there).
VIRTUAL_VIEWS: tuple[str, ...] = ("front", "top", "left", "right", "back")

#: view -> (axis_u, axis_v, axis_depth, flip_depth, flip_u)
#: `axis_u` and `axis_v` say which world axes span the image; `axis_depth` is the
#: one the camera looks along. These are the numbers the geometric decoder
#: inverts, so they are stated once, here, and imported everywhere else.
VIEW_SPEC: dict[str, tuple[int, int, int, bool, bool]] = {
    "front": (0, 2, 1, False, False),
    "back": (0, 2, 1, True, True),
    "top": (0, 1, 2, True, False),
    "left": (1, 2, 0, True, True),
    "right": (1, 2, 0, False, False),
}

_IDX_BITS = 20


def project_to_view(
    points: Tensor,
    view: str,
    centre: Tensor,
    extent: float,
    img_size: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """World points -> (col, row, depth, inside) for one orthographic view.

    `points` is (..., 3). Columns increase with +u, rows increase **downwards**,
    so that +v is up in the rendered image as it is in a photograph.
    """
    au, av, ad, flip_d, flip_u = VIEW_SPEC[view]
    p = points - centre
    u = p[..., au]
    v = p[..., av]
    d = p[..., ad] * (-1.0 if flip_d else 1.0)
    if flip_u:
        u = -u
    half = extent / 2.0
    inside = (u.abs() <= half) & (v.abs() <= half)
    col = ((u + half) / extent * (img_size - 1)).round().long().clamp(0, img_size - 1)
    row = ((half - v) / extent * (img_size - 1)).round().long().clamp(0, img_size - 1)
    return col, row, d, inside


def unproject_from_view(
    col: Tensor, row: Tensor, view: str, centre: Tensor, extent: float, img_size: int
) -> tuple[Tensor, Tensor, int, int]:
    """Inverse of `project_to_view` for the two axes a view resolves.

    Returns ``(value_u, value_v, axis_u, axis_v)``: the two world coordinates the
    view pins down, and which axes they are. The third world axis is the one the
    view looks along and is not recoverable from a single orthographic view --
    which is exactly why there is more than one view.

    `col` and `row` may be fractional, which is what makes a soft-argmax useful.
    """
    au, av, _, _, flip_u = VIEW_SPEC[view]
    half = extent / 2.0
    u = col / (img_size - 1) * extent - half
    v = half - row / (img_size - 1) * extent
    if flip_u:
        u = -u
    return u + centre[au], v + centre[av], au, av


def render_views(
    points: Tensor,
    feats: Tensor,
    valid: Tensor,
    *,
    views: tuple[str, ...] = VIRTUAL_VIEWS,
    centre: Tensor,
    extent: float,
    img_size: int,
) -> Tensor:
    """Rasterise a batch of point clouds into `len(views)` orthographic images.

    Args:
        points: (B, N, 3) world-frame points.
        feats:  (B, N, C) per-point features written to the winning pixel.
        valid:  (B, N) bool; invalid points (sensor holes) are never drawn.
        centre: (3,) centre of the rendered cube.
        extent: side length of that cube, metres.

    Returns:
        (B, V, C + 1, S, S) -- the feature channels followed by a hit mask, in
        image layout (channels first) ready for a convolution or a patch embed.
        Pixels no point reached are zero in every channel including the mask.
    """
    b, n, c = feats.shape
    assert n < (1 << _IDX_BITS), f"{n} points exceeds the {_IDX_BITS}-bit index field"
    dev = points.device
    s2 = img_size * img_size
    out = points.new_zeros(b, len(views), c + 1, s2)

    idx = torch.arange(n, device=dev).expand(b, n)
    batch_off = (torch.arange(b, device=dev) * s2).view(b, 1)

    for vi, view in enumerate(views):
        col, row, depth, inside = project_to_view(points, view, centre, extent, img_size)
        keep = inside & valid
        # depth is signed (the cube straddles the centre); shift to non-negative
        dmin = torch.where(keep, depth, depth.new_full((), 1e9)).amin()
        dq = ((depth - dmin) * 1e6).round().clamp(min=0).long()
        key = (dq << _IDX_BITS) | idx
        key = torch.where(keep, key, key.new_full((), torch.iinfo(torch.int64).max))

        flat = batch_off + row * img_size + col                       # (B, N)
        best = key.new_full((b * s2,), torch.iinfo(torch.int64).max)
        best.scatter_reduce_(0, flat.reshape(-1), key.reshape(-1), reduce="amin")
        best = best.view(b, s2)

        hit = best != torch.iinfo(torch.int64).max
        # Empty pixels carry the int64 sentinel, whose low bits decode to an
        # out-of-range index; clamp before gathering and zero them via `hit`.
        winner = (best & ((1 << _IDX_BITS) - 1)).clamp(0, n - 1)       # (B, S*S)
        gathered = torch.gather(feats, 1, winner.unsqueeze(-1).expand(b, s2, c))
        out[:, vi, :c] = (gathered * hit.unsqueeze(-1)).permute(0, 2, 1)
        out[:, vi, c] = hit.to(out.dtype)

    return out.view(b, len(views), c + 1, img_size, img_size)
