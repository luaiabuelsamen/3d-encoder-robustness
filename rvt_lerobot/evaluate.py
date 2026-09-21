"""Metrics for next-keypose prediction.

Translation error is the headline because it is where 3D structure is supposed
to help and where a wrong answer ends the task: the block is 20 mm across, so a
prediction 20 mm off does not grasp it. Rotation and gripper state are reported
because a method that bought translation accuracy by giving them up has not
bought anything.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .data.batching import make_images
from .models.policy import MultiViewPolicy

#: The block's width. A translation prediction inside this is a grasp; outside
#: it, the jaws close on air.
GRASP_TOLERANCE_M = 0.010


def rot6d_to_matrix(r: Tensor) -> Tensor:
    """(..., 6) -> (..., 3, 3) by Gram-Schmidt on the two predicted columns."""
    a, b = r[..., :3], r[..., 3:]
    e1 = torch.nn.functional.normalize(a, dim=-1)
    e2 = torch.nn.functional.normalize(b - (e1 * b).sum(-1, keepdim=True) * e1, dim=-1)
    return torch.stack([e1, e2, torch.cross(e1, e2, dim=-1)], dim=-1)


def geodesic_degrees(pred6: Tensor, target6: Tensor) -> Tensor:
    """Angle of the relative rotation, in degrees."""
    rp, rt = rot6d_to_matrix(pred6), rot6d_to_matrix(target6)
    cos = ((rp.transpose(-1, -2) @ rt).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.rad2deg(torch.arccos(cos.clamp(-1 + 1e-7, 1 - 1e-7)))


@torch.no_grad()
def evaluate(
    model: MultiViewPolicy,
    cache,
    *,
    batch_size: int = 64,
    virtual_size: int = 96,
    limit: int | None = None,
) -> dict[str, float]:
    """Run a trained arm over a rendered condition and summarise the errors."""
    model.eval()
    n = cache.n if limit is None else min(limit, cache.n)
    errs, rots, grips = [], [], []
    for start in range(0, n, batch_size):
        idx = np.arange(start, min(start + batch_size, n))
        batch = cache.batch(idx)
        images = make_images(model.spec, batch, virtual_size=virtual_size)
        out = model(images, batch["proprio"], calib=batch)
        errs.append((out["pos"] - batch["target_pos"]).norm(dim=-1).cpu())
        rots.append(geodesic_degrees(out["rot6"], batch["target_rot6"]).cpu())
        grips.append(
            ((out["grip_logit"] > 0).float() == batch["target_grip"]).float().cpu()
        )
    model.train()
    e = torch.cat(errs).numpy()
    r = torch.cat(rots).numpy()
    g = torch.cat(grips).numpy()
    return {
        "n": int(len(e)),
        "trans_mm_mean": float(e.mean() * 1000),
        "trans_mm_median": float(np.median(e) * 1000),
        "trans_mm_p90": float(np.percentile(e, 90) * 1000),
        "rot_deg_median": float(np.median(r)),
        "grip_acc": float(g.mean()),
        "success_10mm": float((e < GRASP_TOLERANCE_M).mean()),
        "success_20mm": float((e < 2 * GRASP_TOLERANCE_M).mean()),
        "success_50mm": float((e < 5 * GRASP_TOLERANCE_M).mean()),
    }
