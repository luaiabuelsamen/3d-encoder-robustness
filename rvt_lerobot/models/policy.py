"""The six arms of the study, built from one backbone and one training recipe.

Every arm below shares the same transformer, the same capacity, the same
rotation and gripper heads, and the same optimiser. They differ in exactly two
places, which are the two mechanisms the study separates:

* **what the encoder sees** -- nothing, RGB, RGB+depth, RGB+world-XYZ, or a
  canonical orthographic re-rendering;
* **how the translation is decoded** -- regressed from a pooled token, or read
  off a per-view heatmap and pushed through known calibration.

Naming the mechanisms rather than the papers is deliberate. "RVT" is the
combination of the last row of each list; the rows above it are the parts.

    arm            encoder input                 translation decoder
    ------------   ---------------------------   -------------------------
    proprio        (none)                        regress
    rgb            4 real RGB views              regress
    rgbd           4 real RGB+D views            regress
    rgbd_unproj    4 real RGB+D views            heatmap -> unproject
    xyz_real       4 real RGB+XYZ views          heatmap -> unproject
    rvt            5 canonical RGB+XYZ views     heatmap -> orthographic

`xyz_real` is the arm that makes the comparison sharp. It gets exactly the same
information as `rvt` -- a world-frame point cloud with colour -- and decodes it
through the same explicit geometry. The only difference is *which viewpoints the
cloud is rasterised from*: the real cameras' or the canonical ones'. Whatever
separates those two is canonicalisation proper, and nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..render.rig import FAR, WORKSPACE_CENTRE, WORKSPACE_EXTENT
from ..render.virtual import VIRTUAL_VIEWS, project_to_view, unproject_from_view

#: Depth is normalised by this before entering the network, so the depth channel
#: and the RGB channels arrive at a comparable scale.
DEPTH_SCALE = 1.0


@dataclass(frozen=True)
class ArmSpec:
    """What distinguishes one arm from another. Everything else is shared."""

    source: str        # "none" | "real" | "virtual"
    channels: str      # "rgb" | "rgbd" | "rgbxyz"
    decode: str        # "regress" | "unproject" | "orthographic"
    camera_aug: bool = False

    @property
    def in_channels(self) -> int:
        base = {"rgb": 3, "rgbd": 4, "rgbxyz": 6}[self.channels]
        return base + (1 if self.source == "virtual" else 0)  # virtual adds a hit mask

    @property
    def n_views(self) -> int:
        return {"none": 0, "real": 4, "virtual": 5}[self.source]


ARMS: dict[str, ArmSpec] = {
    "proprio": ArmSpec("none", "rgb", "regress"),
    "rgb": ArmSpec("real", "rgb", "regress"),
    "rgbd": ArmSpec("real", "rgbd", "regress"),
    "rgbd_unproj": ArmSpec("real", "rgbd", "unproject"),
    "xyz_real": ArmSpec("real", "rgbxyz", "unproject"),
    "rvt": ArmSpec("virtual", "rgbxyz", "orthographic"),
    "rgb_aug": ArmSpec("real", "rgb", "regress", camera_aug=True),
    "rvt_aug": ArmSpec("virtual", "rgbxyz", "orthographic", camera_aug=True),
}


# ------------------------------------------------------------------- backbone


class PatchEmbed(nn.Module):
    def __init__(self, in_ch: int, dim: int, patch: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x: Tensor) -> Tensor:            # (B*V, C, H, W)
        return self.proj(x)                            # (B*V, D, h, w)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class HeatmapHead(nn.Module):
    """Token grid -> per-view heatmap logits at `out_size`.

    Upsampling from an 8x8 token grid alone cannot localise better than the
    grid, so the decoder is given the input image again at the output
    resolution and concatenates it. This is the skip connection RVT's
    convolutional decoder uses, in its smallest honest form.
    """

    def __init__(self, dim: int, in_ch: int, out_size: int) -> None:
        super().__init__()
        self.out_size = out_size
        self.up = nn.Sequential(
            nn.Conv2d(dim, 128, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1), nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(64, 32, 3, padding=1), nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(32 + in_ch, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 1, 1)
        )

    def forward(self, tokens: Tensor, image: Tensor) -> Tensor:
        # tokens (B*V, D, h, w); image (B*V, C, H, W)
        x = self.up(tokens)
        x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        skip = F.interpolate(image, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([x, skip], dim=1))[:, 0]


def soft_argmax_2d(logits: Tensor, temperature: float = 1.0) -> tuple[Tensor, Tensor, Tensor]:
    """(B, H, W) logits -> fractional (col, row) and the peak probability.

    The peak probability is returned because it is the natural confidence with
    which to weight one view against another when they disagree.
    """
    b, h, w = logits.shape
    p = F.softmax(logits.reshape(b, -1) / temperature, dim=-1)
    peak = p.max(dim=-1).values
    p = p.reshape(b, h, w)
    rows = torch.arange(h, device=logits.device, dtype=p.dtype)
    cols = torch.arange(w, device=logits.device, dtype=p.dtype)
    return (p.sum(1) * cols).sum(-1), (p.sum(2) * rows).sum(-1), peak


class MultiViewPolicy(nn.Module):
    """One arm. The `spec` decides what it sees and how it decodes."""

    def __init__(
        self,
        spec: ArmSpec,
        *,
        image_size: int = 96,
        patch: int = 12,
        dim: int = 192,
        depth: int = 6,
        heads: int = 6,
        heatmap_size: int = 48,
        proprio_dim: int = 7,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.grid = image_size // patch
        v = spec.n_views

        self.proprio = nn.Sequential(nn.Linear(proprio_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))

        if v:
            self.embed = PatchEmbed(spec.in_channels, dim, patch)
            self.pos = nn.Parameter(torch.zeros(1, self.grid * self.grid, dim))
            self.view_embed = nn.Parameter(torch.zeros(1, v, 1, dim))
            nn.init.trunc_normal_(self.pos, std=0.02)
            nn.init.trunc_normal_(self.view_embed, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        # shared across every arm: rotation and gripper come from the pooled token
        self.rot_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 6))
        self.grip_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

        if spec.decode == "regress":
            self.pos_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 3))
        else:
            self.heatmap = HeatmapHead(dim, spec.in_channels, heatmap_size)
            self.view_conf = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))
            if spec.decode == "unproject":
                # a real-view heatmap gives a ray, not a point: the depth of the
                # TARGET is regressed per view, because the target is usually in
                # free space above a surface and the depth buffer there belongs
                # to whatever is behind it.
                self.depth_head = nn.Sequential(nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1))

    # --------------------------------------------------------------- forward

    def forward(self, images: Tensor | None, proprio: Tensor, calib: dict | None = None) -> dict:
        """images: (B, V, C, H, W) or None for the proprio-only arm."""
        b = proprio.shape[0]
        tokens = [self.cls.expand(b, -1, -1), self.proprio(proprio).unsqueeze(1)]
        per_view = None

        if self.spec.n_views:
            v = images.shape[1]
            flat = images.reshape(b * v, *images.shape[2:])
            f = self.embed(flat)                                   # (B*V, D, h, w)
            g = f.shape[-1]
            seq = f.flatten(2).transpose(1, 2)                     # (B*V, h*w, D)
            seq = seq + self.pos
            seq = seq.reshape(b, v, g * g, -1) + self.view_embed
            tokens.append(seq.reshape(b, v * g * g, -1))
            per_view = (f, flat, v, g)

        x = torch.cat(tokens, dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        pooled = x[:, 0]

        out = {"rot6": self.rot_head(pooled), "grip_logit": self.grip_head(pooled)[:, 0]}

        if self.spec.decode == "regress":
            # Predict an offset from the workspace centre in units of its half
            # extent, not a raw world coordinate. At initialisation the head
            # outputs zero, so the prediction starts at the centre of the
            # workspace -- the same prior the heatmap arms start from, which is
            # what keeps the comparison between them about the mechanism rather
            # than about who got a luckier bias initialisation.
            centre = torch.tensor(WORKSPACE_CENTRE, device=pooled.device, dtype=pooled.dtype)
            out["pos"] = centre + self.pos_head(pooled) * (WORKSPACE_EXTENT / 2)
            return out

        f, flat, v, g = per_view
        vt = x[:, 2:].reshape(b, v, g * g, -1)                     # per-view tokens
        # the heatmap decoder wants the *contextualised* tokens back in 2D
        ctx = vt.reshape(b * v, g, g, -1).permute(0, 3, 1, 2)
        logits = self.heatmap(ctx, flat)                           # (B*V, S, S)
        col, row, peak = soft_argmax_2d(logits)
        conf = self.view_conf(vt.mean(2)).reshape(b, v)            # (B, V)
        out["heatmap_logits"] = logits.reshape(b, v, self.heatmap_size, self.heatmap_size)
        out["view_conf"] = conf
        out["peak"] = peak.reshape(b, v)

        if self.spec.decode == "orthographic":
            out["pos"] = self._decode_orthographic(col, row, conf, b, v)
        else:
            depth = self.depth_head(vt.mean(2)).reshape(b, v)
            out["pred_depth"] = depth
            out["pos"] = self._decode_unproject(col, row, depth, conf, b, v, calib)
        return out

    # --------------------------------------------------------------- decoders

    def _decode_orthographic(self, col, row, conf, b, v) -> Tensor:
        """Each canonical view pins two world axes. Combine them by confidence.

        This is the whole geometric decoder: the virtual cameras are orthographic
        and axis-aligned by construction, so inverting them is arithmetic, not
        optimisation, and no camera parameters are involved at all -- which is
        precisely the property being tested.
        """
        centre = torch.tensor(WORKSPACE_CENTRE, device=col.device, dtype=col.dtype)
        w = F.softmax(conf, dim=1)                                  # (B, V)
        col = col.reshape(b, v)
        row = row.reshape(b, v)
        # accumulate into per-axis lists rather than writing into a tensor: an
        # in-place index assignment on a tensor that is part of the graph is a
        # standing invitation to an autograd error, and this costs nothing
        num: list[list[Tensor]] = [[], [], []]
        den: list[list[Tensor]] = [[], [], []]
        for i, view in enumerate(VIRTUAL_VIEWS[:v]):
            a, c, ax_a, ax_c = unproject_from_view(
                col[:, i], row[:, i], view, centre, WORKSPACE_EXTENT, self.heatmap_size
            )
            for value, axis in ((a, ax_a), (c, ax_c)):
                num[axis].append(w[:, i] * value)
                den[axis].append(w[:, i])
        return torch.stack(
            [
                sum(num[k]) / (sum(den[k]) + 1e-8) if num[k] else col.new_zeros(b)
                for k in range(3)
            ],
            dim=-1,
        )

    def _decode_unproject(self, col, row, depth, conf, b, v, calib) -> Tensor:
        """Real-view heatmap + predicted depth, pushed through the calibration.

        `calib` carries the extrinsics the policy was *told*, which under the
        miscalibration axis are not the ones the camera actually had. That is
        the point: this decoder consumes calibration as truth, and the
        experiment measures what happens when it is not.
        """
        K, T = calib["K"], calib["T"]                               # (B, V, 3, 3), (B, V, 4, 4)
        scale = self.image_size / self.heatmap_size
        u = col.reshape(b, v) * scale
        r = row.reshape(b, v) * scale
        z = F.softplus(depth) + 0.05                                # metres, positive
        x = (u - K[..., 0, 2]) / K[..., 0, 0] * z
        y = (r - K[..., 1, 2]) / K[..., 1, 1] * z
        cam = torch.stack([x, y, z], dim=-1).unsqueeze(-1)          # (B, V, 3, 1)
        world = (T[..., :3, :3] @ cam).squeeze(-1) + T[..., :3, 3]  # (B, V, 3)
        w = F.softmax(conf, dim=1).unsqueeze(-1)
        return (world * w).sum(1)


# ----------------------------------------------------------------------- loss


def heatmap_targets_virtual(pos: Tensor, size: int) -> Tensor:
    """Flat index of the target's projection in each canonical view. (B, V)"""
    centre = torch.tensor(WORKSPACE_CENTRE, device=pos.device, dtype=pos.dtype)
    idx = []
    for view in VIRTUAL_VIEWS:
        col, row, _, _ = project_to_view(pos, view, centre, WORKSPACE_EXTENT, size)
        idx.append(row * size + col)
    return torch.stack(idx, dim=1)


def heatmap_targets_real(pos: Tensor, K: Tensor, T: Tensor, size: int, image_size: int):
    """Flat index of the target's projection in each real view, plus validity.

    A keypose can project outside a camera's frame or behind it, and a view that
    cannot see the target should neither be supervised nor trusted in the
    decode, so the mask is returned alongside.
    """
    b, v = K.shape[:2]
    R = T[..., :3, :3].transpose(-1, -2)
    cam = (R @ (pos.unsqueeze(1) - T[..., :3, 3]).unsqueeze(-1)).squeeze(-1)  # (B,V,3)
    z = cam[..., 2]
    u = K[..., 0, 0] * cam[..., 0] / z.clamp(min=1e-3) + K[..., 0, 2]
    r = K[..., 1, 1] * cam[..., 1] / z.clamp(min=1e-3) + K[..., 1, 2]
    scale = size / image_size
    cu = (u * scale).round().long()
    cr = (r * scale).round().long()
    valid = (z > 0.05) & (cu >= 0) & (cu < size) & (cr >= 0) & (cr < size)
    idx = (cr.clamp(0, size - 1) * size + cu.clamp(0, size - 1))
    return idx, valid, z


def policy_loss(out: dict, batch: dict, spec: ArmSpec, model: MultiViewPolicy) -> tuple[Tensor, dict]:
    """Shared objective. Only the heatmap term differs between arms."""
    pos_l = F.smooth_l1_loss(out["pos"], batch["target_pos"], beta=0.02)
    rot_l = F.mse_loss(out["rot6"], batch["target_rot6"])
    grip_l = F.binary_cross_entropy_with_logits(out["grip_logit"], batch["target_grip"])
    total = pos_l + 0.5 * rot_l + 0.2 * grip_l
    parts = {"pos": pos_l.detach(), "rot": rot_l.detach(), "grip": grip_l.detach()}

    if spec.decode != "regress":
        s = model.heatmap_size
        logits = out["heatmap_logits"].reshape(-1, s * s)
        if spec.decode == "orthographic":
            idx = heatmap_targets_virtual(batch["target_pos"], s).reshape(-1)
            hm = F.cross_entropy(logits, idx)
        else:
            idx, valid, z = heatmap_targets_real(
                batch["target_pos"], batch["K"], batch["T"], s, model.image_size
            )
            flat_valid = valid.reshape(-1)
            hm = (
                F.cross_entropy(logits[flat_valid], idx.reshape(-1)[flat_valid])
                if flat_valid.any()
                else logits.sum() * 0.0
            )
            # the per-view depth head is supervised directly; it is the second
            # half of the geometric decode and would otherwise only get gradient
            # through the weighted average
            d_l = F.smooth_l1_loss(
                F.softplus(out["pred_depth"])[valid] + 0.05, z[valid], beta=0.02
            ) if valid.any() else logits.sum() * 0.0
            total = total + 0.2 * d_l
            parts["depth"] = d_l.detach()
        total = total + 0.5 * hm
        parts["heatmap"] = hm.detach()

    return total, parts
