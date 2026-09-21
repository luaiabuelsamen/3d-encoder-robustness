"""A depth-sensor noise model with the shape real stereo depth actually has.

Additive Gaussian noise on depth is the wrong model and it flatters 3D methods.
Active-stereo sensors (RealSense D4xx and the like) have three characteristics
that matter here, all of which this module reproduces:

1. **Axial noise grows with the square of range.** Disparity error is roughly
   constant, and depth = baseline * focal / disparity, so sigma_z = c * z^2.
   The coefficient is the one dial: a D435 at 1 m has sigma of about 2.5 mm,
   which is c = 0.0025 in these units.
2. **Quantisation.** Depth is reported as uint16 millimetres. Below about 1 mm
   of true noise the quantiser is the noise.
3. **Edges are where the lies are.** Stereo matching fails on depth
   discontinuities, producing both holes (no match) and *flying pixels* (a
   match that interpolates between foreground and background, putting a point
   in mid-air between the two surfaces). Flying pixels are the failure mode
   that hurts a point-cloud method most, because they are not zero-mean --
   every object gets a halo of points floating off its silhouette, and the
   object in this scene is a 2 cm block.

Everything is keyed to the single scalar `c` so the stress axis is one number,
with holes and flying pixels scaled alongside it.
"""

from __future__ import annotations

import numpy as np

#: Depth is reported in integer millimetres by every consumer RGBD sensor.
QUANT_M = 1e-3


def apply_depth_noise(
    depth: np.ndarray,
    c: float,
    rng: np.random.Generator,
    *,
    far: float = 3.0,
    edge_threshold: float = 0.02,
    hole_rate: float = 30.0,
    flying_rate: float = 30.0,
) -> np.ndarray:
    """Corrupt a clean depth image the way a stereo sensor would.

    Args:
        depth: (H, W) clean z-depth in metres.
        c: axial noise coefficient; sigma_z = c * z^2 metres. c=0 disables
            everything except quantisation, which is always on because it is a
            property of the interface rather than of the sensor's quality.
        edge_threshold: depth step, in metres, above which a pixel counts as
            being on a discontinuity.
        hole_rate, flying_rate: fraction of edge pixels dropped / turned into
            flying pixels, per unit of `c`. At c=0.0025 (a D435 at 1 m) this
            gives 7.5% of edge pixels each, which is conservative.

    Returns:
        (H, W) float32 corrupted depth. Holes are set to `far`, the same value
        the renderer uses for "nothing here", so downstream code needs no new
        special case.
    """
    d = depth.astype(np.float32).copy()
    if c > 0:
        d = d + rng.normal(0.0, 1.0, d.shape).astype(np.float32) * (c * d * d)

        gy, gx = np.gradient(depth.astype(np.float32))
        edge = np.hypot(gx, gy) > edge_threshold
        n_edge = int(edge.sum())
        if n_edge:
            ey, ex = np.nonzero(edge)
            frac_hole = min(0.5, hole_rate * c)
            frac_fly = min(0.5, flying_rate * c)
            pick = rng.random(n_edge)
            hole = pick < frac_hole
            fly = (pick >= frac_hole) & (pick < frac_hole + frac_fly)
            d[ey[hole], ex[hole]] = far
            # A flying pixel lands somewhere between the two surfaces the stereo
            # matcher could not choose between. Interpolate toward the local
            # depth range's opposite end.
            if fly.any():
                fy, fx = ey[fly], ex[fly]
                h, w = depth.shape
                y0 = np.clip(fy - 1, 0, h - 1)
                y1 = np.clip(fy + 1, 0, h - 1)
                x0 = np.clip(fx - 1, 0, w - 1)
                x1 = np.clip(fx + 1, 0, w - 1)
                neigh = np.stack(
                    [depth[y0, fx], depth[y1, fx], depth[fy, x0], depth[fy, x1]]
                )
                other = np.where(
                    np.abs(neigh.max(0) - depth[fy, fx])
                    > np.abs(neigh.min(0) - depth[fy, fx]),
                    neigh.max(0),
                    neigh.min(0),
                )
                alpha = rng.uniform(0.25, 0.75, size=len(fy)).astype(np.float32)
                d[fy, fx] = (1 - alpha) * depth[fy, fx] + alpha * other

    np.clip(d, 0.0, far, out=d)
    return np.round(d / QUANT_M).astype(np.float32) * QUANT_M


#: The sweep used in the paper. c=0 is a perfect sensor (quantisation only);
#: c=0.0025 is a RealSense D435 at 1 m; c=0.008 is a cheap sensor, or a good one
#: looking at a dark or oblique surface.
NOISE_LEVELS = (0.0, 0.0005, 0.001, 0.002, 0.004, 0.008)
