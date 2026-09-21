"""Turn results/grid.json into the study's figures and its results table.

House style: no chartjunk, one idea per panel, and error bars are the standard
error over seeds so a reader can see immediately whether a gap is resolved. Arms
are coloured by *mechanism*, not by name -- regression arms in one family,
geometric decoders in another -- because the mechanism is the finding.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

#: Ordered so the legend reads as the ablation ladder it is.
ARM_ORDER = [
    "proprio", "proprio_joints", "rgb", "rgbd", "xyz_cam",
    "dp3_pcd", "dp3_world", "rgbd_unproj", "xyz_real", "rvt",
    "rgb_aug", "rvt_aug",
]
LABEL = {
    "proprio": "proprio only (blind)",
    "proprio_joints": "proprio + joint angles",
    "xyz_cam": "RGB+XYZ camera frame",
    "dp3_pcd": "DP3 point cloud, 1 cam",
    "dp3_world": "DP3 point cloud, 4 cam world",
    "rgb": "RGB, regress",
    "rgbd": "RGB+D, regress",
    "rgbd_unproj": "RGB+D, unproject",
    "xyz_real": "RGB+XYZ real views",
    "rvt": "RVT (canonical views)",
    "rgb_aug": "RGB + camera aug",
    "rvt_aug": "RVT + camera aug",
}
COLOR = {
    "proprio": "#8a8a8a",
    "proprio_joints": "#c0c0c0",
    "xyz_cam": "#9c4fd9",
    "dp3_pcd": "#00a0a0",
    "dp3_world": "#006060",
    "rgb": "#3b7dd8",
    "rgbd": "#7aa9e9",
    "rgbd_unproj": "#e0851f",
    "xyz_real": "#d94f3d",
    "rvt": "#2a9d5c",
    "rgb_aug": "#1f3f7a",
    "rvt_aug": "#14603a",
}
STYLE = {"rgb_aug": "--", "rvt_aug": "--", "proprio_joints": ":"}

METRIC = "trans_mm_median"
METRIC_LABEL = "next-keypose translation error (mm, median)"


def load(path: pathlib.Path) -> list[dict]:
    return json.loads(path.read_text())


def agg(rows, arm, key, value, others):
    """Mean and standard error over seeds at one point of one axis."""
    sel = [
        r for r in rows
        if r["arm"] == arm and abs(r[key] - value) < 1e-12
        and all(abs(r[k] - v) < 1e-12 for k, v in others.items())
    ]
    if not sel:
        return None, None, 0
    v = np.array([r[METRIC] for r in sel])
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0, len(v)


def sweep_panel(ax, rows, key, values, others, xlabel, arms):
    for arm in arms:
        xs, ys, es = [], [], []
        for v in values:
            m, e, n = agg(rows, arm, key, v, others)
            if m is None:
                continue
            xs.append(v)
            ys.append(m)
            es.append(e)
        if not xs:
            continue
        ax.errorbar(
            xs, ys, yerr=es, marker="o", ms=4, lw=1.8, capsize=2.5,
            color=COLOR[arm], ls=STYLE.get(arm, "-"), label=LABEL[arm],
        )
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=pathlib.Path, default=pathlib.Path("results/grid.json"))
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("figures"))
    a = p.parse_args()
    rows = load(a.grid)
    a.out.mkdir(parents=True, exist_ok=True)
    arms = [x for x in ARM_ORDER if any(r["arm"] == x for r in rows)]

    thetas = sorted({r["theta"] for r in rows if r["eps"] == 0 and r["noise"] == 0})
    epss = sorted({r["eps"] for r in rows if r["theta"] == 0 and r["noise"] == 0})
    noises = sorted({r["noise"] for r in rows if r["theta"] == 0 and r["eps"] == 0})

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0), sharey=True)
    sweep_panel(axes[0], rows, "theta", thetas, {"eps": 0.0, "noise": 0.0},
                "camera perturbation $\\theta$ (deg), recalibrated", arms)
    axes[0].set_title("cameras move, calibration follows", fontsize=10)
    axes[0].set_ylabel(METRIC_LABEL)
    sweep_panel(axes[1], rows, "eps", epss, {"theta": 0.0, "noise": 0.0},
                "calibration error $\\epsilon$ (deg)", arms)
    axes[1].set_title("cameras stay put, calibration is wrong", fontsize=10)
    sweep_panel(axes[2], rows, "noise", noises, {"theta": 0.0, "eps": 0.0},
                "depth noise $c$   ($\\sigma_z = c\\,z^2$)", arms)
    axes[2].set_title("depth degrades", fontsize=10)
    axes[2].set_xscale("symlog", linthresh=5e-4)
    axes[1].legend(fontsize=8, frameon=False, ncol=2, loc="upper left")
    for ax in axes:
        ax.set_yscale("log")
    fig.suptitle(
        "What re-rendering buys, and what it costs: three stresses on one task",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(a.out / "fig1_sweeps.png", dpi=180)
    print(f"wrote {a.out/'fig1_sweeps.png'}")

    # --- fig 2: where does the ordering change hands? ---------------------
    cthetas = sorted({r["theta"] for r in rows if r["eps"] == 0})
    cnoises = sorted({r["noise"] for r in rows if r["eps"] == 0})
    if len(cthetas) > 1 and len(cnoises) > 1 and {"rvt", "rgb"} <= set(arms):
        grid_rvt = np.full((len(cnoises), len(cthetas)), np.nan)
        grid_rgb = np.full_like(grid_rvt, np.nan)
        for i, c in enumerate(cnoises):
            for j, t in enumerate(cthetas):
                m1, _, n1 = agg(rows, "rvt", "theta", t, {"noise": c, "eps": 0.0})
                m2, _, n2 = agg(rows, "rgb", "theta", t, {"noise": c, "eps": 0.0})
                if n1 and n2:
                    grid_rvt[i, j], grid_rgb[i, j] = m1, m2
        adv = grid_rgb - grid_rvt          # positive: the 3D arm is ahead
        lim = np.nanmax(np.abs(adv)) if np.isfinite(adv).any() else 1.0
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        im = ax.imshow(adv, cmap="RdBu", vmin=-lim, vmax=lim, origin="lower", aspect="auto")
        ax.set_xticks(range(len(cthetas)), [f"{t:g}" for t in cthetas])
        ax.set_yticks(range(len(cnoises)), [f"{c:g}" for c in cnoises])
        ax.set_xlabel("camera perturbation $\\theta$ (deg)")
        ax.set_ylabel("depth noise $c$")
        ax.set_title("RGB error minus RVT error (mm)\nblue: RGB wins   red: the 3D arm wins", fontsize=10)
        for i in range(len(cnoises)):
            for j in range(len(cthetas)):
                if np.isfinite(adv[i, j]):
                    ax.text(j, i, f"{adv[i, j]:+.0f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="mm")
        fig.tight_layout()
        fig.savefig(a.out / "fig2_crossover.png", dpi=180)
        print(f"wrote {a.out/'fig2_crossover.png'}")

    # --- fig 3: the two keyposes that decide the task ----------------------
    phases = [("trans_mm_median_grasp", "grasp"), ("trans_mm_median_release", "release"),
              ("trans_mm_median_transit", "transit")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (point, title) in zip(axes, (
            ({"theta": 0.0, "eps": 0.0, "noise": 0.0}, "nominal"),
            ({"theta": max(thetas), "eps": 0.0, "noise": 0.0}, f"theta = {max(thetas):g} deg"))):
        width = 0.8 / len(phases)
        for k, (metric, label) in enumerate(phases):
            vals, errs = [], []
            for arm in arms:
                key, val = next(iter(point.items()))
                others = {kk: vv for kk, vv in point.items() if kk != key}
                m, e, n = agg(rows, arm, key, val, others)
                sel = [r[metric] for r in rows if r["arm"] == arm
                       and all(abs(r[kk] - vv) < 1e-12 for kk, vv in point.items())
                       and np.isfinite(r.get(metric, np.nan))]
                vals.append(np.mean(sel) if sel else np.nan)
                errs.append(np.std(sel, ddof=1) / np.sqrt(len(sel)) if len(sel) > 1 else 0.0)
            x = np.arange(len(arms)) + k * width - 0.4 + width / 2
            ax.bar(x, vals, width, yerr=errs, capsize=2, label=label,
                   color=["#2a9d5c", "#d94f3d", "#8a8a8a"][k], alpha=0.9)
        ax.set_xticks(range(len(arms)), [LABEL[x] for x in arms], rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, lw=0.6, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("translation error (mm, median)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Averaging over the trajectory hides the two keyposes that decide the task",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(a.out / "fig3_phases.png", dpi=180)
    print(f"wrote {a.out/'fig3_phases.png'}")

    # table
    lines = ["| arm | nominal | theta=30 | eps=5 | c=0.008 |", "|---|---|---|---|---|"]
    for arm in arms:
        cells = []
        for key, val, others in (
            ("theta", 0.0, {"eps": 0.0, "noise": 0.0}),
            ("theta", 30.0, {"eps": 0.0, "noise": 0.0}),
            ("eps", 5.0, {"theta": 0.0, "noise": 0.0}),
            ("noise", 0.008, {"theta": 0.0, "eps": 0.0}),
        ):
            m, e, n = agg(rows, arm, key, val, others)
            cells.append("--" if m is None else f"{m:.1f} ± {e:.1f}")
        lines.append(f"| {LABEL[arm]} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    (a.out / "table1.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
