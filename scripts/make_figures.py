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
ARM_ORDER = ["proprio", "rgb", "rgbd", "rgbd_unproj", "xyz_real", "rvt", "rgb_aug", "rvt_aug"]
LABEL = {
    "proprio": "proprio only",
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
    "rgb": "#3b7dd8",
    "rgbd": "#7aa9e9",
    "rgbd_unproj": "#e0851f",
    "xyz_real": "#d94f3d",
    "rvt": "#2a9d5c",
    "rgb_aug": "#1f3f7a",
    "rvt_aug": "#14603a",
}
STYLE = {"rgb_aug": "--", "rvt_aug": "--"}

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
