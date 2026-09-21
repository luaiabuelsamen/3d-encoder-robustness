"""State each claim as a number with an interval, and say whether it resolves.

The figures show shapes; this decides what may be written down. Every claim the
study makes is listed here with the comparison that would falsify it, and each
is printed with a seed-level mean difference, a Welch t statistic and a verdict.
Claims that do not resolve are printed as *not resolved* rather than quietly
dropped -- an unresolved comparison is a result about the experiment's power,
and hiding it is how a study ends up asserting things its seeds do not support.

Three seeds is a small n. The verdicts are therefore deliberately coarse:
"resolved" needs |t| > 2.5 and a difference that matters physically (2 mm, a
fifth of the block's width), not a p-value alone.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

METRIC = "trans_mm_median"
MATERIAL_MM = 2.0
T_RESOLVED = 2.5


def seeds_at(rows, arm, theta=0.0, eps=0.0, noise=0.0, metric=METRIC):
    v = [
        r[metric] for r in rows
        if r["arm"] == arm
        and abs(r["theta"] - theta) < 1e-9
        and abs(r["eps"] - eps) < 1e-9
        and abs(r["noise"] - noise) < 1e-9
        and np.isfinite(r.get(metric, np.nan))
    ]
    return np.array(v, dtype=float)


def welch(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    d = np.sqrt(va + vb)
    return float((a.mean() - b.mean()) / d) if d > 0 else float("nan")


def compare(rows, name, arm_a, arm_b, point, expect, out, metric=METRIC):
    """`expect` is "a<b", "b<a" or "same"; everything is reported either way."""
    a = seeds_at(rows, arm_a, metric=metric, **point)
    b = seeds_at(rows, arm_b, metric=metric, **point)
    if len(a) == 0 or len(b) == 0:
        out.append(f"- **{name}**: no data ({arm_a}: {len(a)} seeds, {arm_b}: {len(b)})")
        return
    diff = a.mean() - b.mean()
    t = welch(a, b)
    resolved = abs(t) > T_RESOLVED and abs(diff) > MATERIAL_MM
    at = ", ".join(f"{k}={v:g}" for k, v in point.items() if v)
    if expect == "same":
        verdict = (
            "indistinguishable" if not resolved
            else f"NOT the same -- {arm_a if diff < 0 else arm_b} is better"
        )
    else:
        better = arm_a if diff < 0 else arm_b
        want = arm_a if expect == "a<b" else arm_b
        verdict = (
            "not resolved" if not resolved
            else ("as expected" if better == want else f"INVERTED -- {better} wins")
        )
    out.append(
        f"- **{name}** (at {at or 'nominal'}): {arm_a} {a.mean():.1f} +- {a.std(ddof=1) if len(a)>1 else 0:.1f} mm "
        f"vs {arm_b} {b.mean():.1f} +- {b.std(ddof=1) if len(b)>1 else 0:.1f} mm; "
        f"diff {diff:+.1f} mm, t = {t:.1f}, n = {len(a)}/{len(b)} seeds -> **{verdict}**"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid", type=pathlib.Path, default=pathlib.Path("results/grid.json"))
    p.add_argument("--out", type=pathlib.Path, default=pathlib.Path("results/findings.md"))
    a = p.parse_args()
    rows = json.loads(a.grid.read_text())
    arms = sorted({r["arm"] for r in rows})
    thetas = sorted({r["theta"] for r in rows})
    out: list[str] = ["# Findings", ""]

    out.append(f"{len(rows)} evaluations; arms {', '.join(arms)}.")
    out.append("")
    out.append("## Does vision matter at all here?")
    out.append("")
    out.append(
        "If the proprioceptive baseline is close to the sighted arms, the task does "
        "not test perception and nothing below means anything."
    )
    compare(rows, "vision vs proprioception", "rvt", "proprio", {}, "a<b", out)
    compare(rows, "vision vs proprioception, grasp keyposes only", "rvt", "proprio", {},
            "a<b", out, metric="trans_mm_median_grasp")

    out += ["", "## 1. Cameras move and the calibration follows them", ""]
    hi = max(thetas)
    for arm in ("rgb", "rgbd", "rgbd_unproj", "xyz_real"):
        if arm in arms:
            compare(rows, f"rvt vs {arm} under a moved rig", "rvt", arm,
                    {"theta": hi}, "a<b", out)
    if "rgb_aug" in arms:
        compare(rows, "does camera augmentation substitute for 3D?", "rvt", "rgb_aug",
                {"theta": hi}, "a<b", out)

    out += ["", "## 2. How much of that is canonicalisation, and how much is geometry?", ""]
    out.append(
        "`xyz_real` sees the same world-frame point cloud as `rvt` and decodes through "
        "the same explicit geometry; it differs only in rasterising from the real "
        "cameras rather than canonical ones. Their gap is canonicalisation, alone."
    )
    for th in thetas:
        compare(rows, f"canonicalisation at theta={th:g}", "rvt", "xyz_real",
                {"theta": th}, "same", out)
    compare(rows, "explicit geometry alone (unproject vs regress)", "rgbd_unproj", "rgbd",
            {"theta": hi}, "a<b", out)

    out += ["", "## 3. The cameras have not moved; the calibration is wrong", ""]
    out.append(
        "Nothing an RGB arm consumes changes on this axis. Everything a 3D arm "
        "consumes does, because the calibration enters its encoder, its decoder, or both."
    )
    for e in sorted({r["eps"] for r in rows if r["eps"] > 0}):
        compare(rows, f"rvt vs rgb at eps={e:g} deg", "rvt", "rgb", {"eps": e}, "a<b", out)

    out += ["", "### 3b. Is that cost predictable without training anything?", ""]
    out.append(
        "A rotation of eps about a camera displaces every reconstructed point by "
        "about `d sin(eps)`, where `d` is the camera-to-workspace distance. If a "
        "canonicalised policy's error under miscalibration is just that rigid "
        "displacement carried through, then its calibration budget can be computed "
        "from the rig's geometry before a single epoch is trained. The prediction "
        "below is made from the nominal rig alone -- no fitted constants."
    )
    from rvt_lerobot.render.rig import NOMINAL_RIG, WORKSPACE_CENTRE

    d = float(np.mean([
        np.linalg.norm(pose.eye - WORKSPACE_CENTRE) for pose in NOMINAL_RIG.values()
    ]))
    out.append("")
    out.append(f"Camera-to-workspace distance d = {d*1000:.0f} mm (nominal rig).")
    out.append("")
    out.append("| eps (deg) | predicted d sin(eps) + 1 mm/deg | rvt measured | xyz_real measured | rgb measured |")
    out.append("|---|---|---|---|---|")
    ratios = []
    for e in sorted({r["eps"] for r in rows if r["eps"] > 0}):
        pred = d * np.sin(np.deg2rad(e)) * 1000 + e  # the operator also shifts 1 mm/deg
        cells = []
        for arm in ("rvt", "xyz_real", "rgb"):
            v = seeds_at(rows, arm, eps=e)
            cells.append(f"{v.mean():.1f}" if len(v) else "--")
        v = seeds_at(rows, "rvt", eps=e)
        base = seeds_at(rows, "rvt", eps=0.0)
        if len(v) and len(base):
            excess = max(0.0, v.mean() - base.mean())
            ratios.append(excess / pred)
        out.append(f"| {e:g} | {pred:.1f} | " + " | ".join(cells) + " |")
    if ratios:
        out.append("")
        out.append(
            f"Excess error over nominal, divided by the geometric prediction: "
            f"mean {np.mean(ratios):.2f}, range {min(ratios):.2f}-{max(ratios):.2f}. "
            "A ratio near 1 means the miscalibration cost is exactly the rigid "
            "displacement and nothing else; well below 1 means the policy absorbs "
            "part of it (proprioception, or a prior over where keyposes live)."
        )
        rgb_nom = seeds_at(rows, "rgb", eps=0.0)
        if len(rgb_nom):
            tol = np.rad2deg(np.arcsin(min(1.0, rgb_nom.mean() / 1000 / d)))
            out.append("")
            out.append(
                f"**Calibration budget.** An RGB policy on this task sits at "
                f"{rgb_nom.mean():.1f} mm. Setting `d sin(eps) = {rgb_nom.mean():.1f} mm` "
                f"gives eps* = {tol:.1f} deg: beyond roughly that much calibration "
                "error, the geometry a 3D encoder is built on costs more than it pays, "
                "and it can be computed from a tape measure."
            )

    out += ["", "## 4. Depth noise", ""]
    for c in sorted({r["noise"] for r in rows if r["noise"] > 0}):
        compare(rows, f"rvt vs rgb at c={c:g}", "rvt", "rgb", {"noise": c}, "a<b", out)

    out += ["", "## Crossover", ""]
    cross = []
    for th in sorted({r["theta"] for r in rows}):
        for c in sorted({r["noise"] for r in rows}):
            x = seeds_at(rows, "rvt", theta=th, noise=c)
            y = seeds_at(rows, "rgb", theta=th, noise=c)
            if len(x) and len(y):
                cross.append((th, c, x.mean(), y.mean()))
    if cross:
        out.append("| theta (deg) | c | rvt (mm) | rgb (mm) | winner |")
        out.append("|---|---|---|---|---|")
        for th, c, xm, ym in cross:
            out.append(f"| {th:g} | {c:g} | {xm:.1f} | {ym:.1f} | "
                       f"{'rvt' if xm < ym else 'rgb'} |")

    text = "\n".join(out) + "\n"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
