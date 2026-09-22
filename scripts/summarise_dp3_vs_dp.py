"""Aggregate the DP3 vs Diffusion Policy seeds, refusing to mix datasets.

Seed files accumulate on disk. Re-recording the dataset and re-running does not
remove the old ones, so an aggregate over `results/dp3_vs_dp_s*.json` will
happily average runs from different recordings and report a tidy mean with an
error bar. That happened once here: two seeds from a re-recorded dataset and one
left over from the previous camera, aggregated and published before anyone
noticed. A stale JSON file looks exactly like a fresh one.

So this refuses to aggregate across differing dataset fingerprints, and says
which files disagree.
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import statistics as st
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pattern", default="results/dp3_vs_dp_s*.json")
    p.add_argument("--tail", type=int, default=3,
                   help="average the last N logged losses; a single diffusion "
                        "loss is evaluated at a random timestep and is noisy")
    a = p.parse_args()

    files = sorted(glob.glob(a.pattern))
    if not files:
        print(f"no files matching {a.pattern}")
        return 1

    fingerprints, rows = {}, {}
    for f in files:
        blob = json.loads(pathlib.Path(f).read_text())
        for arm, c in blob.items():
            ds = c.get("dataset")
            key = None if ds is None else (ds.get("total_frames"), ds.get("total_episodes"))
            fingerprints.setdefault(key, []).append(f)
            rows.setdefault(arm, []).append((f, st.mean(x["loss"] for x in c["history"][-a.tail:]),
                                             c["params"], c["history"][-1]["seconds"]))

    if len(fingerprints) > 1:
        print("REFUSING to aggregate: these files describe different datasets")
        for key, fs in fingerprints.items():
            label = "no fingerprint (pre-dates this check)" if key is None else f"{key[1]} episodes, {key[0]} frames"
            print(f"  {label}: {sorted(set(fs))}")
        print("\nRe-run the seeds that are stale, or delete them.")
        return 1

    print(f"{'arm':6s} {'n':>2s} {'final loss':>18s} {'params':>9s} {'wall-clock':>11s}")
    for arm, vals in rows.items():
        losses = [v[1] for v in vals]
        sd = st.stdev(losses) if len(losses) > 1 else 0.0
        print(f"{arm:6s} {len(losses):2d} {st.mean(losses):9.4f} +- {sd:.4f} "
              f"{vals[0][2] / 1e6:8.1f} M {st.mean(v[3] for v in vals):9.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
