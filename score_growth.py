"""Re-score a finished sweep with the three-window growth metric.

    python score_growth.py                             out/tooth_passing_v3
    python score_growth.py --sweep out/tooth_passing_v3 --design grid
    python score_growth.py --envelope grid_fz0p18_rpm1000_z1

Reads only what is already on disk - `raw/coupled.npz` and `plant.npz` in each
run - so it never re-simulates. Writes `growth.csv` beside `sweep.csv` and joins
the two on the run name, so the prediction and the three measurements sit in one
table.

WHAT IT REPLACES, AND WHY

`analysis.report.chatter_metrics` fits ONE exponential across the trimmed middle
of the pass. `analysis.growth` measures the three things that middle is actually
made of - see that module for the argument. The short version: a forced plateau
has a flat envelope whatever the damping, so a single fit over a long pass
returns ~0 and the answer ends up depending on the feed rate rather than on the
machine.

The old columns are left untouched in `sweep.csv`. This writes new ones beside
them rather than overwriting, so the two can be compared on the same runs.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analysis                                                # noqa: E402
from analysis import growth, save                              # noqa: E402


def load_sweep(sweep_dir: Path) -> list:
    f = sweep_dir / "sweep.csv"
    if not f.exists():
        raise SystemExit(f"no sweep.csv under {sweep_dir}")
    with open(f, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def num(row, key, default=np.nan):
    v = row.get(key)
    if v in (None, "", "None"):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def score_sweep(sweep_dir: Path, design="both") -> list:
    rows = load_sweep(sweep_dir)
    if design != "both":
        rows = [r for r in rows if r.get("design") == design]
    out = []
    for r in rows:
        d = sweep_dir / "runs" / r["run"]
        s = growth.score(d, ae_mm=num(r, "ae_mm", 5.0))
        out.append({
            "run": r["run"], "design": r.get("design", ""),
            "spindle_rpm": num(r, "spindle_rpm"), "n_teeth": num(r, "n_teeth"),
            "fz_mm": num(r, "fz_mm"), "tpf_hz": num(r, "tpf_hz"),
            # the prediction, for the comparison this metric exists to make
            "pred_growth_max_1_s": num(r, "pred_growth_max_1_s"),
            "pred_growth_max_trim_1_s": num(r, "pred_growth_max_trim_1_s"),
            "pred_growth_entry_1_s": num(r, "pred_growth_entry_1_s"),
            # what the old single fit said, kept alongside rather than replaced
            "old_sim_growth_1_s": num(r, "sim_growth_1_s"),
            "old_sim_growth_r2": num(r, "sim_growth_r2"),
            **s,
        })
    return out


def table(rows) -> str:
    hdr = (f"{'cell':>26} {'span s':>7} {'taus':>6} | "
           f"{'ring g':>8} {'R2':>5} {'seen':>5} | "
           f"{'plat g':>9} {'floor':>7} {'verdict':>10} | "
           f"{'peak %ae':>9} | {'pred_un':>8} {'old fit':>8}")
    lines = [hdr, "-" * len(hdr)]
    for r in sorted(rows, key=lambda r: (r["design"], r["spindle_rpm"], r["n_teeth"])):
        if not r.get("grow_valid"):
            lines.append(f"{r['run'][:26]:>26}   (nothing measurable)")
            continue
        flag = "" if r["ring_complete"] else "*"
        lines.append(
            f"{r['run'][:26]:>26} {r['engaged_s']:7.2f} {r['engaged_tau']:6.1f} | "
            f"{r['ring_g_1_s']:8.2f} {r['ring_r2']:5.2f} {r['ring_taus_seen']:4.1f}{flag} | "
            f"{r['plateau_g_1_s']:+9.3f} {r['plateau_floor_1_s']:7.3f} "
            f"{r['plateau_verdict']:>10} | {r['peak_pct_ae']:9.2f} | "
            f"{r['pred_growth_max_1_s']:+8.2f} {r['old_sim_growth_1_s']:+8.2f}")
    lines.append("  * the pass is shorter than the ring-down window - "
                 "the decay was truncated, not measured")
    return "\n".join(lines)


def summary(rows) -> str:
    ok = [r for r in rows if r.get("grow_valid")]
    if not ok:
        return "nothing measurable"
    good = [r for r in ok if r["ring_complete"] and r["ring_r2"] >= 0.5]
    lines = [
        f"cells        {len(ok)} scored, {len(good)} with a complete, well-fitted "
        f"ring-down (R2 >= 0.5)",
    ]
    if good:
        for key, lbl in (("pred_growth_max_1_s", "untrimmed"),
                         ("pred_growth_max_trim_1_s", "trimmed  ")):
            p = np.array([r[key] for r in good])
            m = np.array([r["ring_g_1_s"] for r in good])
            fin = np.isfinite(p) & np.isfinite(m)
            if fin.sum() >= 3:
                r_ = np.corrcoef(p[fin], m[fin])[0, 1]
                lines.append(f"ring-down vs pred {lbl}   r = {r_:+.3f}   "
                             f"bias = {np.mean(m[fin] - p[fin]):+.2f} 1/s   "
                             f"rmse = {np.sqrt(np.mean((m[fin]-p[fin])**2)):.2f} 1/s")
    v = {}
    for r in ok:
        v[r["plateau_verdict"]] = v.get(r["plateau_verdict"], 0) + 1
    lines.append("plateau      " + ", ".join(f"{k}: {n}" for k, n in sorted(v.items())))
    resolved = [r for r in ok if np.isfinite(r["plateau_floor_1_s"])]
    if resolved:
        f = np.array([r["plateau_floor_1_s"] for r in resolved])
        lines.append(f"resolving    growth floor {f.min():.3f} .. {f.max():.3f} 1/s "
                     f"(a long pass resolves {f.max()/f.min():.0f}x finer than a short one)")
    worst = max(ok, key=lambda r: r["peak_pct_ae"])
    lines.append(f"severity     worst {worst['run']} at {worst['peak_pct_ae']:.2f}% of ae "
                 f"({worst['peak_dev_um']:.0f} um)")
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default=None,
                   help="sweep directory (default: out/tooth_passing_v3)")
    p.add_argument("--design", default="both", choices=("grid", "rows", "both"))
    p.add_argument("--envelope", default=None, metavar="RUN",
                   help="print the band-passed envelope of one run in slices "
                        "instead of scoring the sweep - the shape, not a fit")
    p.add_argument("--slices", type=int, default=12)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    sweep_dir = Path(a.sweep) if a.sweep else analysis.OUT / "tooth_passing_v3"

    if a.envelope:
        d = sweep_dir / "runs" / a.envelope
        t, rms = growth.envelope(d, a.slices)
        if not len(t):
            raise SystemExit(f"nothing measurable in {d}")
        print(f"{a.envelope}: band-passed envelope, {a.slices} slices")
        print("  t[s] " + " ".join(f"{x:7.2f}" for x in t))
        print("  rms  " + " ".join(f"{x:7.1f}" for x in rms) + "   [um]")
        return t, rms

    rows = score_sweep(sweep_dir, a.design)
    print(summary(rows))
    print()
    print(table(rows))
    out = save.write_csv(sweep_dir / "growth.csv", rows)
    print()
    print(f"out      {out}")
    return rows


if __name__ == "__main__":
    main()
