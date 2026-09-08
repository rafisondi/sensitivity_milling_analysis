"""What a run leaves on disk.

One directory per run under `out/`, and every number in it is either raw or
derived from raw that is also there:

    out/<run>/
      config.json               the RunConfig this run was built from
      summary.json              every scalar: prediction, measurement, agreement
      summary.txt               the console output, kept
      plant.npz                 A/B/C/D of the linearised arm, as driven
      raw/
        coupled.npz             the full SimResult — every sample, every joint
        timeseries.csv          t, command, TCP, deviation, force  (decimated)
      stability/
        stability.npz           the whole StabilityAlongPath
        along_path.csv          one row per node: growth rate, ap_crit, F0
      forces/
        feedforward.npz         F0(s), what the motors carried, G(0) F0
        revolutions.csv         one row per spindle revolution: sim vs F0

WHY BOTH .npz AND .csv. The npz is the record — every sample, full precision,
and it is what a later script should read. The CSV is the copy a human opens, so
the time series is decimated to something a spreadsheet will load. The decimation
is recorded in `summary.json` (`csv_decimate`) so nobody mistakes the CSV's rate
for the simulation's.
"""

import csv
import json
from pathlib import Path

import numpy as np


def _jsonable(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return None if np.isnan(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.ndarray):
        return [_jsonable(x) for x in v.tolist()]
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, Path):
        return str(v)
    return v


def write_csv(path, rows) -> Path:
    """One row per dict. Columns are the union, in first-seen order."""
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fields, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _jsonable(v) for k, v in r.items()})
    return path


def write_json(path, obj) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = ({k: _jsonable(v) for k, v in obj.items()}
               if isinstance(obj, dict) else _jsonable(obj))
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def run_dir(out_root, name) -> Path:
    d = Path(out_root) / name
    for sub in ("raw", "stability", "forces"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def save_run(d, run, *, decimate=1) -> dict:
    """The simulated pass: the full SimResult, and a readable time series."""
    from analysis import report

    written = {}
    run.save(d / "raw" / "coupled.npz")
    written["raw_npz"] = d / "raw" / "coupled.npz"
    written["timeseries_csv"] = write_csv(
        d / "raw" / "timeseries.csv", report.deviation_table(run, decimate))
    return written


def save_stability(d, stab, rows) -> dict:
    stab.save_npz(d / "stability" / "stability.npz")
    return {"stability_npz": d / "stability" / "stability.npz",
            "along_path_csv": write_csv(d / "stability" / "along_path.csv", rows)}


def save_forces(d, ff, rows) -> dict:
    out = {}
    if ff is not None:
        ff.save_npz(d / "forces" / "feedforward.npz")
        out["feedforward_npz"] = d / "forces" / "feedforward.npz"
    out["revolutions_csv"] = write_csv(d / "forces" / "revolutions.csv", rows)
    return out
