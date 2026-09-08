# One coupled robot milling pass, and the linear model meant to replace it

A rectangular workpiece, one edge milled, 25 mm of air before and after — so the
pass contains a real **entry** and a real **exit**, which is where a steady-state
model has the least to say. Two descriptions of that cut are computed side by
side and compared:

| | what it is | what it costs |
|---|---|---|
| **truth** | the regenerative dexel engine closed on the flexible-joint arm, integrated in joint space | minutes |
| **prediction** | the same cut linearised into `F0 + K_cut·x + C_cut·ẋ`, closed on the arm's M/D/K at the tool tip | ~0.2 s |

They share the geometry measurement, the plant and the cutting coefficients, so
what separates them is the **linearisation of the cut** and nothing else.

```bash
conda activate milling_env

python main.py                     # one compensated pass -> out/<name>/
python main.py --no-compensate     # the same pass with the motors idle
python main.py --predict-only      # the linear side only, no simulation
python main.py --ap 2 --rpm 2000 --name deeper
python -m unittest discover -p "test_*.py"
```

## The DC compensation, and why it is not optional

This arm is compliant: `G(0)` is 0.8–2.3 µm/N at the tool tip, and the
revolution-average cutting force on the base job is 144 N. A tool asked to follow
the nominal trajectory is therefore **pressed out of the workpiece by ~160 µm**
before it has done anything wrong — 3.4 % of a 5 mm radial engagement, always in
the same direction, and growing with the cutting coefficients because the force
does.

That breaks the comparison. `analysis.stability` measures its engagement on the
**commanded** path; the engine cuts at the **deflected** one. Uncompensated, the
two sides are not describing the same cut, and any disagreement between them is
contaminated by a geometry error before the linearisation is asked anything.

So the revolution-average force is carried on the **motor torques**, scheduled
along the path:

```
tau_ff(t) = -J(theta(t))^T · F0(s(t))
```

`Solver.step` forms `tau_drive = tau_ext + tau_ff` with `tau_ext = J^T f_ext`, so
this leaves the joint springs carrying `J^T (F_cut - F0)` — the **fluctuation**
about the mean and nothing else, which is exactly the object the linearisation is
about. Tooth-passing ripple, the entry and exit transients and any regenerative
growth all survive it untouched.

**Measured on the base job**, `python main.py` against `python main.py
--no-compensate`:

| | uncompensated | compensated |
|---|---|---|
| DC deviation from the command | **132.3 µm** | **17.1 µm** |
| AC deviation, rms | 22.5 µm | 8.7 µm |
| peak deviation | 160.6 µm | 39.8 µm |
| dominant frequency of the fitted band | 4.2 Hz | 11.0 Hz |
| envelope growth rate | −0.69 ± 0.26 1/s | −0.79 ± 0.17 1/s |
| fit R² | 0.19 | 0.43 |

87 % of the static offset is gone. What is left is the part `F0` cannot cancel:
the ZOA mean force differs from the engine's by about 1 % (below), the Jacobian
is exact but the pose the plant was linearised at is frozen, and the correction
is quasi-static so it does nothing about the transient at each end.

The frequency row is the more interesting one. Uncompensated, the growth-rate fit
sits at 4.2 Hz — below every structural mode — because the quasi-static
deflection following the engagement profile bleeds into the band. With the mean
carried on the motors it lands on 11.0 Hz, between the 8.8 and 14.8 Hz modes,
which is where regenerative growth actually lives. The compensation does not just
tidy the geometry; it is what makes the growth-rate measurement about chatter.

### Why the torque route rather than an offset command

`stabsim/compensate.py` documents the alternative: aim the command off the
nominal by `-G(0)·F0`, so the mean force pushes the tool back onto the geometry
it was supposed to cut. Both cancel the same offset and at DC they must agree
wherever `G(0) = J K_m⁻¹ J^T`. The torque route is used here because

* **it moves no geometry.** The commanded path stays the nominal one, so `F0`,
  `K_cut` and `C_cut` are evaluated at the path the tool is actually asked to
  follow, and there is no second path to keep track of downstream.
* **an offset command can leave the material.** The correction is `G(0)·F0`, and
  once that exceeds `ae` the compensated command steers the tool clear of the
  workpiece and the pass never cuts at all.

`Feedforward.offset_um` reports what the offset route *would* have applied
(157 µm mean, 169 µm max here), so the two can be put side by side without
running both.

## What a run produces

```
out/<name>/
  config.json                 the RunConfig this run was built from
  summary.json                every scalar: prediction, measurement, agreement
  summary.txt                 the console output, kept
  plant.npz                   A/B/C/D of the linearised arm, as driven
  raw/
    coupled.npz               the full SimResult - every sample, every joint
    timeseries.csv            t, command, TCP, deviation, force  (decimated)
  stability/
    stability.npz             the whole StabilityAlongPath
    along_path.csv            one row per node: growth rate, ap_crit, F0
  forces/
    feedforward.npz           F0(s), what the motors carried, G(0)·F0
    revolutions.csv           one row per spindle revolution: sim vs F0
```

The **npz is the record** — every sample, full precision, and what a later script
should read. The **CSV is the copy a human opens**, so the time series is
decimated (`--csv-decimate`, recorded in `summary.json` so nobody mistakes it for
the simulation rate).

`raw/coupled.npz` carries the raw deliverables directly: `force_i` / `force_w`
(cutting force at the TCP, base and workpiece frames), `fk_hist` (TCP pose),
`theta_cmd` / `q_hist` (commanded motor and deflected link angles),
`deflection_w_um` (TCP against the command) and `tcp_error_w_um` (against the
nominal).

### Stability along the trajectory

`stability/along_path.csv` is the linear evaluation, one row per node every
`--ds` millimetres: arc length, position, radial engagement, entry/exit angles,
`growth_rate_1_s`, `mode_hz`, the unstable/chattering/diverging flags,
`ap_crit_mm`, and the surrogate's `F0`. Nothing is reduced to a verdict — that
happens in `summary.json`.

Two things to know before quoting a growth rate:

**There are no speed lobes in this criterion.** Both cut terms truncate the
regenerative delay after one order, so there is no `e^{-jωT}` left in the loop.
Read a growth rate as "at this speed, at this depth", never as "at this speed
rather than that one".

**The entry node owns the maximum.** The first engaged node is the tool part-way
into the material, at partial `ae`, and it is reliably the largest growth rate on
the path — on the base job −1.54 1/s against −4.73 over the trimmed span. The
measurement side throws that span away (`report.TRIM`, 15 % off each end), so
pairing an untrimmed maximum with a trimmed measurement compares different parts
of the cut. `summary.json` reports `pred_growth_max_trim_1_s` as the like-for-like
number and `pred_growth_entry_1_s` beside it, rather than quietly dropping the
entry.

`pred_ap_crit_trim_mm` — the depth the whole trimmed cut survives — is the more
useful output of the two. It is in millimetres, it is what a planner actually
picks, and it needs no threshold on either side.

### The surrogate's mean force against the engine's

`forces/revolutions.csv` puts the two on one axis, one row per spindle
revolution, with `s_mm` so the entry and exit are locatable.

`F0` is a revolution average with no tooth passing in it; the engine's force is
the full history at several kHz, dominated by that ripple. Comparing them sample
against sample would charge the model for a feature it never claimed to have, so
the engine's force is averaged over **one full spindle revolution** first — which
is exactly the quantity `F0` estimates. Magnitude and direction are then reported
separately, because a force 10 % too large and one 10° off point at different
causes.

On the base job, compensated:

| span | engine | surrogate | magnitude | direction |
|---|---|---|---|---|
| plateau (98 revs) | 154.1 N | 155.6 N | +0.98 % | 0.22° |
| entry + exit (40 revs) | 150.6 N | 149.3 N | −0.87 % | 0.11° |

This tests the force model and the engagement geometry, and **nothing else** — it
does not test `K_cut`, `C_cut` or the delay truncation, which live in the growth
rate. That is what makes it the low-noise half of the comparison.

## The operating point

`ap = 1.0 mm`, `ae = 5 mm`, 3333 rpm, 40 mm/s on a 100 × 60 mm part. Chosen so
the cut is comfortably stable but the deflections are in the tens-to-hundreds of
micrometres a compliant arm actually produces; at the stock `ap = 5 mm` the tool
deflects far enough that the cut stops being one.

Every part of it is a flag: `--ap --ae --rpm --feed --ktc --krc --sim-dt
--raster --robot-model`. `--feed` regenerates the toolpath, because the
constant-feed profile has to be replanned for a new maximum rather than rescaled.

### The feed profile is flat across the whole cut

The stock planner chains rest-to-rest quintics, so the tool would come to a full
stop at the edge entry and again at the exit. `analysis/feedplan.py` replaces the
profile: the ramps live entirely in the lead-in and lead-out, through air, and
the engaged span runs at a flat `v_max`.

```
   v                 _________________________
                    /                         \
   ________________/                           \______________
                 |  lead-in |    EDGE    | lead-out |
                 0         25          131        156   s [mm]
```

Each ramp is a smoothstep in time, `v(τ) = v_max(10τ³ − 15τ⁴ + 6τ⁵)`, which has
zero acceleration at both ends. That polynomial integrates to exactly ½ over
`[0,1]`, so a ramp of duration `T = 2·L_ramp/v_max` covers exactly `L_ramp` — no
search, no integration, and the plateau begins precisely where the material does.
Verified on every run: feed is 40.000..40.000 mm/s across the engaged span.

## The plant

`M ẍ + D ẋ + K x = F` at the tool tip, from `tcp_linear_model` — the arm's own
mass matrix, its identified joint springs and the Jacobian, linearised at the
start pose. Nothing measured is read anywhere in this workspace, so the M/D/K and
the arm the coupled pass integrates are the **same model**, one linearised and
one not.

On the base pose it comes out as modes at **8.77 / 14.82 / 23.14 Hz** with

```
G(0) [µm/N], workpiece frame   [[0.795  0.019  0.801]
                                [0.019  2.289 -0.017]
                                [0.801 -0.017  1.204]]
```

**The pose is frozen.** M, D and K are evaluated once and used for the whole
prediction while the arm traverses 156 mm. That is a slow drift across the pass
rather than an entry/exit effect, but it is unmodelled and it bounds how well any
prediction here can do.

Note the mode placement: at 3333 rpm with 4 teeth, tooth passing is **222 Hz**
while the structure is at 9–23 Hz. Chatter here sits *below* the forcing, not
above it. `analysis/report.py` band-passes the deviation to a window around the
modes for that reason — an earlier detector that compared off-harmonic to
on-harmonic energy called every pass unstable, because broadband entry/exit
transient energy swamped the ratio.

## Layout

| directory | holds |
|---|---|
| `fastsim/` | the dexel cutting engine (Numba raster kernel) and the coupled loop |
| `robotsim/` | the arm: URDF, IK, RK4, its linearisation, its receptance |
| `stabsim/` | the cut linearised into `F0`/`K_cut`/`C_cut`, and the closed loop |
| `analysis/` | the job: config, toolpath, plant, prediction, feedforward, scoring |
| `main.py` | the seven steps, in order |
| `configs/`, `data/` | the operating point, the URDF, the workpiece outlines |
| `out/` | one directory per run, gitignored |

### Where this code came from

`fastsim`, `robotsim`, `stabsim` and `runconfig.py` are trimmed copies of
`milling_force_model_validation` (via `milling_sensitivity`), reduced to what one
coupled pass and one stability prediction need. What was dropped and why:

* **`gdsim/` is gone entirely.** It existed to read a *measured* disturbance
  receptance — the closed-loop `(I + PC)⁻¹P` identified with a shaker, servo
  inside, valid only over 1–20 Hz. Nothing here reads that data. Its one useful
  export, the `GdReceptance` container, is now `robotsim/receptance.py` as
  `Receptance`, with a `from_mdk` constructor and the file readers removed.
  `robotsim/test_receptance.py` checks that closing the cut around it through the
  general output-feedback form is algebraically the classic 6×6 second-order
  form, which is what lets `stability_along_path`'s verdict be read as a
  statement about the arm actually simulated.
* **the KUKA KR60 robot models are gone.** They set `K_m_diag = nan` on every
  joint — a fully rigid arm, which has no compliance to deflect, so the coupled
  pass would report a flat zero and the feedforward would have nothing to cancel.
  Only the Staeubli models with identified joint compliance remain.
* **the rigid and reduced-plant backends are gone** (`fastsim/nominal.py`,
  `fastsim/run_linear.py`), along with `stabsim`'s exact-delay work
  (`delay.py`, `compare_delay_*.py`), the time-domain surrogate (`surrogate.py`),
  the feed planner, and every `plotting.py` / `run_*.py` script. This workspace
  runs one thing.

Two files repeat upstream code on purpose, and both say so at the top:

* `analysis/sim_coupled.py` repeats the ten-line coupled loop because
  `fastsim.coupled.run_pass` only supports a **constant** feedforward torque, and
  `F0` is not constant along this job. Every piece it uses is imported; nothing
  is reimplemented.
* `analysis/feedforward.py` uses `stabsim.compensate.f0_along_path` rather than
  `compensate()`, because that function re-sweeps the engagement geometry
  `analysis.stability.prepare` already measured.

## Environment

```bash
conda create -n milling_env -c conda-forge python=3.11 pinocchio numpy scipy \
    numba matplotlib shapely dill
conda activate milling_env
```

`pinocchio` is conda-forge only; everything else is in `requirements.txt`. No
plotting library is imported by a run — results go to disk.
