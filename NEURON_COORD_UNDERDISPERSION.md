# Finding: neuron generation under-produces coordinate spread (size-dependent)

**Date:** 2026-07-25
**Run analysed:** `vivid-thunder-13` (wandb), sample dump `neuron_samples.smol` (1847 graphs)
**Status:** diagnosed. One modelling defect; no code/normalisation bug. Root-cause fix still open.

---

## TL;DR

The neuron flow-matching model generates structurally-plausible trees whose **coordinates
are systematically too compact**, and this is **why the three extent W1 metrics (axial /
radial / total) and branch-length W1 climb over training** while topology metrics improve.

- Overall generated coordinate spread is **~0.84×** the ground truth (std 52.65 vs 62.63 µm).
- The deficit is **size-dependent**, not a uniform scale: small/sparse neurons come out far
  too compact (ratio **0.81** at N=20–40), large ones are basically correct (**0.99** at
  N=120–160). The model emits a near-constant ~50 µm spread regardless of the target's true
  size.
- It is **not** a normalisation/scaling bug (constants audited and self-consistent), **not**
  outliers, **not** disconnected nodes, and **not** `scale_ot`.
- It is **variance contraction** (mean-seeking flow objective), worst when there are few
  nodes to pin down the extent.
- **Node count is exact** (fed in via the prior mask; `val-node_count_w1 ≡ 0`).

---

## What was observed

Validation W1 curves split into two families that move in **opposite** directions:

| family | metrics | behaviour over training | best at |
|---|---|---|---|
| geometry (connectivity-blind) | branch_length, axial/radial/total extent | drop sharply then **climb** | high disconnected_frac |
| topology (root-component) | bifurcation_angle, leaf_count, bifurcation_count | improve **monotonically** | low disconnected_frac |

Representative numbers (start → best → end):

| metric | start | best (step / disc) | end |
|---|---|---|---|
| branch_length | 7.3 | 2.47 (294 / 0.96) | 8.7 |
| axial_extent | 51.8 | 15.0 (412 / 0.94) | 67.6 |
| bifurcation_angle | 27.3 | 4.65 (3008 / 0.13) | 5.3 |
| leaf_count | 21.8 | 0.84 (end) | 0.84 |
| bifurcation_count | 21.7 | 1.01 (end) | 1.01 |

disconnected_frac falls 0.996 → 0.109 over the run (still 11% disconnected at the end).

The early "good" extent W1 is an **artifact**: at init the std-1 noise prior blob happens to
have ~GT physical extent, so W1 is low before the model contracts the scale. Topology metrics
correctly penalise the disconnected early state and improve as real trees form.

## Root cause: size-dependent variance contraction

Per-graph centered coordinate std (physical µm), median per node-count bin, gen vs correct GT:

| N-bin | gen std | GT std | gen/GT |
|---|---|---|---|
| 20–40 | 48.6 | 59.8 | **0.81** |
| 40–60 | 51.8 | 61.1 | 0.85 |
| 60–80 | 51.7 | 56.4 | 0.92 |
| 80–120 | 50.2 | 51.6 | 0.97 |
| 120–160 | 53.0 | 53.4 | **0.99** |

The model outputs a roughly **constant ~50 µm spread regardless of the target size**, and its
per-graph std distribution is compressed (gen p10–p90 = 31–68 vs GT 38–84). Real low-node-count
neurons are often sparse and spatially large (GT std is actually *highest* for small N); the
model hedges few-node structures toward a compact average. This is the mean-seeking flow
objective contracting variance, worst where there are few points to fix the extent.

**Confirmation (single global scale, vs correct GT):** scaling generated coords by ~1.2×
collapses the errors — axial W1 71.7 → 19.5, radial 43.8 → 9.0, total 73.4 → 18.6,
branch-length 7.0 → 3.2 at scale 1.19. Per-metric best scale: extents 1.22–1.23,
branch 1.16 (so mostly one factor, small non-uniform residual).

## What was ruled out

- **Normalisation / scaling bug — NO.** Constants defined in `scriptutil.py:21–42`, measured by
  `preprocess_neurons.py::_coord_std` (pooled per-tree zero-CoM std over train — matches the
  data). The transform divides by `coord_std` (`scriptutil.py:155`) and `_generate` multiplies
  by the same `coord_scale` (`fm.py:922`), so it cancels; the physical output scale depends only
  on whether the model reproduces the standardised distribution. Verified GT std == constant
  (66.04≈66.03 conditional; 62.63≈62.69 unconditional), i.e. standardised data std = 1.000 =
  prior std.
- **`scale_ot` — NO.** The `log(n+1)*0.2` prior scaling (`interpolate.py:91`) is off by default
  (`--optimal_transport equivariant` → `scale_ot=False`) and its per-N signature does not match
  the deficit.
- **Outliers — NO.** Dropping each graph's farthest node keeps 98% of the diameter; gen tails
  are *less* extreme than GT's.
- **Disconnected / stray nodes — NO.** Largest-connected-component extents ≈ all-node extents
  (LCC coverage ~100%).
- **Node-count deficit — NO.** `val-node_count_w1 ≡ 0` at every step; node count is fed in via
  the prior mask and never changed by the model. (An earlier apparent 0.885 deficit was an
  artifact of comparing against the wrong GT.)

## Latent footguns (not active, but adjacent to this)

- `scale_ot` (`interpolate.py:91`) is tuned for **molecular** coord scale (~2.4), not neurons.
  Do **not** pass `--optimal_transport scale` for neuron runs without re-tuning `SCALE_OT_FACTOR`.
- Stale commented constant `# NEURON_COORDS_STD_DEV = 0.0727` at `scriptutil.py:33` — ~862×
  off, a "double-scaling" regression waiting to happen if uncommented.

## Metric-code caveats (context for reading the curves)

- Extents & branch_length are **connectivity-blind** (all nodes/edges); bifurcation_angle,
  leaf_count, bifurcation_count use **root-component-only** traversal (`_root_tree` in
  `structural_metrics.py`).
- `leaf_count` miscounts every out-of-component node as a leaf (children dict initialised for
  all nodes) — so early (disconnected) `leaf_count ≈ total node count`.
- `val-structural-w1-mean` is an **unweighted** mean over metrics of very different scales
  (axial_extent 15–74 dominates 1–27 for the others), so it mostly tracks axial_extent.

## Important: sample provenance to confirm

The correct GT for `neuron_samples.smol` is **`neurons_final_smol/val.smol`** (1847 graphs,
std 62.63, max N=149) — it matches the samples exactly by node count. This is the **unconditional
`neurons`** set (std ≈ `NEURON_COORDS_STD_DEV` 62.69), **not** `neurons_conditional`
(std 66.04, 2527 graphs, max N=217). Confirm the `vivid-thunder-13` curves and this sample dump
are from the same run/dataset before cross-comparing absolute W1 values.

## Open question / next step

Is the contraction **sampling-time** (under-integration) or **trained-in**?

- **Test:** regenerate with more ODE steps (e.g. 200–500 vs the current 100) and re-measure the
  standardised std. If it rises toward 1.0 → cheap fix (`--num_inference_steps`). If it stays
  ~0.84 → trained-in, needs an OT-coupling / coord-loss / stochastic-sampler change.
- Any fix must be evaluated **per node-count**, not as one global scalar (the deficit is
  concentrated in small/sparse neurons).

## Reproduction

Scripts used for this analysis live in the session scratchpad (extent diagnostic, scale
confirmation, per-N std). Core comparison: load `neuron_samples.smol` via
`GeometricMolBatch.from_bytes`, GT via `GeometricDataset.load(..., transform=None)`, build graphs
with `validation.convert.geometric_mol_to_nx(coord_scale=1.0)`, and compare per-graph centered
coord std / extents binned by node count.
