# Neuron / Tree Morphology Generation

This document describes the neuron-generation extension to SemlaFlow: how neuron
morphologies are adapted into the model, how to preprocess data, train, sample,
and evaluate, and how to use the two optional conditioning paths — **TMD
(Topological Morphology Descriptor) conditioning** and **neuron-type (cell-class)
conditioning**. The two are orthogonal: either, both, or neither may be enabled.

The upstream [`README.md`](README.md) still describes the original molecular
(QM9 / GEOM Drugs) workflow. This file covers the neuron additions only.

---

## Overview

A neuron morphology is a rooted tree of 3D points (nodes) connected by branches
(edges) — structurally the same object SemlaFlow already generates for molecules
(atoms + bonds). The flow-matching core, the Semla generator, the interpolants,
and the optimal-transport machinery are therefore **unchanged**. The adaptation
lives entirely at the edges of the pipeline:

| Concern | Molecular path | Neuron path |
| --- | --- | --- |
| Data source | RDKit mols / SMILES | SWC files (`id type x y z radius parent_id`) |
| Adapter | RDKit → `GeometricMol` | `data/swc.py`: SWC → `GeometricMol` (no RDKit) |
| Vocab | `build_vocab()` (elements) | `build_neuron_vocab()` = `<PAD>, <MASK>, NODE` |
| Transform | `mol_transform` | `neuron_mol_transform` |
| Coord scale | `QM9/GEOM_COORDS_STD_DEV` | `DATASET_CONFIGS[…].coord_std` (physical units) |
| Model class | `MolecularCFM` | `NeuronCFM` (RDKit metrics stripped) |
| Eval | validity/energy/stability | Wasserstein-1 structural distribution metrics |

**Node/edge encoding** (`data/swc.py`): every node is the single token `NODE`
(vocab index 2); every edge is a single real class (index 1, reusing the
molecular "single bond" slot so `BOND_MASK_INDEX=5` stays valid). The root
(soma, `parent<=0` in the SWC) is reordered to node index 0.

Everything below is selected by `--dataset <name>`, where `<name>` is any entry in
`scriptutil.DATASET_CONFIGS`: `neurons` (the original unlabelled corpus),
`neurons_conditional` (soma-rooted, binarized, **cell-class-labelled**), or
`trees_genus_d10` / `d15` / `d20` (botanical trees, **genus-labelled** — see
[§6](#6-tree-genus-datasets)). They all share the neuron vocab / transform /
`NeuronCFM` and differ only in their registry entry: measured coord scale, size
buckets, node cap and class names.

Both conditioning paths are **optional and off by default**; with both off the
output is identical to the unconditional pipeline:
- **TMD conditioning** — a continuous per-graph persistence-image vector ([§4](#4-tmd-conditioning)).
- **Neuron-type (cell-class) conditioning** — a discrete per-graph class label ([§5](#5-neuron-type-cell-class-conditioning)).

---

## 1. Data preparation (SWC → `.smol`)

Input layout (as produced by dendrite_gen's `prepare_neurons_final.py`):

```
<input_dir>/train/*.swc
<input_dir>/val_extended/*.swc      # preferred; falls back to <input_dir>/val/
```

Convert to SemlaFlow's `.smol` binary format:

```bash
python -m semlaflow.preprocess_neurons \
    --input_dir  /path/to/neurons_final \
    --output_dir /path/to/neurons_final/smol
```

This writes `<output_dir>/train.smol` and `<output_dir>/val.smol` — plus
`<output_dir>/test.smol` when the input has a `test/` subfolder — skips graphs
with more than `--max_atoms` (default 256) nodes, and does a round-trip sanity
check.

**Important — coord std-dev and sizes.** The script prints a measured `coord_std`
over the zero-CoM'd training coordinates, together with per-split node counts:

```
== Measured coord_std (over train, post zero-CoM): 62.6894 ==
   train size: min=8, max=242, total=22740
   val   size: min=12, max=217, total=2527

Paste into semlaflow.scriptutil.DATASET_CONFIGS: coord_std=62.6894, max_nodes=256.
Top bucket limit must be >= 242 (largest graph across all splits).
```

Every dataset gets one entry in `DATASET_CONFIGS` in `semlaflow/scriptutil.py`,
which is the single source of truth for its coord scale, bucket limits and class
names. If the corpus changes, re-run preprocessing and update that entry. Note the
top bucket must cover the largest graph in **every** split, not just train — the
bucket sampler raises for the val loader too.

Useful flags:
- `--val_dir_name NAME` — force a specific validation subfolder (default
  auto-picks `val_extended`, else `val`).
- `--test_dir_name NAME` — force a specific test subfolder (default auto-detects
  `test/`; no `test.smol` is written when absent).
- `--max_atoms N` — cap graph size (default 256).
- `--compute_tmd` / `--tmd_filtrations ...` — attach TMD conditioning vectors
  (see [§4](#4-tmd-conditioning)).

**Class-labelled corpus (`neurons_conditional`).** If the input SWCs carry a
`# cell_class N` header (as `/path/to/neurons_conditional` does), the per-graph
label is captured automatically — no flag needed — and preprocessing prints a
class histogram so you can confirm every graph is labelled:

```bash
python -m semlaflow.preprocess_neurons \
    --input_dir  /path/to/neurons_conditional \
    --output_dir /path/to/neurons_conditional/smol \
    --val_dir_name val
```

Paste that run's coord_std into `DATASET_CONFIGS["neurons_conditional"].coord_std`
(currently `66.0298`) and make sure its `bucket_limits` cover the printed max node
count (currently `242`, top bucket `256`). See
[§5](#5-neuron-type-cell-class-conditioning) to condition on the labels. The
botanical-tree corpora use exactly this path — see [§6](#6-tree-genus-datasets).

---

## 2. Training

```bash
python -m semlaflow.train \
    --dataset neurons \
    --data_path /path/to/neurons_final/smol
```

`--dataset neurons` is the single switch that routes everything down the neuron
path in `train.py`:

- vocab → `build_neuron_vocab()`, transform → `neuron_mol_transform`
- coord scale → `NEURON_COORDS_STD_DEV`, buckets → `NEURON_BUCKET_LIMITS`
- model → `NeuronCFM` (no RDKit / SMILES novelty path)
- **checkpoint selection by `val-loss`** (there is no validity metric), plus
  periodic **weights-only** `snap-{epoch}` checkpoints (`save_top_k=-1`) so a
  better checkpoint can be picked post-hoc from the logged structural-metric
  trajectories.

Otherwise it is stock SemlaFlow flow matching: equivariant OT, self-conditioning
on, EMA on, `uniform-sample` categorical strategy, size-bucketed batching,
constant LR with warm-up.

Selected flags (see the bottom of `train.py` for the full list):

| Flag | Default | Notes |
| --- | --- | --- |
| `--epochs` | 300 | |
| `--batch_cost` | 1024 | cost-based batch size |
| `--lr` | 3e-4 | |
| `--val_check_epochs` | 20 | validation + checkpoint cadence |
| `--n_validation_mols` | 1800 | val subset size |
| `--num_inference_steps` | 100 | ODE steps used in structural-metric rollout |
| `--no_val_structural_metrics` | (on) | disable the per-validation generation rollout (faster; loses the trajectory) |
| `--no_ema` | (EMA on) | |
| `--trial_run` | off | 1 epoch, no logger — smoke test |
| `--tmd_conditioning` / `--tmd_hidden` | off / 64 | see [§4](#4-tmd-conditioning) |
| `--type_conditioning` / `--class_hidden` | off / 16 | cell-class conditioning, see [§5](#5-neuron-type-cell-class-conditioning) |
| `--no_per_cell_class` | (on) | disable per-class stratified val metrics |
| `--per_cell_class_min_count` | 20 | min val graphs per class for stratified metrics |

Pass `--dataset neurons_conditional` to train on the class-labelled corpus (with
or without `--type_conditioning` — the dataset key and the conditioning flag are
orthogonal, so you can train an unconditional baseline on the same corpus).

Checkpoints written per run: `best-{epoch}-{val-loss}` (monitored, top-1),
`last`, and `snap-{epoch}` (all epochs, weights-only). The EMA weights used for
sampling live in the state dict.

> Training uses `WandbLogger`. `--trial_run` disables logging entirely; adjust
> `build_trainer` in `train.py` if you need a different logging setup.

---

## 3. Sampling & evaluation

```bash
python -m semlaflow.sample_neurons \
    --ckpt_path /path/to/best-###.ckpt \
    --data_path /path/to/neurons_final/smol \
    --save_dir  /path/to/out
```

What it does:
1. Loads the `NeuronCFM` checkpoint.
2. Samples `--n_molecules` (default 256) graphs from a **pure-noise prior**; the
   node-count distribution is drawn from `--dataset_split` (default `val`). The
   real graphs are only consulted for their sizes.
3. Writes the raw predicted graphs (coords + binary-edge adjacency, no tree
   extraction) to `<save_dir>/<save_file>` (default `neuron_samples.smol`).
4. Unless `--skip_eval`, runs `evaluate_samples`: computes Wasserstein-1
   structural distribution metrics vs the ground-truth split → `metrics.json` +
   a printed table, and renders multi-azimuth 3D plot grids of generated and GT
   samples as PNGs.

**Structural metrics** (`validation/dist_metrics.py`). Every graph — generated *and*
ground-truth — is first reduced to a clean critical tree by `validation/sanitise.py`
(largest connected component → minimum spanning tree → contract degree-2 nodes), and
structural health is measured on the **raw** graph before that repair. See
[`COMPATIBLE.md` §4.12](COMPATIBLE.md) for why, and for the cost.

Read the two blocks together — they answer different questions:

> **Health** answers *"is it a valid tree?"*
> **Morphometrics** answer *"given a valid tree, is the morphology right?"*
>
> Sanitisation deliberately makes the morphometrics blind to structural failure, so a
> low `mmd_morpho` on its own means nothing until the health block is clean.

*Health* — **every one of these is exactly 0.0 on ground truth for both corpora**
(1.0 for `lcc_node_frac`), so any non-zero value is a generator failure, never a data
property. Values shown are from a real run (`vivid-thunder-13`):

| Key | Meaning | real run |
| --- | --- | ---: |
| `disconnected_frac` | fraction of graphs that are not connected | 0.125 |
| `multifurcation_frac` | fraction with a **non-root** node of degree > 3 | 0.062 |
| `isolated_node_frac` | fraction containing a degree-0 node | 0.049 |
| `cycle_frac` | fraction with at least one cycle | 0.038 |
| `non_critical_node_frac` | fraction with a non-root degree-2 node | 0.135 |
| `excess_edge_frac` / `degree2_node_frac` | magnitudes of the two above | 0.0008 / 0.003 |
| `lcc_node_frac` | mean fraction of nodes in the largest component | 0.980 |

`multifurcation_frac` **excludes the root** — including it reads 99.6% on real neuron GT,
because the soma is a legitimate high-degree hub.

*Morphometrics* (lower is better for `*_w1` and `mmd_*`; higher for `density_*` /
`coverage_*`):

| Key | Meaning |
| --- | --- |
| `mmd_morpho`, `density_morpho`, `coverage_morpho` | joint distance / fidelity / diversity over a 9-D per-tree morphometric vector. **`mmd_morpho` is the single best monitor.** |
| `mmd_tmd` | branching topology weighted by spatial reach (persistence image, `radial_root` filtration) |
| `branch_length_w1` | per-edge Euclidean branch lengths |
| `bifurcation_angle_w1` | pairwise sibling-branch angles at branch points |
| `radial_to_root_w1` | straight-line distance from root to each node |
| `contraction_w1` | per leaf: straight-line ÷ along-cable distance. Tortuosity. |
| `branch_order_w1` | on this data, exactly the node-depth distribution |
| `partition_asymmetry_w1` | Van Pelt asymmetry: 0 = balanced forks, 1 = caterpillar |
| `degree_w1` | non-root node degrees. GT is exactly `{1, 3}`, so this is sharp. |
| `node_count_w1`, `bifurcation_count_w1` | **branch-point counts, not size** — see below |
| `axial_extent_w1`, `radial_span_w1` | extents along / perpendicular to the root→centroid axis |
| `sholl_critical_radius_w1` | where the arbor is densest, as a fraction of its own reach |
| `tmd_barlen_w1` | persistence bar lengths |
| `w1_pooled_mean_normalized`, `w1_pertree_mean_normalized` | mean of W1 ÷ GT spread. **Use these, not a raw mean of the `*_w1` keys** — those mix microns, degrees and counts, and which key dominates flips between corpora (90% extents on neurons, counts on the metre-scale tree corpora). |
| `gen_degenerate_frac`, `morpho_nan_frac` | imputation disclosure; ~0.0 here because sanitisation removes the causes |

**`node_count` is topology, not size.** These are critical trees, so
`1 + leaves + bifurcations == N` holds for 100.00% of real graphs and
`corr(node_count, bifurcation_count) = 0.9970`. Fewer nodes means fewer branch points.
In-loop it used to read exactly 0.0 (the prior mask is paired with the GT batch); after
sanitisation it measures how many of the fed-in nodes became real branch points. That
change away from 0.0 is the metric starting to work, not a regression.

> ⚠️ **Score against the split the prior drew its sizes from.** Comparing to a different
> corpus turns `node_count_w1` into a corpus-mismatch measurement. On the real run,
> against its paired GT it reads 1.27 (0.052 GT-sd); against the wrong corpus, 11.55.

All metrics are rotation/translation invariant (verified to ~1e-16).

**Interpreting a number: use the real-vs-real floor.** Two disjoint halves of the *same*
real dataset differ by this much; a run near the floor is at ceiling performance, not
failing. The floor is **per corpus** — recompute it for whichever split you score against.
Measured over 6 disjoint splits on this repo's code:

| corpus | n/side | `mmd_morpho` | `w1_pooled_mean_normalized` | `w1_pertree_mean_normalized` |
| --- | ---: | ---: | ---: | ---: |
| `neurons` | 923 | −0.00009 ± 0.00027 | 0.0162 ± 0.0044 | 0.0503 ± 0.0062 |
| `neurons_conditional` | 1263 | +0.00012 ± 0.00038 | 0.0133 ± 0.0036 | 0.0549 ± 0.0115 |
| `trees_genus_d10` | 168 | +0.00027 ± 0.00203 | 0.0585 ± 0.0137 | 0.1512 ± 0.0378 |

For scale, `vivid-thunder-13` against its paired GT sits at `mmd_morpho` 0.1034
(z ≈ +384), `w1_pooled` 0.194 (z ≈ +41), `w1_pertree` 0.265 (z ≈ +35) — far above floor,
so the aggregate has plenty of headroom to track.

`mmd_*` is an **unbiased** estimator, so it is legitimately slightly negative when the two
sets match. Do not clip it and do not read a negative value as an error.

⚠️ **`mmd_morpho` here is not numerically comparable to dendrite_gen's** — different axial
frame and different sanitisation. Never put them on one plot axis; compare against the
floor above instead. See [`COMPATIBLE.md` §4.14](COMPATIBLE.md).

> **Coord scaling note:** `_generate` already rescales its output by
> `coord_scale` (= `NEURON_COORDS_STD_DEV`) back to physical microns, and the GT
> split is loaded untransformed, so both sides are compared with
> `coord_scale=1.0`. Do not double-scale.

Selected flags:

| Flag | Default | Notes |
| --- | --- | --- |
| `--dataset_split` | `val` | `train` / `val` / `test` — size-prior + GT source |
| `--n_molecules` | 256 | number of samples |
| `--integration_steps` | 100 | ODE steps |
| `--ode_sampling_strategy` | `log` | `linear` or `log` time grid |
| `--n_plot_examples` | 8 | samples per plot grid |
| `--save_raw` | off | also dump raw per-batch tensors as `<save_file>.raw.pt` |
| `--skip_eval` | off | pure sampling, no metrics/plots (no networkx/matplotlib) |
| `--tmd_cond` | off | paired conditional generation (see [§4](#4-tmd-conditioning)) |
| `--type_cond` | off | paired class-conditional generation + per-class metrics (see [§5](#5-neuron-type-cell-class-conditioning)) |
| `--per_cell_class_min_count` | 20 | with `--type_cond`: skip classes below this many gen/GT graphs |
| `--metric_report_level` | `standard` | `headline` / `standard` / `full`. Printing filter only — every metric is computed and written to `metrics.json` regardless. |

The same metrics are logged **inline during training** by `NeuronCFM` (a full ODE rollout
over the val set each validation) as `val-<key>`. With a class-conditioned model they are
additionally logged **per cell class** as `val-class_<name>-<key>`
(see [§5](#5-neuron-type-cell-class-conditioning)).

**Checkpoint selection.** `--ckpt_monitor` defaults to `val-loss`. Pass
`val-morpho-selection` to select on `mmd_morpho` instead — gated on the health fractions,
because sanitisation makes `mmd_morpho` blind to structural failure and a model emitting
garbage with a plausible spanning tree would otherwise win:

```bash
--ckpt_monitor val-morpho-selection \
--selection_max_disconnected_frac 0.05 --selection_max_cycle_frac 0.02
```

Ceilings default to 1.0 (no gating); an epoch that breaches one scores `+inf` and loses to
any healthy epoch. ⚠️ **Single-GPU only** — under DDP `sync_dist` averages per-rank
scalars, and the mean of per-rank MMDs is not the MMD of the union. `train.py` prints a
warning if it sees more than one visible GPU.

The GT-derived fit (morphometric mean/std, MMD bandwidths, TMD PCA) is computed **once**
on the first real validation epoch and reused for the whole run, so the `mmd_morpho`
trajectory is comparable across checkpoints. This is valid despite the per-epoch random
rotation because every feature is rotation invariant. Per-run constants
(`mmd_bandwidth_*`, `tmd_eff_rank`, `morpho_version`) go to the logger **config**, not the
time series.

---

## 4. TMD conditioning

TMD = **Topological Morphology Descriptor**. Conditioning lets the model
generate a neuron whose topology matches a target descriptor. The feature is
**optional and off by default** end-to-end.

**The descriptor** (`semlaflow/tmd/`): `compute_neuron_tmd(mol)` builds a rooted
networkx tree (root = soma at node 0) and computes a **persistence image** using
the rotation-invariant path-length-from-root filtration — default 16×16 =
**256-dim** vector. Degenerate/non-tree graphs yield a zero vector, so no sample
ever crashes the pipeline.

The vector is a *global* per-graph feature. In the model
(`SemlaGenerator`), when `tmd_hidden > 0` a small MLP (`Linear → SiLU →
Linear`) projects it to `tmd_hidden`, **broadcasts it to every node** (mirroring
`size_emb`), and concatenates it onto the invariant atom features. It is carried
through the data pipeline via a `_tmd` slot on `GeometricMol` (serialized into
the `.smol`), copied by the interpolant onto both the prior and interpolated
mols so it reaches the model from whichever dict is active (`interpolated` at
training, `prior` at sampling).

### 4a. Preprocess with TMD vectors

```bash
python -m semlaflow.preprocess_neurons \
    --input_dir  /path/to/neurons_final \
    --output_dir /path/to/neurons_final/smol \
    --compute_tmd \
    --tmd_filtrations path
```

- `--tmd_filtrations` accepts `path` (default, rotation-invariant), `height`,
  `rho`, or a space-separated combination. Each filtration adds a
  16×16 = 256-dim block to the vector.
- The script prints the resulting `TMD vector dim`.
- Without `--compute_tmd`, the output `.smol` is byte-identical to the
  unconditional one.

### 4b. Train with conditioning

```bash
python -m semlaflow.train \
    --dataset neurons \
    --data_path /path/to/neurons_final/smol \
    --tmd_conditioning \
    --tmd_hidden 64
```

- `tmd_dim` is inferred from the training data; training raises a clear error if
  the `.smol` has no TMD vectors (re-run preprocessing with `--compute_tmd`).
- `tmd_dim` / `tmd_hidden` are saved into the checkpoint hparams, so the
  conditioning MLP is reconstructed automatically on load.

### 4c. Sample with conditioning

```bash
python -m semlaflow.sample_neurons \
    --ckpt_path /path/to/tmd-best-###.ckpt \
    --data_path /path/to/neurons_final/smol \
    --save_dir  /path/to/out \
    --tmd_cond
```

This performs **paired conditional generation**: each generated sample is
conditioned on the TMD vector of a real graph from `--dataset_split`. Consistency
guards (all fatal):
- checkpoint is conditional (`tmd_dim > 0`) but `--tmd_cond` was not passed;
- `--tmd_cond` passed but the checkpoint is unconditional;
- `--tmd_cond` passed but the `.smol` has no TMD vectors (re-run preprocessing
  with `--compute_tmd`).

---

## 5. Neuron-type (cell-class) conditioning

An **alternative** conditioning path to TMD: condition generation on a discrete
neuron cell-class label instead of (or alongside) a continuous TMD vector. It
mirrors the cell-type conditioning in the primary `dendrite_gen` method and is
structurally parallel to the TMD path — **optional and off by default** end-to-end.

**The classes are per-dataset**, declared as `class_names` on that dataset's
`DATASET_CONFIGS` entry, with id = list index. For `neurons_conditional` there are 7
(`scriptutil.NEURON_CELL_CLASS_NAMES`):

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| name | 23P | 4P | 5P-IT | 5P-ET | 5P-NP | 6P-IT | 6P-CT |

The tree corpora declare 6 genera instead (`TREE_GENUS_NAMES`); see
[§6](#6-tree-genus-datasets). A dataset with `class_names=None` (e.g. `neurons`)
rejects `--type_conditioning` outright.

**The label** is parsed from a `# cell_class N` SWC header (`data/swc.py`) and
carried as a per-graph `_cell_class` slot on `GeometricMol` — exactly like `_tmd`:
serialized into the `.smol`, preserved across transforms, copied by the interpolant
onto the prior + interpolated mols, and stacked into `data["cell_class"]` by the
collator. In the model (`SemlaGenerator`), when `class_hidden > 0` a
`one_hot(n_classes) → Linear(class_hidden)` embedding (a learned embedding kept
explicit, matching the real method) is **broadcast to every node and concatenated**
onto the invariant features (mirroring `size_emb`/TMD). Conditioning is **always-on
when enabled** — no classifier-free guidance; a missing label raises.

### 5a. Preprocess (labels captured automatically)

No flag needed — see [§1](#1-data-preparation-swc--smol). Any corpus whose SWCs
carry a `# cell_class N` header (e.g. `neurons_conditional`) has the label attached
during preprocessing, and the printed class histogram confirms full coverage.

### 5b. Train with conditioning

```bash
python -m semlaflow.train \
    --dataset neurons_conditional \
    --data_path /path/to/neurons_conditional/smol \
    --type_conditioning \
    --class_hidden 16
```

- `n_classes` comes from the dataset's `class_names` in `DATASET_CONFIGS` (7 here,
  6 for the tree corpora). Training raises a clear error if the dataset declares no
  class names, if the `.smol` has no class labels, or if the data contains a class id
  outside the declared range.
- `n_classes` / `class_hidden` are saved into the checkpoint hparams, so the class
  embedding is reconstructed automatically on load.
- Orthogonal to TMD: `--type_conditioning` and `--tmd_conditioning` can be combined.

### 5c. Sample with conditioning

```bash
python -m semlaflow.sample_neurons \
    --ckpt_path /path/to/type-best-###.ckpt \
    --data_path /path/to/neurons_conditional/smol \
    --save_dir  /path/to/out \
    --type_cond
```

**Paired conditional generation**: each sample is conditioned on the cell class of
a real graph from `--dataset_split`. Consistency guards (all fatal): checkpoint
conditional but `--type_cond` absent; `--type_cond` on an unconditional checkpoint;
`--type_cond` but the `.smol` has no labels.

### 5d. Per-cell-class stratified metrics

When the model is class-conditioned, the structural distribution metrics are
additionally computed **per class** (generated graphs grouped by their conditioning
class, GT grouped by its own label, scored distribution-vs-distribution). Classes
with fewer than `--per_cell_class_min_count` (default 20) graphs on either side are
skipped.

- **Inline during training** (`NeuronCFM`): logged as `val-class_<name>-<key>` for every
  key at the current `--metric_report_level` (monitoring only; disable with
  `--no_per_cell_class`).
- **Offline** (`sample_neurons.py --type_cond`): written to `metrics.json` under
  `class_<name>` blocks and printed as a per-class table (`n_gen`, `n_gt`, `mmd_morpho`,
  `w1_pooled_mean_normalized`).

Each class is scored against **its own** GT subset, but the standardization, MMD
bandwidths and TMD PCA stay **run-wide** (`subset_gt_cache`). Refitting per class would
make the classes comparable neither to each other nor to the overall run.

---

## 6. Tree (genus) datasets

Three botanical-tree corpora — `trees_genus_d10`, `trees_genus_d15`,
`trees_genus_d20` — the same 3368 QSM reconstructions binarized to three different
depth caps. They are base-rooted, binarized, and carry `# cell_class N` genus
headers, so they use the `neurons_conditional` path unchanged: same vocab, same
transform, same `NeuronCFM`, same metrics.

```
/Users/umer/Documents/trees_genus_d{10,15,20}/
├── dataset_stats.csv
├── train/  2695 .swc
├── val/     337 .swc
└── test/    336 .swc      <- a real test split, unlike the neuron corpora
```

**The 6 genera** (`scriptutil.TREE_GENUS_NAMES`, id = list index, matching
`dendrite_gen`'s `utils.data_loading.TREE_GENUS_NAMES`):

| id | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| name | Fagus | Quercus | Acer | Carpinus | Fraxinus | Betula |
| train count | 2377 | 142 | 57 | 52 | 47 | 20 |

**Measured constants** (all three keep 100% of their graphs — nothing is dropped):

| dataset | coord_std | units | train median / max N | `max_nodes` |
| --- | ---: | --- | ---: | ---: |
| `trees_genus_d10` | 1.6417 | m | 72 / 378 | 384 |
| `trees_genus_d15` | 1.8062 | m | 178 / 1456 | 1536 |
| `trees_genus_d20` | 1.9999 | m | 338 / 3056 | 3072 |

Note the coords are in **metres**, not the neurons' microns — hence a coord_std
near 2 rather than near 63. Absolute W1 values for the extent metrics are therefore
in metres for these datasets.

### 6a. Preprocess

```bash
python -m semlaflow.preprocess_neurons \
    --input_dir  /Users/umer/Documents/trees_genus_d10 \
    --output_dir /Users/umer/Documents/trees_genus_d10/smol \
    --val_dir_name val --max_atoms 384          # 1536 for d15, 3072 for d20
```

`--val_dir_name val` is required: auto-detect prefers `val_extended/`, which the
tree corpora do not have. This writes `test.smol` as well, so
`sample_neurons.py --dataset_split test` works for these datasets.

### 6b. Memory — read this before training d15 or d20

SemlaFlow materialises dense `[B, N, N, d]` tensors in every layer, so activation
memory is **O(N²)**: measured at **~57 KB per node-pair** in fp32 at the default
model size. These trees are far larger than any neuron, so the defaults do not fit:

| | KB/pair | max N on a 40 GB A100 | d10 (384) | d15 (1536) | d20 (3072) |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 (the neuron default) | 57.0 | 785 | 8.6 GB | 137.7 GB ❌ | 550.8 GB ❌ |
| `--precision bf16-mixed` | 28.5 | 1110 | 4.3 GB | 68.9 GB ❌ | 275.4 GB ❌ |
| **+ `--grad_checkpointing`** | **4.0** | **2970** | **0.6 GB** | **9.6 GB** | **38.5 GB** |

**Reducing `--batch_cost` does not help.** `_round_batch_size` floors the batch at 1
(`data/util.py`), and these buckets are already there — a single N=3072 graph costs
551 GB in fp32 on its own. `--acc_batches` changes the effective batch, not the
micro-batch, so it does not help either. Gradient checkpointing (recomputing each
layer in the backward pass, ~30% extra compute) is the only lever below batch-size-1.
It is numerically exact — verified to produce bit-identical losses and gradients.

`--batch_cost` is best read as a memory budget: peak pairs ≈ `256 × batch_cost`.
`16384` is a reasonable start; read the `items per bucket / bucket batch sizes /
batches per bucket` printout to tune it.

**d20 has almost no headroom** at 38.5 GB of a 40 GB card, and exactly one train
graph (N=3056) sits above the 2970-node ceiling. If it OOMs, drop d20's `max_nodes`
and top bucket to 2816 (99.9% retention, 32.4 GB) and re-run preprocessing for d20.

### 6c. Train

```bash
python -m semlaflow.train \
    --dataset trees_genus_d10 \
    --data_path /Users/umer/Documents/trees_genus_d10/smol \
    --max_atoms 385 \
    --n_validation_mols 320 \
    --type_conditioning --class_hidden 16
```

For d15 / d20 add `--precision bf16-mixed --grad_checkpointing --batch_cost 16384`
and `--max_atoms 1537 / 3073`.

- **`--max_atoms` must be strictly greater than the largest graph.** `size_emb` is an
  `Embedding(max_atoms)` indexed by the raw node count, so `max_atoms == N` runs off
  the end of the table. The model now raises a clear error rather than an `IndexError`.
- **`--n_validation_mols`** defaults to 1800 but the val split has 337 graphs. It is
  clamped automatically with a printed notice, so the default is safe.

### 6d. Caveats specific to these corpora

- **Class balance is extreme.** Fagus is 88.2% of the corpus and Betula has 20 train /
  2 val / 2 test graphs. With `--per_cell_class_min_count 20` only Fagus clears the bar
  in val, so `val-class_*` effectively reports one class. Per-class W1 for
  Acer/Carpinus/Fraxinus/Betula at n ≤ 8 is not statistically meaningful — do not
  report it.
- **Bond-loss weight.** The bond head predicts N² logits of which only N−1 are
  positive: a 2/N positive rate, so 2.8% at d10's median of 72 but 0.6% at d20's 338.
  `--bond_loss_weight 1.0` was tuned at N≈50 and likely needs retuning for d15/d20.
- **Compute.** At these caps d15 is ~8× and d20 ~35× d10's FLOPs per epoch
  (`sum(N²)` over train), before the ~30% checkpointing overhead.
- **Root selection is estimated, not known.** These trees have `root_deg == 1` and no
  degree-2 nodes, so `choose_root`'s soma-hub heuristic never fires and it falls back
  to the PCA base node. PC1 is reliably the trunk axis (`|PC1 · trunk| ≈ 0.99`) and one
  of its two ends is the true root 96–98% of the time, so the only real question is
  *which* end — and `np.linalg.eigh` returns eigenvectors of arbitrary sign, which used
  to make that a coin flip (~50%). `_orient_axis` now picks the end by node density
  (the crown is dense, the trunk base is sparse), recovering the true root:

  | dataset | exact root, raw sign | with `_orient_axis` | ceiling | `bifurcation_angle` error |
  | --- | ---: | ---: | ---: | ---: |
  | `trees_genus_d10` | 50.7% | **86.5%** | 96.2% | 0.54° on a 76.5° mean |
  | `trees_genus_d15` | 46.0% | **90.0%** | 97.0% | 0.20° on a 75.8° mean |
  | `trees_genus_d20` | 51.5% | **93.8%** | 98.0% | 0.05° on a 75.9° mean |

  The last column is the W1 between the bifurcation-angle distribution under the true
  root and under the estimated root — i.e. the actual metric-level cost, ≤0.7%. It is
  small because rooting at the wrong end only changes the parent pointer for nodes on
  the path between the two candidates (~depth nodes out of N), not for the whole tree,
  and because the error is applied symmetrically to generated and GT graphs. Only
  `bifurcation_angle_w1` is affected at all: `branch_length_w1` is root-free, the three
  extent metrics use the axis sign-invariantly, and `leaf_count`/`bifurcation_count`
  shift by at most 1. The neuron path is untouched — it short-circuits at the soma hub
  100% of the time.

  **Why GT does not just use its known root** (it is exactly available at index 0):
  doing so would put an estimator on one side of the comparison and the truth on the
  other, so a perfect generator would never score zero. The full argument, with the
  per-metric measurements, is [`COMPATIBLE.md` §4.11](COMPATIBLE.md#411-the-root-is-estimated-for-ground-truth-graphs-too-never-taken-from-the-data).

---

## File / constant reference

| Path | Role |
| --- | --- |
| `semlaflow/data/swc.py` | SWC parser (incl. `# cell_class` header) + SWC → `GeometricMol` adapter (RDKit-free) |
| `semlaflow/preprocess_neurons.py` | SWC dir → `train.smol` / `val.smol` / `test.smol` (+ optional TMD, + auto cell-class) |
| `semlaflow/scriptutil.py` | `build_neuron_vocab`, `neuron_mol_transform`, and `DATASET_CONFIGS` — the per-dataset registry |
| `semlaflow/models/semla.py` | `SemlaGenerator` — `tmd_proj` (TMD) + `class_proj` (cell-class) embeddings |
| `semlaflow/models/fm.py` | threads `cond_tmd`/`cond_class` through forward + `_generate` |
| `semlaflow/models/neuron_cfm.py` | `NeuronCFM` — eval = loss + structural metrics (+ per-class stratified) |
| `semlaflow/sample_neurons.py` | offline sampling + evaluation (+ per-class metrics) |
| `semlaflow/tmd/` | TMD persistence-image descriptor (numpy + networkx). `radial_root` is the evaluation filtration; `path` is the conditioning one — deliberately different. |
| `semlaflow/validation/sanitise.py` | structural health block + LCC → MST → degree-2 contraction |
| `semlaflow/validation/dist_metrics.py` | W1 marginals, joint MMD/density/coverage, normalised aggregates, report tiers |
| `semlaflow/validation/structural_metrics.py` | per-graph morphometric extractors (branch length, angles, Sholl, Strahler, asymmetry, …) |
| `semlaflow/util/dist_helper.py` | RBF-kernel MMD + Naeem et al. density/coverage |
| `semlaflow/validation/convert.py` | `GeometricMol` ↔ networkx, `samples_to_mols` |
| `semlaflow/validation/plot.py` | multi-azimuth 3D plot grids |
| `tests/validation_metrics.py` | unit tests for the above; the real-corpus tests skip if the `.smol` files are absent |

**`DATASET_CONFIGS` in `scriptutil.py` is the single source of truth** for every SWC
corpus. Each entry is a `DatasetConfig(coord_std, bucket_limits, max_nodes,
class_names, units)`; registering a new corpus means adding one entry and nothing
else, since `NEURON_DATASETS = tuple(DATASET_CONFIGS)` and every branch point reads
from the table.

| dataset | coord_std | units | top bucket | classes |
| --- | ---: | --- | ---: | ---: |
| `neurons` | 62.6894 | µm | 220 | — (unconditional) |
| `neurons_conditional` | 66.0298 | µm | 256 | 7 cell types |
| `trees_genus_d10` | 1.6417 | m | 384 | 6 genera |
| `trees_genus_d15` | 1.8062 | m | 1536 | 6 genera |
| `trees_genus_d20` | 1.9999 | m | 3072 | 6 genera |

Accessors: `get_dataset_config(name)` (strict, for build-time paths),
`class_names_for_dataset(name)` and `class_label(dataset, idx)` (tolerant, for
checkpoint-driven metric labelling — they fall back to `id<N>`). The
`NEURON_COORDS_STD_DEV` / `NEURON_BUCKET_LIMITS` / `NEURON_CELL_CLASS_NAMES` /
`NEURON_NUM_CLASSES` names still exist as derived aliases for the two neuron entries.

Also in `scriptutil.py`: `BOND_MASK_INDEX = 5`. Vocab ordering
`<PAD>=0, <MASK>=1, NODE=2` is coupled to `NEURON_NODE_TOKEN_INDEX` in
`data/swc.py` — keep them in sync.
