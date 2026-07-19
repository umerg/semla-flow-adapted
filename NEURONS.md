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
| Coord scale | `QM9/GEOM_COORDS_STD_DEV` | `NEURON_COORDS_STD_DEV` (physical microns) |
| Model class | `MolecularCFM` | `NeuronCFM` (RDKit metrics stripped) |
| Eval | validity/energy/stability | Wasserstein-1 structural distribution metrics |

**Node/edge encoding** (`data/swc.py`): every node is the single token `NODE`
(vocab index 2); every edge is a single real class (index 1, reusing the
molecular "single bond" slot so `BOND_MASK_INDEX=5` stays valid). The root
(soma, `parent<=0` in the SWC) is reordered to node index 0.

Everything below is selected by passing `--dataset neurons` (the original
unlabelled corpus) or `--dataset neurons_conditional` (a soma-rooted, binarized,
**cell-class-labelled** corpus). Both use the same neuron vocab / transform /
`NeuronCFM`; they differ only in their measured coord scale and size buckets
(`NEURON_CONDITIONAL_COORDS_STD_DEV` / `NEURON_CONDITIONAL_BUCKET_LIMITS`).

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

This writes `<output_dir>/train.smol` and `<output_dir>/val.smol`, skips graphs
with more than `--max_atoms` (default 256) nodes, and does a round-trip sanity
check.

**Important — coord std-dev.** The script prints a measured `coord_std` over the
zero-CoM'd training coordinates, e.g.:

```
== Measured neuron coord_std (over train, post zero-CoM): 62.6894 ==
Paste this into semlaflow.scriptutil as NEURON_COORDS_STD_DEV.
```

If the training corpus changes, paste the new value into
`semlaflow/scriptutil.py` as `NEURON_COORDS_STD_DEV` (and refresh
`NEURON_BUCKET_LIMITS` if the size distribution shifts).

Useful flags:
- `--val_dir_name NAME` — force a specific validation subfolder (default
  auto-picks `val_extended`, else `val`).
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

Paste that run's coord_std into `NEURON_CONDITIONAL_COORDS_STD_DEV` (currently
`66.0298`) and make sure `NEURON_CONDITIONAL_BUCKET_LIMITS` covers the printed
train `max` node count (currently `242`, top bucket `256`). See
[§5](#5-neuron-type-cell-class-conditioning) to condition on the labels.

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

**Structural metrics** (`validation/dist_metrics.py`, lower is better):

| Key | Meaning |
| --- | --- |
| `branch_length_w1` | per-edge Euclidean branch lengths |
| `bifurcation_angle_w1` | pairwise sibling-branch angles at branch points |
| `leaf_count_w1` | number of leaves per tree |
| `bifurcation_count_w1` | number of branch points per tree |
| `axial_extent_w1` | spread along the graph's own PCA principal axis |
| `total_extent_w1` | 3D diameter (max pairwise distance) |

All metrics are rotation/translation invariant. A `disconnected_frac`
diagnostic is also reported (fraction of generated graphs that are not a single
connected component).

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

The same structural metrics are also logged **inline during training** by
`NeuronCFM` (a full ODE rollout over the val set each validation, logged as
`val-branch_length_w1`, …, `val-structural-w1-mean`, `val-disconnected-frac`).
With a class-conditioned model these are additionally logged **per cell class** as
`val-class_<name>-<key>` and `val-class_<name>-w1-mean` (see [§5](#5-neuron-type-cell-class-conditioning)).
All are for monitoring only — checkpoint selection still uses `val-loss`.

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

**The 7 classes** (`scriptutil.NEURON_CELL_CLASS_NAMES`, id = list index):

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| name | 23P | 4P | 5P-IT | 5P-ET | 5P-NP | 6P-IT | 6P-CT |

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

- `n_classes` is fixed at `NEURON_NUM_CLASSES` (7); training raises a clear error if
  the `.smol` has no class labels.
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

- **Inline during training** (`NeuronCFM`): logged as `val-class_<name>-<key>` for
  each `METRIC_KEYS` plus `val-class_<name>-w1-mean` (monitoring only; disable with
  `--no_per_cell_class`).
- **Offline** (`sample_neurons.py --type_cond`): written to `metrics.json` under
  `class_<name>` blocks and printed as a per-class table (`n_gen`, `n_gt`, `w1_mean`).

---

## File / constant reference

| Path | Role |
| --- | --- |
| `semlaflow/data/swc.py` | SWC parser (incl. `# cell_class` header) + SWC → `GeometricMol` adapter (RDKit-free) |
| `semlaflow/preprocess_neurons.py` | SWC dir → `train.smol` / `val.smol` (+ optional TMD, + auto cell-class) |
| `semlaflow/scriptutil.py` | `build_neuron_vocab`, `neuron_mol_transform`, `NEURON_*` constants, `NEURON_CELL_CLASS_NAMES` |
| `semlaflow/models/semla.py` | `SemlaGenerator` — `tmd_proj` (TMD) + `class_proj` (cell-class) embeddings |
| `semlaflow/models/fm.py` | threads `cond_tmd`/`cond_class` through forward + `_generate` |
| `semlaflow/models/neuron_cfm.py` | `NeuronCFM` — eval = loss + structural metrics (+ per-class stratified) |
| `semlaflow/sample_neurons.py` | offline sampling + evaluation (+ per-class metrics) |
| `semlaflow/tmd/` | TMD persistence-image descriptor (numpy + networkx) |
| `semlaflow/validation/dist_metrics.py` | Wasserstein-1 structural distribution metrics |
| `semlaflow/validation/structural_metrics.py` | branch-length / bifurcation-angle extractors |
| `semlaflow/validation/convert.py` | `GeometricMol` ↔ networkx, `samples_to_mols` |
| `semlaflow/validation/plot.py` | multi-azimuth 3D plot grids |

Key constants in `scriptutil.py`: `NEURON_COORDS_STD_DEV` (`62.6894`) /
`NEURON_CONDITIONAL_COORDS_STD_DEV` (`66.0298`), physical microns;
`NEURON_BUCKET_LIMITS` / `NEURON_CONDITIONAL_BUCKET_LIMITS` (top bucket `256`,
covers the labelled corpus max of 242); `NEURON_CELL_CLASS_NAMES` (7 classes) /
`NEURON_NUM_CLASSES`; `NEURON_DATASETS = ("neurons", "neurons_conditional")`;
`BOND_MASK_INDEX = 5`. Vocab ordering `<PAD>=0, <MASK>=1, NODE=2` is coupled to
`NEURON_NODE_TOKEN_INDEX` in `data/swc.py` — keep them in sync.
