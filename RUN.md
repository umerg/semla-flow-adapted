# RUN.md — command reference for the SWC datasets

Copy-paste commands for the corpora that get trained, with the flags that differ
between them and *why*. Conceptual background lives in [`NEURONS.md`](NEURONS.md);
design rationale in [`COMPATIBLE.md`](COMPATIBLE.md). This file is just the runbook.

`neurons` (the original unlabelled corpus) and `neurons_conditional` are registered too but are
superseded by `neurons_conditional_full`; see [§0a](#0a-superseded-corpora) and
[§6](#6-legacy-the-unlabelled-neurons-corpus).

The three `trees_genus_d*_capped` corpora are node-capped, sample-matched duplicates of the
tree corpora and are the cheaper thing to train — see [§3a](#3a-the-capped-tree-corpora).

---

## 0. The one-screen summary

These are the five corpora currently trained. `neurons_conditional`, `trees_genus_d15` and
`trees_genus_d20` are superseded — see [§0a](#0a-superseded-corpora).

| | `neurons_conditional_full` | `trees_genus_d10` | `d10_capped` | `d15_capped` | `d20_capped` |
| --- | --- | --- | --- | --- | --- |
| what | mouse cortical neurons | trees, depth ≤10 | ≤10, capped | ≤15, capped | ≤20, capped |
| classes | 7 cell types | 6 genera | 6 genera | 6 genera | 6 genera |
| units | µm | **m** | **m** | **m** | **m** |
| `coord_std` | 66.1040 | 1.6417 | 1.6346 | 1.7935 | 1.9658 |
| train / val / test | 22773 / 2529 / 1167 | 2695 / 337 / 336 | 2538 / 316 / 315 | 2538 / 316 / 315 | 2538 / 316 / 315 |
| median / max N | 53 / 537 | 72 / 378 | 70 / 268 | 168 / 666 | 312 / 1110 |
| **preprocess `--max_atoms`** | `537` | `384` | `268` | `666` | `1110` |
| **train `--max_atoms`** | `538` | `385` | `269` | `667` | `1111` |
| **`--n_validation_mols`** | `1800` | `320` | `316` | `316` | `316` |
| **`--precision`** | `32` | `32` | `32` | `bf16-mixed` | `bf16-mixed` |
| **`--grad_checkpointing`** | no | no | no | **no** | **yes** |
| **`--batch_cost`** | `1024` | `1024` | `1024` | `2048` | `16384` |
| peak GPU (est.) | ~18 GB | ~18 GB | ~26 GB | ~26 GB | ~28 GB |
| epoch cost vs d10 | — | 1× | 0.75× | 5× | 20× |

Everything else — `--epochs`, `--lr`, architecture, `--optimal_transport` — stays at the repo
defaults. The flags in bold are the whole story.

**Every corpus above has one `smol/` built with `--compute_tmd`, and that single directory serves
unconditional, class-conditioned and TMD-conditioned runs alike** — there are no `smol_tmd/`
directories. See [§1a](#1a-tmd-conditioning).

The peaks are sampler-aware (`max over buckets of batch_size × limit² × KB/pair`), not
single-graph figures; see [§4](#4-memory-what-to-do-when-it-ooms).

### 0a. Superseded corpora

| corpus | superseded by | why |
| --- | --- | --- |
| `neurons_conditional` | `neurons_conditional_full` | strict subset — same splits, 24 fewer graphs, and its `.smol` also drops the 33 train graphs above its 256 cap |
| `trees_genus_d15` | `trees_genus_d15_capped` | 1456-node tail forces `--grad_checkpointing`; the ladder is also lossy (§3a) |
| `trees_genus_d20` | `trees_genus_d20_capped` | 3056-node tail, ~38 GB of a 40 GB card, lossy ladder |
| `neurons` | `neurons_conditional_full` | no class labels |

They stay registered and their entries are unchanged, so old runs remain reproducible. Their
original flags are in the git history of this file.

### Why those four flags differ

1. **`--max_atoms` on `train.py` must be strictly greater than the largest graph in
   any split.** `size_emb` is an `nn.Embedding(max_atoms)` indexed by the raw node
   count, so a graph with exactly `max_atoms` nodes runs one past the end of the table.
   The model now raises a clear error rather than an `IndexError`. Actual headroom with
   the values in the table:

   | dataset | largest graph | `--max_atoms` | headroom |
   | --- | ---: | ---: | ---: |
   | `neurons` | 217 | 256 | 39 |
   | `neurons_conditional` | 242 | 256 | 14 |
   | `neurons_conditional_full` | 537 | 538 | 1 |
   | `trees_genus_d10` | 378 | 385 | 7 |
   | `trees_genus_d15` | 1456 | 1537 | 81 |
   | `trees_genus_d20` | 3056 | 3073 | 17 |
   | `trees_genus_d10_capped` | 268 | 269 | 1 |
   | `trees_genus_d15_capped` | 666 | 667 | 1 |
   | `trees_genus_d20_capped` | 1110 | 1111 | 1 |

   The capped rows run at headroom 1 deliberately: their `--max_atoms` is `measured max + 1`, so
   any future corpus change that grows a graph by even one node fails loudly at
   `models/semla.py:880` instead of training on a silently different corpus.

   `neurons` and `neurons_conditional` keep the default `256` — `max_atoms` is a saved hparam
   that sizes the embedding, so changing it would break loading their existing checkpoints.
   `neurons_conditional_full` is a new corpus with no such legacy, so it uses `537 + 1`. The tree
   values are `cap + 1`, the smallest safe choice.

   Preprocess's `--max_atoms` is a *different* knob: it *drops* graphs above the cap.
   At the caps in the table nothing is dropped from **any** corpus — every current `.smol`
   reports `dropped 0 graphs` on every split.
2. **`--n_validation_mols` 1800 > the tree val splits** (337 uncapped, 316 capped). It is clamped
   automatically with a printed notice, so the default will not crash — but passing the real count
   keeps the logs honest. `neurons_conditional_full` has 2,529 val graphs, so `1800` is not clamped
   there.
3. **`--precision` / `--grad_checkpointing`** — activation memory is O(N²) at ~57 KB
   per node-pair in fp32. d15 and d20 do not fit on a 40 GB card without both. See
   [§4](#4-memory-what-to-do-when-it-ooms).
4. **`--batch_cost`** is a memory budget, not a batch size: peak pairs ≈
   `256 × batch_cost`. Once checkpointing frees up headroom you can afford a much
   larger budget, which is where the throughput comes back. It also sets each bucket's batch
   size, which used to mean that raising it could drop under-full buckets from training
   entirely; that is fixed (`drop_last=False` on the train loader), but it is still the knob
   that decides how ragged the last batch of each bucket is — see
   [§3a](#--batch_cost-used-to-silently-exclude-small-graphs-fixed-2026-08-15).

---

## 1. `neurons_conditional_full` — 7 cortical cell types

The neuron corpus of record and the baseline the tree runs are compared against.

```bash
# preprocess (smol/ already exists; rebuild only if the corpus changed)
python -m semlaflow.preprocess_neurons \
    --input_dir  /Users/umer/Documents/neurons_conditional_full \
    --output_dir /Users/umer/Documents/neurons_conditional_full/smol \
    --val_dir_name val --test_dir_name test --max_atoms 537 \
    --compute_tmd --tmd_filtrations path radial_root

# train, class-conditioned  (drop the two flags for unconditional; see 1a for TMD)
python -m semlaflow.train \
    --dataset neurons_conditional_full \
    --data_path /Users/umer/Documents/neurons_conditional_full/smol \
    --max_atoms 538 \
    --type_conditioning --class_hidden 16

# sample + evaluate
python -m semlaflow.sample_neurons \
    --ckpt_path <ckpt> \
    --data_path /Users/umer/Documents/neurons_conditional_full/smol \
    --save_dir  <out> \
    --dataset_split test --type_cond
```

Notes:
- It is a **strict superset of `neurons_conditional`**: all 26,445 of that corpus's graphs with
  **identical split assignment**, plus 24 more (23 train, 1 val). Those 24 are the somata that
  `neurons_conditional`'s `MAX_CHILDREN=16` primary-dendrite cap dropped — dendrite_gen raised the
  cap to 23, the corpus maximum, so no neuron is filtered by soma degree any more (23P 20, 5P-ET 3,
  5P-IT 1; see `dendrite_gen/docs/NEURON_DATASET_STATS.md` §2). That cap change widens a
  *dendrite_gen* input feature but is inert here — `swc_to_geometric_mol` reads only x/y/z and
  adjacency. Its `.smol` additionally keeps the 33 train graphs that `neurons_conditional`'s
  256-node cap discards, since nothing is capped here (max N 537, `--max_atoms 537` drops zero).
  Prior `neurons_conditional` results stay broadly comparable, but the corpora are not identical —
  say which one a number came from.
- It has a **real `test.smol`** (1,167 graphs), unlike `neurons_conditional`. `--dataset_split`
  still defaults to `val`, so pass `--dataset_split test` explicitly when you want the test split.
- Expect `Class conditioning enabled: n_classes=7 (23P, 4P, ...)`.
- **All 7 classes clear `--per_cell_class_min_count 20` in val** (866 / 642 / 336 / 65 / 28 / 353 /
  239), so unlike the tree corpora the `val-class_*` metrics are meaningful for every class.
- It is *cheaper* than `neurons_conditional` — ~18 GB against ~23 GB at `--batch_cost 1024` —
  because its bucket ladder folds the 224 bucket into 256, and that bucket was setting the peak.

### 1a. TMD conditioning

Orthogonal to `--type_conditioning`; both can be on at once.

**One `smol/` per corpus serves all three run modes.** The descriptor is computed and stored at
preprocess time, but it is *additive*: `--compute_tmd` writes `mol._tmd` and `mol._tmd_filtrations`
next to the coordinates and class labels every `.smol` already carries, and nothing reads them
unless the model was built with `--tmd_conditioning`. So build the corpus once, with descriptors,
and pick the mode at train time. **There are no `smol_tmd/` directories** — every corpus in §0 was
built with `--compute_tmd`.

```bash
# train, on the SAME smol/ used by the unconditional and class-conditioned runs
python -m semlaflow.train \
    --dataset neurons_conditional_full \
    --data_path /Users/umer/Documents/neurons_conditional_full/smol \
    --max_atoms 538 \
    --tmd_conditioning --tmd_hidden 64

# sample + evaluate (paired: each sample gets a real graph's descriptor)
python -m semlaflow.sample_neurons \
    --ckpt_path <ckpt> \
    --data_path /Users/umer/Documents/neurons_conditional_full/smol \
    --save_dir  <out> \
    --dataset_split test --tmd_cond
```

Why it is safe to leave the descriptor in place on a run that does not use it:

- `train.py:107-119` leaves `tmd_dim = 0` unless `--tmd_conditioning`, and
  `semla.py:892-900` gates the projection on `self.tmd_proj is not None`, which is `None` when
  `tmd_hidden == 0`. The collate *does* stack the vectors unconditionally
  (`data/datamodules.py:211-215`), so an unused `data["tmd"]` rides along in each batch and is
  dropped unread — a few MB of host→device traffic per epoch, no effect on results.
- The distribution metrics (`mmd_tmd`, `tmd_barlen_w1`, `tmd_eff_rank`) recompute their own
  embeddings from the graphs and never read the stored vectors, so they are unchanged either way.
- Verified end to end: unconditional, `--type_conditioning` and `--tmd_conditioning` runs all
  train off one `neurons_conditional_full/smol`, with `tmd_dim=0` in the first two.

Cost of carrying it: 512 × float32 = **2 KB per graph** (~+28% `.smol` size, ~53 MB on the neuron
train split) and **0.6–2.5 ms per graph** of one-time preprocessing — ~14 s for all 22,773 neuron
graphs. `compute_neuron_tmd` is pure numpy/networkx; `persim` is only needed for the *evaluation*
metrics below, not for building the descriptor.

Notes:
- Expect `TMD conditioning enabled: tmd_dim=512 (path, radial_root), tmd_hidden=64`.
  A `tmd_dim=256` here means the `.smol` was built with one filtration — re-preprocess.
- `--tmd_filtrations` takes only `path` and `radial_root`. `height`/`rho` are rejected:
  they need a fixed anatomical axis that the random-rotation transform destroys
  (COMPATIBLE.md §4.15).
- Sampling **fails** if the checkpoint's filtrations differ from the dataset's. Both
  sets are 512-dim, so this guard is the only thing that catches a mismatched `.smol`.
- Watch `val-tmd_cond-pd_wasserstein_path_mean` for whether the conditioning is being
  followed. On a conditioned run `mmd_tmd` cannot tell you that — its filtration is one
  the model trained on (COMPATIBLE.md §4.14).
- PD distances need `persim` (already in the `NEURO2` env). Without it they read `nan`
  and `pd_nan_frac_*` reads 1.0; the extent and W1 entries still compute.

Cost knobs for that block, all training-side and ignored on an unconditional run:

| flag | default | effect |
| --- | --- | --- |
| `--tmd_cond_every` | `5` | run the matched-pair evaluation every N validation epochs |
| `--tmd_cond_max_pairs` | `64` | cap on pairs scored per run (a persim Wasserstein per pair per filtration) |
| `--no-tmd_cond_eval` | — | skip it entirely |

`sample_neurons.py` has its own `--tmd_cond_max_pairs` (default `256`) for the offline
`"tmd_cond"` block in `metrics.json`.

---

## 2. `trees_genus_d10` — the cheap one, use it to iterate

Fits comfortably in fp32 with no special flags. This is the corpus to debug on.

```bash
# smol/ already exists, built with descriptors; rebuild only if the corpus changes
python -m semlaflow.preprocess_neurons \
    --input_dir  /Users/umer/Documents/trees_genus_d10 \
    --output_dir /Users/umer/Documents/trees_genus_d10/smol \
    --val_dir_name val --test_dir_name test --max_atoms 384 \
    --compute_tmd --tmd_filtrations path radial_root

python -m semlaflow.train \
    --dataset trees_genus_d10 \
    --data_path /Users/umer/Documents/trees_genus_d10/smol \
    --max_atoms 385 \
    --n_validation_mols 320 \
    --type_conditioning --class_hidden 16

python -m semlaflow.sample_neurons \
    --ckpt_path <ckpt> \
    --data_path /Users/umer/Documents/trees_genus_d10/smol \
    --save_dir  <out> \
    --dataset_split test --type_cond
```

Pass `--val_dir_name val` on **all** tree corpora. It is not strictly required — auto-detect
prefers `val_extended/` but falls back to `val/`, which is what they have — it just pins the
behaviour and keeps the log honest.

Expect `Class conditioning enabled: n_classes=6 (Fagus, Quercus, Acer, Carpinus,
Fraxinus, Betula)` — if you see 7 cell-type names, the wrong dataset is selected.

**Its bucket ladder changed on 2026-08-15** and no longer uses `_SWC_BUCKET_PREFIX`. The old
ladder's 24-node bucket held 89 of the 2,695 train graphs but was given batch size 312 at the
default `--batch_cost 1024`, so it produced zero batches and those graphs never entered training
(§3a). Runs before that date trained on **2,606 of 2,695 graphs**; do not compare them directly
with runs after it. Peak memory is unchanged (18.2 GB at `--batch_cost 1024`, either ladder).

---

## 3. `trees_genus_d15` and `trees_genus_d20` — need bf16 + checkpointing

> **Superseded by [§3a](#3a-the-capped-tree-corpora).** Prefer `trees_genus_d15_capped` /
> `trees_genus_d20_capped`: same trees minus a 199-graph tail, 37–43% cheaper per epoch, d15 no
> longer needs `--grad_checkpointing`, and their bucket ladders do not silently drop small graphs
> (these two do — 2.23% and 13.47% at the settings below). Kept here so existing runs stay
> reproducible; their `smol/` was **not** rebuilt with `--compute_tmd`, so TMD runs need one.

```bash
# --- d15 ---
python -m semlaflow.preprocess_neurons \
    --input_dir  /Users/umer/Documents/trees_genus_d15 \
    --output_dir /Users/umer/Documents/trees_genus_d15/smol \
    --val_dir_name val --max_atoms 1536

python -m semlaflow.train \
    --dataset trees_genus_d15 \
    --data_path /Users/umer/Documents/trees_genus_d15/smol \
    --max_atoms 1537 \
    --n_validation_mols 320 \
    --precision bf16-mixed --grad_checkpointing --batch_cost 16384 \
    --type_conditioning --class_hidden 16

# --- d20 --- (same, with the larger cap)
python -m semlaflow.preprocess_neurons \
    --input_dir  /Users/umer/Documents/trees_genus_d20 \
    --output_dir /Users/umer/Documents/trees_genus_d20/smol \
    --val_dir_name val --max_atoms 3072

python -m semlaflow.train \
    --dataset trees_genus_d20 \
    --data_path /Users/umer/Documents/trees_genus_d20/smol \
    --max_atoms 3073 \
    --n_validation_mols 320 \
    --precision bf16-mixed --grad_checkpointing --batch_cost 16384 \
    --type_conditioning --class_hidden 16
```

Sampling is unchanged from d10 apart from `--data_path` — `--precision` and
`--grad_checkpointing` are training-only, and neither is stored as a weight-affecting
hparam, so a checkpoint trained with them samples fine without them.

**Consider `--no_val_structural_metrics` for d20.** The structural validation metrics
run a full ODE rollout over the val set every `--val_check_epochs` (default **10**, so 30
validations over a 300-epoch run). At d20 sizes that is expensive; disabling it trades the
metric trajectory for wall-clock. Raising `--val_check_epochs` is the softer version of
the same trade — it thins the trajectory instead of removing it.

---

## 3a. The capped tree corpora

`trees_genus_d{10,15,20}_capped` are node-count-capped, **sample-matched** duplicates of the three
tree corpora. The rule: drop every tree whose **d20** node count exceeds **1110** — 199 treeIDs —
and remove that same ID set from all three depths. One shared ID set is what keeps the depth
ablation clean: all three hold the same **3,169** graphs (2,538 / 316 / 315), so a d10-vs-d15-vs-d20
comparison measures depth, not composition. It also bounds d15 at 666 and d10 at 268, because d15's
own tail above 785 is a strict subset of d20's above 1110.

The tail is cheap to lose. The top 5% of trees carry ~18% of the nodes but **~39% of the N² compute**,
and `corr(nodes, realized_depth)` is only **0.37** at d20 — the tail is *wide, not deep*.

| | `d10_capped` | `d15_capped` | `d20_capped` |
| --- | --- | --- | --- |
| train / val / test | 2538 / 316 / 315 | same | same |
| median / max N | 70 / **268** | 168 / **666** | 312 / **1110** |
| `coord_std` | 1.6346 | 1.7935 | 1.9658 |
| — uncapped was | 1.6417 | 1.8062 | 1.9999 |
| ΣN² vs uncapped | 75.3% | 63.0% | 57.2% |
| depth mean / p95 / max | 10.75 / 12 / 14 | 16.01 / 18 / 20 | 21.16 / 23 / 26 |
| **preprocess `--max_atoms`** | `268` | `666` | `1110` |
| **train `--max_atoms`** | `269` | `667` | `1111` |
| **`--n_validation_mols`** | `316` | `316` | `316` |
| **`--precision`** | `32` | `bf16-mixed` | `bf16-mixed` |
| **`--grad_checkpointing`** | no | **no** | **yes** |
| **`--batch_cost`** | `1024` | `2048` | `16384` |
| peak (sampler-aware, est.) | ~26 GB | ~26 GB | ~28 GB |

Class balance *improves*: the dropped 199 are 192 Fagus, 6 Acer, 1 Carpinus, so Fagus falls from
88.18% to **87.66%** and **no rare-genus validation tree is lost** (val stays Quercus 18, Acer 7,
Carpinus 6, Fraxinus 6, Betula 2). Per-genus metrics are still only meaningful for Fagus — see §5.

### Do I only need bf16 now?

**d10_capped: no special flags. d15_capped: `--precision bf16-mixed` alone — drop
`--grad_checkpointing`. d20_capped: keep both.**

d20's max N of 1110 sits *exactly* at §4's 40 GB bf16 ceiling: 1110² × 28.5 KB = **35.1 GB** for a
single graph at batch size 1, before ~0.4 GB of model/Adam/EMA state and any fragmentation. And the
top bucket is not the only constraint — mid-bucket peak scales with `--batch_cost`, so dropping
checkpointing would *also* pin `batch_cost` at ≤2048 and still sit at 35.1 GB, versus **27.6 GB at
`batch_cost 16384`** with checkpointing on. Checkpointing is numerically exact (verified
bit-identical), so keeping it is a strict win. What the cap buys d20 is 43% of the epoch cost and
real headroom — not one fewer flag. d15_capped is different: its largest graph is 12.6 GB in bf16
(25.3 GB even in fp32), so checkpointing is pure overhead there.

### `--batch_cost` used to silently exclude small graphs (fixed 2026-08-15)

**Fixed at the source — `train_dataloader` now passes `drop_last=False`
(`data/datamodules.py:79`).** All corpora are at 100% train coverage per epoch, verified by
iterating the real `BucketBatchSampler` and taking the union of emitted indices. Read this section
for what it means for runs made *before* that date.

The defect: with `drop_last=True`, `BucketBatchSampler` computes
`n_batches = len(bucket) // batch_size` (`data/util.py:48`), so **a bucket holding fewer graphs than
its own batch size yields zero batches**. `__iter__` draws buckets with
`random.choices(weights=remaining_batches)`, so a zero-weight bucket is never drawn and its graphs
never enter training *in any epoch*. Because a bucket's batch size is set by `--batch_cost`, raising
that knob quietly shrank the corpus. On the uncapped `trees_genus_d20` at the `--batch_cost 16384`
prescribed in §3, the six lowest buckets of `_SWC_BUCKET_PREFIX` all went to zero: **363 of 2,695
train graphs (13.5%) — everything under ~128 nodes.**

This never affected validation or sampling: `val_dataloader` / `test_dataloader` already passed
`drop_last=False`, so `sample_neurons.py` (which uses `test_dataloader()`, `:486`) always saw every
graph, and its `DEFAULT_BATCH_COST = 8192` was always harmless.

Train-split coverage per epoch, measured by iterating the sampler:

| dataset | `--batch_cost` | before (`drop_last=True`) | now |
| --- | ---: | ---: | ---: |
| `neurons_conditional_full` | 1024 | 22,528 / 22,773 (98.9%) | **100%** |
| `trees_genus_d10` | 1024 | 2,684 / 2,695 (99.6%) | **100%** |
| `trees_genus_d10_capped` | 1024 | 2,514 / 2,538 (99.1%) | **100%** |
| `trees_genus_d15_capped` | 2048 | 2,503 / 2,538 (98.6%) | **100%** |
| `trees_genus_d20_capped` | 16384 | 2,365 / 2,538 (93.2%) | **100%** |
| `trees_genus_d15` | 1024 | 2,536 / 2,695 (94.1%) | **100%** |
| `trees_genus_d20` | 16384 | 2,208 / 2,695 (81.9%) | **100%** |

Two distinct losses hid behind the one flag, and only the second was serious. The **remainder**
(`len(bucket) % batch_size`) was benign: `RandomSampler` reshuffles each epoch, so a different
random subset was skipped each time and the chance a graph was never seen over 300 epochs was
~0 (worst case `0.188³⁰⁰`). Its only real cost was that an "epoch" ran 0.4–6.8% short of a full
pass, which matters when epochs are a budget-parity unit. The **zero-batch buckets** were permanent.

Consequences for existing results, in decreasing order of severity:

- **Uncapped `trees_genus_d20` at `--batch_cost 16384`: 13.5% of the train split (363 graphs)
  permanently never seen**, plus a further ~4.6% skipped per epoch as re-randomising remainder —
  the 81.9% coverage above is the two combined. Uncapped `d15` at the default: 2.2% permanent.
- `trees_genus_d10` also had its ladder coarsened on 2026-08-15 (it had 89 graphs of ≤24 nodes in a
  bucket that starved at the default `batch_cost`). Between the ladder change and this one, runs on
  it before that date trained on 2,606 of 2,695 graphs.
- Everything else was ≥98.6% and re-randomising, i.e. statistically indistinguishable from full.

The coarse bucket ladders on the capped corpora and `trees_genus_d10` (`[96, 128, …]`,
`[128, 200, …]`, `[160, 200, …]`) are **no longer load-bearing for correctness** — `drop_last=False`
alone guarantees full coverage. They are kept because they still batch small graphs better than
`_SWC_BUCKET_PREFIX`'s fine low end does on a ~2,500-graph corpus.

`drop_last=True` came in with the original upstream SemlaFlow import (`d2193e9`), where it is
standard and harmless: on ~300k-molecule sets no bucket can starve. It was flipped here because
this repo trains only SWC corpora. If you ever train `qm9` / `geom-drugs` from this tree, note the
train loader now emits one short batch per bucket per epoch.

The startup line `items per bucket / bucket batch sizes / batches per bucket` is still the fastest
check: **every bucket with a non-zero item count must now show ≥1 batch.**

### Commands

```bash
# build the corpora (dendrite_gen side; reads the uncapped corpora read-only)
cd /Users/umer/Documents/dendrite_gen
python preprocessing/make_capped_tree_corpora.py --expect-kept 3169 --expect-dropped 199

# preprocess (one per depth; --max_atoms is a tripwire, nothing should be dropped)
cd /Users/umer/Documents/semla-flow
python -m semlaflow.preprocess_neurons \
    --input_dir  /Users/umer/Documents/trees_genus_d20_capped \
    --output_dir /Users/umer/Documents/trees_genus_d20_capped/smol \
    --val_dir_name val --test_dir_name test --max_atoms 1110

# train
python -m semlaflow.train \
    --dataset trees_genus_d20_capped \
    --data_path /Users/umer/Documents/trees_genus_d20_capped/smol \
    --max_atoms 1111 \
    --n_validation_mols 316 \
    --precision bf16-mixed --grad_checkpointing --batch_cost 16384 \
    --type_conditioning --class_hidden 16

# sample + evaluate
python -m semlaflow.sample_neurons \
    --ckpt_path <ckpt> \
    --data_path /Users/umer/Documents/trees_genus_d20_capped/smol \
    --save_dir  <out> \
    --dataset_split test --type_cond
```

For d15 swap the paths, `--max_atoms 667`, and use `--precision bf16-mixed --batch_cost 2048` with
**no** `--grad_checkpointing`. For d10 use `--max_atoms 269` and no precision/checkpointing flags.

All three capped corpora (and uncapped `trees_genus_d10`) have their `smol/` built with
`--compute_tmd --tmd_filtrations path radial_root`, so the same directory serves unconditional,
`--type_conditioning` and `--tmd_conditioning` runs — see [§1a](#1a-tmd-conditioning). Add
`--compute_tmd --tmd_filtrations path radial_root` to the preprocess command above if you ever
rebuild one.

If a run must be rebuilt, `make_capped_tree_corpora.py --force` moves the old tree aside rather than
mutating it, and refuses outright if a stale `smol/` sits beside it — a rebuilt SWC tree with an old
`.smol` next to it is the one failure mode where everything downstream still runs.

### Caveats specific to the capped corpora

- **The size distribution is truncated.** The dropped trees are the big mature canopy beeches:
  median height 22.9 m vs 16.1 m and median 1,464 nodes vs 320 at d20. Every extent and
  branch-length W1, and the node-count prior `sample_neurons.py` draws from, therefore describe a
  narrower corpus. **Score capped runs against capped splits only** — `w1_*` and `node_count_w1`
  are not comparable across capped/uncapped (§5 already documents the same trap for `--dataset_split`).
- **The real-vs-real floor in §5 does not transfer.** Both the per-side n and the size distribution
  changed, so the `trees_genus_d10` row is not a floor for `trees_genus_d10_capped`. Recompute per
  corpus before quoting any gap to it.
- **Sample loss.** The cap drops 199 of the 3,368 trainable trees (**5.91%**); counted from the
  3,386 graphs in the source archive, together with the 18 rare-genus/blank-split drops that
  produced the uncapped corpora, total loss is **6.41%** (3,169 kept). Full accounting in
  `dendrite_gen/docs/TREE_DATASET_STATS.md` §3.
- dendrite_gen's `pos_scale_factor` / `prior_std_pos` are **depth- and corpus-specific and were not
  re-measured** for the capped corpora. Re-run `tests/analyse_c0_distribution.py` before training
  dendrite_gen on them.

---

## 4. Memory: what to do when it OOMs

Measured on the real model at repo defaults (fp32, batch 1, fwd+bwd): **~57 KB of
retained activations per node-pair**. Peak memory ≈ `batch × N² × KB_per_pair`.

| | KB/pair | max N on 40 GB | d10 (384) | d15 (1536) | d20 (3072) |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 | 57.0 | 785 | 8.6 GB | 137.7 GB ❌ | 550.8 GB ❌ |
| `--precision bf16-mixed` | 28.5 | 1110 | 4.3 GB | 68.9 GB ❌ | 275.4 GB ❌ |
| **+ `--grad_checkpointing`** | **4.0** | **2970** | **0.6 GB** | **9.6 GB** | **38.5 GB** |

**Those three right-hand columns are single-graph figures** — the cost of one graph at the corpus
cap, which is what dominates for d15/d20 because their top buckets run at batch size 1. The true
peak is `max over buckets of (batch_size × limit² × KB/pair)`, and for a corpus whose cap is small
enough that the top bucket batches (d10, and all three capped corpora) it is a *mid* bucket that
peaks, not the top one. The capped corpora's peaks in [§3a](#3a-the-capped-tree-corpora) are
computed that way and are the ones to plan against.

**Lowering `--batch_cost` will not fix an OOM at the top bucket.** `_round_batch_size`
floors the batch at 1 (`data/util.py`), and the large buckets are already there — a
single N=3072 graph costs 551 GB in fp32 on its own. `--acc_batches` changes the
effective batch, not the micro-batch, so it does not help either. Gradient
checkpointing (recompute each layer in the backward pass, ~30% extra compute) is the
only lever below batch-size-1. It is numerically exact — verified bit-identical losses
and gradients.

In order, when d20 OOMs:

1. Confirm both `--precision bf16-mixed` **and** `--grad_checkpointing` are set. This
   is by far the most common cause.
2. Lower `--batch_cost` (`16384` → `8192` → `4096`). This only helps the *mid* buckets,
   but those are where most of the epoch's time goes, so it is often enough.
3. **Lower d20's cap to 2816.** 38.5 GB of a 40 GB card is almost no headroom, and
   exactly one train graph (N=3056) sits above the 2970-node ceiling. Edit
   `DATASET_CONFIGS["trees_genus_d20"]` in `semlaflow/scriptutil.py` — set
   `max_nodes=2816` and change the top `bucket_limits` entry from `3072` to `2816` —
   then re-run preprocessing for d20 with `--max_atoms 2816`. Costs 0.1% of the data
   and brings the peak to 32.4 GB.

To check before committing to a long run, read `torch.cuda.max_memory_allocated()`
after a few steps, and read the `items per bucket / bucket batch sizes / batches per
bucket` line the sampler prints at startup — if the large buckets show batch size 1 and
the small ones show hundreds, that is expected and correct.

---

## 5. Reading the results

`metrics.json` in `--save_dir`, plus `<save_file>_gen3d.png` / `_ref3d.png` grids.

**Read the health block first, then the morphometrics.** Generated graphs are sanitised
(largest connected component → minimum spanning tree → contract degree-2 nodes) before any
morphometric is computed, so the morphometrics answer *"given a valid tree, is the
morphology right?"* and cannot see structural failure. The health keys are the only answer
to *"is it a valid tree?"* — and every one of them is exactly 0.0 on ground truth for all
four corpora (1.0 for `lcc_node_frac`), so any non-zero value is a generator failure:

```
disconnected_frac  multifurcation_frac  isolated_node_frac  cycle_frac
non_critical_node_frac   excess_edge_frac   degree2_node_frac   lcc_node_frac
```

Measured: at a 1e-5 edge false-positive rate, `cycle_frac` reaches 0.053 while
`mmd_morpho` is *indistinguishable from clean*. Do not read a good `mmd_morpho` as a
healthy model without checking these.

### 5a. Reading the W&B curves

**Everything plots against `epoch`.** Runs set `define_metric("*", step_metric="epoch")`, so
new projects open on an epoch axis rather than wandb's "Step". That matters because "Step" is
not the optimizer step: Lightning never passes `step=` to `wandb.log`, so wandb's counter just
tallies log calls and its scale is an artifact of `log_every_n_steps` and `val_check_epochs`.
Two runs with different values for either do not line up on it.

**Prefer `train-loss-epoch` over `train-loss`.** Both are logged. `train-loss` is the original
per-step series — one batch, and with `bucket_cost_scale="quadratic"` a batch is anywhere from
1 to 312 graphs depending on which bucket the sampler drew, so most of that curve's visible
noise is bucket-identity noise rather than optimisation. `train-loss-epoch` is the
sample-weighted epoch mean (and the only DDP-correct one), with one point per epoch that lines
up 1:1 with the `val-*` series. Same for each component: `train-coord-loss-epoch`, etc.

**`lr` replaces `lr-Adam`.** `LearningRateMonitor` logged straight to the logger, bypassing
Lightning's connector, so its payload carried no `epoch` key and the series plotted empty on an
epoch axis. The LR now comes from the module as `lr` — an epoch mean, since the schedule steps
per batch. Old `lr-Adam` panels will not populate on new runs.

**Sample images: `val-plot-*`**, logged once per validation epoch. Graphs are raw, *not*
sanitised — the point is to see the fragments and cycles the morphometrics cannot — and
colour-coded by what sanitisation would do to them: **grey** = outside the largest component
(dashed edges), **orange** = an edge the MST cuts, **half-size** = a degree-2 node contraction
collapses, full row colour = the critical tree every metric actually scored. GT rows should be
entirely plain, since sanitisation is a no-op on ground truth. Same overlay on
`sample_neurons.py`'s `_gen3d.png` / `_ref3d.png`.

| key | when | rows |
| --- | --- | --- |
| `val-plot-class`, `val-plot-class-gt` | `--type_conditioning` | one per cell class, labelled by name |
| `val-plot-tmd_pairs` | `--tmd_conditioning` | `Gen #i` (red) above the `GT #i` (blue) that conditioned it |
| `val-plot-examples`, `val-plot-examples-gt` | neither | first N |

The modes are independent: a run with both conditioners gets the class grid *and* the pair
grid. `--no-val_plots` disables them; `--val_plot_max_rows` (default 8) bounds the cost.

**Use the normalised aggregates, not a raw mean of the `*_w1` keys.**
`w1_pooled_mean_normalized` and `w1_pertree_mean_normalized` divide each W1 by the GT
spread. A raw mean mixes microns, degrees and counts, and **which key dominates flips
between corpora**: on neurons it is ~90% the three extent keys, on the metre-scale tree
corpora the counts take over. That is why the old `val-structural-w1-mean` was retired.

**Score against the split the prior drew its node counts from.** `sample_neurons.py` takes
its sizes from `--dataset_split` of `--data_path`; pointing the evaluation at a different
corpus silently turns `node_count_w1` into a corpus-mismatch measurement (11.55 vs 1.27 on
the same real run) and shifts every extent metric.

**Compare against the real-vs-real floor, not against dendrite_gen.** Two disjoint halves
of the same real dataset differ by this much. The floor is **per corpus** — recompute it
for whatever you score against:

| corpus | n/side | `mmd_morpho` | `w1_pooled_mean_normalized` | `w1_pertree_mean_normalized` |
| --- | ---: | ---: | ---: | ---: |
| `neurons` | 923 | −0.00009 ± 0.00027 | 0.0162 ± 0.0044 | 0.0503 ± 0.0062 |
| `neurons_conditional` | 1263 | +0.00012 ± 0.00038 | 0.0133 ± 0.0036 | 0.0549 ± 0.0115 |
| `trees_genus_d10` | 168 | +0.00027 ± 0.00203 | 0.0585 ± 0.0137 | 0.1512 ± 0.0378 |
| `neurons_conditional_full` | 1264 | **not measured** | **not measured** | **not measured** |
| `trees_genus_d*_capped` | 158 | **not measured** | **not measured** | **not measured** |

The capped rows are deliberately blank: both the per-side n and the size distribution differ, so the
`trees_genus_d10` floor is **not** a floor for `trees_genus_d10_capped`. Recompute before quoting a
gap.

`mmd_morpho` here is **not** comparable to dendrite_gen's — different axial frame and
different sanitisation ([`COMPATIBLE.md` §4.14](COMPATIBLE.md)). `mmd_*` is an unbiased
estimator and is legitimately slightly negative on a match; do not clip it.

**Units.** Tree extent and branch-length W1 values are in **metres**; neuron ones are
in **microns**. Do not compare them across corpora without converting — this is exactly
what the normalised aggregates fix.

**Checkpoints.** Every neuron run writes `best-` (by `--ckpt_monitor`, default `val-loss`),
`morpho-` (by the gated `mmd_morpho`) and `snap-` (weights-only, every validation, all
kept). So **both selection criteria come out of one run** with no extra flags — compare
them at the end. `--save_top_k` (default 1) raises how many of the two full-checkpoint
families are kept; at ~437 MB each, mind the disk.

**Where they go, and why they are not uploaded to W&B.** The `ModelCheckpoint` callbacks set no
`dirpath` and the Trainer sets no `default_root_dir`, so Lightning derives the location from the
logger: **`wandb/equinv-<dataset>/<run-id>/checkpoints/`**, relative to the working directory the
run was launched from. That happens whether or not W&B is involved — the files are always local.

`WandbLogger(log_model=...)` controls only the *additional* upload of those files as W&B artifacts,
and it is now **off by default** (`--wandb_log_model` turns it on). It used to be `True`, which is
expensive here for a non-obvious reason: `after_save_checkpoint` in Lightning's
`loggers/wandb.py` uploads on every save when a callback has `save_top_k == -1` —

```python
if self._log_model == "all" or self._log_model is True and checkpoint_callback.save_top_k == -1:
    self._scan_and_log_checkpoints(checkpoint_callback)
```

— and the `snap-` callback is exactly that. So at the defaults (300 epochs, `--val_check_epochs 10`)
every one of the 30 weights-only snapshots went up during training, ~175 MB each, plus the two full
checkpoints at the end: **~6 GB per run**. Nothing is lost by leaving it off except being able to
pull a checkpoint from the W&B UI instead of the training filesystem; metrics, curves and sample
images are unaffected. Note there is no middle setting — turning it on re-enables the per-snapshot
upload too, because that is a property of the `snap-` callback rather than of the flag.

**Making morphology the primary criterion.** Add `--ckpt_monitor val-morpho-selection`
*plus at least one ceiling*, e.g. `--selection_max_disconnected_frac 0.05`. Without a
ceiling the gate never fires and you are selecting on raw `mmd_morpho`, which is blind to
structural failure by design. Single-GPU only — MMD is quadratic in the samples, so DDP's
per-rank averaging gives a different quantity, not an approximation.

**Per-class metrics are unreliable on the tree corpora.** Fagus is 88.2% of the data;
per split the counts are:

| | Fagus | Quercus | Acer | Carpinus | Fraxinus | Betula |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 2377 | 142 | 57 | 52 | 47 | 20 |
| val | 298 | 18 | 7 | 6 | 6 | 2 |
| test | 295 | 18 | 8 | 7 | 6 | 2 |

With `--per_cell_class_min_count 20` (the default) only Fagus clears the bar in val, so
`val-class_*` effectively reports one class. Lowering it surfaces the rare genera but
their W1 at n ≤ 8 is not statistically meaningful — **do not report those numbers.**
Either treat this as Fagus-vs-rest, or use the pooled metrics only.

**Two known caveats** carried over from `NEURONS.md`:
- `bifurcation_angle_w1` is computed from an *estimated* root, for ground-truth graphs
  as well as generated ones — deliberately, so the metric's zero point stays meaningful
  ([`COMPATIBLE.md` §4.11](COMPATIBLE.md#411-the-root-is-estimated-for-ground-truth-graphs-too-never-taken-from-the-data)).
  These trees have no high-degree hub to anchor on; the estimate is right 86–94% of the
  time and the metric-level cost is ≤0.7%. The other seven metrics are provably
  unaffected. Summary in [`NEURONS.md` §6d](NEURONS.md#6d-caveats-specific-to-these-corpora).
  The same root feeds the `axial_extent`/`radial_span` frame, so those two are noisier on
  trees (root accuracy 78.9%) than on neurons (99.6%, where the soma hub rule fires).
- Generated coordinate spread is systematically ~0.84× ground truth on neurons, which
  is why the extent W1 metrics climb over training while topology improves. See
  [`NEURON_COORD_UNDERDISPERSION.md`](NEURON_COORD_UNDERDISPERSION.md). Expect the same
  on trees.

**`--bond_loss_weight` probably needs retuning for d15/d20.** The bond head predicts N²
logits of which only N−1 are positive — a 2/N positive rate, so 2.8% at d10's median of
72 but 0.6% at d20's 338. The default `1.0` was tuned at N≈50.

---

## 6. Legacy: the unlabelled `neurons` corpus

Still registered, but has no class labels, so `--type_conditioning` is rejected with an
explicit error. Use `neurons_conditional_full` instead unless you are reproducing an old run.

```bash
python -m semlaflow.train \
    --dataset neurons \
    --data_path /Users/umer/Documents/neurons_final_smol
```

### Identifying which corpus an old `.smol` came from

Generated samples carry no dataset name, and the two neuron corpora are easy to confuse.
`sample_neurons.py` feeds **ground-truth node counts into the prior**, so the generated
node-count distribution is an exact fingerprint of the split it was sampled from:

| corpus | `.smol` | n | mean N | max N |
| --- | --- | ---: | ---: | ---: |
| `neurons` | `neurons_final_smol/val.smol` | 1847 | 49.3 | 149 |
| `neurons_conditional` | `neurons_conditional/smol/val.smol` | 2527 | 60.1 | 217 |
| `neurons_conditional_full` | `neurons_conditional_full/smol/val.smol` | 2529 | 60.3 | 495 |
| `neurons_conditional_full` | `neurons_conditional_full/smol/test.smol` | 1167 | 83.2 | 397 |

Match `len(samples)` and `max(seq_length)` against that table before scoring anything —
picking the wrong reference silently shifts every W1. The two conditional val splits differ by
only **2 graphs** in count and 0.2 in mean N, so **`max N` is the discriminator**: 217 means
`neurons_conditional`, 495 means `neurons_conditional_full` (whose `.smol` is uncapped).

---

## 7. Gotchas that cost time

- **Never `--optimal_transport scale`** on any SWC corpus. `SCALE_OT_FACTOR` is tuned
  for a molecular coord scale.
- **`predict.py` and `evaluate.py` do not work on these datasets** and will raise
  `Unknown dataset`. That is intentional — they decode RDKit molecules and compute
  chemistry metrics. Use `sample_neurons.py`.
- **A checkpoint with no `dataset` hparam** now fails loudly in `sample_neurons.py`
  rather than silently using the `neurons` coord scale. Pass `--dataset <name>` to fix.
  If you ever see plausible-but-wrong extents from an old checkpoint, this was why.
- **W&B creates one project per dataset** (`equinv-trees_genus_d10`, etc.). For
  cross-depth comparison in a single workspace, use a shared project and a `depth` tag.
  The chart x-axis is a **per-project UI setting**, so changing `--dataset` used to drop you
  into a fresh project whose charts defaulted back to "Step". Runs now call
  `define_metric("*", step_metric="epoch")` so every new project opens on epochs; older
  projects keep whatever axis you set by hand.
- **Local (macOS) runs**: `NEURO2` has no `wandb`, and the dataset holds a thread lock
  so `spawn` workers fail — set `dm._num_workers = 0`. A full-size d15/d20 graph is not
  runnable on the laptop; smoke-test on d10 or on graphs under ~200 nodes.
- **Radii are discarded.** The tree SWCs carry meaningful radii in column 6, but
  `swc_to_geometric_mol` reads only x/y/z. If you want them, that is new work. Capping changes
  neither this nor the metre units — a `*_capped` corpus is a strict subset, nothing else.
- **The `*_capped` corpora are derived, not rebuilt.** They come from
  `dendrite_gen/preprocessing/make_capped_tree_corpora.py`, which reads the uncapped corpora
  read-only; each carries a `BUILD_MANIFEST.json` (rule, counts, measured constants) and a
  `DROPPED_IDS.csv`. Never hand-edit their contents — re-derive.
- **Do not mix capped and uncapped paths in one command.** The only difference between the two
  `smol/` directories is one path component, and `sample_neurons.py` takes its node counts from
  `--data_path`/`--dataset_split`. Check `len(samples)` and `max(seq_length)` (≤ 268 / 666 / 1110)
  before trusting any metric — see §3a.
