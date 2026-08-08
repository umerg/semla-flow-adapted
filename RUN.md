# RUN.md — command reference for the four SWC datasets

Copy-paste commands for the four corpora that get trained, with the flags that differ
between them and *why*. Conceptual background lives in [`NEURONS.md`](NEURONS.md);
design rationale in [`COMPATIBLE.md`](COMPATIBLE.md). This file is just the runbook.

`neurons` (the original unlabelled corpus) is registered too but is superseded by
`neurons_conditional`; see [§6](#6-legacy-the-unlabelled-neurons-corpus).

---

## 0. The one-screen summary

| | `neurons_conditional` | `trees_genus_d10` | `trees_genus_d15` | `trees_genus_d20` |
| --- | --- | --- | --- | --- |
| what | mouse cortical neurons | botanical trees, depth ≤10 | trees, depth ≤15 | trees, depth ≤20 |
| classes | 7 cell types | 6 genera | 6 genera | 6 genera |
| units | µm | **m** | **m** | **m** |
| `coord_std` | 66.0298 | 1.6417 | 1.8062 | 1.9999 |
| train / val / test | 22750 / 2528 / — | 2695 / 337 / 336 | 2695 / 337 / 336 | 2695 / 337 / 336 |
| median / max N | 46 / 242 | 72 / 378 | 178 / 1456 | 338 / 3056 |
| **preprocess `--max_atoms`** | `256` (default) | `384` | `1536` | `3072` |
| **train `--max_atoms`** | `256` (default) | `385` | `1537` | `3073` |
| **`--n_validation_mols`** | `1800` | `320` | `320` | `320` |
| **`--precision`** | `32` | `32` | `bf16-mixed` | `bf16-mixed` |
| **`--grad_checkpointing`** | no | no | **yes** | **yes** |
| **`--batch_cost`** | `1024` | `1024` | `16384` | `16384` |
| peak GPU (est.) | ~23 GB | ~9 GB | ~10 GB | ~38 GB ⚠ |
| epoch cost vs d10 | — | 1× | ~8× | ~35× |

Everything else — `--epochs`, `--lr`, architecture, `--optimal_transport` — stays at the
repo defaults for all four. The four flags in bold are the whole story.

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
   | `trees_genus_d10` | 378 | 385 | 7 |
   | `trees_genus_d15` | 1456 | 1537 | 81 |
   | `trees_genus_d20` | 3056 | 3073 | 17 |

   The neuron corpora keep the default `256` — `max_atoms` is a saved hparam that sizes
   the embedding, so changing it would break loading existing checkpoints. The tree
   values are `cap + 1`, which is the smallest safe choice.

   Preprocess's `--max_atoms` is a *different* knob: it *drops* graphs above the cap.
   At the caps in the table nothing is dropped from any tree corpus.
2. **`--n_validation_mols` 1800 > the 337 tree val graphs.** It is clamped
   automatically with a printed notice, so the default will not crash — but passing
   `320` keeps the logs honest.
3. **`--precision` / `--grad_checkpointing`** — activation memory is O(N²) at ~57 KB
   per node-pair in fp32. d15 and d20 do not fit on a 40 GB card without both. See
   [§4](#4-memory-what-to-do-when-it-ooms).
4. **`--batch_cost`** is a memory budget, not a batch size: peak pairs ≈
   `256 × batch_cost`. Once checkpointing frees up headroom you can afford a much
   larger budget, which is where the throughput comes back.

---

## 1. `neurons_conditional` — 7 cortical cell types

The reference configuration. Nothing unusual; this is the baseline the tree runs are
compared against.

```bash
# preprocess (only if the corpus changed -- smol/ already exists)
python -m semlaflow.preprocess_neurons \
    --input_dir  /Users/umer/Documents/neurons_conditional \
    --output_dir /Users/umer/Documents/neurons_conditional/smol \
    --val_dir_name val

# train, class-conditioned
python -m semlaflow.train \
    --dataset neurons_conditional \
    --data_path /Users/umer/Documents/neurons_conditional/smol \
    --type_conditioning --class_hidden 16

# sample + evaluate
python -m semlaflow.sample_neurons \
    --ckpt_path <ckpt> \
    --data_path /Users/umer/Documents/neurons_conditional/smol \
    --save_dir  <out> \
    --type_cond
```

Notes:
- `--dataset_split` defaults to `val`. There is **no `test.smol`** for this corpus
  (`neurons_conditional/test/` exists but preprocessing was run before test-split
  support; re-run preprocessing if you want it).
- Expect `Class conditioning enabled: n_classes=7 (23P, 4P, ...)`.

---

## 2. `trees_genus_d10` — the cheap one, use it to iterate

Fits comfortably in fp32 with no special flags. This is the corpus to debug on.

```bash
python -m semlaflow.preprocess_neurons \
    --input_dir  /Users/umer/Documents/trees_genus_d10 \
    --output_dir /Users/umer/Documents/trees_genus_d10/smol \
    --val_dir_name val --max_atoms 384

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

`--val_dir_name val` is required for **all** tree corpora: auto-detect prefers
`val_extended/`, which they do not have.

Expect `Class conditioning enabled: n_classes=6 (Fagus, Quercus, Acer, Carpinus,
Fraxinus, Betula)` — if you see 7 cell-type names, the wrong dataset is selected.

---

## 3. `trees_genus_d15` and `trees_genus_d20` — need bf16 + checkpointing

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
run a full ODE rollout over the val set every `--val_check_epochs`. At d20 sizes that
is expensive; disabling it trades the metric trajectory for wall-clock.

---

## 4. Memory: what to do when it OOMs

Measured on the real model at repo defaults (fp32, batch 1, fwd+bwd): **~57 KB of
retained activations per node-pair**. Peak memory ≈ `batch × N² × KB_per_pair`.

| | KB/pair | max N on 40 GB | d10 (384) | d15 (1536) | d20 (3072) |
| --- | ---: | ---: | ---: | ---: | ---: |
| fp32 | 57.0 | 785 | 8.6 GB | 137.7 GB ❌ | 550.8 GB ❌ |
| `--precision bf16-mixed` | 28.5 | 1110 | 4.3 GB | 68.9 GB ❌ | 275.4 GB ❌ |
| **+ `--grad_checkpointing`** | **4.0** | **2970** | **0.6 GB** | **9.6 GB** | **38.5 GB** |

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
explicit error. Use `neurons_conditional` instead unless you are reproducing an old run.

```bash
python -m semlaflow.train \
    --dataset neurons \
    --data_path /Users/umer/Documents/neurons_final_smol
```

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
- **Local (macOS) runs**: `NEURO2` has no `wandb`, and the dataset holds a thread lock
  so `spawn` workers fail — set `dm._num_workers = 0`. A full-size d15/d20 graph is not
  runnable on the laptop; smoke-test on d10 or on graphs under ~200 nodes.
- **Radii are discarded.** The tree SWCs carry meaningful radii in column 6, but
  `swc_to_geometric_mol` reads only x/y/z. If you want them, that is new work.
