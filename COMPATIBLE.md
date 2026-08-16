# SemlaFlow → Neuron/Tree Adaptation: Compatibility Notes

This document explains exactly how SemlaFlow (an E(3)-equivariant flow-matching
model for 3D molecules) was adapted to generate neuron/botanical-tree morphologies
from SWC data. It covers every assumption, every mapping between molecular and
tree concepts, and the argument for why the model architecture is still valid
for this domain.

Audience: you (the owner of `dendrite_gen`) and anyone reviewing the baseline
before publishing. The goal is to make it possible to challenge every design
decision individually.

---

## 1. Problem Statement Recap

**Source domain (SemlaFlow original):** 3D molecular graphs with N atoms. Joint
flow-matching over:
- **Coordinates** `x ∈ R^{N×3}` (continuous, Gaussian prior → data)
- **Atom types** `a ∈ {1..V}^N` (discrete, categorical via DFM)
- **Bond types** `b ∈ {0..B}^{N×N}` (discrete, dense adjacency via DFM)
- **Formal charges** `c ∈ {-3..+3}^N` (discrete, categorical via DFM)

**Target domain (your use case):** 3D tree graphs representing neurons. Each
node has a 3D position. Edges are binary (edge / no-edge). No per-node
categorical information (all nodes are type=3 dendrite, all radii=1.0). Trees
are rooted, acyclic, and binary-branching below the root; root degree varies
(observed range 1–26, median 7).

**Goal:** run SemlaFlow unmodified at the architecture level, with the minimal
adapters needed to feed it the right tensor shapes. Trust the model to learn
"topology as a one-shot classification over all pairs" and "geometry as a
flow-matched point cloud." Its failure modes (cycles, disconnections, high-
degree violations, no tree invariant) are the point — they are the baseline
SemlaFlow-can't-do-X motivation for your K-Root-Children approach.

---

## 2. One-Line Summary of the Adaptation

> Build `GeometricMol` objects directly (bypassing RDKit) with a trivial
> 1-type vocabulary and a trivial 1-class edge set. Wire a new
> `--dataset neurons` branch through `train.py` that (a) swaps in a neuron
> vocab, (b) skips the atomic-number ↔ element-symbol remapping, (c) replaces
> the RDKit-driven validation metrics with loss-only validation, and
> (d) monitors `val-loss` instead of `val-validity` for checkpointing.
> **No architecture changes. No loss-function changes. No flow-matching
> changes.**

---

## 3. Concept-by-Concept Mapping

| Molecular concept           | Neuron concept                       | How it's wired                                                                                 |
|-----------------------------|--------------------------------------|------------------------------------------------------------------------------------------------|
| Atom coordinate             | Soma/dendrite node position          | `GeometricMol.coords[N, 3]` holds the SWC x,y,z directly.                                     |
| Atom type (e.g. C, N, O)    | (Nothing — single node type)         | `atomics[N]` filled with a constant index pointing at the single real token `"NODE"`.         |
| Bond type (single/double/…) | Edge exists                          | `bond_types[M]` filled with index 1 (reusing the "single bond" slot; see §6 for why).         |
| "No bond" between two atoms | No edge                              | Implicit in the dense adjacency: pairs not listed in `bond_indices` default to class 0.       |
| Formal charge               | (Nothing)                            | `charges[N]` filled with 0. The charge head exists but its loss weight is 0.                  |
| Atomic-number → vocab map   | Vocab index already stored           | `neuron_mol_transform` skips the `PT.symbol_from_atomic(...)` step of `mol_transform`.        |
| `CHARGE_IDX_MAP` remap      | (Not needed — all charges are 0)     | `neuron_mol_transform` skips the `CHARGE_IDX_MAP[charge]` lookup.                             |
| QM9/GEOM coord std (1.72 / 2.41) | measured per dataset (62.6894 for `neurons`) | Stored in `scriptutil.DATASET_CONFIGS[…].coord_std`. Re-run `preprocess_neurons.py` if data changes. |
| QM9/GEOM bucket limits      | per dataset (`[…, 200, 220]` for `neurons`) | `DATASET_CONFIGS[…].bucket_limits`, sized to that corpus (see §4.6).                    |
| Validity / novelty / energy metrics | W1 over structural statistics | `NeuronCFM` empties `gen_metrics` and `stability_metrics`; `validation/dist_metrics.py` supplies the replacements. |
| (No molecular analogue)     | Rooted-tree statistics               | Require a root; it is **estimated** for GT as well as generated graphs — see §4.11.           |
| RDKit-based `predict.py`    | (Broken as-is; out of scope)         | `sample_neurons.py` is the neuron/tree equivalent. See §9.                                    |

---

## 4. Assumptions (Explicit List)

Every assumption below is one you can override with a small code change; I've
noted where.

### 4.1 Single node type
**Assumption:** all nodes are functionally identical.
**Why:** your SWC corpus has `type=3` uniformly and `radius=1.0` uniformly.
No categorical signal exists per-node.
**Where encoded:** `semlaflow/data/swc.py:NEURON_NODE_TOKEN_INDEX=2` and
`scriptutil.build_neuron_vocab()` returning `["<PAD>", "<MASK>", "NODE"]`.
**Override path:** add tokens to `build_neuron_vocab()` and change the adapter
to assign per-node indices (e.g. "ROOT" vs "BRANCH" vs "LEAF") if that signal
becomes useful.

### 4.2 Binary edges
**Assumption:** every edge in a training neuron is the same "single edge"
class; the generation task is edge-exists vs edge-absent.
**Why:** your SWC has no edge attributes; the data is pure tree connectivity.
**Where encoded:** `swc.py:NEURON_EDGE_CLASS_INDEX=1`.
**Override path:** if you later want multi-class edges (e.g. axon/dendrite),
extend the adapter and possibly increase `n_bond_types` (see §6 for the
subtlety here).

### 4.3 Use the `mask` categorical strategy (recommended, not enforced)
**Assumption:** my training command suggests `--categorical_strategy mask`.
**Why:** it cleanly handles the "most pairs have no edge" class imbalance —
during training, most of the target adjacency is the background (absent)
class, and mask DFM only computes loss on *currently-masked* positions, so the
gradient signal is concentrated where the model is uncertain.
**Consequences:** `get_n_bond_types("mask") = 6` (4 molecular bond classes +
background + mask). For `uniform-sample` / `dirichlet` strategies,
`n_bond_types = 5` and you should verify my `BOND_MASK_INDEX=5` constant isn't
read (it is only referenced when strategy is `mask`).
**Override path:** CLI flag — nothing in the adapter forces `mask`.

### 4.4 Zero center-of-mass before training
**Assumption:** every tree is translated so that its CoM (arithmetic mean of
node coords) is at the origin, then rotated by a random 3D rotation, then
scaled by `1/coord_std`.
**Why:** this is the existing SemlaFlow `mol_transform` pipeline and it's what
makes E(3)-equivariance empirically stable (targets always live near origin
with unit std).
**Consequence:** the root is *not* at origin (unlike `dendrite_gen`'s
convention, which root-centers). This doesn't matter for SemlaFlow because
E(3)-equivariance means the model can generate rotated/translated versions of
the same graph equally well — it will internally learn to place the
root wherever the data says.
**Override path:** subtract root instead of CoM inside `neuron_mol_transform`
(one-liner), but there's no reason to: zero-CoM is model-friendly and you're
not conditioning on root position anyway.

> **Note (sections 4.5–4.7).** These three were originally written as global
> assumptions for the single `neurons` corpus. They are now **per-dataset**, declared
> in `scriptutil.DATASET_CONFIGS`; the values below are the `neurons` entry. See
> `NEURONS.md` §6 for the botanical-tree corpora, whose coords are in metres and whose
> graphs run to 3056 nodes.

### 4.5 Coordinate scaling factor (per dataset; 62.6894 for `neurons`)
**Assumption:** `DATASET_CONFIGS[<dataset>].coord_std`, measured as
`np.concatenate([coords - coords.mean(0, keepdims=True) for coords in train]).std()`.
Currently 62.6894 (`neurons`, µm), 66.0298 (`neurons_conditional`, µm),
1.6417 / 1.8062 / 1.9999 (`trees_genus_d10/d15/d20`, **metres**).
**Why:** matches what `mol_transform` does for molecules — scales coords so
post-transform std is roughly 1. Prior sampler is standard-Gaussian, so we
need data to be in the same scale.
**Consequence:** generated samples must be multiplied by that dataset's factor before
saving/rendering, and compared against data at the same scale. The
`coord_scale` attribute of `MolecularCFM` tracks this automatically.
**Override path:** re-run `semlaflow/preprocess_neurons.py` whenever the
training corpus changes and paste the new number into that dataset's registry entry.

### 4.6 Bucket limits (per dataset)
**Assumption:** each dataset's `bucket_limits` covers its size distribution. All SWC
corpora share the prefix `[24, 40, 56, 72, 96, 128, 160, 200]` and extend it: `neurons`
ends at 220, `neurons_conditional` at 256, and the tree corpora at 384 / 1536 / 3072.
**Why:** for `neurons`, p5=18, median=46, p75=65, p95=108, train-max=217. Drug buckets
end at 192 and would silently drop the largest graphs.
**Constraint:** the top bucket must cover the largest graph in **every** split, not
just train — `BucketBatchSampler` raises for the val loader too. Preprocessing prints
the per-split maxima for exactly this reason.
**Override path:** edit that dataset's `bucket_limits` in `scriptutil.py`. If you
shard/downsample the dataset, re-check the distribution.

### 4.7 Node cap (per dataset)
**Assumption:** graphs above the dataset's `max_nodes` are dropped at preprocess time
via `--max_atoms`. 256 for the neuron corpora (SemlaFlow's `DEFAULT_MAX_ATOMS`);
384 / 1536 / 3072 for the tree corpora, which retain 100% of their graphs.
**Consequence:** for `neurons`, 1 validation graph (N=258) was silently dropped.
**Caution:** `train.py --max_atoms` is a *different* knob and must be strictly greater
than the largest graph — `size_emb` is an `Embedding(max_atoms)` indexed by the raw
node count. This is now checked with an explicit error.
**Cost:** activation memory is O(N²) at ~57 KB per node-pair in fp32, so the cap is a
memory decision as much as a data one. Beyond ~800 nodes on a 40 GB card you need
`--precision bf16-mixed --grad_checkpointing`; see the table in `NEURONS.md` §6b.

### 4.8 Charges: keep the head, zero the loss
**Assumption:** the charge classifier head (fixed at 7 classes) stays in the
model; `charges = torch.zeros(N, dtype=long)` in every adapter output; the
training command sets `--charge_loss_weight 0.0`.
**Why:** the head is hardcoded at 7 in `SemlaGenerator` (`semla.py:773`,
`n_charges=7`). Refactoring the head size would require touching the model
code; zeroing the loss weight is a one-flag change and avoids the risk of
introducing a bug in the architecture. The head still runs each forward pass
(wasted compute: a tiny MLP per node), but the loss it produces is multiplied
by 0 and contributes no gradient.
**Override path:** if compute matters, guard the head creation in
`semla.py:SemlaGenerator.__init__` behind an `if self.n_charges > 0` and pass
`n_charges=0` for neurons. Optional cleanup; not required for correctness.

### 4.9 Validation skips generation-quality metrics
**Assumption:** during validation, we compute the same four losses used at
training (coord MSE + type CE + bond CE + charge CE) and log them; we do NOT
call `_generate_mols` / `_generate_stabilities` to decode RDKit molecules.
**Why:** those two methods try to build chemically valid molecules from the
predicted (coords, atom_types, bonds, charges) tuple via RDKit; they would
error on a 1-token vocab and binary edges. Also: the generative-quality
metrics they feed (`Validity`, `Uniqueness`, `Novelty`, `EnergyValidity`,
`MoleculeStability`, etc.) are chemistry-specific and have no analogue in
neuron evaluation.
**Consequence:** you don't get generation metrics during training. You WILL
need a separate evaluation pipeline (see §9) to sample and compute
tree-specific metrics (degree distribution, cycle presence, tree-depth,
end-to-end MMD against training data, etc.).
**Override path:** implement neuron metrics inside
`NeuronCFM.on_validation_epoch_end` if you want live metrics. Straightforward
but not done here.

### 4.10 Checkpoint monitors `val-loss` instead of `val-validity`
**Assumption:** `ModelCheckpoint(monitor="val-loss", mode="min")` when
`--dataset neurons`.
**Why:** `val-validity` is an RDKit-validity rate and doesn't exist in
`NeuronCFM`. The sum of the 4 per-step losses is a reasonable "is the model
fitting?" signal.
**Consequence:** the "best" checkpoint is the lowest-loss one, which is
usually the longest-trained EMA model. Fine for a baseline. If you want to
select based on a more task-relevant metric, add one in
`on_validation_epoch_end` and change the monitor.

### 4.11 The root is *estimated* for ground-truth graphs too, never taken from the data

This is the least obvious decision in the evaluation path, and the one most likely
to look like a bug to a reviewer. It is deliberate.

**Assumption:** both generated and ground-truth graphs get their root from the same
estimator, `choose_root` (`validation/convert.py:152`), even though the ground-truth
root is exactly known — `swc_to_geometric_mol` reorders the SWC root to index 0, so
passing `root=0` for GT would be free and exact. We don't. Neither
`neuron_cfm.py:134-135` nor `sample_neurons.py:286-287` passes `root=`.

**Why:** it protects the **zero point of the metric**. Generated graphs carry no root
marker — SemlaFlow emits an unordered point cloud plus an adjacency, so their root can
only ever be estimated. If GT used the true root while generated graphs used an
estimator, then `bifurcation_angle_w1` would measure *generation error + estimator
bias*, with no way to separate the two, and **a perfect generator would still score a
non-zero W1**. A metric whose floor is an unknown positive constant cannot be used to
judge convergence or to compare checkpoints.

Applying the same estimator to both sides keeps the floor at zero. The cost is
attenuation: a symmetric error blurs both distributions and mildly *understates* real
differences. That is the right direction to be wrong in — an under-sensitive metric
with a trustworthy zero beats a sensitive one with a floating floor.

**Where the estimator is weak, and what it costs.** `choose_root` first looks for a
unique strict-maximum-degree node of degree ≥ 3 — the soma. That fires for 100% of
neurons, so the neuron corpora are effectively using the true root anyway and none of
this matters there. The botanical-tree corpora are the opposite case: `root_deg == 1`
and no degree-2 nodes, so a binarized tree has hundreds of tied degree-3 nodes and the
hub branch *never* fires. Every tree falls through to `pca_base_root`
(`convert.py:143`), the extremal node along PC1.

That decomposes into three questions, measured over 400–600 trees per corpus:

1. Is PC1 the trunk axis? Yes — `|PC1 · trunk| ≈ 0.99` median.
2. Is the true root one of PC1's *two* ends? Yes, 96–98% of the time. This is the
   ceiling for any sign convention.
3. Which of the two ends? `np.linalg.eigh` returns eigenvectors of arbitrary sign, so
   originally this was a **coin flip** (46–52%). `_orient_axis` (`convert.py:90`) now
   picks the end by node density — a crown is dense, a trunk base is sparse — with a
   median-tail tie-break for tiny graphs.

| corpus | exact root, raw `eigh` sign | with `_orient_axis` | ceiling | `bifurcation_angle` cost |
|---|---:|---:|---:|---:|
| `trees_genus_d10` | 50.7% | 86.5% | 96.2% | 0.54° on a 76.5° mean |
| `trees_genus_d15` | 46.0% | 90.0% | 97.0% | 0.20° on a 75.8° mean |
| `trees_genus_d20` | 51.5% | 93.8% | 98.0% | 0.05° on a 75.9° mean |

The last column is the W1 between the bifurcation-angle distribution computed from the
true root and from the estimated root — the actual metric-level error, ≤0.7%.

**Consequence:** the residual error is smaller than the root-accuracy figure suggests,
for a structural reason worth knowing: rooting at the wrong end of a tree only flips
the parent pointer for nodes **on the path between the two candidate roots** (~`depth`
nodes out of `N`), not for the whole tree. Every other node's parent is still "the
neighbour towards the root" and is unchanged.

**Only one metric is exposed at all.** Measured over 600 trees by recomputing every
statistic from the true root and from the *opposite* end of PC1 — i.e. the worst case,
not the average one:

| metric | max change under a fully flipped root | why |
|---|---|---|
| `bifurcation_angle_w1` | **the affected metric** | the root picks which pair of a degree-3 node's three edges count as siblings |
| `branch_length_w1` | exactly 0 | root-free — just per-edge Euclidean lengths |
| `axial_extent_w1` | exactly 0 | calls `pca_axis` but consumes it **sign-invariantly** (`max - min`) |
| `radial_span_w1` | exactly 0 | sign-invariant (`pdist` of the perpendicular components) |
| `total_extent_w1` | exactly 0 | sign-invariant (`pdist`), rotation-invariant anyway |
| `leaf_count_w1` | 1 | only whether the root itself counts as a leaf, against counts of 40–1500 |
| `bifurcation_count_w1` | 0 | a degree-3 node has ≥2 children either way |
| `node_count_w1` | exactly 0 | root-free. It used to also be ≈0 *by construction*, since N is fed in via the prior mask — see §4.12, that is no longer true once sanitisation lands |

So the three extent metrics — the ones that actually move during training, per
`NEURON_COORD_UNDERDISPERSION.md` — are provably untouched by the root question, and
`_orient_axis` changed nothing for them.

**Override path:** if you ever get a root oracle for *generated* graphs — e.g. a model
that predicts the root, or the K-Root-Children approach where the root is explicit by
construction — then pass `root=` on both sides and the attenuation disappears. Do not
pass it on only one side. If you want to quantify the current attenuation for a paper,
compute the metrics twice for GT (true root vs estimated) and report the difference;
that is exactly the last column above.

---

### 4.12 Generated graphs are *sanitised* before any morphometric is computed

The second least obvious decision, and the one with the most surface area.

**Assumption:** `validation/sanitise.py:sanitise_graph` reduces every graph — generated
*and* ground-truth — to `largest connected component → relabel → minimum spanning tree →
contract non-root degree-2 nodes`, and the morphometrics score *that*. Structural health
is measured on the **raw** graph first, and only then is the graph repaired.

**Why it is necessary.** dendrite_gen's generator emits trees by construction. SemlaFlow's
bond head emits N² independent edge logits, so it emits cycles, multifurcations, degree-2
chain nodes and fragments. On such a graph the ported metrics are not merely noisy, they
are undefined or wrong:

- `branch_order_values` raises `KeyError` on any node unreachable from the root — 99/200
  graphs at only 1% edge dropout.
- `_root_tree` initialises `children` for *every* node but fills only the root's
  component, so every node in another fragment counts as a **leaf**. Isolating 5 nodes of
  a 130-node tree gave `leaf_count = 130`, `strahler = 1`, `partition_asymmetry = nan` —
  while `radial_to_root` still returned 129 values. The topology and geometry halves of
  the suite were scoring *different objects*.
- `compute_tmd_barcode_diagram` hard-asserts `nx.is_tree`, and both call sites swallow the
  exception, so `tmd_barlen_w1`/`mmd_tmd` would silently score only the graphs that came
  out perfect — survivorship bias that *flatters* a worse model.

**Why non-tree output is the normal case, not an edge case.** Keeping 95% of graphs
cycle-free needs an edge false-positive rate below 2.1e-5 at N=70, 1.0e-6 at N=320 and
1.1e-8 at N=3000. A 3-way softmax will not deliver 1e-8.

**Why MST rather than a BFS spanning tree.** A spurious edge joins a random pair and is
therefore long; a real branch segment is short. Measured over injected false positives the
MST drops 85–94% of them at 99.5–100% true-edge recall, where BFS-from-root keeps ~78% and
displaces real edges to do it.

**The cost, stated plainly: sanitisation launders the defect.** `branch_length_w1` at
fp 1e-3 falls from 0.1242 (raw) to 0.0026 (MST). This is a deliberate division of labour,
not an oversight:

> The morphometrics answer **"given a valid tree, is the morphology right?"**
> The health block is the **only** answer to **"is it a valid tree?"**

Measured on the shipped code, at low defect rates the morphometrics are *indistinguishable
from clean* and only the health keys move:

| defect | `disconnected` | `cycle` | `multifurcation` | `mmd_morpho` |
|---|---:|---:|---:|---:|
| clean | 0.000 | 0.000 | 0.000 | −0.0024 |
| fp 1e-5 | 0.000 | **0.053** | **0.012** | −0.0024 |
| fn 1% | **0.499** | 0.000 | 0.000 | −0.0019 |

This is why `--ckpt_monitor val-morpho-selection` gates `mmd_morpho` on the health
fractions. Selecting on `mmd_morpho` alone would let a model emitting garbage with a
plausible spanning tree win.

**Why applying it to GT too is safe, and required.** It is a verified no-op there —
**0/2527** neuron and **0/337** tree val graphs are altered — so it costs nothing, and
running both sides through the identical path makes a gen/GT asymmetry structurally
impossible. The same argument as §4.11.

**Three implementation traps, all of which bit during development:**

1. `pca_base_root` returns a **positional index** into the coords array, and
   `geometric_mol_to_nx` guarantees node id == coords row to make that valid. A plain
   `G.subgraph(component)` keeps the original labels, so the returned "root" addresses a
   different node or none at all. Every stage must relabel to contiguous `0..M-1`.
2. `choose_root` must be called **exactly once, before contraction**, and carried through.
   Calling it again afterwards runs it over a different node set, so the node protected
   during contraction can fail to become the final root — leaving a non-root degree-2 node
   alive and breaking the identity in §4.13. Choosing before is also the stable option:
   contracting a degree-2 node joins its two neighbours, so every surviving node keeps its
   degree and the hub rule cannot change its answer.
3. The GT fit (`build_gt_cache`) must **not** be built during Lightning's sanity-check
   loop, which runs 2 batches — the fit would come from ~16 graphs and poison every
   subsequent epoch.

**Why degree-2 contraction is in at all**, given it changes almost nothing (every metric
shifts by ≤0.006 GT-sd at the observed 0.37% rate, and seven of them are *structurally*
immune because they key on child counts or degree ≥ 3): it makes `node_count` exact
(§4.13); it removes a real inconsistency, since `compute_tmd_barcode_diagram` already runs
with `simplify_to_critical_tree=True` so the TMD keys were already seeing a contracted
tree while `branch_length_w1` was not; and the distortion is linear in the rate
(≈1.15 GT-sd of `branch_length` per unit rate), which matters for the far larger d15/d20
graphs.

### 4.13 `node_count` is a branch-point count, and `multifurcation` must exclude the root

Two metric definitions that look like generic graph statistics but are corpus-specific.

**`node_count` is topology, not size.** These are *critical trees*: every non-root node is
a bifurcation or a terminal. Verified exactly — `1 + leaves + bifurcations + deg2 == N`
holds for **100.00%** of graphs on all corpora, and GT `deg2` is **0.0000**, so on ground
truth `N = 1 + leaves + bifurcations` and `corr(node_count, bifurcation_count) = 0.9970`.
A node deficit *is* a branch-point deficit; there is no separate "size" reading to
confound it.

Consequently `node_count_w1` and `bifurcation_count_w1` both stay in the `standard` tier,
where dendrite_gen demotes them. They are near-duplicates of *each other* (r = 0.997) and
should be read as one signal, but they are redundant with nothing else.

**It stops being trivially zero, and that is the point.** In-loop `val-node_count_w1` was
0.0 at all 60 logged steps of the real run, because validation pairs the prior mask with
the GT batch. After sanitisation it measures *how many of the fed-in nodes the model turned
into real branch points*. Treat the change away from 0.0 as the metric starting to work,
not as a regression.

On the real run (`vivid-thunder-13` vs its paired GT) that measures **0.052 GT-sd**, i.e.
node budget is being spent on branch points almost exactly as often as in real data:
nodes 0.985× GT, leaves 1.002×, bifurcations 0.962×. It is currently one of the *weakest*
signals in the suite, and that is the correct reading — this model's problem is geometry,
not branching. It becomes informative precisely when a model starts wasting its node
budget on fragments.

> ⚠️ **Score against the *paired* GT split.** The prior draws its node counts from a
> specific split, so comparing to a different corpus silently turns `node_count_w1` into a
> corpus-mismatch measurement. `neuron_samples.smol` pairs with
> `neurons_final_smol/val.smol` (1847 graphs, max N = 149), **not** with
> `neurons_conditional` (2527 graphs, max N = 217) — against the wrong one the same run
> reads 11.55 instead of 1.27, and the bifurcation ratio reads 0.778 instead of 0.962.

(The key now means four different things across contexts: 0.0 in-loop pre-sanitisation,
the prior's size draw offline in `sample_neurons.py`, fragmentation post-LCC, and learned
stopping in dendrite_gen. `lcc_node_frac` is the unambiguous fragmentation reading.)

**`multifurcation_frac` excludes the root.** Including it reads **99.6%** on real neuron
ground truth, because the soma is a legitimate high-degree hub (per-graph max degree:
median 8, max 16). Excluding the root it is a perfect discriminator — **0.0000** on GT for
*both* corpora, **0.0623** on the real generations. The pooled non-root degree distribution
on real neurons is exactly `{1: 0.5655, 3: 0.4345}`, with zero mass at degree 0, 2, 4, 5
and 6.

This is **not** circular even though `choose_root` prefers the unique maximum-degree node
and so absorbs the offending hub 97.4% of the time: the *second-highest* node degree, which
consults no root at all, flags the identical set of graphs. That equivalence is pinned by
`tests/validation_metrics.py:test_multifurcation_matches_root_free_cross_check`. The
root-excluding form is the one shipped because the second-highest form would miss a lone
degree-4 node on the tree corpora, where the root has degree 1.

### 4.14 `mmd_morpho` is not numerically comparable to dendrite_gen's

Three independent reasons, any one of which is sufficient:

1. **Different axial frame.** dendrite_gen decomposes extents along a fixed anatomical
   `uhat`. SemlaFlow is fully E(3)-equivariant and the training transform applies a random
   rotation, so no fixed axis exists; `axial_extent`/`radial_span` use a per-graph
   root→centroid axis instead. Measured z under a perpendicular squash: fixed `uhat` 4.1,
   root→centroid 3.3, per-graph PCA 2.6 — so the frame demonstrably changes sensitivity.
2. **Different sanitisation** (§4.12).
3. **`MORPHO_VERSION`** already forbids sharing an axis across a `MORPHO_KEYS` change; this
   is the same class of break.

Compare against the **real-vs-real floor** computed on this repo's own code instead, which
is the only meaningful reference. The floor is per-corpus; measured over 6 disjoint splits:

| corpus | `mmd_morpho` floor | n/side |
| --- | ---: | ---: |
| `neurons` (`neurons_final_smol/val`) | −0.00009 ± 0.00027 | 923 |
| `neurons_conditional` | +0.00012 ± 0.00038 | 1263 |
| `trees_genus_d10` | +0.00027 ± 0.00203 | 168 |

Two related notes:

- **`gen_degenerate_frac` and `morpho_nan_frac` will read ~0.0 here**, because sanitisation
  removes their causes (0.0053/0.0271/0.1263 → 0.0000 at 1/5/20% edge dropout). They are
  kept for field-for-field parity with dendrite_gen; the §4.12 health block is the live
  disclosure for SemlaFlow's own failure modes.
- **On a TMD-conditioned run, `mmd_tmd` and `tmd_barlen_w1` are NOT independent evidence.**
  dendrite_gen's blind spot §8.3 — the evaluation filtration sitting inside the model's
  conditioning set — now applies here too. It did not when SemlaFlow conditioned on `path`
  alone, but the conditioning set is now `("path", "radial_root")` and evaluation still uses
  `radial_root`, so both metrics partly score the model reproducing its own input. See §4.15
  for why no fourth filtration is available to move evaluation onto.

  This is disclosed rather than fixed, and it is a real limitation: on a conditioned run,
  read those two as consistency checks, not as fidelity. The metrics that *are* independent
  evidence of conditioning fidelity are the matched-pair ones in
  `validation/tmd_conditional_eval.py`, logged as `val-tmd_cond-*` (in-loop) and written to
  `metrics.json` under `"tmd_cond"` (offline). They pair each generated graph with the
  specific GT graph whose descriptor produced it, so no amount of population-level mimicry
  can flatter them. On an unconditional run nothing changes: `mmd_tmd` remains independent
  and the `val-tmd_cond-*` block is not computed at all.

### 4.15 Only rotation-invariant filtrations can be conditioned on

`compute_tmd_mixed` implements four filtrations — `path`, `height`, `rho`, `radial_root` —
but `NEURON_TMD_SUPPORTED_FILTRATIONS` admits only `path` and `radial_root`, and
`preprocess_neurons.py --tmd_filtrations` will not accept the other two.

`height` (projection onto an axis) and `rho` (distance from that axis) are defined relative
to a fixed anatomical frame. dendrite_gen has one — it is SO(2)-equivariant about a
configured `so2_axis` (`[0., 1., 0.]` for neurons), so the descriptor and the geometry share
a frame and "reaches far along the apical axis" is a statement the model can act on.

SemlaFlow has no such frame. `scriptutil.neuron_mol_transform` applies a uniformly random 3D
rotation on every `__getitem__`, and `SemlaGenerator` is E(3)-equivariant. The TMD vector is
computed once at preprocess time in the SWC's own frame and is *not* recomputed per epoch, so
an axis-dependent channel would tell the model about an axis its input coordinates no longer
agree with. The axis half of the signal is not merely noisy — it is unresolvable, and the
model can only learn to ignore it. This is the same frame problem as §4.14's, applied to
conditioning instead of metrics.

`path` (geodesic distance from the soma) and `radial_root` (straight-line distance from the
soma) are invariant under rotation about the root, so they survive the augmentation intact —
verified to ~1e-7 in `tests/tmd_conditioning.py`. They are also complementary rather than
redundant: their ratio is what contraction/tortuosity measures.

Making `height`/`rho` usable would mean canonicalising the frame — replacing the random
rotation with a deterministic alignment — which changes the symmetry the model is trained
under and invalidates comparison with existing checkpoints. That is a separate decision, not
a filtration-set change.

---

## 5. What Survived Unchanged (and Why That's Safe)

The parts of SemlaFlow that were **not** modified, with the argument that they
remain correct for neuron data:

### 5.1 `SemlaGenerator` / `EquiInvDynamics` (`models/semla.py`)
These are the actual neural networks. They take:
- `coords [B, N, 3]` — 3D point cloud (data-agnostic)
- `features [B, N, d]` — invariant per-node features (time + atomics one-hot)
- `edge_feats [B, N, N, d_edge]` — pairwise edge features (bond one-hot)
- `atom_mask [B, N]` — which nodes are real vs padding

None of these have molecular semantics. The network assumes E(3) equivariance
over coords (translations and rotations of the input cause the same
transformation on the output) and permutation equivariance over nodes. Both
assumptions hold for trees as much as for molecules.

**Why I'm sure:** I traced the input shape through `build_model` → `SemlaGenerator` → `EquiInvDynamics` and verified the model only consumes
the four fields above, never references atomic-number-specific logic or
valency constraints. Searched `semla.py` for chemistry-specific identifiers
(valency, aromatic, atomic_number) and found none.

### 5.2 Flow-matching interpolant & prior sampler (`data/interpolate.py`)
The interpolant builds the noisy `interpolated` sample at time `t ∈ [0, 1]`
by mixing data with prior:
- Coords: linear interpolation with Gaussian noise (data-agnostic).
- Atomics: categorical DFM. The number of categories is our `vocab.size=3`.
- Bonds: categorical DFM over `n_bond_types=6`.
- Charges: categorical DFM over 7 classes (unused via loss weight).

**Why I'm sure:** the interpolant receives `vocab.size` and `n_bond_types` as
arguments in `GeometricInterpolantDM`. I verified that `build_dm` passes our
neuron values through (`vocab.size=3`, `n_bond_types=6`). The interpolant
itself doesn't care what the categories mean.

### 5.3 Losses (`models/fm.py:_loss`)
- Coord loss: MSE on masked positions (data-agnostic).
- Type loss: CE over vocab.size classes (data-agnostic).
- Bond loss: CE over n_bond_types classes on a mask-weighted dense adjacency
  (data-agnostic).
- Charge loss: CE over 7 classes (unused; weighted 0).

All four losses are tensor operations with no domain semantics. Multiplying
`type_loss_weight=0` zeroes the gradient contribution cleanly.

### 5.4 Optimal transport (`--optimal_transport equivariant`)
OT aligns prior samples to data samples via Hungarian matching with
equivariant distance — this is just an energy minimization over a cost matrix
of pairwise L2 distances on coords. Data-agnostic.

### 5.5 EMA + self-conditioning
Both are meta-training tricks that operate on the model's parameters and
outputs respectively. Neither references the data domain.

### 5.6 `GeometricDataset` and `GeometricMolBatch`
Pure-container classes. They pad coords, atomics, bonds, charges, and masks
into batched tensors. They do not interpret any of the contents chemically.

---

## 6. The One Subtle Compatibility Decision: `n_bond_types = 6`

This deserves its own section because it's the one place where the adaptation
is "not quite clean" and could bite someone editing the code later.

### What happened
`scriptutil.get_n_bond_types("mask")` returns
`len(BOND_IDX_MAP) + 1 + 1 = 4 + 1 + 1 = 6`. The class labels are:
- 0: no-bond (background, implicit)
- 1: single bond
- 2: double bond
- 3: triple bond
- 4: aromatic bond
- 5: `<BOND_MASK>` (absorbing token for the mask DFM strategy)

For neurons, we only use class 0 ("no edge") and class 1 ("edge"), leaving
classes 2, 3, 4 unused during training. The model's edge classifier head
has 6 logits; three of them will simply learn to always emit low scores since
they never appear as targets.

### Why I didn't fix this
**Option A (taken):** leave `n_bond_types=6`, waste 3 logits.
- Pro: zero code changes to `get_n_bond_types`, `BOND_MASK_INDEX`, model.
- Pro: if you later add multi-class edges, the slots are already there.
- Con: slightly wasteful compute (3 extra logits per N² pair).

**Option B (rejected):** thread `n_bond_types=2` through as a CLI override and
also move `BOND_MASK_INDEX` (currently the fixed value 5) to `n_bond_types - 1`.
- Pro: clean, no wasted capacity.
- Con: touches more files; `BOND_MASK_INDEX` is imported by name in
  `train.py`, `fm.py`'s `Integrator`, and the interpolant. Introduces risk.

For a baseline where getting-it-running is the priority, Option A is the
right tradeoff. Flagging here for anyone who wants to squeeze efficiency.

### Why this is still correct
The model sees the ground-truth adjacency with class 0 (no-edge) and class 1
(edge). Cross-entropy with `n_bond_types=6` logits is well-defined: the
softmax normalizes over all 6 dimensions, and the CE target is always 0 or 1.
Gradients for dimensions 2/3/4 just push those logits downward uniformly
(they'll converge to very negative values). Nothing breaks.

---

## 7. File-by-File Change Summary

### Added
| Path                                           | Role                                                         |
|------------------------------------------------|--------------------------------------------------------------|
| `semlaflow/data/swc.py`                        | Parse SWC (incl. `# cell_class`) → build `GeometricMol` directly (no RDKit). |
| `semlaflow/preprocess_neurons.py`              | Walk `<input_dir>/{train, val[_extended], test}`, emit `.smol`. |
| `semlaflow/models/neuron_cfm.py`               | `NeuronCFM(MolecularCFM)` — strips RDKit metrics, adds structural validation. |
| `semlaflow/sample_neurons.py`                  | Offline sampling + evaluation (the neuron/tree `predict.py`). |
| `semlaflow/validation/`                        | W1 structural distribution metrics, `GeometricMol` ↔ networkx, root selection (§4.11), plots. |
| `semlaflow/tmd/`                               | TMD persistence-image descriptor for optional conditioning.  |
| `COMPATIBLE.md`                                | This file — design decisions, open to challenge.             |
| `NEURONS.md`                                   | Conceptual guide + per-dataset details.                      |
| `RUN.md`                                       | Runbook: exact commands and per-dataset flags.               |

### Modified
| Path                                           | Change                                                                                    |
|------------------------------------------------|-------------------------------------------------------------------------------------------|
| `semlaflow/scriptutil.py`                      | Added `build_neuron_vocab`, `neuron_mol_transform`, and `DATASET_CONFIGS` — the per-dataset registry of coord scale / bucket limits / node cap / class names (the `NEURON_*` constants are now derived aliases). |
| `semlaflow/train.py`                           | `--dataset neurons` branch in `build_dm`, `build_model`, `main`, `build_trainer`. Swap to `NeuronCFM`. Monitor `val-loss` for checkpointing. Skip `train_smiles` for neurons. |
| `semlaflow/data/datamodules.py`                | macOS compat: fall back from `os.sched_getaffinity` to `os.cpu_count()`.                  |

### Untouched (for good reason)
- `semlaflow/models/semla.py` — architecture; data-agnostic.
- `semlaflow/models/fm.py` — flow-matching + losses; data-agnostic.
- `semlaflow/data/interpolate.py` — interpolant; data-agnostic.
- `semlaflow/data/datasets.py` — `.smol` loader; data-agnostic.
- `semlaflow/util/molrepr.py` — `GeometricMol`/`GeometricMolBatch`; data-agnostic.
- `semlaflow/util/functional.py` — padding, CoM, rotations; data-agnostic.
- `semlaflow/util/rdkit.py` — untouched (imported but RDKit code paths
  are never hit on the neuron code path; it's still imported by modules like
  `molrepr.py`, which is why you still need `rdkit` in the env).

---

## 8. End-to-End Data Flow (For Audit)

Tracing a single batch from SWC files all the way to a training loss, so you
can spot-check every step:

```
neurons_final/train/*.swc                                 (16,639 files)
        │
        ▼  parse_swc (line-by-line text parser)
(coords [N,3] tuples, edges [(parent, child), ...], root_id)
        │
        ▼  swc_to_geometric_mol
GeometricMol:
  coords       = torch.tensor [N, 3], float32
  atomics      = torch.full((N,), 2, long)     ← NODE token index
  bond_indices = torch.tensor [N-1, 2], long   ← parent→child pairs
  bond_types   = torch.ones (N-1,), long       ← class "1" = edge
  charges      = torch.zeros (N,), long
        │
        ▼  GeometricMolBatch.to_bytes → pickle → .smol file
train.smol, val.smol
        │
        ▼  GeometricDataset.load(..., transform=neuron_mol_transform)
(each __getitem__:)
  rotate(random 3D rotation)
  scale(1/62.6894)
  zero_com()
  atomics: [N] → [N, 3] one-hot
  bond_types: [N-1] → [N-1, 6] one-hot
        │
        ▼  GeometricInterpolantDM._batch_to_dict (collate)
  pad to max(N_in_batch)
  expand bonds to dense [N, N, 6] adjacency via adj_from_edges(symmetric=True)
  charges: [N] → [N, 7] one-hot
(now have dict: coords[B,N,3], atomics[B,N,3], bonds[B,N,N,6],
                charges[B,N,7], mask[B,N])
        │
        ▼  GeometricInterpolant.forward
Sample t ∈ [0,1]; build interpolated = lerp(prior, data, t).
Prior = (Gaussian coords, all-mask atomics/bonds/charges).
Output: (prior_dict, data_dict, interpolated_dict, times)
        │
        ▼  MolecularCFM (aka NeuronCFM).training_step
forward(interpolated, times) → (coords_pred, atomics_logits, bonds_logits, charges_logits)
losses = _loss(data, interpolated, predicted):
  coord_loss  = MSE(pred_coords, data_coords) * mask                  [weight=1]
  type_loss   = CE(atomics_logits, argmax(data_atomics))              [weight=0]  ← zeroed
  bond_loss   = CE(bonds_logits, argmax(data_bonds))                  [weight=1]
  charge_loss = CE(charges_logits, argmax(data_charges))              [weight=0]  ← zeroed
loss = sum(weighted losses) → backward → step
```

Verified in the smoke test (see §10).

---

## 9. Known Limitations / What You'll Still Need to Build

These are explicitly **not** adapted; the baseline is training-only.

1. **Sampling/inference (`semlaflow/predict.py`)** still calls
   `_generate_mols` → RDKit. To sample neurons, write a new script that:
   a. Calls `model._generate(prior_batch, steps)` to get the generated tensors.
   b. For each item, extracts `argmax(bonds)` — class 1 = edge, else no edge.
   c. Builds an adjacency matrix, computes connected components, and writes an
      SWC file (or whatever you want).
   d. Handles the "dropped no-edge pairs leave a non-tree graph" case
      gracefully (spanning tree? largest connected component? report cycles?).

2. **Evaluation (`semlaflow/evaluate.py`)** is chemistry-only. For neurons
   you'll want MMD over degree distributions, TMD distance if you have TMD
   embeddings, graph-edit-distance to nearest training neuron, etc. None of
   these are in SemlaFlow; they live in `dendrite_gen` tooling.

3. **Tree-structure invariants are not enforced.** The baseline will produce
   disconnected components, cycles, high-degree nodes, etc. By design. That's
   the baseline's weakness and your K-Root-Children approach's justification.

4. **Node count N is an input, not generated.** At inference time you must
   decide N (e.g. sample from the training-set size distribution). SemlaFlow
   does not predict N.

5. **No conditioning on root degree k.** Your K-Root approach conditions on
   `num_root_children`; this baseline has no such hook. All generations will
   have a freely-varying root degree.

6. **E(3) equivariance includes reflections.** Your data presumably has an
   anatomical "up" (the SO(2)-axis in your config); this baseline doesn't
   respect that. It'll happily produce upside-down neurons. Acceptable for a
   baseline; noted.

---

## 10. Verification Performed

Before declaring this done, I verified on the actual corpus:

- **SWC parsing correctness:** one file → 50 nodes, 49 edges, root at id=1,
  all non-root degrees ≤ 3, file IDs contiguous 1..N. All match SWC spec.
- **Round-trip:** `GeometricMol.to_bytes → from_bytes` preserves `coords`
  (`torch.allclose` passes) and `bond_indices` (exact equality).
- **Dense adjacency shape:** for N=50, adjacency is [50, 50] with 98 nonzeros
  (2×49, symmetric, one per edge direction).
- **Preprocess end-to-end:** 16,639 train / 1,847 val graphs serialized to
  `.smol`. 1 val graph > 256 nodes dropped. Measured `coord_std = 62.6894`.
  Round-trip check on first mol passed.
- **Transform output:** for dataset items 0, 100, 1000: post-transform CoM is
  `~[1e-7, 1e-7, 1e-7]` (numerically zero), std is 1.03–1.41 (close to 1 as
  expected — individual trees vary around the global mean).
- **Batch shapes:** one training batch (size 16, max 72 nodes in bucket):
  `coords[16, 72, 3]`, `atomics[16, 72, 3]`, `bonds[16, 72, 72, 6]`,
  `charges[16, 72, 7]`, `mask[16, 72]`.
- **Atomics class check:** real-node positions all have `argmax == 2` (NODE);
  padded positions all have `argmax == 0` (PAD). Correct.
- **Bond class check:** dense adjacency contains only classes {0, 1} as
  expected (no stray 2/3/4/5 showing up where they shouldn't).
- **Bucketing:** 9 buckets with counts [1858, 5054, 4679, 2338, 1631, 894, 165, 17, 3], summing to 16,639 (full train set, nothing lost).

What I could **not** verify locally:
- Full training loop (macOS doesn't support the Linux-native multiprocessing
  `spawn` start method for dataloader workers with `threading.Lock` objects
  in the dataset — a pre-existing SemlaFlow issue unrelated to the adapter).
  On Linux + GPU this won't occur.
- Actual generation quality (requires training to convergence + a sampling
  script; see §9).

---

## 11. Suggested First Training Command

```bash
python -m semlaflow.train \
  --data_path /Users/umer/Documents/neurons_final/smol \
  --dataset neurons \
  --max_atoms 220 \
  --type_loss_weight 0.0 \
  --charge_loss_weight 0.0 \
  --bond_loss_weight 1.0 \
  --categorical_strategy mask \
  --epochs 300 \
  --lr 3e-4 \
  --batch_cost 4096 \
  --optimal_transport equivariant
```

Rationale:
- `--type_loss_weight 0.0`: single node type, no signal.
- `--charge_loss_weight 0.0`: all-zero charges, no signal.
- `--bond_loss_weight 1.0`: topology is the learning target.
- `--categorical_strategy mask`: class-imbalanced "mostly no-edge" is handled
  cleanly (see §4.3).
- `--max_atoms 220`: covers all observed training graphs.

Start with a trial run first:

```bash
python -m semlaflow.train --trial_run \
  --data_path /Users/umer/Documents/neurons_final/smol \
  --dataset neurons \
  --max_atoms 220 \
  --type_loss_weight 0.0 --charge_loss_weight 0.0 --bond_loss_weight 1.0 \
  --categorical_strategy mask
```

to smoke-test the full loop (one epoch, no wandb) before committing compute.

---

## 12. TL;DR Compatibility Argument

SemlaFlow's architecture treats the domain as: *3D points + invariant per-node
categoricals + invariant per-edge categoricals + E(3) equivariance*. Neurons
are exactly that structure with impoverished categoricals (one class each).
The adaptation therefore reduces to: (1) data format conversion, (2) dummy
vocab, (3) zero-weight the losses that have no signal, (4) replace the
RDKit-flavored validation metrics with loss-based validation. The flow-
matching math, the equivariance guarantees, and the optimization pipeline are
unchanged — which is why I'm confident this will train and produce
*something*. Whether that "something" is a useful baseline depends on how
badly it violates the implicit tree invariants (degree cap, connectivity,
acyclicity) — and that's precisely the experiment you want to run.

---

## Appendix A: Mechanics Deep-Dive

This appendix unpacks three mechanisms often conflated: the `mask` categorical
strategy, the roles of `<PAD>` vs `<MASK>` in the vocab, and the sparse→dense
handling of bond indices. Every claim below is tied to a specific file and
line in the SemlaFlow source.

### A.1 The `mask` categorical strategy, end-to-end

This is **Discrete Flow Matching (DFM)** with an absorbing "mask" state. Three
moving parts: the prior, the interpolation, and the loss.

#### A.1.1 The prior (`data/interpolate.py:100-102`)
```python
elif self.type_noise == "mask":
    atomics = torch.zeros((n_atoms, self.vocab_size), dtype=torch.float32)
    atomics[:, self.type_mask_index] = 1.0
```
At `t=0` (pure noise), **every node is one-hot at the `<MASK>` index**. For
bonds, same thing — every pair in the dense adjacency is set to the mask
class (index 5 for bonds when `categorical_strategy="mask"`). Coords are
standard Gaussian in parallel; the discrete and continuous priors are
independent.

#### A.1.2 The interpolation at time t (`data/interpolate.py:303-308`, 316-321)
```python
elif self.type_interpolation == "unmask":
    atom_mask = torch.rand(from_mol.seq_length) > t
    to_atomics = torch.argmax(to_mol.atomics, dim=-1)
    from_atomics = torch.argmax(from_mol.atomics, dim=-1)
    to_atomics[atom_mask] = from_atomics[atom_mask]
    atomics = one_hot_encode_tensor(to_atomics, ...)
```
For each node (or pair, independently), flip a biased coin:
- With probability `t`, keep the **real** token (from data).
- With probability `(1-t)`, revert to the **mask** token (from prior).

At `t=0` everything is mask. At `t=1` everything is revealed. The forward
process is literally just random per-token unmasking. Each node is unmasked
independently of all others — no structure in the noising.

#### A.1.3 The loss on masked positions only (`models/fm.py:818-823`, 840-842)
```python
if self.type_strategy == "mask":
    masked_types = torch.argmax(interpolated["atomics"], dim=-1) == self.type_mask_index
    n_atoms = masked_types.sum(dim=-1) + eps
    type_loss = type_loss * masked_types.float().unsqueeze(-1)
```
Cross-entropy is **only computed on positions that are currently masked** in
the interpolated view. Positions already revealed at time `t` contribute zero
gradient — the model isn't asked to "predict" them again. Normalization is by
the number of masked positions (`n_atoms`), not total positions.

The analogous bond loss (`fm.py:840-842`) does the same thing at the
`(i, j)` pair level.

#### A.1.4 Why this matters specifically for neurons
The bond tensor is *dominated* by the no-edge class 0. For a tree with
N nodes, 2(N-1) positions out of N² are edges — ~4% for N=50, ~1% for N=220.

With the alternative `uniform-sample` strategy (which computes CE on every
pair unconditionally), the model gets an easy-win signal "predict 0
everywhere" and the gradient on the rare class 1 gets averaged down by the
abundant class 0. With `mask`, the model is explicitly asked "given these
pairs are masked, predict their class" — so class imbalance doesn't
marginalize the rare-but-important edge signal.

Alternatives are still valid; `mask` is just the most sample-efficient
default for class-imbalanced structure, which is why the suggested training
command uses it.

#### A.1.5 What sampling looks like (for completeness, not used during training)
The `Integrator` (`fm.py`) reverses the process: start with all-mask, predict
logits per position, sample from the softmax, unmask a fraction proportional
to the step size, repeat. At `t=1` everything is decided. The sampling side
isn't wired for neurons yet (see §9) but the math is identical.

### A.2 `<PAD>` vs `<MASK>` — two different jobs

Easy to conflate; they do different things and live in different scopes.

#### A.2.1 `<PAD>` (vocab index 0): a **batching** concern
Different neurons in one batch have different N (e.g. [40, 37, 45]). The
datamodule pads them up to the max in the bucket — say 72. Padded positions
need *some* value in every tensor; `<PAD>` is that filler token.

- `data["atomics"]` has shape `[B, 72, 3]`. For a graph with only 45 real
  nodes, positions 45..71 get atomics `[1, 0, 0]` (one-hot at `<PAD>` = 0).
- The `mask` tensor `[B, 72]` has 1 for real positions and 0 for padded.
- **Losses are multiplied by this mask** (see `_type_loss`, `_bond_loss`,
  `_charge_loss` in `fm.py`). Padded positions never contribute gradient.

The model *sees* `<PAD>` nodes in the forward pass (they participate in
attention and message passing) but because their gradient contribution is
zeroed, it learns to treat them as inert. Verified in the smoke test:
```
real-node atomics all 2: True   # argmax == NODE token
padded atomics all 0: True      # argmax == PAD token
```

#### A.2.2 `<MASK>` (vocab index 1): a **DFM** concern
This is the absorbing state in discrete flow matching (see A.1). It
represents "this token hasn't been decided yet; the model must predict it."
It only appears in the *interpolated* view (the noisy intermediate between
prior and data), never in the data itself. It's a flow-matching concept, not
a batching concept.

Orthogonality to PAD: a padded position never flips to MASK (it has no real
token for DFM to try to reveal); a real position can be MASK at time t and
NODE at time t+dt. The two tokens live in the same vocabulary slot count
because they're both just possible one-hot values the model may receive as
input, but they're produced by different pipelines.

#### A.2.3 How the one-hot feeds into the model (`models/fm.py:520-521`)
The transform produces atomics as `[N, vocab_size]` one-hot. Collation stacks
these to `[B, N, vocab_size]`. Then in `MolecularCFM.forward`:
```python
times = t.view(-1, 1, 1).expand(-1, coords.size(1), -1)  # [B, N, 1]
features = torch.cat((times, atom_types), dim=2)         # [B, N, V+1]
```
For neurons, `V=3`, so the per-node invariant feature vector has
**4 dimensions**: `[t, one_hot_PAD, one_hot_MASK, one_hot_NODE]`. These
features are passed into `SemlaGenerator` as the scalar (rotation-invariant)
stream that flows alongside the equivariant coordinate stream.

| Node state                     | Feature vector          |
|--------------------------------|-------------------------|
| Real node (always, for neurons) | `[t, 0, 0, 1]`          |
| Padded node                    | `[t, 1, 0, 0]`          |
| Node currently masked in DFM   | `[t, 0, 1, 0]`          |

Because `type_loss_weight=0.0` on neurons, the model's **predicted** atom-
type logits are discarded (not used for gradient) — but the atom-type **input
features** still exist and still carry the PAD/MASK/NODE distinction the
model needs to function. You can't just set the feature dim to 0; you need
at least those three tokens even though the vocab is degenerate.

(An analogous argument applies to edges: `<BOND_MASK>` at index 5 plays the
same role for bonds that `<MASK>` plays for atoms. The implicit "no-edge"
class 0 is just absence — see A.3.)

### A.3 Bond indices: sparse storage, dense densification, implicit class 0

At storage time, bond_indices are a **sparse** edge list. At train time,
they're implicitly expanded to a **dense** N×N tensor. "No-edge" is class 0
by construction.

#### A.3.1 Storage (GeometricMol, my adapter)
```python
bond_indices = [[parent_0, child_0], [parent_1, child_1], ...]   # [N-1, 2]
bond_types   = [1, 1, 1, ..., 1]                                  # [N-1]
```
**Only edges that exist** are recorded. No-edge is never stored explicitly —
it's defined by absence. Serialised into `.smol` as-is (see `to_bytes`).

#### A.3.2 Densification at collate time (`data/datamodules.py:196` → `util/molrepr.py:644-648`)
```python
adjs = [
    adj_from_edges(mol.bond_indices, mol.bond_types, n_atoms, symmetric=True)
    for mol in self._mols
]
```
`adj_from_edges` initializes an N×N×n_bond_types tensor of zeros, then writes
`bond_types[i]` at `adj[from, to]` and (via `symmetric=True`) also at
`adj[to, from]`. Every other position stays zero.

After one-hot encoding (done in `neuron_mol_transform`), `bond_types` is
`[N-1, 6]` one-hot-at-class-1. So the dense adjacency becomes:

| Position                    | Value                           | Count        |
|-----------------------------|---------------------------------|--------------|
| Real edge (either direction) | `[0, 1, 0, 0, 0, 0]`            | 2(N-1)       |
| Diagonal `(i, i)`           | `[0, 0, 0, 0, 0, 0]`            | N            |
| All other pairs             | `[0, 0, 0, 0, 0, 0]`            | N² − 2(N-1) − N |

#### A.3.3 How the loss target falls out (`models/fm.py:829`)
```python
bonds = torch.argmax(data["bonds"], dim=-1)   # [B, N, N]
```
- Real-edge positions: `argmax([0,1,0,0,0,0]) = 1` → "edge" class
- Absent-pair positions: `argmax([0,0,0,0,0,0]) = 0` → "no-edge" class
  (argmax breaks ties by taking the first index, which is 0)

So **bond_indices are kept purely to flag which pairs are edges; everything
else falls into class 0 by construction.** The model's output head produces
`[B, N, N, 6]` logits and the CE loss matches them against this implicit
"absence = 0" target.

#### A.3.4 Two wrinkles worth knowing

**(a) Self-loops.** The diagonal `adj[i, i]` is never written (a node doesn't
have a self-edge in my adapter), so it stays all zeros → argmax 0 → "no-edge"
target. Correct; a node is not its own parent. Note that in `_bond_loss` the
loss is masked by `adj_from_node_mask(..., self_connect=True)`
(`fm.py:835`), which *does* include the diagonal — so the model is trained
to predict "no-edge" on the diagonal too. Harmless; this just makes "no self-
loops" a learned constraint rather than a structural one.

**(b) Interpolated vs data representation.** In the **data** mol,
`bond_indices` is sparse (the `N-1` real edges). But inside
`GeometricInterpolant._interpolate_mol` (`interpolate.py:323-324`):
```python
bond_indices = torch.ones((N, N)).nonzero()
bond_types = interp_adj[bond_indices[:, 0], bond_indices[:, 1]]
```
The **interpolated** mol stores a dense N² bond list — after mask-noise is
applied, most pairs have non-trivial "bond types" (either mask class 5 or a
revealed data value) and you can't encode that sparsely. So the sparse-vs-
dense distinction is a quirk of the data path; at the model input,
everything is dense `[B, N, N, 6]` either way.

#### A.3.5 Why this is efficient
For a tree with N=100, storage cost:
- Sparse: `N-1 = 99` edges × 2 cols = 198 entries.
- Dense one-hot: `N² × n_bond_types = 100 × 100 × 6 = 60,000` entries.

A ~300× reduction on disk. The dense form is only realised transiently in
GPU memory per batch. For larger trees (your N=220 upper end) the savings
grow quadratically. This is the same reason SemlaFlow serialises molecules
this way in the first place — molecules are sparse graphs too.
