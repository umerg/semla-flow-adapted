"""Distribution-level validation metrics for generated vs ground-truth neuron trees.

Port of dendrite_gen's `validation/dist_metrics.py`. We do NOT expect a generated tree to
match a specific GT tree node-for-node, so we compare the *distribution* of summary
statistics pooled over the generated set against the same statistics over the GT set:

  * per-feature marginals reduced to a Wasserstein-1 scalar;
  * joint distribution via MMD + Density/Coverage on two per-tree embeddings -- a
    standardized morphometric vector and a radial-from-root TMD persistence image --
    which catch broken cross-feature correlations that every marginal misses;
  * a structural health block that no morphometric can see (see below).

The invisible robustness rule kept from the original: the MMD bandwidth and the
morphometric standardization / TMD PCA are FIT ON THE GT SET and reused, so the MMD
trajectory is comparable across training steps. Callers cache this via `build_gt_cache`.

Differences from dendrite_gen, all deliberate:

  * **Graphs are sanitised first** (`validation.sanitise`). dendrite_gen's generator emits
    trees by construction; SemlaFlow's N**2 bond head does not, and the morphometrics are
    undefined-to-crashing on cycles and fragments. Health is measured on the RAW graph
    before sanitisation, so the defect is reported rather than hidden.
  * **Extents use a root->centroid axis** (`G.graph["axis"]`), not dendrite_gen's fixed
    anatomical `uhat`. SEMLA is fully E(3)-equivariant and the training transform applies
    a random rotation, so no fixed axis exists. This makes `mmd_morpho` here NOT
    numerically comparable to dendrite_gen's -- do not share a plot axis.
  * Tree-edit distance, KS twins and ZCA whitening are dropped (cost / out of scope /
    off in every dendrite_gen config).

Returned dict is flat ``{str: float}``. Keys with insufficient data are nan.
"""

from __future__ import annotations

from typing import Callable, Iterable

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance

from semlaflow.util.dist_helper import (
    density_coverage,
    median_heuristic_bandwidth,
    mmd2_unbiased,
)

from .sanitise import HEALTH_KEYS, graph_health, sanitise_graph
from .structural_metrics import (
    SHOLL_N_SHELLS,
    _pos_to_xyz,
    _root_tree,
    bifurcation_angle_values,
    branch_length_values,
    branch_order_values,
    contraction_ratio_values,
    degree_values,
    partition_asymmetry,
    path_length_to_root_values,
    radial_distance_to_root_values,
    sholl_summary,
    strahler_number,
)

# Fixed-order morphometric feature vector behind the joint metrics. Copied verbatim from
# dendrite_gen's MORPHO_KEYS v2 so the vector definition stays shared between the repos.
#
# The v1 16-D vector was rank-deficient (`node_count`/`leaf_count`/`bifurcation_count` at
# r = 1.000, `axial_extent`/`total_extent` at 0.999, the two root-distance means at
# 0.996). Because the RBF kernel runs on raw z-scores -- which equalise per-feature
# variance but NOT correlation -- each such block contributed ~m x to the squared
# distance, so size and reach dominated while shape features got 1 x each.
#
# `node_count` is EXCLUDED so the joint block is a shape-only comparison; size is
# reported separately as the `node_count_w1` marginal, which on this data is a
# branch-point count (see `_per_tree_scalars`).
MORPHO_KEYS = (
    "axial_extent",
    "radial_span",
    "max_branch_order",
    "partition_asymmetry",
    "mean_branch_length",
    "mean_bifurcation_angle",
    "mean_radial_to_root",
    "mean_contraction",
    "sholl_critical_radius",
)

# Bump whenever MORPHO_KEYS changes. mmd/density/coverage_morpho are NOT comparable
# across versions -- runs either side of a bump must not share a plot axis.
MORPHO_VERSION = 2

# Report tiers. This is a PURE LOGGING FILTER: `compute_distribution_metrics` does not
# take it as an argument, so it is structurally impossible for it to change a computed
# number. `standard` mirrors dendrite_gen's 19-key dashboard plus the SemlaFlow-specific
# health block.
_HEADLINE = (
    "mmd_morpho",
    "w1_pooled_mean_normalized",
    "w1_pertree_mean_normalized",
    "disconnected_frac",
)

_STANDARD = _HEADLINE + (
    # pooled marginals
    "branch_length_w1",
    "bifurcation_angle_w1",
    "radial_to_root_w1",
    "contraction_w1",
    "branch_order_w1",
    "tmd_barlen_w1",
    "degree_w1",
    # per-tree marginals
    "node_count_w1",
    "bifurcation_count_w1",
    "axial_extent_w1",
    "radial_span_w1",
    "partition_asymmetry_w1",
    "sholl_critical_radius_w1",
    # joint
    "density_morpho",
    "coverage_morpho",
    "mmd_tmd",
    # health
    "gen_degenerate_frac",
    "morpho_nan_frac",
) + tuple(k for k in HEALTH_KEYS if k != "disconnected_frac")

_FULL_EXTRA = (
    "path_to_root_w1",
    "leaf_count_w1",
    "total_extent_w1",
    "strahler_w1",
    "sholl_peak_w1",
    "sholl_auc_w1",
    "density_tmd",
    "coverage_tmd",
)

# Per-run constants: pushed to logger config once, never as a time series.
CONSTANT_KEYS = (
    "mmd_bandwidth_morpho",
    "mmd_bandwidth_tmd",
    "tmd_eff_rank",
    "morpho_gt_nan_frac",
    "morpho_version",
)

METRIC_TIERS = {"headline": _HEADLINE, "standard": _STANDARD, "full": _STANDARD + _FULL_EXTRA}

# Backwards compatibility: the previous 8-key tuple. Kept so older analysis scripts and
# `notebooks/` keep importing successfully.
METRIC_KEYS = (
    "branch_length_w1",
    "bifurcation_angle_w1",
    "node_count_w1",
    "leaf_count_w1",
    "bifurcation_count_w1",
    "axial_extent_w1",
    "radial_span_w1",
    "total_extent_w1",
)


def keys_for_level(level: str = "standard") -> tuple[str, ...]:
    """Keys to log at a given report level. Pure filter -- see `METRIC_TIERS`."""
    try:
        return METRIC_TIERS[level]
    except KeyError:
        raise ValueError(
            f"Unknown metric_report_level {level!r}; expected one of {sorted(METRIC_TIERS)}"
        ) from None


# --- small helpers ---------------------------------------------------------------------


def _root_of(G: nx.Graph) -> int | None:
    root = G.graph.get("root")
    if root is None or root not in G.nodes:
        return None
    return int(root)


def _axis_of(G: nx.Graph) -> np.ndarray:
    """The graph's axial direction, as stashed by `sanitise_graph`."""
    axis = G.graph.get("axis")
    if axis is None:
        return np.array([0.0, 0.0, 1.0])
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])


def _w1(gen_vals, gt_vals) -> float:
    """Wasserstein-1 between two pooled value arrays; nan if either is empty."""
    gen_vals = np.asarray(gen_vals, dtype=np.float64).reshape(-1)
    gt_vals = np.asarray(gt_vals, dtype=np.float64).reshape(-1)
    gen_vals = gen_vals[np.isfinite(gen_vals)]
    gt_vals = gt_vals[np.isfinite(gt_vals)]
    if gen_vals.size == 0 or gt_vals.size == 0:
        return float("nan")
    return float(wasserstein_distance(gen_vals, gt_vals))


def _safe_mean(arr) -> float:
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def _nan_frac(mat: np.ndarray) -> float:
    """Fraction of non-finite entries in an (N, d) morpho matrix; nan if empty."""
    mat = np.asarray(mat, dtype=np.float64)
    if mat.size == 0:
        return float("nan")
    return float(np.mean(~np.isfinite(mat)))


def _pool(graphs: Iterable[nx.Graph], fn) -> np.ndarray:
    arrs = []
    for G in graphs:
        try:
            a = np.asarray(fn(G), dtype=np.float64).reshape(-1)
        except Exception:
            continue
        a = a[np.isfinite(a)]
        if a.size:
            arrs.append(a)
    return np.concatenate(arrs) if arrs else np.zeros((0,), dtype=np.float64)


# --- per-graph statistic extractors ----------------------------------------------------


def _bifurcation_angles(G: nx.Graph) -> np.ndarray:
    root = _root_of(G)
    if root is None:
        return np.zeros((0,), dtype=np.float64)
    try:
        return bifurcation_angle_values(G, root=root)
    except ValueError:
        return np.zeros((0,), dtype=np.float64)


def _max_branch_order(G: nx.Graph) -> float:
    vals = branch_order_values(G)
    return float(vals.max()) if vals.size else float("nan")


def _tmd_bar_lengths(G: nx.Graph) -> np.ndarray:
    """|death - birth| for each persistence interval (raw scale, no per-graph norm).

    Uses the same filtration as the joint TMD block so the two share one source of truth
    -- dendrite_gen shipped a version where this took the barcode default (`path`) while
    the joint block used `radial_root`, silently putting `tmd_barlen_w1` in different
    units from every other TMD metric.

    Returns empty on failure. That is safe here *only because graphs are sanitised
    first*: on a raw generated graph `compute_tmd_barcode_diagram` raises on any cycle or
    fragment, and swallowing that would silently restrict the pooled comparison to the
    graphs that happened to come out perfect -- survivorship bias that flatters a worse
    model.
    """
    from semlaflow.tmd import TMD_EVAL_FILTRATION
    from semlaflow.tmd.tmd import compute_tmd_barcode_diagram

    if _root_of(G) is None:
        return np.zeros((0,), dtype=np.float64)
    try:
        _barcode, diagram = compute_tmd_barcode_diagram(
            G, filtration=TMD_EVAL_FILTRATION, normalize_mode="none"
        )
    except Exception:
        return np.zeros((0,), dtype=np.float64)
    pairs = np.asarray(diagram.as_pairs(), dtype=np.float64).reshape(-1, 2)
    if pairs.size == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.abs(pairs[:, 1] - pairs[:, 0])


def _size_extent(G: nx.Graph) -> dict[str, float]:
    """Per-tree size/extent stats decomposed in the graph's root->centroid frame.

    ``node_count`` is NOT a size nuisance on this data. These are critical trees, so
    every non-root node is a bifurcation or a terminal and ``1 + leaves + bifurcations ==
    N`` holds exactly (100.00% of real graphs, both corpora), with
    ``corr(node_count, bifurcation_count) = 0.9979``. Node count *is* the branching
    topology, and it is among the strongest signals in the suite (0.36 GT-sd on real
    generations, vs 0.21-0.31 for most keys).

    In-loop it used to read exactly 0.0, because validation pairs the prior mask with the
    GT batch. Post-sanitisation it measures how many of the fed-in nodes the model turned
    into real branch points -- the change away from 0.0 is the intended behaviour, not a
    regression.
    """
    out = {
        "node_count": float(G.number_of_nodes()),
        "leaf_count": float("nan"),
        "bifurcation_count": float("nan"),
        "axial_extent": float("nan"),
        "radial_span": float("nan"),
        "total_extent": float("nan"),
    }
    root = _root_of(G)
    if root is not None:
        _parent, children = _root_tree(G, root)
        out["leaf_count"] = float(sum(1 for k, ch in children.items() if len(ch) == 0 and k != root))
        out["bifurcation_count"] = float(
            sum(1 for k, ch in children.items() if len(ch) >= 2 and k != root)
        )
    n = G.number_of_nodes()
    if n > 0:
        pts = np.stack([_pos_to_xyz(G.nodes[k].get("pos", np.zeros(3))) for k in G.nodes()], axis=0)
        u = _axis_of(G)
        s = pts @ u
        out["axial_extent"] = float(s.max() - s.min())
        if n >= 2:
            pts_perp = pts - np.outer(s, u)
            out["radial_span"] = float(pdist(pts_perp).max())
            out["total_extent"] = float(pdist(pts).max())
        else:
            out["radial_span"] = 0.0
            out["total_extent"] = 0.0
    return out


def _per_tree_scalars(G: nx.Graph) -> dict[str, float]:
    out = _size_extent(G)
    out["strahler"] = strahler_number(G)
    out["partition_asymmetry"] = partition_asymmetry(G)
    out.update(sholl_summary(G, n_shells=SHOLL_N_SHELLS))
    return out


def _degenerate_frac(graphs: list[nx.Graph]) -> float:
    """Fraction of trees that are structurally or spatially degenerate.

    Degenerate = no bifurcation at all (so partition asymmetry and fork angle are
    undefined) OR zero spatial extent. These are exactly the cases whose nan features get
    imputed to the GT mean by `standardize_vectors` and would otherwise be invisible.

    Computed on the SANITISED graphs, matching dendrite_gen's semantics. It reads ~0.0
    here because sanitisation removes the causes; the `sanitise.graph_health` block is
    the live disclosure for SemlaFlow's own failure modes.
    """
    if not graphs:
        return float("nan")
    bad = 0
    for G in graphs:
        root = _root_of(G)
        if root is None or G.number_of_nodes() < 2:
            bad += 1
            continue
        _parent, children = _root_tree(G, root)
        if not any(len(ch) >= 2 for ch in children.values()):
            bad += 1
            continue
        if not (float(_size_extent(G).get("total_extent", 0.0) or 0.0) > 0.0):
            bad += 1
    return float(bad / len(graphs))


# (key, extractor) for statistics pooled over every element of every tree.
_POOLED_FEATURES = (
    ("branch_length", branch_length_values),
    ("bifurcation_angle", _bifurcation_angles),
    ("tmd_barlen", _tmd_bar_lengths),
    ("path_to_root", path_length_to_root_values),
    ("radial_to_root", radial_distance_to_root_values),
    ("contraction", contraction_ratio_values),
    ("branch_order", branch_order_values),
    ("degree", degree_values),
)

# Per-tree scalars: one value per tree -> a distribution over trees.
_PERTREE_KEYS = (
    "node_count",
    "leaf_count",
    "bifurcation_count",
    "axial_extent",
    "radial_span",
    "total_extent",
    "strahler",
    "partition_asymmetry",
    "sholl_peak",
    "sholl_critical_radius",
    "sholl_auc",
)


# --- morphometric vector + GT-fit cache ------------------------------------------------


def assemble_morpho_vector(G: nx.Graph) -> np.ndarray:
    """Fixed-order per-tree morphometric vector (see MORPHO_KEYS). May contain nan."""
    ext = _size_extent(G)
    sh = sholl_summary(G, n_shells=SHOLL_N_SHELLS)
    vals = {
        "axial_extent": ext["axial_extent"],
        "radial_span": ext["radial_span"],
        "max_branch_order": _max_branch_order(G),
        "partition_asymmetry": partition_asymmetry(G),
        "mean_branch_length": _safe_mean(branch_length_values(G)),
        "mean_bifurcation_angle": _safe_mean(_bifurcation_angles(G)),
        "mean_radial_to_root": _safe_mean(radial_distance_to_root_values(G)),
        "mean_contraction": _safe_mean(contraction_ratio_values(G)),
        "sholl_critical_radius": sh["sholl_critical_radius"],
    }
    return np.asarray([vals[k] for k in MORPHO_KEYS], dtype=np.float64)


def standardize_vectors(vecs: np.ndarray, *, mean, std, eps: float = 1e-8) -> np.ndarray:
    """z-score by GT mean/std, then impute non-finite entries to 0 (the GT mean)."""
    vecs = np.asarray(vecs, dtype=np.float64)
    if vecs.size == 0:
        return vecs.reshape(0, len(mean))
    z = (vecs - mean) / (std + eps)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_pca(X: np.ndarray, ncomp: int | None):
    """Fit a top-``ncomp`` PCA (centered SVD). Returns None when reduction is a no-op."""
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape if X.ndim == 2 else (0, 0)
    if not ncomp or n < 2 or ncomp >= min(n, d):
        return None
    mean = X.mean(axis=0)
    _U, _S, Vt = np.linalg.svd(X - mean, full_matrices=False)
    return {"mean": mean, "components": Vt[:ncomp]}


def _apply_pca(X: np.ndarray, pca) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if pca is None:
        return X
    return (X - pca["mean"]) @ pca["components"].T


def _effective_rank(X: np.ndarray) -> float:
    """Participation-ratio effective rank of a (centered) embedding matrix."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return float("nan")
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    s = s[s > 0]
    if s.size == 0:
        return float("nan")
    return float((s.sum() ** 2) / (s**2).sum())


def _embed_matrix(graphs: Iterable[nx.Graph], embed_fn) -> np.ndarray:
    """Stack per-tree TMD embeddings, skipping graphs whose embedding fails."""
    rows = []
    for G in graphs:
        try:
            e = np.asarray(embed_fn(G), dtype=np.float64).reshape(-1)
        except Exception:
            continue
        if e.size and np.all(np.isfinite(e)):
            rows.append(e)
    if not rows:
        return np.zeros((0, 0), dtype=np.float64)
    return np.stack(rows, axis=0)


def build_gt_cache(
    gt_graphs: list[nx.Graph],
    *,
    embed_fn: Callable[[nx.Graph], np.ndarray] | None = None,
    tmd_pca_ncomp: int | None = 64,
    already_sanitised: bool = False,
) -> dict:
    """Precompute the GT-derived objects the joint metrics need, ONCE on the fixed GT set.

    Morphometric mean/std + standardized GT vectors + MMD bandwidth, and the TMD
    persistence-image PCA + reduced GT embeddings + bandwidth. Reusing these across
    training steps is what keeps the MMD trajectory comparable -- a per-step bandwidth
    would silently make the curve meaningless.

    Safe to fit once and reuse for the whole run even though the training transform
    applies a fresh random rotation each epoch: every morphometric here is rotation
    invariant to ~1e-16, and `choose_root` was measured to flip on 0.0% of graphs under
    rotation.

    Callers MUST NOT build this during Lightning's sanity-check loop -- it runs 2 batches,
    so the fit would come from ~16 graphs and poison every subsequent epoch.
    """
    if embed_fn is None:
        from semlaflow.tmd import compute_tmd_embedding

        embed_fn = compute_tmd_embedding

    if not already_sanitised:
        gt_graphs = [sanitise_graph(G) for G in gt_graphs]

    morpho = (
        np.stack([assemble_morpho_vector(G) for G in gt_graphs], axis=0)
        if gt_graphs
        else np.zeros((0, len(MORPHO_KEYS)), dtype=np.float64)
    )
    morpho_mean = np.nanmean(morpho, axis=0) if morpho.shape[0] else np.zeros(len(MORPHO_KEYS))
    morpho_std = np.nanstd(morpho, axis=0) if morpho.shape[0] else np.ones(len(MORPHO_KEYS))
    morpho_mean = np.nan_to_num(morpho_mean, nan=0.0)
    morpho_std = np.nan_to_num(morpho_std, nan=1.0)
    # Guard (near-)zero-variance features: a constant GT feature would otherwise turn any
    # deviation into a huge z-score and dominate the RBF kernel.
    morpho_std = np.where(morpho_std < 1e-8, 1.0, morpho_std)
    morpho_z = standardize_vectors(morpho, mean=morpho_mean, std=morpho_std)
    # Should be 0.0 on any healthy dataset. Non-zero means some GT trees are themselves
    # degenerate, which would invalidate reading the gen-side `morpho_nan_frac` as a pure
    # generator-failure signal.
    morpho_gt_nan_frac = _nan_frac(morpho)
    morpho_sigma = median_heuristic_bandwidth(morpho_z) if morpho_z.shape[0] > 1 else 1.0

    tmd_raw = _embed_matrix(gt_graphs, embed_fn)
    pca = _fit_pca(tmd_raw, tmd_pca_ncomp)
    tmd_reduced = _apply_pca(tmd_raw, pca) if tmd_raw.shape[0] else tmd_raw
    tmd_sigma = median_heuristic_bandwidth(tmd_reduced) if tmd_reduced.shape[0] > 1 else 1.0

    # The GT side of every marginal is also fixed for the whole run, so precompute it
    # here rather than re-deriving it each validation epoch. This is the bulk of the
    # per-epoch cost (Sholl and two Dijkstras per graph), so caching it roughly halves
    # validation overhead.
    pooled = {name: _pool(gt_graphs, fn) for name, fn in _POOLED_FEATURES}
    gt_ext = [_per_tree_scalars(G) for G in gt_graphs]
    pertree = {
        key: np.array([d[key] for d in gt_ext], dtype=np.float64) if gt_ext else np.zeros((0,))
        for key in _PERTREE_KEYS
    }

    return {
        "graphs": gt_graphs,
        "pooled": pooled,
        "pertree": pertree,
        "morpho_mean": morpho_mean,
        "morpho_std": morpho_std,
        "morpho_z": morpho_z,
        "morpho_sigma": morpho_sigma,
        "morpho_gt_nan_frac": morpho_gt_nan_frac,
        "morpho_version": MORPHO_VERSION,
        "embed_fn": embed_fn,
        "pca": pca,
        "tmd_reduced": tmd_reduced,
        "tmd_sigma": tmd_sigma,
        "tmd_eff_rank": _effective_rank(tmd_raw),
    }


def subset_gt_cache(cache: dict, graphs: list[nx.Graph], *, already_sanitised: bool = False) -> dict:
    """Re-target a GT cache at a subset of the GT graphs, keeping the run-wide fit.

    Used for the per-class stratified metrics. The GT *reference set* must be that class's
    own graphs -- comparing one class against all classes would be meaningless -- but the
    standardization (mean/std), the MMD bandwidths and the TMD PCA must stay run-wide, or
    the per-class numbers become incomparable both to each other and to the run-wide
    number.
    """
    if not already_sanitised:
        graphs = [sanitise_graph(G) for G in graphs]

    morpho = (
        np.stack([assemble_morpho_vector(G) for G in graphs], axis=0)
        if graphs
        else np.zeros((0, len(MORPHO_KEYS)), dtype=np.float64)
    )
    tmd_raw = _embed_matrix(graphs, cache.get("embed_fn"))
    gt_ext = [_per_tree_scalars(G) for G in graphs]

    out = dict(cache)
    out["graphs"] = graphs
    out["pooled"] = {name: _pool(graphs, fn) for name, fn in _POOLED_FEATURES}
    out["pertree"] = {
        key: np.array([d[key] for d in gt_ext], dtype=np.float64) if gt_ext else np.zeros((0,))
        for key in _PERTREE_KEYS
    }
    out["morpho_z"] = standardize_vectors(
        morpho, mean=cache["morpho_mean"], std=cache["morpho_std"]
    )
    out["tmd_reduced"] = (
        _apply_pca(tmd_raw, cache.get("pca")) if tmd_raw.shape[0] else tmd_raw
    )
    return out


def joint_metrics_from_vectors(
    gen_vecs: np.ndarray, gt_vecs: np.ndarray, *, prefix: str, sigma: float, k: int
) -> dict[str, float]:
    """MMD + Density/Coverage between two already-transformed embedding matrices."""
    gen_vecs = np.asarray(gen_vecs, dtype=np.float64)
    gt_vecs = np.asarray(gt_vecs, dtype=np.float64)
    out = {
        f"mmd_{prefix}": float("nan"),
        f"density_{prefix}": float("nan"),
        f"coverage_{prefix}": float("nan"),
    }
    if gen_vecs.shape[0] < 2 or gt_vecs.shape[0] < 2:
        return out
    out[f"mmd_{prefix}"] = mmd2_unbiased(gen_vecs, gt_vecs, sigma)
    dens, cov = density_coverage(gen_vecs, gt_vecs, k=k)
    out[f"density_{prefix}"] = dens
    out[f"coverage_{prefix}"] = cov
    return out


# --- main entry point -------------------------------------------------------------------


def compute_distribution_metrics(
    gen_graphs: list[nx.Graph],
    gt_graphs: list[nx.Graph],
    *,
    gt_cache: dict | None = None,
    enable_joint: bool = True,
    dc_k: int = 5,
    tmd_pca_ncomp: int | None = 64,
) -> dict[str, float]:
    """Compare distributions of summary statistics between generated and GT trees.

    Takes **raw** graphs (as built by `convert.geometric_mol_to_nx`) and does the health
    measurement and sanitisation internally, in that order -- the ordering is load-bearing
    and easy to get wrong, so it is not left to the caller.

    Pass `gt_cache` from `build_gt_cache` to keep the MMD comparable across steps; without
    one it is rebuilt from `gt_graphs` (deterministic, convenient for tests, but a
    per-step refit).

    Returns a flat dict of floats. Keys with insufficient data are nan.
    """
    metrics: dict[str, float] = {}

    # 1. Health on the RAW graphs. Must precede sanitisation, or sanitisation hides
    #    exactly the defects it repairs.
    metrics.update(graph_health(gen_graphs))

    # 2. Sanitise both sides identically. A verified no-op on ground truth, so this
    #    cannot introduce a gen/GT asymmetry.
    gen_s = [sanitise_graph(G) for G in gen_graphs]
    if gt_cache is not None and gt_cache.get("graphs"):
        gt_s = gt_cache["graphs"]
    else:
        gt_s = [sanitise_graph(G) for G in gt_graphs]

    pooled_norms: list[float] = []
    pertree_norms: list[float] = []

    # 3. Pooled marginals. Each also contributes W1/GT-spread to the normalized
    #    aggregate, so that units cannot dominate: the raw mean of these keys is ~90%
    #    extents on neurons (microns) but is dominated by the counts on the tree corpora
    #    (metres). The artifact flips direction between corpora, which is exactly what
    #    normalising removes.
    gt_pooled_cache = (gt_cache or {}).get("pooled") or {}
    for name, fn in _POOLED_FEATURES:
        gen_pool = _pool(gen_s, fn)
        gt_pool = gt_pooled_cache.get(name)
        if gt_pool is None:
            gt_pool = _pool(gt_s, fn)
        w1 = _w1(gen_pool, gt_pool)
        metrics[f"{name}_w1"] = w1
        scale = float(np.nanstd(gt_pool)) if gt_pool.size else float("nan")
        if np.isfinite(w1) and np.isfinite(scale) and scale > 1e-12:
            pooled_norms.append(w1 / scale)

    # 4. Per-tree scalars.
    gen_ext = [_per_tree_scalars(G) for G in gen_s]
    gt_pertree_cache = (gt_cache or {}).get("pertree") or {}
    gt_ext = None if gt_pertree_cache else [_per_tree_scalars(G) for G in gt_s]
    for key in _PERTREE_KEYS:
        gen_vals = np.array([d[key] for d in gen_ext], dtype=np.float64) if gen_ext else np.zeros((0,))
        gt_vals = gt_pertree_cache.get(key)
        if gt_vals is None:
            gt_vals = np.array([d[key] for d in gt_ext], dtype=np.float64) if gt_ext else np.zeros((0,))
        w1 = _w1(gen_vals, gt_vals)
        metrics[f"{key}_w1"] = w1
        finite_gt = gt_vals[np.isfinite(gt_vals)]
        scale = float(np.nanstd(finite_gt)) if finite_gt.size else float("nan")
        if np.isfinite(w1) and np.isfinite(scale) and scale > 1e-12:
            pertree_norms.append(w1 / scale)

    if pooled_norms:
        metrics["w1_pooled_mean_normalized"] = float(np.mean(pooled_norms))
    if pertree_norms:
        metrics["w1_pertree_mean_normalized"] = float(np.mean(pertree_norms))

    if gen_s:
        metrics["gen_degenerate_frac"] = _degenerate_frac(gen_s)

    # 5. Joint metrics on two per-tree embeddings. Standardization / PCA / bandwidth come
    #    from the GT-fit cache so the MMD is comparable across steps.
    if enable_joint:
        if gt_cache is None:
            gt_cache = build_gt_cache(
                gt_s, tmd_pca_ncomp=tmd_pca_ncomp, already_sanitised=True
            )
        embed_fn = gt_cache.get("embed_fn")
        k = int(dc_k)

        gen_morpho = (
            np.stack([assemble_morpho_vector(G) for G in gen_s], axis=0)
            if gen_s
            else np.zeros((0, len(MORPHO_KEYS)))
        )
        metrics["morpho_nan_frac"] = _nan_frac(gen_morpho)
        gen_morpho_z = standardize_vectors(
            gen_morpho, mean=gt_cache["morpho_mean"], std=gt_cache["morpho_std"]
        )
        metrics.update(
            joint_metrics_from_vectors(
                gen_morpho_z,
                gt_cache["morpho_z"],
                prefix="morpho",
                sigma=gt_cache["morpho_sigma"],
                k=k,
            )
        )

        gen_tmd_raw = _embed_matrix(gen_s, embed_fn)
        gen_tmd = _apply_pca(gen_tmd_raw, gt_cache["pca"]) if gen_tmd_raw.shape[0] else gen_tmd_raw
        metrics.update(
            joint_metrics_from_vectors(
                gen_tmd,
                gt_cache["tmd_reduced"],
                prefix="tmd",
                sigma=gt_cache["tmd_sigma"],
                k=k,
            )
        )

        metrics["mmd_bandwidth_morpho"] = float(gt_cache["morpho_sigma"])
        metrics["mmd_bandwidth_tmd"] = float(gt_cache["tmd_sigma"])
        metrics["tmd_eff_rank"] = float(gt_cache.get("tmd_eff_rank", float("nan")))
        metrics["morpho_gt_nan_frac"] = float(gt_cache.get("morpho_gt_nan_frac", float("nan")))
        metrics["morpho_version"] = float(gt_cache.get("morpho_version", MORPHO_VERSION))

    return metrics
