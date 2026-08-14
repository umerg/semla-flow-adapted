"""Matched-pair, TMD-conditioned fidelity metrics (companion to `dist_metrics`).

The distributional suite pools statistics over the whole generated set against the whole
GT set. It answers "does the population look right?" but is blind to *per-conditioning*
failure: a generator can match every population marginal while mapping individual TMD
vectors to the wrong tree. That blindness is total here, because SemlaFlow now conditions
on both rotation-invariant filtrations, so `mmd_tmd`/`tmd_barlen_w1` are computed with a
filtration the model was trained on and partly measure it echoing its own input (see
`tmd/__init__.py:TMD_EVAL_FILTRATION`). These matched-pair metrics are not subject to that:
they ask whether generated graph *i* realises the specific descriptor it was handed.

Pairing is by index. `NeuronCFM._val_gen_graphs` and `._val_gt_graphs` are appended in the
same order from the same batch, and generation is conditioned on the paired GT graph's
vector, so `gen_graphs[i]` was conditioned on `gt_graphs[i]`. (dendrite_gen has to undo a
size-balancing permutation to recover this; SemlaFlow never applies one.)

What is compared, per pair:

  - Persistence-diagram Wasserstein (optionally bottleneck) per filtration -- "did the
    generated tree realise the specific barcode we fed it?"
  - Absolute differences of the per-tree extent scalars.
  - Wasserstein-1 between the two trees' branch-length and bifurcation-angle value sets --
    the per-tree analogue of the pooled `branch_length_w1` / `bifurcation_angle_w1`.

`compute_conditional_pairwise_metrics` is a pure function: it returns a flat `{str: float}`
(`{}` when there are no usable pairs), never raises on a bad pair (that pair contributes
nan and is dropped by the nan-aware aggregation), and degrades to nan PD distances rather
than crashing when `persim` is not installed.

Two deliberate departures from dendrite_gen's version:

  - **Extent scalars are SemlaFlow's, not dendrite_gen's.** It uses `height_z_range` /
    `span_xy_diameter`, which project onto a fixed anatomical `uhat`. There is no such axis
    here (`COMPATIBLE.md` §4.14), so this uses `_size_extent`'s `axial_extent` /
    `radial_span` / `total_extent`, each measured in the graph's own root->centroid frame.
  - **No persistent GT cache.** dendrite_gen precomputes the GT side once because it scores
    every step. Here `max_pairs` bounds the work and `tmd_cond_every` spaces it out, so
    recomputing avoids a whole class of stale-cache bugs for no meaningful cost.
"""
from __future__ import annotations

import numpy as np
import networkx as nx

from .dist_metrics import _bifurcation_angles, _root_of, _size_extent, _w1
from .sanitise import sanitise_graph
from .structural_metrics import branch_length_values

try:  # optional dependency; PD distances degrade to nan when absent (never a crash)
    import persim as _persim
except ImportError:  # pragma: no cover - exercised by monkeypatching _persim to None
    _persim = None


# Extent scalars compared per pair. Keys of `_size_extent`, measured in each graph's own
# root->centroid frame -- so the comparison is of magnitudes, not of a shared frame.
_EXTENT_KEYS = ("axial_extent", "radial_span", "total_extent")


# --------------------------------------------------------------------------- helpers
def _finite_mean_median(vals: list[float]) -> tuple[float, float]:
    """(mean, median) over the finite entries; (nan, nan) if none are finite."""
    a = np.asarray(vals, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    return float(a.mean()), float(np.median(a))


def _nan_frac(vals: list[float]) -> float:
    """Fraction of entries that are non-finite; nan if the list is empty."""
    a = np.asarray(vals, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return float("nan")
    return float(np.mean(~np.isfinite(a)))


def _pd_pairs(G: nx.Graph, filtration: str, normalize_mode: str) -> np.ndarray | None:
    """(M,2) canonical (birth <= death) diagram pairs for one filtration; None on failure.

    `compute_tmd_barcode_diagram` requires a rooted tree with 3D `pos` and raises on a
    cycle or fragment. Sanitisation should have removed those, but a pathological
    generated graph can still get through -- returning None makes that pair contribute nan
    instead of aborting the sweep, and it is counted in `n_pairs_skipped`.
    """
    from semlaflow.tmd.tmd import compute_tmd_barcode_diagram

    if _root_of(G) is None:
        return None
    try:
        _barcode, diagram = compute_tmd_barcode_diagram(
            G,
            filtration=filtration,
            normalize_mode=normalize_mode,
            weight_edges_by_euclidean=True,
            simplify_to_critical_tree=True,
        )
    except Exception:
        return None
    return np.asarray(diagram.as_pairs(), dtype=np.float64).reshape(-1, 2)


def _pd_distance(pairs_a: np.ndarray, pairs_b: np.ndarray, *, kind: str) -> float:
    """persim Wasserstein/bottleneck between two diagrams; nan if persim is missing.

    Two empty diagrams are distance 0. persim matches a lone non-empty diagram against the
    diagonal by itself; any error falls back to nan.
    """
    if _persim is None:
        return float("nan")
    if pairs_a.size == 0 and pairs_b.size == 0:
        return 0.0
    try:
        fn = _persim.wasserstein if kind == "wasserstein" else _persim.bottleneck
        return float(fn(pairs_a, pairs_b))
    except Exception:
        return float("nan")


def _pairwise_w1(a: np.ndarray, b: np.ndarray) -> float:
    """W1 between two per-tree value sets; nan if either side is empty."""
    if a.size == 0 or b.size == 0:
        return float("nan")
    return _w1(a, b)


def _absdiff(a: float, b: float) -> float:
    """|a - b|, propagating nan (so an unmeasurable pair is dropped, not scored as 0)."""
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return float(abs(a - b))


# --------------------------------------------------------------------------- entry point
def compute_conditional_pairwise_metrics(
    gen_graphs: list[nx.Graph],
    gt_graphs: list[nx.Graph],
    *,
    pd_filtrations=None,
    max_pairs: int | None = 64,
    enable_wasserstein: bool = True,
    enable_bottleneck: bool = False,
    normalize_mode: str = "minmax",
    already_sanitised: bool = False,
) -> dict[str, float]:
    """Index-matched fidelity of `gen_graphs[i]` against its conditioning `gt_graphs[i]`.

    Graphs are sanitised first (largest connected component -> MST -> root -> contract),
    exactly as `compute_distribution_metrics` does, so both sides are the critical trees
    the morphometrics assume. `sanitise_graph` never drops a graph, so index alignment --
    the whole basis of the pairing -- survives it.

    `pd_filtrations` defaults to the conditioning set (`NEURON_TMD_FILTRATIONS`): the point
    is to score the descriptor the model was actually given.

    Returns a flat dict of Python floats, or `{}` if there are no pairs.
    """
    if pd_filtrations is None:
        from semlaflow.tmd import NEURON_TMD_FILTRATIONS

        pd_filtrations = NEURON_TMD_FILTRATIONS
    filts = tuple(pd_filtrations)

    n = min(len(gen_graphs), len(gt_graphs))
    if max_pairs is not None and int(max_pairs) > 0:
        n = min(n, int(max_pairs))
    if n == 0:
        return {}

    gen_s = gen_graphs[:n] if already_sanitised else [sanitise_graph(G) for G in gen_graphs[:n]]
    gt_s = gt_graphs[:n] if already_sanitised else [sanitise_graph(G) for G in gt_graphs[:n]]

    pd_w: dict[str, list[float]] = {f: [] for f in filts}
    pd_bn: dict[str, list[float]] = {f: [] for f in filts}
    extent_d: dict[str, list[float]] = {k: [] for k in _EXTENT_KEYS}
    bl_w1: list[float] = []
    ba_w1: list[float] = []
    n_skipped = 0

    for gp, gt in zip(gen_s, gt_s):
        # Extent scalars, each in its own graph's frame; nan-safe on degenerate input.
        ext_p, ext_t = _size_extent(gp), _size_extent(gt)
        for key in _EXTENT_KEYS:
            extent_d[key].append(_absdiff(ext_p[key], ext_t[key]))

        # Per-pair W1 between the two trees' own value sets.
        bl_w1.append(_pairwise_w1(branch_length_values(gp), branch_length_values(gt)))
        ba_w1.append(_pairwise_w1(_bifurcation_angles(gp), _bifurcation_angles(gt)))

        # Persistence-diagram distances, per filtration.
        gp_bad = False
        for f in filts:
            pairs_g = _pd_pairs(gp, f, normalize_mode)
            pairs_t = _pd_pairs(gt, f, normalize_mode)
            if pairs_g is None:
                gp_bad = True
            if pairs_g is None or pairs_t is None:
                if enable_wasserstein:
                    pd_w[f].append(float("nan"))
                if enable_bottleneck:
                    pd_bn[f].append(float("nan"))
                continue
            if enable_wasserstein:
                pd_w[f].append(_pd_distance(pairs_g, pairs_t, kind="wasserstein"))
            if enable_bottleneck:
                pd_bn[f].append(_pd_distance(pairs_g, pairs_t, kind="bottleneck"))
        if gp_bad:
            n_skipped += 1

    out: dict[str, float] = {}
    for f in filts:
        if enable_wasserstein:
            mean, median = _finite_mean_median(pd_w[f])
            out[f"pd_wasserstein_{f}_mean"] = mean
            out[f"pd_wasserstein_{f}_median"] = median
            out[f"pd_nan_frac_{f}"] = _nan_frac(pd_w[f])
        if enable_bottleneck:
            mean, median = _finite_mean_median(pd_bn[f])
            out[f"pd_bottleneck_{f}_mean"] = mean
            out[f"pd_bottleneck_{f}_median"] = median
            if not enable_wasserstein:
                out[f"pd_nan_frac_{f}"] = _nan_frac(pd_bn[f])

    for key in _EXTENT_KEYS:
        mean, median = _finite_mean_median(extent_d[key])
        out[f"{key}_absdiff_mean"] = mean
        out[f"{key}_absdiff_median"] = median

    for name, vals in (("branch_length", bl_w1), ("bifurcation_angle", ba_w1)):
        mean, median = _finite_mean_median(vals)
        out[f"{name}_w1_pairwise_mean"] = mean
        out[f"{name}_w1_pairwise_median"] = median

    out["n_pairs"] = float(n)
    out["n_pairs_skipped"] = float(n_skipped)
    return out


def conditional_metric_keys(pd_filtrations=None, *, enable_bottleneck: bool = False) -> tuple[str, ...]:
    """Keys `compute_conditional_pairwise_metrics` emits, for logger registration."""
    if pd_filtrations is None:
        from semlaflow.tmd import NEURON_TMD_FILTRATIONS

        pd_filtrations = NEURON_TMD_FILTRATIONS

    keys: list[str] = []
    for f in tuple(pd_filtrations):
        keys += [f"pd_wasserstein_{f}_mean", f"pd_wasserstein_{f}_median", f"pd_nan_frac_{f}"]
        if enable_bottleneck:
            keys += [f"pd_bottleneck_{f}_mean", f"pd_bottleneck_{f}_median"]
    for key in _EXTENT_KEYS:
        keys += [f"{key}_absdiff_mean", f"{key}_absdiff_median"]
    for name in ("branch_length", "bifurcation_angle"):
        keys += [f"{name}_w1_pairwise_mean", f"{name}_w1_pairwise_median"]
    keys += ["n_pairs", "n_pairs_skipped"]
    return tuple(keys)
