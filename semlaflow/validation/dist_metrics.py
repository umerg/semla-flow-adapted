"""Distribution-level validation metrics for generated vs ground-truth neuron trees.

Geometry-only port of dendrite_gen's `validation/dist_metrics.py`. We do NOT expect a
generated tree to match a specific GT tree node-for-node, so we compare the
*distribution* of summary statistics pooled over the generated set against the same
statistics over the GT set, reducing each to a single Wasserstein-1 (Earth-Mover)
scalar.

Differences from the dendrite_gen original:
  * TMD-barcode and tree-edit-distance metrics are dropped (geometry-only).
  * Orientation-dependent extents are decomposed in each graph's own PCA principal
    axis (see `convert.pca_axis`) rather than a model symmetry axis, because SEMLA is
    fully E(3)-equivariant and generated graphs are arbitrarily oriented.

Returned dict is flat ``{str: float}``. Keys with insufficient data are nan.
"""

from __future__ import annotations

from typing import Iterable

import networkx as nx
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import wasserstein_distance

from .convert import pca_axis
from .structural_metrics import (
    _pos_to_xyz,
    _root_tree,
    bifurcation_angle_values,
    branch_length_values,
)

# Keys produced by compute_distribution_metrics (stable order for printing/JSON).
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


def _root_of(G: nx.Graph) -> int | None:
    root = G.graph.get("root")
    if root is None or root not in G.nodes:
        return None
    return int(root)


def _w1(gen_vals: np.ndarray, gt_vals: np.ndarray) -> float:
    """Wasserstein-1 between two pooled value arrays; nan if either is empty."""
    gen_vals = np.asarray(gen_vals, dtype=np.float64)
    gt_vals = np.asarray(gt_vals, dtype=np.float64)
    gen_vals = gen_vals[np.isfinite(gen_vals)]
    gt_vals = gt_vals[np.isfinite(gt_vals)]
    if gen_vals.size == 0 or gt_vals.size == 0:
        return float("nan")
    return float(wasserstein_distance(gen_vals, gt_vals))


# --- per-graph statistic extractors --------------------------------------------------


def _branch_lengths(G: nx.Graph) -> np.ndarray:
    return branch_length_values(G)


def _bifurcation_angles(G: nx.Graph) -> np.ndarray:
    root = _root_of(G)
    if root is None:
        return np.zeros((0,), dtype=np.float64)
    try:
        return bifurcation_angle_values(G, root=root)
    except ValueError:
        return np.zeros((0,), dtype=np.float64)


def _size_extent(G: nx.Graph) -> dict[str, float]:
    """Per-tree size/extent stats decomposed in the graph's PCA principal-axis frame.

    The axis (``uhat``) is each graph's own top principal component, so the extents
    are rotation invariant -- essential because generated graphs are arbitrarily
    oriented (E(3)-equivariant model):
      - axial_extent : spread along the principal axis ("height")
      - radial_span  : planar diameter in the plane orthogonal to the axis
      - total_extent : 3D diameter (max pairwise distance) -- rotation invariant anyway

    Returns a nan-filled dict on degenerate (empty/unrooted) trees.
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
        out["leaf_count"] = float(sum(1 for ch in children.values() if len(ch) == 0))
        out["bifurcation_count"] = float(sum(1 for ch in children.values() if len(ch) >= 2))
    n = G.number_of_nodes()
    if n > 0:
        pts = np.stack([_pos_to_xyz(G.nodes[k].get("pos", np.zeros(3))) for k in G.nodes()], axis=0)
        u = pca_axis(pts)  # per-graph principal axis (already unit-norm)
        s = pts @ u  # axial coordinate along the principal axis
        out["axial_extent"] = float(s.max() - s.min())
        if n >= 2:
            pts_perp = pts - np.outer(s, u)  # components orthogonal to the axis
            out["radial_span"] = float(pdist(pts_perp).max())  # planar diameter
            out["total_extent"] = float(pdist(pts).max())  # 3D diameter
        else:
            out["radial_span"] = 0.0
            out["total_extent"] = 0.0
    return out


# --- main entry point -----------------------------------------------------------------


def compute_distribution_metrics(
    gen_graphs: list[nx.Graph],
    gt_graphs: list[nx.Graph],
) -> dict[str, float]:
    """Compare distributions of summary statistics between generated and GT trees.

    Returns a flat dict of float scalars (see ``METRIC_KEYS``). Keys with insufficient
    data are nan.
    """
    metrics: dict[str, float] = {}

    def _pool(graphs: Iterable[nx.Graph], fn) -> np.ndarray:
        arrs = [np.asarray(fn(G), dtype=np.float64).reshape(-1) for G in graphs]
        arrs = [a for a in arrs if a.size > 0]
        return np.concatenate(arrs) if arrs else np.zeros((0,), dtype=np.float64)

    # Pooled-distribution statistics (every value across every tree contributes).
    metrics["branch_length_w1"] = _w1(
        _pool(gen_graphs, _branch_lengths), _pool(gt_graphs, _branch_lengths)
    )
    metrics["bifurcation_angle_w1"] = _w1(
        _pool(gen_graphs, _bifurcation_angles), _pool(gt_graphs, _bifurcation_angles)
    )

    # Per-tree size/extent statistics (one value per tree -> distribution over trees),
    # decomposed in each tree's PCA-axis frame.
    gen_ext = [_size_extent(G) for G in gen_graphs]
    gt_ext = [_size_extent(G) for G in gt_graphs]
    for key in ("node_count", "leaf_count", "bifurcation_count", "axial_extent", "radial_span", "total_extent"):
        gen_vals = np.array([d[key] for d in gen_ext], dtype=np.float64)
        gt_vals = np.array([d[key] for d in gt_ext], dtype=np.float64)
        metrics[f"{key}_w1"] = _w1(gen_vals, gt_vals)

    return metrics
