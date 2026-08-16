"""Structural metrics for rooted tree graphs (geometry-only port).

Ported from dendrite_gen's `validation/structural_metrics.py`. The tree-edit-distance
(zss) and persistence (persim) paths are intentionally dropped; everything the
distribution metrics need is here.

Assumes graphs are "critical" skeletons (branch/termination points only), so per-edge
lengths correspond to branch segment lengths. `validation.sanitise.sanitise_graph`
guarantees that property for generated graphs; ground truth already satisfies it
(`1 + leaves + bifurcations == N` holds for 100.00% of real graphs on both corpora).

Every extractor is nan/empty-safe: a missing or degenerate root yields an empty array
(pooled metrics) or nan (per-tree scalars) rather than raising, so a whole generated set
can be pooled without guarding each call.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

# Default Sholl shell count. Profiles use PER-TREE shells (see `sholl_intersection_profile`).
SHOLL_N_SHELLS = 32


def _pos_to_xyz(pos) -> np.ndarray:
    arr = np.asarray(pos, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), mode="constant", constant_values=0.0)
    return arr[:3]


def branch_length_values(G: nx.Graph) -> np.ndarray:
    """Per-edge Euclidean branch lengths. Empty array if the graph has no edges."""
    if G.number_of_edges() == 0:
        return np.zeros((0,), dtype=np.float64)
    lengths: list[float] = []
    for u, v in G.edges():
        pu = _pos_to_xyz(G.nodes[u].get("pos", np.zeros(3)))
        pv = _pos_to_xyz(G.nodes[v].get("pos", np.zeros(3)))
        lengths.append(float(np.linalg.norm(pu - pv)))
    return np.asarray(lengths, dtype=np.float64)


def _root_tree(G: nx.Graph, root: int) -> Tuple[Dict[int, int], Dict[int, List[int]]]:
    """Return parent and children maps for an undirected tree rooted at root.

    Only the connected component containing `root` is traversed; nodes outside it keep
    empty children lists.
    """
    parent: Dict[int, int] = {}
    children: Dict[int, List[int]] = {n: [] for n in G.nodes}
    stack: List[int] = [root]
    seen = {root}
    while stack:
        u = stack.pop()
        for v in G.neighbors(u):
            if v in seen:
                continue
            seen.add(v)
            parent[v] = u
            children[u].append(v)
            stack.append(v)
    return parent, children


def bifurcation_angle_values(
    G: nx.Graph,
    *,
    root: int | None = None,
    degrees: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """Collect all pairwise sibling-branch angles at bifurcation nodes.

    Rooting assigns a parent direction so child branches are well defined. Returns an
    empty array if no bifurcations are present.
    """
    if G.number_of_nodes() == 0:
        return np.zeros((0,), dtype=np.float64)

    if root is None:
        root = G.graph.get("root")
    if root is None or root not in G.nodes:
        raise ValueError("Root node is required for bifurcation angle computation.")

    _parent, children = _root_tree(G, root)
    angles: list[float] = []

    for u, ch in children.items():
        if len(ch) < 2:
            continue
        pu = _pos_to_xyz(G.nodes[u].get("pos", np.zeros(3)))
        vecs: list[np.ndarray] = []
        for c in ch:
            pc = _pos_to_xyz(G.nodes[c].get("pos", np.zeros(3)))
            v = pc - pu
            if float(np.linalg.norm(v)) > eps:
                vecs.append(v)
        if len(vecs) < 2:
            continue

        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                v1 = vecs[i]
                v2 = vecs[j]
                denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
                if denom <= eps:
                    continue
                cos = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
                ang = float(math.acos(cos))
                if degrees:
                    ang = float(math.degrees(ang))
                angles.append(ang)

    return np.asarray(angles, dtype=np.float64)


# --- root-anchored geometry + topology (ported from dendrite_gen) --------------------


def _resolve_root_or_none(G: nx.Graph, root: int | None) -> int | None:
    if root is None:
        root = G.graph.get("root")
    if root is None or root not in G.nodes:
        return None
    return int(root)


def _edge_length(G: nx.Graph, u: int, v: int) -> float:
    pu = _pos_to_xyz(G.nodes[u].get("pos", np.zeros(3)))
    pv = _pos_to_xyz(G.nodes[v].get("pos", np.zeros(3)))
    return float(np.linalg.norm(pu - pv))


def degree_values(G: nx.Graph, *, root: int | None = None) -> np.ndarray:
    """Per-**non-root** node degree.

    The root is excluded because it is not comparable across corpora: a neuron soma is a
    legitimate high-degree hub (median 8, max 16) while a tree root has degree 1.
    Excluding it, real ground truth is startlingly sharp -- the pooled non-root degree
    distribution on neurons is exactly {1: 0.5655, 3: 0.4345}, with zero mass at degree
    0, 2, 4, 5 and 6. Any other degree in a generated graph is a structural violation.
    """
    root = _resolve_root_or_none(G, root)
    return np.asarray(
        [float(d) for k, d in G.degree() if k != root], dtype=np.float64
    )


def path_length_to_root_values(G: nx.Graph, *, root: int | None = None) -> np.ndarray:
    """Per-non-root-node path length from root along the tree (Euclidean-weighted)."""
    root = _resolve_root_or_none(G, root)
    if root is None or G.number_of_nodes() < 2:
        return np.zeros((0,), dtype=np.float64)
    path_map = nx.single_source_dijkstra_path_length(
        G, root, weight=lambda u, v, _d: _edge_length(G, u, v)
    )
    return np.asarray(
        [float(path_map[n]) for n in G.nodes() if n != root and n in path_map],
        dtype=np.float64,
    )


def radial_distance_to_root_values(G: nx.Graph, *, root: int | None = None) -> np.ndarray:
    """Per-non-root-node straight-line (Euclidean) distance from the root position."""
    root = _resolve_root_or_none(G, root)
    if root is None or G.number_of_nodes() < 2:
        return np.zeros((0,), dtype=np.float64)
    root_pos = _pos_to_xyz(G.nodes[root].get("pos", np.zeros(3)))
    return np.asarray(
        [
            float(np.linalg.norm(_pos_to_xyz(G.nodes[n].get("pos", np.zeros(3))) - root_pos))
            for n in G.nodes()
            if n != root
        ],
        dtype=np.float64,
    )


def contraction_ratio_values(
    G: nx.Graph, *, root: int | None = None, eps: float = 1e-12
) -> np.ndarray:
    """Per-leaf contraction = radial(root->leaf) / path(root->leaf), in (0, 1].

    A robust "tortuosity" surrogate for critical (branch-point-only) trees, where
    per-branch geometric tortuosity is ~1 by construction. 1 means a straight reach from
    the root; smaller means the dendrite wanders before terminating.
    """
    root = _resolve_root_or_none(G, root)
    if root is None or G.number_of_nodes() < 2:
        return np.zeros((0,), dtype=np.float64)
    _parent, children = _root_tree(G, root)
    leaves = [n for n, ch in children.items() if len(ch) == 0 and n != root]
    if not leaves:
        return np.zeros((0,), dtype=np.float64)
    path_map = nx.single_source_dijkstra_path_length(
        G, root, weight=lambda u, v, _d: _edge_length(G, u, v)
    )
    root_pos = _pos_to_xyz(G.nodes[root].get("pos", np.zeros(3)))
    out: list[float] = []
    for n in leaves:
        path = float(path_map.get(n, 0.0))
        if path <= eps:
            continue
        radial = float(np.linalg.norm(_pos_to_xyz(G.nodes[n].get("pos", np.zeros(3))) - root_pos))
        out.append(min(radial / path, 1.0))
    return np.asarray(out, dtype=np.float64)


def branch_order_values(G: nx.Graph, *, root: int | None = None) -> np.ndarray:
    """Per-non-root-node branch order (number of bifurcations on the root->node path).

    Order increments only when passing through a node of degree >= 3 (a true branch
    point), so degree-2 chain nodes are transparent to it.

    NOTE ON WHAT THIS MEASURES: our datasets are strictly binary away from the root with
    zero degree-2 non-root nodes, so every non-root internal node increments and
    ``branch_order == hop_depth - 1`` exactly. This is therefore the node-depth
    distribution. On `trees_genus_d{10,15,20}` it is concentrated at the preprocessing
    depth cap (97% of d10 trees fall in {9,10,11}), so there it partly measures "did the
    generator reach the dataset's ceiling" rather than free morphological depth.

    Unlike the dendrite_gen original this skips nodes unreachable from the root instead
    of raising `KeyError`. dendrite_gen never hits that because its generator emits trees
    by construction; SemlaFlow can emit fragments, and the unguarded version fails on
    99/200 graphs at only 1% edge dropout. `sanitise_graph` also prevents it -- this is
    defence in depth, because it is the one function that turns a malformed graph into a
    crash rather than a bad number.
    """
    root = _resolve_root_or_none(G, root)
    if root is None or G.number_of_nodes() < 2:
        return np.zeros((0,), dtype=np.float64)
    order = {root: 0}
    stack = [root]
    seen = {root}
    while stack:
        u = stack.pop()
        parent_order = order[u]
        for v in G.neighbors(u):
            if v in seen:
                continue
            seen.add(v)
            order[v] = parent_order + 1 if (G.degree(u) >= 3 and u != root) else parent_order
            stack.append(v)
    return np.asarray(
        [float(order[n]) for n in G.nodes() if n != root and n in order],
        dtype=np.float64,
    )


def _postorder_subtree_stats(
    G: nx.Graph, root: int
) -> tuple[dict[int, list[int]], dict[int, int], dict[int, int]]:
    """Bottom-up subtree leaf counts and Strahler numbers for a rooted tree.

    Iterative (reverse pre-order gives a valid post-order) to avoid recursion limits on
    deep path-like trees.
    """
    _parent, children = _root_tree(G, root)
    pre: list[int] = []
    stack = [root]
    while stack:
        u = stack.pop()
        pre.append(u)
        for c in children[u]:
            stack.append(c)
    subtree_leaves: dict[int, int] = {}
    strahler: dict[int, int] = {}
    for u in reversed(pre):
        ch = children[u]
        if not ch:
            subtree_leaves[u] = 1
            strahler[u] = 1
            continue
        subtree_leaves[u] = sum(subtree_leaves[c] for c in ch)
        child_orders = [strahler[c] for c in ch]
        m = max(child_orders)
        strahler[u] = m + 1 if child_orders.count(m) >= 2 else m
    return children, subtree_leaves, strahler


def strahler_number(G: nx.Graph, *, root: int | None = None) -> float:
    """Horton-Strahler order of the whole rooted tree. nan if empty/unrooted."""
    root = _resolve_root_or_none(G, root)
    if root is None or G.number_of_nodes() == 0:
        return float("nan")
    _children, _leaves, strahler = _postorder_subtree_stats(G, root)
    return float(strahler[root])


def partition_asymmetry(
    G: nx.Graph, *, root: int | None = None, eps: float = 1e-12
) -> float:
    """Van Pelt tree asymmetry index.

    Mean over branch points of the local partition asymmetry |r-s|/(r+s-2) of the subtree
    leaf counts (r, s); a partition with r+s == 2 contributes 0. For multifurcations,
    averaged over all child pairs. Per-tree scalar in [0, 1]; nan if there is no
    qualifying branch point. 0 = every fork splits evenly, 1 = caterpillar.
    """
    root = _resolve_root_or_none(G, root)
    if root is None or G.number_of_nodes() == 0:
        return float("nan")
    children, subtree_leaves, _strahler = _postorder_subtree_stats(G, root)
    node_vals: list[float] = []
    for _u, ch in children.items():
        if len(ch) < 2:
            continue
        counts = [subtree_leaves[c] for c in ch]
        pair_vals: list[float] = []
        for i in range(len(counts)):
            for j in range(i + 1, len(counts)):
                r, s = counts[i], counts[j]
                denom = r + s - 2
                pair_vals.append(0.0 if denom <= 0 else abs(r - s) / float(denom))
        if pair_vals:
            node_vals.append(float(np.mean(pair_vals)))
    return float(np.mean(node_vals)) if node_vals else float("nan")


def sholl_intersection_profile(
    G: nx.Graph,
    *,
    root: int | None = None,
    radii: np.ndarray | None = None,
    n_shells: int = SHOLL_N_SHELLS,
) -> tuple[np.ndarray, np.ndarray]:
    """Sholl analysis: edges crossing each concentric sphere centred at the root.

    An edge (u,v) crosses radius r iff min(d_u,d_v) < r <= max(d_u,d_v), where d_* is the
    node's radial distance from the root.

    With ``radii=None`` (the default, and what the metrics use) the shells are PER-TREE:
    ``n_shells`` evenly spaced over (0, this tree's own max radial extent]. Crossing
    counts are inherently scale-invariant, but a *shared* grid breaks that by
    undersampling small trees -- dendrite_gen measured sholl_peak 7/13/13 on one tree
    scaled x1/x1.5/x3 with a shared grid, where per-tree shells give 14/14/14. Nothing
    here compares profiles pointwise, so a shared grid buys nothing.
    """
    root = _resolve_root_or_none(G, root)
    if root is None or G.number_of_edges() == 0:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    root_pos = _pos_to_xyz(G.nodes[root].get("pos", np.zeros(3)))
    dist = {
        n: float(np.linalg.norm(_pos_to_xyz(G.nodes[n].get("pos", np.zeros(3))) - root_pos))
        for n in G.nodes()
    }
    if radii is None:
        max_r = max(dist.values()) if dist else 0.0
        if max_r <= 0.0:
            return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
        radii = np.linspace(0.0, max_r, int(n_shells) + 1, dtype=np.float64)[1:]
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    counts = np.zeros(radii.shape, dtype=np.float64)
    for u, v in G.edges():
        lo, hi = sorted((dist[u], dist[v]))
        counts += ((radii > lo) & (radii <= hi)).astype(np.float64)
    return radii, counts


def sholl_summary(
    G: nx.Graph,
    *,
    root: int | None = None,
    radii: np.ndarray | None = None,
    n_shells: int = SHOLL_N_SHELLS,
) -> dict[str, float]:
    """Reduce a Sholl profile to three per-tree scalars. nan-filled on degenerate trees.

      sholl_peak            maximum intersection count
      sholl_critical_radius radius of the peak, normalised by THIS TREE's radial extent
      sholl_auc             area under the profile (trapezoid)

    The critical radius divides by the tree's own extent, NOT by ``r.max()`` (the outer
    edge of the shell grid). The two coincide for per-tree shells but diverge for a
    shared grid, where the grid's outer edge is the whole set's max -- which turned the
    ratio into an absolute-size feature rather than the documented shape fraction.
    """
    out = {
        "sholl_peak": float("nan"),
        "sholl_critical_radius": float("nan"),
        "sholl_auc": float("nan"),
    }
    r, counts = sholl_intersection_profile(G, root=root, radii=radii, n_shells=n_shells)
    if r.size == 0 or counts.size == 0 or float(counts.max()) <= 0.0:
        return out
    peak_idx = int(np.argmax(counts))
    own = radial_distance_to_root_values(G, root=root)
    own_rmax = float(own.max()) if own.size else 0.0
    out["sholl_peak"] = float(counts.max())
    out["sholl_critical_radius"] = (
        float(r[peak_idx] / own_rmax) if own_rmax > 0 else float("nan")
    )
    _trapezoid = getattr(np, "trapezoid", np.trapz)
    out["sholl_auc"] = float(_trapezoid(counts, r))
    return out
