"""Structural metrics for rooted tree graphs (geometry-only port).

Ported from dendrite_gen's `validation/structural_metrics.py`, trimmed to the
extractors needed for distribution metrics: per-edge branch lengths and per-bifurcation
sibling-branch angles. The tree-edit-distance (zss) and persistence (persim) paths are
intentionally dropped.

Assumes graphs are "critical" skeletons (branch/termination points only), so per-edge
lengths correspond to branch segment lengths.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


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
