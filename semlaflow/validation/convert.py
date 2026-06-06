"""GeometricMol -> networkx.Graph adapter and shared orientation/root rules.

SEMLA is fully E(3)-equivariant (trained with random rotation augmentation), so
generated neuron graphs come out in arbitrary global orientation and carry no soma
node. To make structural statistics comparable to the ground-truth graphs (which
live in their original SWC frame, soma at index 0), we:

  * measure orientation-dependent extents relative to each graph's own PCA principal
    axis (`pca_axis`) rather than world x/y/z -- rotation invariant; and
  * choose a root via `choose_root`: an unambiguous high-degree branch hub when one
    exists (the soma normally has the most primary neurites), else the geometric
    "base" node (`pca_base_root`).

These rules are defined once here and reused by `dist_metrics` and `plot`.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def _coords_array(mol, coord_scale: float = 1.0) -> np.ndarray:
    """Return an (N, 3) float64 coord array from a GeometricMol, optionally rescaled."""
    coords = mol.coords.detach().cpu().numpy().astype(np.float64)
    if coord_scale != 1.0:
        coords = coords * float(coord_scale)
    return coords


def pca_axis(coords: np.ndarray) -> np.ndarray:
    """Unit principal axis (top eigenvector of the centred coord covariance).

    Falls back to +z for degenerate inputs (<2 points or zero variance).
    """
    pts = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 2:
        return np.array([0.0, 0.0, 1.0])
    centred = pts - pts.mean(axis=0, keepdims=True)
    cov = centred.T @ centred
    if not np.all(np.isfinite(cov)) or np.allclose(cov, 0.0):
        return np.array([0.0, 0.0, 1.0])
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return axis / norm


def pca_base_root(coords: np.ndarray) -> int:
    """Index of the node with the minimum projection onto the PCA principal axis."""
    pts = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] == 0:
        return 0
    axis = pca_axis(pts)
    return int(np.argmin(pts @ axis))


def choose_root(G: nx.Graph, coords: np.ndarray) -> int:
    """Pick a root node for a neuron graph.

    Prefer an unambiguous branch hub -- a *unique* node whose degree is the strict
    maximum and is >= 3 (soma-like). Otherwise fall back to the geometric base node
    (min projection on the PCA principal axis). Node ids index `coords` row-for-row.
    """
    if G.number_of_nodes() == 0:
        return 0
    degrees = dict(G.degree())
    max_deg = max(degrees.values())
    if max_deg >= 3:
        hubs = [n for n, d in degrees.items() if d == max_deg]
        if len(hubs) == 1:
            return int(hubs[0])
    return pca_base_root(coords)


def geometric_mol_to_nx(mol, *, coord_scale: float = 1.0) -> nx.Graph:
    """Build an undirected networkx graph from a GeometricMol.

    Node `i` carries `pos = coords[i] * coord_scale` (length-3 np.float64); edges come
    from `mol.bond_indices`; `G.graph["root"]` is set via `choose_root`. Pass
    `coord_scale=NEURON_COORDS_STD_DEV` for generated (standardised) mols to recover
    physical microns; leave `coord_scale=1.0` for ground-truth mols (already physical).
    """
    coords = _coords_array(mol, coord_scale)
    n = coords.shape[0]

    G = nx.Graph()
    for i in range(n):
        G.add_node(i, pos=coords[i])

    bonds = mol.bond_indices.detach().cpu().numpy().astype(int)
    for a, b in bonds.reshape(-1, 2):
        if a == b:
            continue
        G.add_edge(int(a), int(b))

    G.graph["root"] = choose_root(G, coords)
    return G
