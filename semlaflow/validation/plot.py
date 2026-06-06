"""Plot helpers for generated/GT neuron-tree visualisation (geometry-only port).

Produces a grid figure (one row per graph, one column per viewing angle) rendered in
3D. Ported from dendrite_gen's `validation/plot.py`, trimmed to the multi-azimuth grid.

Because SEMLA is E(3)-equivariant (graphs arbitrarily oriented), azimuths orbit each
graph's own PCA principal axis: we rotate that axis to +z before plotting so sweeping
the matplotlib azimuth is meaningful and consistent across samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"axes.labelsize": 24, "axes.titlesize": 24})
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402

from .convert import pca_axis  # noqa: E402

DEFAULT_ANGLES = [(20, 30), (20, 120), (20, 210)]
GT_COLOR = "#1f77b4"
PRED_COLOR = "#8b1e3f"
NODE_SIZE = 18
EDGE_WIDTH = 1.4
SKELETON_WIDTH = 1.8


def _pos_to_xyz(pos) -> np.ndarray:
    arr = np.asarray(pos, dtype=float).reshape(-1)
    if arr.size < 3:
        arr = np.pad(arr, (0, 3 - arr.size), mode="constant", constant_values=0.0)
    return arr[:3]


def _graph_positions(G: nx.Graph) -> dict[int, np.ndarray]:
    return {n: _pos_to_xyz(G.nodes[n].get("pos", np.zeros(3))) for n in G.nodes()}


def _set_axes_tight(ax, pts: np.ndarray, pad_frac: float = 0.04) -> None:
    if pts.size == 0:
        return
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    ranges = maxs - mins
    pad = np.maximum(ranges * pad_frac, 1e-3)
    mins = mins - pad
    maxs = maxs + pad
    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_box_aspect(maxs - mins)


def _plot_graph(
    ax,
    G: nx.Graph,
    title: str,
    *,
    node_color: str,
    edge_color: str,
    show_nodes: bool = True,
    show_edges: bool = True,
) -> None:
    pos = _graph_positions(G)
    if not pos:
        ax.set_title(title)
        return
    pts = np.stack(list(pos.values()), axis=0)
    if show_edges:
        lw = SKELETON_WIDTH if not show_nodes else EDGE_WIDTH
        for u, v in G.edges():
            p0 = pos[u]
            p1 = pos[v]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                [p0[2], p1[2]],
                color=edge_color,
                linewidth=lw,
            )
    if show_nodes:
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=NODE_SIZE,
            c=node_color,
            edgecolors="k",
            linewidths=0.3,
        )
    ax.set_title(title)
    _set_axes_tight(ax, pts)


def _nice_title(label: str, n_nodes: int, suffix: str = "") -> str:
    suffix_str = f" - {suffix}" if suffix else ""
    return f"{label}{suffix_str} (n={n_nodes})"


def _rotation_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix R (3x3) with R @ a == b for unit vectors a, b (Rodrigues)."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-8:  # already aligned
        return np.eye(3)
    if c < -1.0 + 1e-8:  # antipodal: rotate 180 deg about any perpendicular axis
        p = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, np.array([0.0, 1.0, 0.0]))
        p = p / (np.linalg.norm(p) + 1e-12)
        return 2.0 * np.outer(p, p) - np.eye(3)
    v = np.cross(a, b)
    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def align_pca_to_z(G: nx.Graph) -> nx.Graph:
    """Return a copy of ``G`` rotated so its PCA principal axis points along +z.

    Plotting this copy and sweeping the matplotlib azimuth then orbits the graph's
    principal axis. The original graph's positions are left untouched.
    """
    pos = _graph_positions(G)
    if not pos:
        return G
    pts = np.stack(list(pos.values()), axis=0)
    R = _rotation_align(pca_axis(pts), np.array([0.0, 0.0, 1.0]))
    H = G.copy()
    for n in H.nodes():
        H.nodes[n]["pos"] = R @ _pos_to_xyz(H.nodes[n].get("pos", np.zeros(3)))
    if "root" in G.graph:
        H.graph["root"] = G.graph["root"]
    return H


def plot_graph_grid_angles(
    graphs: list[nx.Graph],
    *,
    out_dir: Path,
    stem: str,
    file_tag: str,
    angles: Iterable[tuple[float, float]] = DEFAULT_ANGLES,
    align_pca: bool = True,
    title_prefix: str = "",
    node_color: str = PRED_COLOR,
    edge_color: str = "lightgray",
    max_graphs: int = 8,
):
    """Build a single figure: one row per graph, one column per (elev, azim) angle.

    When ``align_pca`` is set, each graph is rotated so its PCA principal axis points
    along +z before rendering, so the azimuth sweep orbits that axis consistently.
    Saves to ``out_dir/<stem>_<file_tag>.png`` and returns (fig, out_path). The figure
    is NOT closed here so callers may log it; release it with ``plt.close`` afterwards.
    """
    angles = list(angles)
    graphs = list(graphs)[:max_graphs]
    n_rows = max(len(graphs), 1)
    n_cols = max(len(angles), 1)
    fig = plt.figure(figsize=(n_cols * 3.2, n_rows * 3.0))
    for r, G in enumerate(graphs):
        Gp = align_pca_to_z(G) if align_pca else G
        n_nodes = G.number_of_nodes()
        for c, (elev, azim) in enumerate(angles):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, projection="3d")
            _plot_graph(
                ax,
                Gp,
                _nice_title(title_prefix, n_nodes, f"az{int(azim)}"),
                node_color=node_color,
                edge_color=edge_color,
            )
            ax.title.set_fontsize(9)  # grid is dense; module default (24) overlaps
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
    fig.tight_layout()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{file_tag}.png"
    fig.savefig(out_path, dpi=150)
    return fig, out_path
