"""Plot helpers for generated/GT neuron-tree visualisation (geometry-only port).

Produces a grid figure (one row per graph, one column per viewing angle) rendered in
3D. Ported from dendrite_gen's `validation/plot.py`, trimmed to the multi-azimuth grid.

Because SEMLA is E(3)-equivariant (graphs arbitrarily oriented), azimuths orbit each
graph's own PCA principal axis: we rotate that axis to +z before plotting so sweeping
the matplotlib azimuth is meaningful and consistent across samples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

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

# Sanitisation-overlay palette. Colour carries membership of the critical tree the
# morphometrics scored, size carries level of detail: same colour = same tree, smaller =
# detail contraction collapsed, grey = the metrics never saw it, orange = an edge the MST
# cut, violet = a branch binarisation dropped. Only genuine defects are visually loud.
FRAGMENT_NODE_COLOR = "#9e9e9e"   # outside the largest connected component
# Dashed, not just paler: the default kept-edge `lightgray` is #d3d3d3, so any "slightly
# lighter grey" is indistinguishable at figure scale and a detached edge would read as a
# real long branch. The linestyle is what actually separates them.
FRAGMENT_EDGE_COLOR = "#9e9e9e"
FRAGMENT_EDGE_STYLE = (0, (4, 3))
EXCESS_EDGE_COLOR = "#e07b39"     # inside the LCC, cut by the MST (cycle-closing)
# Dropped by `_binarise` at a multifurcation. Solid and full-size, not shrunk like a
# contraction: the generator emitting a trifurcation is a real defect, not a detail the
# critical tree folds away. Violet reads as distinct from both the fragment grey and the
# excess orange at figure scale.
PRUNED_NODE_COLOR = "#7b52ab"
PRUNED_EDGE_COLOR = "#7b52ab"
CONTRACTED_SIZE_FRAC = 0.45       # non-root degree-2 nodes, collapsed to a critical tree


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


def _overlay_styles(G: nx.Graph, provenance: dict, node_color: str):
    """Per-node colours/sizes and a per-edge colour map from a `sanitise_provenance` dict.

    Node arrays are built in ``G.nodes()`` order, matching how `_plot_graph` stacks `pts` --
    that is insertion order, NOT sorted id, so indexing by id would mis-colour any graph
    whose nodes were not added in order.
    """
    colors, sizes = [], []
    for n in G.nodes():
        if n in provenance["fragment_nodes"]:
            colors.append(FRAGMENT_NODE_COLOR)
            sizes.append(NODE_SIZE)
        elif n in provenance["pruned_nodes"]:
            colors.append(PRUNED_NODE_COLOR)
            sizes.append(NODE_SIZE)
        elif n in provenance["contracted_nodes"]:
            colors.append(node_color)
            sizes.append(NODE_SIZE * CONTRACTED_SIZE_FRAC)
        else:
            colors.append(node_color)
            sizes.append(NODE_SIZE)

    edge_colors, edge_styles = {}, {}
    for key in provenance["excess_edges"]:
        edge_colors[key] = EXCESS_EDGE_COLOR
    for key in provenance["pruned_edges"]:
        edge_colors[key] = PRUNED_EDGE_COLOR
    for key in provenance["fragment_edges"]:
        edge_colors[key] = FRAGMENT_EDGE_COLOR
        edge_styles[key] = FRAGMENT_EDGE_STYLE
    return colors, sizes, edge_colors, edge_styles


def _plot_graph(
    ax,
    G: nx.Graph,
    title: str,
    *,
    node_color: str,
    edge_color: str,
    show_nodes: bool = True,
    show_edges: bool = True,
    node_colors=None,
    node_sizes=None,
    edge_colors: Optional[dict] = None,
    edge_styles: Optional[dict] = None,
) -> None:
    """Draw one graph into a 3D axes.

    ``node_colors`` / ``node_sizes`` are per-node sequences in ``G.nodes()`` order and
    override the scalar ``node_color`` / `NODE_SIZE`. ``edge_colors`` and ``edge_styles`` map
    ``frozenset({u, v})`` to a colour / linestyle, falling back to ``edge_color`` and solid.
    All four come from `_overlay_styles`; passing none of them reproduces the original
    single-colour drawing.
    """
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
                color=(edge_colors or {}).get(frozenset((u, v)), edge_color),
                linestyle=(edge_styles or {}).get(frozenset((u, v)), "-"),
                linewidth=lw,
            )
    if show_nodes:
        # Deliberately ONE scatter call: `tests/validation_plots.py` reads
        # `ax.collections[-1]` to find the node artist, and edges above are Line2D rather
        # than a collection, so this stays the last collection on the axes.
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            s=NODE_SIZE if node_sizes is None else node_sizes,
            c=node_color if node_colors is None else node_colors,
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


def _add_overlay_legend(fig, keep_color: str) -> None:
    """One figure-level key for the sanitisation overlay (a 7x3 grid has 21 axes)."""
    from matplotlib.lines import Line2D

    def marker(color, size, label):
        return Line2D([], [], linestyle="none", marker="o", markersize=size,
                      markerfacecolor=color, markeredgecolor="k", markeredgewidth=0.3,
                      label=label)

    handles = [
        marker(keep_color, 6, "kept (critical tree)"),
        marker(keep_color, 6 * (CONTRACTED_SIZE_FRAC ** 0.5), "contracted (deg-2)"),
        marker(PRUNED_NODE_COLOR, 6, "pruned (multifurcation)"),
        marker(FRAGMENT_NODE_COLOR, 6, "fragment (outside LCC)"),
        Line2D([], [], color=EXCESS_EDGE_COLOR, linewidth=2.0, label="excess edge (MST-cut)"),
        Line2D([], [], color=FRAGMENT_EDGE_COLOR, linewidth=1.4,
               linestyle=FRAGMENT_EDGE_STYLE, label="fragment edge"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8)


def plot_graph_grid_angles(
    graphs: list[nx.Graph],
    *,
    out_dir: Optional[Path] = None,
    stem: str = "",
    file_tag: str = "",
    angles: Iterable[tuple[float, float]] = DEFAULT_ANGLES,
    align_pca: bool = True,
    title_prefix: str = "",
    per_graph_titles: Optional[list[str]] = None,
    node_color: str = PRED_COLOR,
    per_graph_colors: Optional[list[str]] = None,
    edge_color: str = "lightgray",
    max_graphs: int = 8,
    sanitise_overlay: bool = False,
):
    """Build a single figure: one row per graph, one column per (elev, azim) angle.

    When ``align_pca`` is set, each graph is rotated so its PCA principal axis points
    along +z before rendering, so the azimuth sweep orbits that axis consistently.

    ``per_graph_titles`` overrides ``title_prefix`` per row -- used to label rows by cell
    class (``Gen 5P-IT``) or by matched pair. ``per_graph_colors`` does the same for
    ``node_color``, so a grid that interleaves generated and ground-truth rows can colour
    them apart rather than relying on the title alone. Shorter lists fall back to the
    scalar argument for the remaining rows.

    ``sanitise_overlay`` colours each graph by what `sanitise_graph` would do to it, so the
    critical tree the morphometrics actually scored is visible inside the raw emission:
    fragments outside the largest component go grey, edges the MST cuts go orange,
    branches binarisation drops go violet, and degree-2 nodes that contraction collapses
    shrink. Off by default, so the plain single-colour drawing is unchanged for any caller
    that does not ask for it.

    With ``out_dir`` set, saves to ``out_dir/<stem>_<file_tag>.png``; with ``out_dir=None``
    nothing is written and ``out_path`` comes back as ``None`` (the in-training logging path
    wants the figure only). Returns ``(fig, out_path)``. The figure is NOT closed here so
    callers may log it; release it with ``plt.close`` afterwards.
    """
    angles = list(angles)
    graphs = list(graphs)[:max_graphs]
    titles = list(per_graph_titles or [])
    colors = list(per_graph_colors or [])
    n_rows = max(len(graphs), 1)
    n_cols = max(len(angles), 1)
    fig = plt.figure(figsize=(n_cols * 3.2, n_rows * 3.0))
    for r, G in enumerate(graphs):
        Gp = align_pca_to_z(G) if align_pca else G
        n_nodes = G.number_of_nodes()
        label = titles[r] if r < len(titles) else title_prefix
        row_color = colors[r] if r < len(colors) else node_color

        node_colors = node_sizes = edge_colors = edge_styles = None
        if sanitise_overlay:
            from .sanitise import sanitise_provenance

            # Computed once per row and reused across all angle columns. `align_pca_to_z`
            # copies the graph and rewrites only `pos`, so ids and iteration order are
            # identical between G and Gp and the styles carry over unchanged.
            node_colors, node_sizes, edge_colors, edge_styles = _overlay_styles(
                G, sanitise_provenance(G), row_color
            )

        for c, (elev, azim) in enumerate(angles):
            ax = fig.add_subplot(n_rows, n_cols, r * n_cols + c + 1, projection="3d")
            _plot_graph(
                ax,
                Gp,
                _nice_title(label, n_nodes, f"az{int(azim)}"),
                node_color=row_color,
                edge_color=edge_color,
                node_colors=node_colors,
                node_sizes=node_sizes,
                edge_colors=edge_colors,
                edge_styles=edge_styles,
            )
            ax.title.set_fontsize(9)  # grid is dense; module default (24) overlaps
            ax.view_init(elev=elev, azim=azim)
            ax.set_axis_off()
    if sanitise_overlay and graphs:
        _add_overlay_legend(fig, node_color if not colors else colors[0])
        # tight_layout ignores figure-level legends, so reserve a strip for it explicitly
        # or the bottom row of axes draws over the key.
        legend_frac = min(0.06, 1.2 / (n_rows * 3.0))
        fig.tight_layout(rect=(0.0, legend_frac, 1.0, 1.0))
    else:
        fig.tight_layout()
    if out_dir is None:
        return fig, None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}_{file_tag}.png"
    fig.savefig(out_path, dpi=150)
    return fig, out_path
