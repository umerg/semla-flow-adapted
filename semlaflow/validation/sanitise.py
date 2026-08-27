"""Structural health reporting and graph sanitisation for generated neuron/tree graphs.

dendrite_gen's generator emits trees *by construction*. SemlaFlow's bond head predicts
N**2 independent edge logits, so a generated graph can be disconnected, cyclic,
multifurcating, or carry degree-2 chain nodes -- none of which the ported morphometric
functions handle well (`branch_order_values` raises `KeyError` on an unreachable node;
`_root_tree` counts every node outside the root's component as a *leaf*).

This module splits that problem in two:

  * `graph_health` measures the structural violations. Every key it returns is exactly 0.0
    (or 1.0 for `lcc_node_frac`) on real ground truth for both the neuron and the tree
    corpora, so any deviation is unambiguously a generator failure -- the same "sanity
    record" logic as dendrite_gen's `morpho_gt_nan_frac`. All but one key are measured on
    the **raw** graph; `multifurcation_frac` is measured one stage in, immediately before
    the repair it describes, for the reason given in `graph_health`'s docstring.
  * `sanitise_graph` reduces the graph to the well-defined critical tree that every
    morphometric is written for: largest component, spanning tree, binarised, contracted.

**Order matters**: health must be measured before sanitisation, or sanitisation hides
exactly the defects it is repairing. `dist_metrics.compute_distribution_metrics` does
this for you.

Sanitisation is deliberately a *choice to score the sanitised object*. It makes the
morphometrics answer "given a valid tree, is the morphology right?", and leaves
`graph_health` as the only answer to "is it a valid tree?".
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from .convert import choose_root
from .structural_metrics import _root_tree

__all__ = ["graph_health", "sanitise_graph", "axial_axis", "HEALTH_KEYS"]


# Keys returned by `graph_health`, in a stable order for printing/JSON.
HEALTH_KEYS = (
    # Incidence -- fraction of graphs exhibiting the violation at all.
    "disconnected_frac",
    "multifurcation_frac",
    "isolated_node_frac",
    "cycle_frac",
    "non_critical_node_frac",
    # Magnitude -- how badly, averaged over graphs. Each pairs with an incidence key
    # above; incidence says "how often", magnitude says "how much".
    "excess_edge_frac",      # pairs with cycle_frac
    "degree2_node_frac",     # pairs with non_critical_node_frac
    "lcc_node_frac",         # pairs with disconnected_frac
)

# A node of this degree or higher is a branch point. Ground truth away from the root is
# strictly binary: the pooled non-root degree distribution on real neurons is exactly
# {1: 0.5655, 3: 0.4345} with 0.00000 mass at every other degree.
_MAX_NONROOT_DEGREE = 3


def _degrees(G: nx.Graph) -> dict:
    return dict(G.degree())


def graph_health(graphs: list[nx.Graph]) -> dict[str, float]:
    """Structural violation rates over a set of raw (unsanitised) graphs.

    On real ground truth every value here is 0.0 except `lcc_node_frac`, which is 1.0 --
    verified over 2527 `neurons_conditional` and 337 `trees_genus_d10` val graphs. So a
    non-zero reading is always a generator failure, never a data property.

    Keys split into incidence ("how many graphs are affected") and magnitude ("how badly,
    per graph"). Both are needed: on real generations 13.5% of graphs contain a degree-2
    node but only 0.37% of nodes are degree-2, and the distortion tracks the latter.

      disconnected_frac      fraction of graphs that are not connected
      multifurcation_frac    fraction whose *spanning tree* has a non-root node of
                             degree > 3 -- see the stage note below
      isolated_node_frac     fraction containing a degree-0 node
      cycle_frac             fraction with at least one independent cycle
      non_critical_node_frac fraction with a non-root node of degree exactly 2
      excess_edge_frac       mean over graphs of (E - (N - C)) / E -- cycle magnitude
      degree2_node_frac      mean fraction of non-root nodes of degree exactly 2
      lcc_node_frac          mean |largest component| / N -- fragmentation magnitude

    **`multifurcation_frac` is measured one stage in, not on the raw graph.** Every other
    key here describes the graph as generated; this one describes what `_binarise` is
    about to repair, so it is read off `_spanning_stage`'s output -- after the largest
    component and the MST, immediately before binarisation. The two stages differ a lot:
    over 400 graphs of `semla_tmd_samples` the raw rate is 0.0700 and the post-MST rate is
    0.0225, because a cycle-closing edge inflates a node's degree and the MST then cuts
    it. Reporting the raw number would credit binarisation with repairs the MST already
    made, and would gate model selection on defects that never reach the scored object.
    Both readings are still 0.0000 on all four GT corpora.

    There is deliberately no node-level magnitude for `multifurcation_frac`: the
    `degree_w1` marginal in `dist_metrics` already compares the full degree distribution,
    which subsumes it.

    Why `multifurcation_frac` excludes the root: including it reads **99.6%** on real
    neuron ground truth, because the soma is a legitimate high-degree hub (per-graph max
    degree: median 8, max 16). Excluding the root it is 0.0000 on GT for every corpus.

    This is *not* circular even though `choose_root` prefers the unique maximum-degree
    node (and so absorbs an offending hub 97.4% of the time): the second-highest node
    degree, which involves no root selection at all, flags the identical set of graphs.
    `tests/validation_metrics.py` pins that equivalence. The root-excluding form is the
    one shipped because the second-highest form would miss a lone degree-4 node on the
    tree corpora, where the root has degree 1.
    """
    n = len(graphs)
    if n == 0:
        return {k: float("nan") for k in HEALTH_KEYS}

    disconnected = multifurcating = isolated = cyclic = non_critical = 0
    excess_fracs: list[float] = []
    lcc_fracs: list[float] = []
    deg2_fracs: list[float] = []

    for G in graphs:
        n_nodes = G.number_of_nodes()
        if n_nodes == 0:
            # An empty graph is degenerate on every axis.
            disconnected += 1
            isolated += 1
            lcc_fracs.append(0.0)
            excess_fracs.append(0.0)
            deg2_fracs.append(0.0)
            continue

        n_edges = G.number_of_edges()
        components = list(nx.connected_components(G))
        n_comp = len(components)

        if n_comp > 1:
            disconnected += 1
        lcc_fracs.append(max(len(c) for c in components) / n_nodes)

        # Independent cycles = E - (N - C) for an undirected graph.
        excess = n_edges - (n_nodes - n_comp)
        if excess > 0:
            cyclic += 1
        excess_fracs.append(excess / n_edges if n_edges else 0.0)

        degrees = _degrees(G)
        if min(degrees.values()) == 0:
            isolated += 1

        root = G.graph.get("root")
        non_root = [d for k, d in degrees.items() if k != root]
        if non_root:
            n_deg2 = sum(1 for d in non_root if d == 2)
            if n_deg2:
                non_critical += 1
            deg2_fracs.append(n_deg2 / len(non_root))
        else:
            deg2_fracs.append(0.0)

        # The one key read off the spanning tree rather than the raw graph. It uses that
        # stage's own root, not `G.graph["root"]`, because that is the root binarisation
        # will exempt.
        T, mst_root, _keep = _spanning_stage(G)
        if _has_multifurcation(T, mst_root):
            multifurcating += 1

    def _mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    return {
        "disconnected_frac": disconnected / n,
        "multifurcation_frac": multifurcating / n,
        "isolated_node_frac": isolated / n,
        "cycle_frac": cyclic / n,
        "non_critical_node_frac": non_critical / n,
        "excess_edge_frac": _mean(excess_fracs),
        "degree2_node_frac": _mean(deg2_fracs),
        "lcc_node_frac": _mean(lcc_fracs),
    }


def second_highest_degree_multifurcation_frac(graphs: list[nx.Graph]) -> float:
    """Root-free cross-check for `multifurcation_frac`; see `graph_health`.

    Uses the second-highest node degree, so it never consults `choose_root`. Exists to
    prove the shipped root-excluding metric is not an artefact of root selection: it flags
    the identical set of graphs. Measured at the same stage as the metric it checks --
    `_spanning_stage`'s output -- or the two would disagree purely because one sees the
    cycle-closing edges and the other does not.
    """
    if not graphs:
        return float("nan")
    hits = 0
    for G in graphs:
        if G.number_of_nodes() < 3:
            continue
        T, _root, _keep = _spanning_stage(G)
        ordered = sorted(_degrees(T).values(), reverse=True)
        if len(ordered) > 1 and ordered[1] > _MAX_NONROOT_DEGREE:
            hits += 1
    return hits / len(graphs)


def _largest_component_relabelled(G: nx.Graph) -> tuple[nx.Graph, list]:
    """Largest connected component, relabelled to contiguous ids 0..M-1.

    Relabelling is mandatory, not cosmetic. `pca_base_root` returns
    ``argmin(coords @ axis)`` -- a *positional* index into the coordinate array -- and
    `geometric_mol_to_nx` guarantees node id == coords row to make that valid. A plain
    `G.subgraph(component)` keeps the original labels, so the index `choose_root` returns
    would address a different node, or none at all.

    Returns ``(H, keep)``. ``keep`` is the sorted list of surviving *original* ids, so
    ``keep[i]`` is the original id of new id ``i`` -- the inverse of the remap. Nothing else
    in the pipeline records it, and `sanitise_provenance` needs it to map results back onto
    the raw graph; returning it is cheaper and less fragile than reconstructing it.
    """
    components = list(nx.connected_components(G))
    keep = sorted(max(components, key=len))
    remap = {old: i for i, old in enumerate(keep)}

    H = nx.Graph()
    for old in keep:
        H.add_node(remap[old], pos=np.asarray(G.nodes[old]["pos"], dtype=np.float64))
    for u, v in G.subgraph(keep).edges():
        H.add_edge(remap[u], remap[v])
    return H, keep


def _min_spanning_tree(G: nx.Graph) -> nx.Graph:
    """Minimum spanning tree weighted by Euclidean edge length.

    A spurious edge joins an essentially random pair of nodes and is therefore *long*; a
    real branch segment is short. Measured over injected false-positive edges, the MST
    drops 85-94% of them at 99.5-100% true-edge recall, where a BFS-from-root spanning
    tree keeps ~78% of them (it retains whichever edge the traversal reaches first) and
    displaces real edges to do it.
    """
    if G.number_of_edges() == 0:
        return G.copy()

    for u, v in G.edges():
        G[u][v]["_len"] = float(np.linalg.norm(G.nodes[u]["pos"] - G.nodes[v]["pos"]))
    T = nx.minimum_spanning_tree(G, weight="_len")
    for node, data in G.nodes(data=True):
        if node in T:
            T.nodes[node].update(data)
    for _u, _v, data in T.edges(data=True):
        data.pop("_len", None)
    return T


def _spanning_stage(G: nx.Graph) -> tuple[nx.Graph, int, list]:
    """Everything before binarisation: LCC -> relabel -> MST -> root.

    Returns ``(T, root, keep)``, where ``keep[i]`` is the raw id of relabelled id ``i``
    (see `_largest_component_relabelled`).

    Three callers need this prefix -- `sanitise_graph`, `sanitise_provenance`, and
    `graph_health`, the last of them because `multifurcation_frac` is measured on ``T``.
    Sharing one helper is what stops the three replays from drifting apart.

    ``G`` must be non-empty; every caller already guards that.
    """
    H, keep = _largest_component_relabelled(G)
    T = _min_spanning_tree(H)
    pts = np.stack([T.nodes[k]["pos"] for k in sorted(T.nodes())])
    return T, int(choose_root(T, pts)), keep


def _has_multifurcation(T: nx.Graph, root: int) -> bool:
    """Does any non-root node exceed the binary branching the corpora are cleaned to?"""
    non_root = [d for k, d in T.degree() if k != root]
    return bool(non_root) and max(non_root) > _MAX_NONROOT_DEGREE


def _subtree_nodes(children: dict, u: int) -> list:
    """Every node at or below `u`, given a rooted `children` map."""
    out: list = []
    stack = [u]
    while stack:
        v = stack.pop()
        out.append(v)
        stack.extend(children[v])
    return out


def _binarise(T: nx.Graph, root: int) -> tuple[nx.Graph, set]:
    """Drop the smallest subtrees at every non-root node with more than two children.

    Both GT corpora were binarised at preprocessing by dendrite_gen's
    `clean_trees.normalize_high_degree`, so every GT critical tree is strictly binary away
    from the root -- the pooled non-root degree distribution is exactly {1: 0.5655,
    3: 0.4345}. The MST removes cycles, not degree, so without this a generated
    trifurcation survives into the scored object and metrics like `partition_asymmetry`
    (which averages over all child pairs) compare a strictly binary reference against a
    multi-ary sample.

    dendrite_gen ranked branches by radius, which we do not have on a generated graph.
    Subtree node count is the stand-in: radius tracks the size of the arbour a branch
    supports, and so does subtree size. Ties are the common case rather than the exception
    -- 7 of the 9 offenders over 400 `semla_tmd_samples` graphs are ``[1, 1, 3]`` -- so
    the tie-break is load-bearing: total branch cable length (stem included, so competing
    leaves are separated by how far they reach) first, then node id for determinism. All
    three keys are rotation invariant, which the rotation-invariance test in
    `tests/validation_metrics.py` requires.

    dendrite_gen splits a degree-4 node into two bifurcations instead of pruning it, and
    only prunes at degree > 4. One uniform rule is used here instead: splitting would have
    to invent a node position, and `sanitise_provenance` maps sanitised ids back to raw
    ids positionally, so a node with no raw counterpart has nowhere to go. The cost is
    small and measurable -- 9 nodes of 16724 (0.05%) over those 400 graphs, and only on
    graphs that already violate the tree prior.

    The root is exempt: a high-degree soma is legitimate, and counting it reads 99.6% on
    real neuron GT (see `graph_health`).

    Returns ``(H, dropped)``; ``H`` is ``T`` itself when nothing was dropped.
    """
    _parent, children = _root_tree(T, root)

    # Reverse pre-order is a valid post-order, so subtree stats accumulate in one pass
    # without recursing -- the same shape as `_postorder_subtree_stats`.
    pre: list = []
    stack = [root]
    while stack:
        u = stack.pop()
        pre.append(u)
        stack.extend(children[u])

    # `cable[u]` includes the stem edge from u's parent, so it measures the whole branch
    # a parent would be dropping -- for a leaf that is the stem alone, not zero.
    size: dict = {}
    cable: dict = {}
    for u in reversed(pre):
        size[u] = 1 + sum(size[c] for c in children[u])
        p = _parent.get(u)
        stem = 0.0 if p is None else float(
            np.linalg.norm(T.nodes[u]["pos"] - T.nodes[p]["pos"])
        )
        cable[u] = stem + sum(cable[c] for c in children[u])

    # Pre-order, so a node inside an already-dropped subtree is skipped when reached.
    dropped: set = set()
    for u in pre:
        if u == root or u in dropped:
            continue
        kids = children[u]
        if len(kids) <= 2:
            continue
        ranked = sorted(kids, key=lambda c: (-size[c], -cable[c], c))
        for c in ranked[2:]:
            dropped.update(_subtree_nodes(children, c))

    if not dropped:
        return T, dropped
    H = T.copy()
    H.remove_nodes_from(dropped)
    return H, dropped


def _contract_degree_two(T: nx.Graph, root: int) -> nx.Graph:
    """Contract every non-root degree-2 node, giving a critical tree.

    In a critical tree each non-root node is a bifurcation or a terminal, which is what
    makes `node_count` a *branch-point count* rather than a size proxy: on ground truth
    ``1 + leaves + bifurcations == N`` holds for 100.00% of graphs, and
    ``corr(node_count, bifurcation_count) = 0.9979``. Real generations carry 0.37%
    degree-2 nodes, so this restores that identity on the generated side too.

    Numerically the contraction is near-invisible at that rate (every metric shifts by
    <= 0.006 GT standard deviations; seven of them are structurally immune because they
    key on child counts or degree >= 3). It is here for three other reasons: it makes
    `node_count` exact, it matches what the TMD path already does
    (`compute_tmd_barcode_diagram(..., simplify_to_critical_tree=True)`), and the
    distortion grows linearly with the rate (~1.15 GT-sd of `branch_length` per unit
    rate), which matters for the far larger d15/d20 graphs.
    """
    H = T.copy()
    # Chains contract from the ends inward, so iterate to a fixed point.
    while True:
        victims = [
            node for node in H.nodes()
            if node != root and H.degree(node) == 2
        ]
        if not victims:
            return H
        progressed = False
        for node in victims:
            if node not in H or H.degree(node) != 2:
                continue
            a, b = list(H.neighbors(node))
            if a == b:
                continue
            H.remove_node(node)
            if not H.has_edge(a, b):
                H.add_edge(a, b)
            progressed = True
        if not progressed:
            return H


def axial_axis(G: nx.Graph) -> np.ndarray:
    """Unit vector from the root to the node centroid -- the graph's own axial direction.

    Replaces the per-graph PCA principal axis for decomposing extents. Both are rotation
    invariant (which is required: SEMLA is E(3)-equivariant and the training transform
    applies a random rotation), but PCA's eigenvector sign is arbitrary and its direction
    is unstable when the top two eigenvalues are close. Root->centroid has neither
    problem and is closer in meaning to dendrite_gen's fixed anatomical `uhat`: measured
    z(`mmd_morpho`) under a perpendicular squash is 4.1 for fixed `uhat`, 3.3 here, and
    2.6 for PCA.

    It inherits root accuracy, which is 99.6% on neurons (the soma triggers
    `choose_root`'s unique-hub rule) but 78.9% on the binarized tree corpora, where that
    rule never fires. On real ground truth it is never degenerate -- the smallest
    |centroid - root| / diameter over 336 trees is 0.269 -- but fall back to +z anyway.

    Note there is no circularity with `pca_axis`: PCA is used only to *pick a root*
    (`choose_root` -> `pca_base_root`), and this axis only to *decompose extents*.
    """
    if G.number_of_nodes() == 0:
        return np.array([0.0, 0.0, 1.0])
    root = G.graph.get("root")
    if root is None or root not in G.nodes:
        return np.array([0.0, 0.0, 1.0])

    pts = np.stack([np.asarray(G.nodes[k]["pos"], dtype=np.float64) for k in G.nodes()])
    v = pts.mean(axis=0) - np.asarray(G.nodes[root]["pos"], dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return v / norm


def sanitise_graph(G: nx.Graph) -> nx.Graph:
    """Reduce a raw generated graph to the critical tree the morphometrics assume.

    Steps, in a load-bearing order:
      1. largest connected component
      2. relabel to contiguous ids 0..M-1        (see `_largest_component_relabelled`)
      3. minimum spanning tree by edge length    (see `_min_spanning_tree`)
      4. `choose_root` -- ONCE, here (see below)     (steps 1-4: `_spanning_stage`)
      5. binarise: drop the smallest subtrees    (see `_binarise`)
      6. contract non-root degree-2 nodes        (see `_contract_degree_two`)
      7. relabel again, carrying the root, and stash the root->centroid axis

    **The root is chosen exactly once, before contraction, and carried through.** Calling
    `choose_root` again afterwards would run it over a different node set, so the node
    protected during contraction could fail to be the final root -- leaving a non-root
    degree-2 node alive and breaking the ``1 + leaves + bifurcations == N`` identity that
    makes `node_count` a branch-point count. Binarisation never touches the root, so it
    cannot invalidate that choice either.

    Choosing before contraction is also the stable option: contracting a degree-2 node
    joins its two neighbours, so every surviving node keeps its degree. The degree
    sequence is therefore identical before and after, and `choose_root`'s hub rule cannot
    change its answer. Only the `pca_base_root` fallback would move, because the point
    set it averages over shrinks.

    A degree-2 *root* is fine and is preserved: it has two children, so it is neither a
    leaf nor a violation of the identity above (which excludes the root).

    **Binarisation runs before contraction, not after.** dendrite_gen's own pipeline
    normalises degree on an already-collapsed tree, but placing it here buys two things:
    the dropped set comes out in spanning-tree ids, which is what `sanitise_provenance`
    needs to map it back onto raw nodes, and contraction stays the last structural step so
    the "no non-root degree-2 survives" invariant needs no fresh argument. The set of
    offending nodes is identical either way -- a degree-2 chain node has one child and can
    never be a multifurcation -- so all that moves is the subtree-size ranking basis, by
    the 0.37% of nodes that are degree-2 on real generations.

    Binarisation cannot itself create a non-root degree-2 node: a node with more than two
    children keeps exactly two, so it lands at degree 3. Contraction is not run twice.

    Ground-truth graphs must be passed through this too. Every step is a verified no-op
    there -- 0/2527 neuron and 0/337 tree val graphs altered, and `multifurcation_frac` is
    0.0000 on all four corpora both raw and post-MST -- so it costs nothing and makes a
    gen/GT asymmetry structurally impossible.

    Returns an empty graph unchanged.
    """
    if G.number_of_nodes() == 0:
        H = G.copy()
        H.graph["root"] = 0
        H.graph["axis"] = np.array([0.0, 0.0, 1.0])
        return H

    T, root, _keep = _spanning_stage(G)
    H, _dropped = _binarise(T, root)
    H = _contract_degree_two(H, root)

    # Binarisation and contraction both leave gaps in the id sequence, and
    # `pca_base_root`'s positional-index contract requires node id == coords row. Carry
    # the root through the remap rather than re-deriving it.
    order = sorted(H.nodes())
    if order != list(range(len(order))):
        remap = {old: i for i, old in enumerate(order)}
        R = nx.Graph()
        for old in order:
            R.add_node(remap[old], pos=H.nodes[old]["pos"])
        for u, v in H.edges():
            R.add_edge(remap[u], remap[v])
        root = remap[root]
        H = R

    H.graph["root"] = int(root)
    H.graph["axis"] = axial_axis(H)
    return H


# Node and edge states reported by `sanitise_provenance`.
PROVENANCE_NODE_STATES = ("kept", "contracted", "pruned", "fragment")
PROVENANCE_EDGE_STATES = ("kept", "excess", "pruned", "fragment")


def sanitise_provenance(G: nx.Graph) -> dict:
    """Classify every raw node and edge by what `sanitise_graph` does to it.

    The morphometrics all score the sanitised graph, so they structurally cannot show a
    disconnected or over-connected generation. This is what lets a plot of the RAW graph
    mark the critical tree the metrics actually saw. Returns raw node ids and raw
    ``frozenset`` edges:

        nodes   kept       survives into the critical tree
                contracted in the spanning tree, collapsed by `_contract_degree_two`
                pruned     in the spanning tree, dropped by `_binarise`
                fragment   outside the largest connected component
        edges   kept       in the spanning tree, and not inside a pruned subtree
                excess     inside the LCC, cut by the MST -- the cycle-closing edges
                pruned     in the spanning tree, incident to a pruned node
                fragment   outside the LCC

    Each quadruple partitions the raw graph exactly.

    ``pruned`` wins over ``contracted`` where they overlap: a degree-2 chain node inside a
    dropped branch is gone because the branch was dropped, which is the more informative
    thing to show.

    Edges follow the MST, not the contraction: a raw edge inside a contracted chain counts
    as *kept*, because its geometry is represented in the critical tree by the single
    substitute edge contraction puts in its place. Only the chain's interior nodes are
    collapsed, and those are reported as ``contracted``.

    This REPLAYS `sanitise_graph` through the same private helpers rather than restating
    their logic, so the two cannot drift apart; `tests/validation_plots.py` pins that the
    kept counts match `sanitise_graph` exactly. Note the state names describe the *actual*
    pipeline, which is not always what the health metrics count -- contraction runs after
    the MST, so `contracted` is not the raw-graph degree-2 set behind `degree2_node_frac`
    (cutting an edge can turn a degree-3 node into a degree-2 one).
    """
    def _key(u, v):
        return frozenset((u, v))

    all_nodes = set(G.nodes())
    all_edges = {_key(u, v) for u, v in G.edges()}
    empty = {
        "kept_nodes": set(), "contracted_nodes": set(), "pruned_nodes": set(),
        "fragment_nodes": all_nodes,
        "kept_edges": set(), "excess_edges": set(), "pruned_edges": set(),
        "fragment_edges": all_edges,
    }
    if G.number_of_nodes() == 0:
        return empty

    # Steps 1-4: largest connected component, MST, root. `keep[i]` is the raw id of
    # relabelled id `i`, so every result below maps straight back onto the raw graph.
    T, root, keep = _spanning_stage(G)
    lcc_nodes = set(keep)
    lcc_edges = {_key(u, v) for u, v in G.subgraph(keep).edges()}
    mst_edges = {_key(keep[u], keep[v]) for u, v in T.edges()}

    # Step 5: binarisation drops whole subtrees, so an edge goes with either endpoint.
    _B, dropped = _binarise(T, root)
    pruned_nodes = {keep[i] for i in dropped}
    pruned_edges = {e for e in mst_edges if e & pruned_nodes}

    # Step 6: degree-2 contraction. `_contract_degree_two` only ever removes nodes, so
    # whatever is left in the LCC and not pruned is the collapsed set.
    C = _contract_degree_two(_B, root)
    kept_nodes = {keep[i] for i in C.nodes()}

    return {
        "kept_nodes": kept_nodes,
        "contracted_nodes": lcc_nodes - kept_nodes - pruned_nodes,
        "pruned_nodes": pruned_nodes,
        "fragment_nodes": all_nodes - lcc_nodes,
        "kept_edges": mst_edges - pruned_edges,
        "excess_edges": lcc_edges - mst_edges,
        "pruned_edges": pruned_edges,
        "fragment_edges": all_edges - lcc_edges,
    }
