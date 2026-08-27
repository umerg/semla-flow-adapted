"""Tests for the structural validation suite (sanitise + dist_metrics).

Data-free tests run anywhere. Tests that need the real corpora are skipped when the
`.smol` files are absent, so this suite is still useful on a machine without the data.
"""

import math
import unittest
from pathlib import Path

import networkx as nx
import numpy as np

from semlaflow.validation.dist_metrics import (
    MORPHO_KEYS,
    build_gt_cache,
    compute_distribution_metrics,
    keys_for_level,
)
from semlaflow.validation.sanitise import (
    HEALTH_KEYS,
    graph_health,
    sanitise_graph,
    second_highest_degree_multifurcation_frac,
)
from semlaflow.validation.structural_metrics import (
    branch_order_values,
    degree_values,
    partition_asymmetry,
    strahler_number,
)

NEURON_VAL = Path("/Users/umer/Documents/neurons_conditional/smol/val.smol")
TREE_VAL = Path("/Users/umer/Documents/trees_genus_d10/smol/val.smol")


def _binary_tree(depth=4, spacing=1.0, seed=0, root_children=4):
    """A clean rooted critical tree -- stands in for real ground truth.

    The root is given `root_children` neighbours so it is the unique strict degree
    maximum, which is what makes `choose_root`'s hub rule fire deterministically. That
    mirrors the real corpora: a neuron soma has median degree 8 (max 16) and fires the
    hub rule for 99.6% of graphs. Away from the root the tree is strictly binary with no
    degree-2 nodes, exactly like ground truth (pooled non-root degrees are {1, 3} only).
    """
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.add_node(0, pos=np.zeros(3))
    frontier, nid = [0], 1
    for d in range(depth):
        nxt = []
        for parent in frontier:
            n_children = root_children if parent == 0 else 2
            for _ in range(n_children):
                offset = rng.normal(size=3)
                offset /= np.linalg.norm(offset)
                G.add_node(nid, pos=G.nodes[parent]["pos"] + spacing * offset)
                G.add_edge(parent, nid)
                nxt.append(nid)
                nid += 1
        frontier = nxt
    G.graph["root"] = 0
    return G


def _load(path, limit=None):
    from semlaflow.data.datasets import GeometricDataset
    from semlaflow.validation.convert import geometric_mol_to_nx

    ds = GeometricDataset.load(str(path))
    n = min(limit, len(ds)) if limit else len(ds)
    return [geometric_mol_to_nx(ds[i], coord_scale=1.0) for i in range(n)]


class SanitiseTests(unittest.TestCase):
    def test_clean_tree_is_a_fixed_point(self):
        G = _binary_tree()
        S = sanitise_graph(G)
        self.assertEqual(G.number_of_nodes(), S.number_of_nodes())
        self.assertEqual(G.number_of_edges(), S.number_of_edges())
        self.assertTrue(nx.is_tree(S))

    def test_output_ids_are_contiguous_and_root_is_valid(self):
        """The relabel contract: pca_base_root returns a POSITIONAL index, so node id
        must equal the coords row or root selection silently addresses the wrong node."""
        G = _binary_tree()
        G.remove_node(3)  # leave a gap in the id sequence
        S = sanitise_graph(G)
        self.assertEqual(sorted(S.nodes()), list(range(S.number_of_nodes())))
        self.assertIn(S.graph["root"], S.nodes)
        self.assertEqual(np.asarray(S.graph["axis"]).shape, (3,))

    def test_disconnected_graph_reduces_to_one_component(self):
        G = _binary_tree()
        edges = list(G.edges())
        G.remove_edge(*edges[len(edges) // 2])
        self.assertFalse(nx.is_connected(G))
        S = sanitise_graph(G)
        self.assertTrue(nx.is_connected(S))
        self.assertLess(S.number_of_nodes(), G.number_of_nodes())

    def test_cycles_are_removed(self):
        G = _binary_tree()
        nodes = list(G.nodes())
        G.add_edge(nodes[3], nodes[-1])
        self.assertGreater(G.number_of_edges(), G.number_of_nodes() - 1)
        S = sanitise_graph(G)
        self.assertTrue(nx.is_tree(S))

    def test_degree_two_nodes_are_contracted(self):
        G = _binary_tree()
        # Split an edge with a midpoint, creating a degree-2 chain node.
        u, v = list(G.edges())[3]
        mid = (G.nodes[u]["pos"] + G.nodes[v]["pos"]) / 2
        new = max(G.nodes()) + 1
        G.remove_edge(u, v)
        G.add_node(new, pos=mid)
        G.add_edge(u, new)
        G.add_edge(new, v)
        self.assertEqual(G.degree(new), 2)
        S = sanitise_graph(G)
        non_root = [d for k, d in S.degree() if k != S.graph["root"]]
        self.assertNotIn(2, non_root)

    def test_no_non_root_degree_two_node_survives(self):
        """Regression: the root must be chosen ONCE, before contraction.

        With two `choose_root` calls the second ran over a different node set, so the node
        protected during contraction could fail to become the final root -- leaving a
        non-root degree-2 node alive and breaking `1 + leaves + bifurcations == N`.
        """
        rng = np.random.default_rng(7)
        for seed in range(30):
            G = _binary_tree(seed=seed)
            G.graph.pop("root", None)
            # Sprinkle degree-2 nodes by splitting random edges.
            for _ in range(4):
                u, v = list(G.edges())[rng.integers(G.number_of_edges())]
                new = max(G.nodes()) + 1
                G.remove_edge(u, v)
                G.add_node(new, pos=(G.nodes[u]["pos"] + G.nodes[v]["pos"]) / 2)
                G.add_edge(u, new)
                G.add_edge(new, v)
            S = sanitise_graph(G)
            non_root = [d for k, d in S.degree() if k != S.graph["root"]]
            self.assertNotIn(2, non_root, f"seed {seed}")

    def test_contraction_protects_the_root_and_only_the_root(self):
        """A degree-2 root is legitimate (it has two children, so the critical-tree
        identity still holds); every other degree-2 node must go."""
        from semlaflow.validation.sanitise import _contract_degree_two

        # 0-1-2 chain off the root, plus a fork, so node 0 and node 1 are both degree 2.
        G = nx.Graph()
        for i, p in enumerate([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 1, 0), (3, -1, 0), (-1, 0, 0)]):
            G.add_node(i, pos=np.array(p, dtype=float))
        G.add_edges_from([(0, 1), (1, 2), (2, 3), (2, 4), (0, 5)])
        self.assertEqual(G.degree(0), 2)
        self.assertEqual(G.degree(1), 2)

        H = _contract_degree_two(G, root=0)
        self.assertIn(0, H.nodes, "the root must survive")
        self.assertEqual(H.degree(0), 2, "a degree-2 root keeps its two children")
        self.assertNotIn(1, H.nodes, "a non-root degree-2 node must be contracted")
        self.assertNotIn(2, [d for k, d in H.degree() if k != 0])

    def test_empty_graph_is_handled(self):
        S = sanitise_graph(nx.Graph())
        self.assertEqual(S.number_of_nodes(), 0)
        self.assertIn("axis", S.graph)

    def test_sanitise_is_idempotent(self):
        G = _binary_tree()
        nodes = list(G.nodes())
        G.add_edge(nodes[2], nodes[-3])
        once = sanitise_graph(G)
        twice = sanitise_graph(once)
        self.assertEqual(once.number_of_nodes(), twice.number_of_nodes())
        self.assertEqual(once.number_of_edges(), twice.number_of_edges())


def _extra_child(G, node, offset):
    """Attach a new leaf to `node`, `offset` away. Returns its id."""
    nid = max(G.nodes()) + 1
    G.add_node(nid, pos=np.asarray(G.nodes[node]["pos"], dtype=float)
               + np.asarray(offset, dtype=float))
    G.add_edge(node, nid)
    return nid


def _hub_multifurcation(offsets, hub_children=5):
    """Hub root -> node `u` -> one leaf per offset. Returns ``(G, u, leaves)``.

    The hub is load-bearing. `choose_root` takes the *unique* strict degree maximum, so
    the root must out-rank `u` by more than one: give `u` an extra child in a fixture
    built on `_binary_tree(root_children=4)` and `u` ties the root at degree 4, root
    selection falls back to `pca_base_root`, the tree is re-rooted at some leaf, and the
    old root becomes a non-root multifurcation of its own. Every offset here hangs off a
    node the hub out-ranks by a clear margin, so only `u` is ever the offender.
    """
    G = nx.Graph()
    G.add_node(0, pos=np.zeros(3))
    nid = 1
    for i in range(hub_children):
        G.add_node(nid, pos=np.array([float(i + 1), -5.0, 0.0]))
        G.add_edge(0, nid)
        nid += 1

    u = nid
    G.add_node(u, pos=np.array([0.0, 0.0, 1.0]))
    G.add_edge(0, u)
    nid += 1

    leaves = []
    for off in offsets:
        G.add_node(nid, pos=G.nodes[u]["pos"] + np.asarray(off, dtype=float))
        G.add_edge(u, nid)
        leaves.append(nid)
        nid += 1

    G.graph["root"] = 0
    assert G.degree(0) > G.degree(u), "hub must stay the unique degree maximum"
    return G, u, leaves


class BinariseTests(unittest.TestCase):
    """The corpora are binarised away from the root; generations must be scored the same.

    dendrite_gen's `clean_trees.normalize_high_degree` did this to the ground truth at
    preprocessing, so a generated trifurcation reaching the morphometrics would be
    compared against a reference that structurally cannot contain one.
    """

    @staticmethod
    def _positions(G):
        return {tuple(np.round(G.nodes[k]["pos"], 9)) for k in G.nodes()}

    def test_trifurcation_loses_only_its_smallest_subtree(self):
        """Subtree size decides: the two-node branch outranks both single leaves."""
        G, u, leaves = _hub_multifurcation(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.5]]
        )
        # Two children, not one: a single child would leave leaves[0] at degree 2 and
        # contraction would remove it as well, confusing what binarisation did.
        big = _extra_child(G, leaves[0], [1.0, 0.0, 0.0])
        _extra_child(G, leaves[0], [1.0, 1.0, 0.0])
        self.assertEqual(G.degree(u), 4)

        S = sanitise_graph(G)
        kept = self._positions(S)
        self.assertEqual(S.number_of_nodes(), G.number_of_nodes() - 1)
        self.assertTrue(nx.is_tree(S))
        for survivor in (leaves[0], big, leaves[1]):
            self.assertIn(tuple(np.round(G.nodes[survivor]["pos"], 9)), kept)
        self.assertNotIn(
            tuple(np.round(G.nodes[leaves[2]]["pos"], 9)), kept,
            "the smallest branch is the one that goes",
        )

    def test_four_children_land_at_exactly_two(self):
        G, u, _leaves = _hub_multifurcation(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            hub_children=6,
        )
        self.assertEqual(G.degree(u), 5)

        S = sanitise_graph(G)
        non_root = [d for k, d in S.degree() if k != S.graph["root"]]
        self.assertLessEqual(max(non_root), 3)
        self.assertEqual(S.number_of_nodes(), G.number_of_nodes() - 2)

    def test_a_high_degree_root_is_exempt(self):
        """The soma is a legitimate hub: a degree-6 root must survive untouched."""
        G = _binary_tree(depth=2, root_children=6)
        S = sanitise_graph(G)
        self.assertEqual(S.number_of_nodes(), G.number_of_nodes())
        self.assertEqual(S.degree(S.graph["root"]), 6)

    def test_cable_length_breaks_a_subtree_size_tie(self):
        """Equal node counts are the common case -- 7 of the 9 real offenders over 400
        `semla_tmd_samples` graphs are [1, 1, 3] -- so the tie-break decides the outcome.
        The branch that reaches further survives."""
        G, u, leaves = _hub_multifurcation(
            [[0.5, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 3.0, 0.0]]
        )
        near, mid, far = leaves
        self.assertEqual(G.degree(u), 4)

        kept = self._positions(sanitise_graph(G))
        for survivor in (mid, far):
            self.assertIn(tuple(np.round(G.nodes[survivor]["pos"], 9)), kept)
        self.assertNotIn(
            tuple(np.round(G.nodes[near]["pos"], 9)), kept, "the shortest branch goes"
        )

    def test_exact_ties_fall_back_to_node_id(self):
        """Three equidistant leaves tie on size AND cable; the result must still be one
        fixed answer rather than whatever set iteration happened to yield."""
        G, _u, leaves = _hub_multifurcation(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        first = sanitise_graph(G)
        kept = self._positions(first)
        for survivor in leaves[:2]:
            self.assertIn(tuple(np.round(G.nodes[survivor]["pos"], 9)), kept)
        self.assertNotIn(tuple(np.round(G.nodes[leaves[2]]["pos"], 9)), kept)

        for _ in range(3):
            again = sanitise_graph(G)
            self.assertEqual(sorted(again.nodes()), sorted(first.nodes()))
            self.assertEqual(sorted(again.edges()), sorted(first.edges()))

    def test_binarisation_creates_no_non_root_degree_two_node(self):
        """A node with >2 children keeps exactly 2, so it lands at degree 3, never 2.

        This is why contraction is not run a second time after binarisation.
        """
        from semlaflow.validation.sanitise import _binarise, _spanning_stage

        rng = np.random.default_rng(3)
        for seed in range(20):
            G = _binary_tree(seed=seed, depth=3, root_children=8)
            for _ in range(3):
                victim = int(rng.choice([k for k, d in G.degree() if k != 0 and d == 3]))
                _extra_child(G, victim, rng.normal(0, 1, 3))
            T, root, _keep = _spanning_stage(G)
            B, dropped = _binarise(T, root)
            self.assertTrue(dropped, f"seed {seed}: fixture should need binarising")
            self.assertNotIn(2, [d for k, d in B.degree() if k != root], f"seed {seed}")

    def test_no_non_root_multifurcation_survives_sanitisation(self):
        rng = np.random.default_rng(11)
        for seed in range(20):
            G = _binary_tree(seed=seed, depth=3, root_children=8)
            for _ in range(4):
                victim = int(rng.choice([k for k in G.nodes() if k != 0]))
                _extra_child(G, victim, rng.normal(0, 1, 3))
            S = sanitise_graph(G)
            non_root = [d for k, d in S.degree() if k != S.graph["root"]]
            self.assertLessEqual(max(non_root), 3, f"seed {seed}")
            self.assertNotIn(2, non_root, f"seed {seed}")
            self.assertTrue(nx.is_tree(S), f"seed {seed}")


class HealthTests(unittest.TestCase):
    def test_clean_tree_scores_zero_on_every_key(self):
        graphs = [_binary_tree(seed=s) for s in range(5)]
        h = graph_health(graphs)
        for key in HEALTH_KEYS:
            expected = 1.0 if key == "lcc_node_frac" else 0.0
            self.assertAlmostEqual(h[key], expected, places=9, msg=key)

    def test_multifurcation_ignores_the_root(self):
        """A high-degree root is legitimate (the soma); a high-degree branch node is not.

        Including the root reads 99.6% on real neuron GT, which is why this must exclude it.
        """
        G = _binary_tree(depth=3)
        # Give the root extra children -- soma-like, must NOT count.
        for i in range(4):
            nid = max(G.nodes()) + 1
            G.add_node(nid, pos=np.array([float(i), 0.0, -1.0]))
            G.add_edge(0, nid)
        self.assertGreater(G.degree(0), 3)
        self.assertEqual(graph_health([G])["multifurcation_frac"], 0.0)

        # Now give a NON-root node a fourth neighbour -- must count.
        victim = next(k for k, d in G.degree() if k != 0 and d == 3)
        nid = max(G.nodes()) + 1
        G.add_node(nid, pos=np.array([9.0, 9.0, 9.0]))
        G.add_edge(victim, nid)
        self.assertEqual(graph_health([G])["multifurcation_frac"], 1.0)

    def test_multifurcation_matches_root_free_cross_check(self):
        """Proves the metric is not an artefact of root selection.

        `choose_root` prefers the unique maximum-degree node, so it absorbs an offending
        hub most of the time. The second-highest degree consults no root at all; if the
        two agree, root selection is not doing the work.

        The offender has to be a NEW leaf, not a chord between existing nodes: both
        metrics are now read off the spanning tree, and the MST cuts a chord before either
        of them sees it. A leaf's single edge is in every spanning tree.
        """
        graphs = []
        for s in range(25):
            G = sanitise_graph(_binary_tree(seed=s))
            if s % 3 == 0:
                victim = next(
                    k for k, d in G.degree() if k != G.graph["root"] and d == 3
                )
                nid = max(G.nodes()) + 1
                G.add_node(nid, pos=G.nodes[victim]["pos"] + np.array([0.1, 0.0, 0.0]))
                G.add_edge(victim, nid)
            graphs.append(G)

        multifurcation_frac = graph_health(graphs)["multifurcation_frac"]
        self.assertGreater(
            multifurcation_frac, 0.0, "the fixture must actually contain multifurcations"
        )
        self.assertAlmostEqual(
            multifurcation_frac,
            second_highest_degree_multifurcation_frac(graphs),
            places=9,
        )

    def test_multifurcation_is_measured_after_the_mst(self):
        """A cycle-closing edge inflates degree, but the MST cuts it before binarisation.

        Reporting the raw degree would credit binarisation with a repair the MST already
        made. Measured over 400 `semla_tmd_samples` graphs the two stages read 0.0700 and
        0.0225.
        """
        G, u, _leaves = _hub_multifurcation([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        # Close a cycle back to a hub leaf. The chord spans the hub's whole radius, so it
        # is the longest edge on its cycle and is the one the MST cuts.
        G.add_edge(u, 1)

        raw_non_root = [d for k, d in G.degree() if k != 0]
        self.assertGreater(max(raw_non_root), 3, "raw graph has a degree-4 node")
        self.assertEqual(graph_health([G])["multifurcation_frac"], 0.0,
                         "the MST cuts the chord, so nothing is left to binarise")
        self.assertEqual(graph_health([G])["cycle_frac"], 1.0,
                         "the cycle itself is still reported, on the raw graph")

    def test_incidence_and_magnitude_differ(self):
        """`non_critical_node_frac` counts graphs; `degree2_node_frac` counts nodes."""
        G = _binary_tree()
        u, v = list(G.edges())[0]
        new = max(G.nodes()) + 1
        G.remove_edge(u, v)
        G.add_node(new, pos=(G.nodes[u]["pos"] + G.nodes[v]["pos"]) / 2)
        G.add_edge(u, new)
        G.add_edge(new, v)
        h = graph_health([G])
        self.assertEqual(h["non_critical_node_frac"], 1.0)
        self.assertLess(h["degree2_node_frac"], 0.1)


class StructuralMetricTests(unittest.TestCase):
    def test_branch_order_survives_disconnection(self):
        """The dendrite_gen original raises KeyError here -- 99/200 graphs at 1% dropout."""
        G = _binary_tree()
        edges = list(G.edges())
        G.remove_edge(*edges[len(edges) // 2])
        vals = branch_order_values(G)
        self.assertGreater(vals.size, 0)
        self.assertTrue(np.all(np.isfinite(vals)))

    def test_degenerate_graphs_return_nan_not_raise(self):
        single = nx.Graph()
        single.add_node(0, pos=np.zeros(3))
        single.graph["root"] = 0
        self.assertEqual(branch_order_values(single).size, 0)
        self.assertTrue(math.isnan(partition_asymmetry(nx.Graph())))
        self.assertTrue(math.isnan(strahler_number(nx.Graph())))

    def test_degree_values_exclude_root(self):
        G = _binary_tree(depth=3)
        self.assertEqual(degree_values(G).size, G.number_of_nodes() - 1)


class MetricSuiteTests(unittest.TestCase):
    def test_identical_sets_score_at_the_floor(self):
        graphs = [_binary_tree(seed=s) for s in range(30)]
        m = compute_distribution_metrics(graphs, graphs)
        for key in ("branch_length_w1", "bifurcation_angle_w1", "node_count_w1"):
            self.assertAlmostEqual(m[key], 0.0, places=9, msg=key)
        self.assertAlmostEqual(m["w1_pooled_mean_normalized"], 0.0, places=9)
        self.assertAlmostEqual(m["w1_pertree_mean_normalized"], 0.0, places=9)
        self.assertEqual(m["gen_degenerate_frac"], 0.0)

        # mmd2_unbiased excludes the diagonal self-terms, so on identical sets it is
        # legitimately NEGATIVE, not zero -- exactly 2(S - n^2) / (n^2 (n-1)). Clipping it
        # to 0 would reintroduce bias and break comparison against the real-vs-real floor,
        # so assert the sign rather than the magnitude.
        self.assertLessEqual(m["mmd_morpho"], 0.0)
        self.assertLessEqual(m["mmd_tmd"], 0.0)

    def test_report_level_is_a_pure_filter(self):
        """The tier must not be reachable from the compute path, by construction."""
        import inspect

        sig = inspect.signature(compute_distribution_metrics)
        self.assertNotIn("metric_report_level", sig.parameters)
        self.assertNotIn("level", sig.parameters)
        self.assertLess(len(keys_for_level("headline")), len(keys_for_level("standard")))
        self.assertLess(len(keys_for_level("standard")), len(keys_for_level("full")))
        with self.assertRaises(ValueError):
            keys_for_level("nope")

    def test_rotation_invariance(self):
        graphs = [_binary_tree(seed=s) for s in range(20)]
        rng = np.random.default_rng(1)
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, 0] *= -1
        rotated = []
        for G in graphs:
            R = nx.Graph()
            for n, d in G.nodes(data=True):
                R.add_node(n, pos=q @ d["pos"])
            R.add_edges_from(G.edges())
            R.graph["root"] = G.graph["root"]
            rotated.append(R)

        ref = compute_distribution_metrics(graphs[:10], graphs[10:])
        rot = compute_distribution_metrics(rotated[:10], rotated[10:])
        for key in keys_for_level("standard"):
            a, b = ref.get(key), rot.get(key)
            if a is None or b is None or not math.isfinite(a):
                continue
            self.assertAlmostEqual(a, b, places=6, msg=key)

    def test_morpho_vector_width_matches_keys(self):
        from semlaflow.validation.dist_metrics import assemble_morpho_vector

        v = assemble_morpho_vector(sanitise_graph(_binary_tree()))
        self.assertEqual(v.shape, (len(MORPHO_KEYS),))


@unittest.skipUnless(NEURON_VAL.exists() and TREE_VAL.exists(), "real corpora not present")
class RealCorpusTests(unittest.TestCase):
    """The checks that only real ground truth can make."""

    @classmethod
    def setUpClass(cls):
        cls.neurons = _load(NEURON_VAL)
        cls.trees = _load(TREE_VAL)

    def test_health_is_exactly_zero_on_ground_truth(self):
        for name, graphs in (("neurons", self.neurons), ("trees", self.trees)):
            h = graph_health(graphs)
            for key in HEALTH_KEYS:
                expected = 1.0 if key == "lcc_node_frac" else 0.0
                self.assertAlmostEqual(h[key], expected, places=9, msg=f"{name}/{key}")

    def test_sanitisation_is_a_no_op_on_ground_truth(self):
        for name, graphs in (("neurons", self.neurons), ("trees", self.trees)):
            for G in graphs:
                S = sanitise_graph(G)
                self.assertEqual(G.number_of_nodes(), S.number_of_nodes(), name)
                self.assertEqual(G.number_of_edges(), S.number_of_edges(), name)

    def test_critical_tree_identity(self):
        """1 + leaves + bifurcations == N. This is what makes node_count a branch-point
        count rather than a size proxy."""
        from semlaflow.validation.structural_metrics import _root_tree

        for name, graphs in (("neurons", self.neurons), ("trees", self.trees)):
            for G in graphs:
                S = sanitise_graph(G)
                root = S.graph["root"]
                _p, children = _root_tree(S, root)
                leaves = sum(1 for k, c in children.items() if len(c) == 0 and k != root)
                bifs = sum(1 for k, c in children.items() if len(c) >= 2 and k != root)
                self.assertEqual(1 + leaves + bifs, S.number_of_nodes(), name)

    def test_non_root_degrees_are_one_or_three(self):
        for name, graphs in (("neurons", self.neurons), ("trees", self.trees)):
            degs = np.unique(np.concatenate([degree_values(G) for G in graphs]))
            self.assertEqual(sorted(degs.tolist()), [1.0, 3.0], name)

    def test_corrupted_graphs_never_raise(self):
        """Independent per-pair edge flips, as the N**2 bond head would produce."""
        from semlaflow.validation.convert import choose_root

        rng = np.random.default_rng(0)
        base = self.trees[:60]
        for fp, fn in ((0.0, 0.05), (0.0, 0.20), (1e-3, 0.0), (1e-2, 0.01)):
            corrupted = []
            for G in base:
                H = nx.Graph()
                for n, d in G.nodes(data=True):
                    H.add_node(n, pos=d["pos"])
                nodes = list(G.nodes())
                for u, v in G.edges():
                    if rng.random() >= fn:
                        H.add_edge(u, v)
                for _ in range(int(rng.poisson(fp * len(nodes) * (len(nodes) - 1) / 2))):
                    a, b = rng.choice(len(nodes), size=2, replace=False)
                    H.add_edge(nodes[a], nodes[b])
                H.graph["root"] = choose_root(
                    H, np.stack([H.nodes[k]["pos"] for k in sorted(H.nodes())])
                )
                corrupted.append(H)

            m = compute_distribution_metrics(corrupted, base)  # must not raise
            for key in keys_for_level("standard"):
                self.assertIn(key, m, f"fp={fp} fn={fn} missing {key}")

    def test_gt_cache_reuse_matches_a_fresh_fit(self):
        gt = self.trees
        cache = build_gt_cache(gt)
        cached = compute_distribution_metrics(gt[:150], gt, gt_cache=cache)
        fresh = compute_distribution_metrics(gt[:150], gt, gt_cache=None)
        for key in keys_for_level("standard"):
            a, b = cached.get(key), fresh.get(key)
            if a is None or b is None or not math.isfinite(a):
                continue
            self.assertAlmostEqual(a, b, places=9, msg=key)


if __name__ == "__main__":
    unittest.main()
