"""Tests for the validation-time sample plots and the plot helper's new options.

Data-free: synthetic graphs only.

`NeuronCFM._log_validation_plots` is exercised against a stub self rather than a real
LightningModule -- building one needs a generator, vocab, interpolant and trainer, none of
which the plotting logic touches. What matters is the mode selection, the row labelling, the
rank guard, and that every figure is closed.
"""

import types
import unittest

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from semlaflow.models.neuron_cfm import NeuronCFM
from semlaflow.validation.plot import plot_graph_grid_angles
from semlaflow.validation.sanitise import sanitise_graph, sanitise_provenance


def _tree(seed: int = 0, depth: int = 3) -> nx.Graph:
    rng = np.random.default_rng(seed)
    G = nx.Graph()
    G.graph["root"] = 0
    G.add_node(0, pos=np.zeros(3))
    frontier = [0]
    for _ in range(depth):
        new_frontier = []
        for parent in frontier:
            for _ in range(2):
                idx = G.number_of_nodes()
                G.add_node(idx, pos=np.asarray(G.nodes[parent]["pos"]) + rng.normal(0, 10, 3))
                G.add_edge(parent, idx)
                new_frontier.append(idx)
        frontier = new_frontier
    return G


class _StubLogger:
    """Minimal stand-in for WandbLogger: records log_image calls."""

    def __init__(self):
        self.images = {}

    def log_image(self, key, images, caption=None):
        self.images[key] = (images, caption)


def _stub_model(*, n_gen=6, class_hidden=0, tmd_dim=0, classes=None,
                is_global_zero=True, val_plots=True, max_rows=8):
    """A NeuronCFM-shaped object carrying only what _log_validation_plots reads."""
    logger = _StubLogger()
    model = types.SimpleNamespace(
        val_plots=val_plots,
        val_plot_max_rows=max_rows,
        _val_gen_graphs=[_tree(seed=i) for i in range(n_gen)],
        _val_gt_graphs=[_tree(seed=100 + i) for i in range(n_gen)],
        _val_gen_classes=list(classes or []),
        _val_gt_classes=list(classes or []),
        gen=types.SimpleNamespace(class_hidden=class_hidden, tmd_dim=tmd_dim),
        trainer=types.SimpleNamespace(is_global_zero=is_global_zero),
        loggers=[logger],
        current_epoch=7,
        hparams={"dataset": "neurons_conditional"},
    )
    model._emit_figure = types.MethodType(NeuronCFM._emit_figure, model)
    model._log_validation_plots = types.MethodType(NeuronCFM._log_validation_plots, model)
    return model, logger


class TestPlotHelperOptions(unittest.TestCase):
    def test_out_dir_none_writes_nothing(self):
        fig, path = plot_graph_grid_angles([_tree()], out_dir=None)
        self.assertIsNone(path)
        plt.close(fig)

    def test_still_writes_a_png_when_out_dir_given(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            fig, path = plot_graph_grid_angles(
                [_tree()], out_dir=Path(tmp), stem="s", file_tag="t"
            )
            self.assertTrue(path.exists())
            plt.close(fig)

    def test_per_graph_titles_label_each_row(self):
        graphs = [_tree(seed=i) for i in range(3)]
        fig, _ = plot_graph_grid_angles(
            graphs, per_graph_titles=["Gen 23P", "Gen 4P", "Gen 5P-IT"], max_graphs=3
        )
        titles = [ax.get_title() for ax in fig.axes]
        self.assertTrue(any(t.startswith("Gen 23P") for t in titles))
        self.assertTrue(any(t.startswith("Gen 5P-IT") for t in titles))
        plt.close(fig)

    def test_short_title_list_falls_back_to_prefix(self):
        graphs = [_tree(seed=i) for i in range(3)]
        fig, _ = plot_graph_grid_angles(
            graphs, per_graph_titles=["Gen 23P"], title_prefix="Gen", max_graphs=3
        )
        titles = [ax.get_title() for ax in fig.axes]
        self.assertTrue(any(t.startswith("Gen 23P") for t in titles))
        self.assertTrue(any(t.startswith("Gen - ") for t in titles))
        plt.close(fig)


class TestSanitiseProvenance(unittest.TestCase):
    """The overlay is only honest if the classifier matches the real pipeline.

    `sanitise_provenance` replays `sanitise_graph` through the same private helpers rather
    than restating their logic; `test_kept_counts_match_sanitise_graph` is what pins that.
    Change a sanitise step without updating the classifier and it fails.
    """

    @staticmethod
    def _line(n=4):
        """Path graph 0-1-...-n-1: the interior nodes are degree-2."""
        G = nx.Graph()
        for i in range(n):
            G.add_node(i, pos=np.array([float(i) * 10.0, 0.0, 0.0]))
        G.add_edges_from((i, i + 1) for i in range(n - 1))
        G.graph["root"] = 0
        return G

    def _perturbed(self, seed, extra_edges=0, n_frag=0):
        r = np.random.default_rng(seed)
        G = nx.Graph()
        G.add_node(0, pos=np.zeros(3))
        n = int(r.integers(6, 40))
        for i in range(1, n):
            p = int(r.integers(0, i))
            G.add_node(i, pos=np.asarray(G.nodes[p]["pos"]) + r.normal(0, 10, 3))
            G.add_edge(p, i)
        for _ in range(extra_edges):
            a, b = r.integers(0, n, 2)
            if a != b:
                G.add_edge(int(a), int(b))
        for k in range(n_frag):
            a, b = n + 2 * k, n + 2 * k + 1
            G.add_node(a, pos=r.normal(0, 400, 3))
            G.add_node(b, pos=r.normal(0, 400, 3))
            G.add_edge(a, b)
        G.graph["root"] = 0
        return G

    def test_kept_counts_match_sanitise_graph(self):
        """The anti-drift guarantee."""
        for seed in range(25):
            G = self._perturbed(seed, extra_edges=seed % 7, n_frag=seed % 3)
            prov = sanitise_provenance(G)
            self.assertEqual(
                len(prov["kept_nodes"]), sanitise_graph(G).number_of_nodes(),
                f"seed {seed}: classifier disagrees with sanitise_graph",
            )

    def test_states_partition_the_raw_graph(self):
        for seed in range(15):
            G = self._perturbed(seed, extra_edges=seed % 5, n_frag=seed % 3)
            prov = sanitise_provenance(G)
            nodes = prov["kept_nodes"] | prov["contracted_nodes"] | prov["fragment_nodes"]
            self.assertEqual(nodes, set(G.nodes()))
            self.assertEqual(
                len(prov["kept_nodes"]) + len(prov["contracted_nodes"])
                + len(prov["fragment_nodes"]),
                G.number_of_nodes(), "node states overlap",
            )
            self.assertEqual(
                len(prov["kept_edges"]) + len(prov["excess_edges"])
                + len(prov["fragment_edges"]),
                G.number_of_edges(), "edge states overlap",
            )

    def test_clean_tree_is_all_kept(self):
        G = _tree(seed=0, depth=3)
        # Bifurcating tree: only the root can be degree-2, and choose_root may relocate it.
        prov = sanitise_provenance(G)
        self.assertEqual(prov["fragment_nodes"], set())
        self.assertEqual(prov["excess_edges"], set())
        self.assertEqual(prov["fragment_edges"], set())

    def test_one_chord_is_one_excess_edge(self):
        G = _tree(seed=0, depth=3)
        leaves = [n for n, d in G.degree() if d == 1]
        G.add_edge(leaves[0], leaves[-1])
        self.assertEqual(len(sanitise_provenance(G)["excess_edges"]), 1)

    def test_detached_pair_is_two_fragment_nodes(self):
        G = _tree(seed=0, depth=3)
        a, b = G.number_of_nodes(), G.number_of_nodes() + 1
        G.add_node(a, pos=np.array([500.0, 0.0, 0.0]))
        G.add_node(b, pos=np.array([510.0, 0.0, 0.0]))
        G.add_edge(a, b)
        prov = sanitise_provenance(G)
        self.assertEqual(prov["fragment_nodes"], {a, b})
        self.assertEqual(prov["fragment_edges"], {frozenset((a, b))})

    def test_degree_two_chain_is_contracted(self):
        """A path graph collapses to its two endpoints."""
        prov = sanitise_provenance(self._line(5))
        self.assertEqual(len(prov["kept_nodes"]), 2)
        self.assertEqual(len(prov["contracted_nodes"]), 3)
        # Every edge is on the tree path, so none is excess or fragment.
        self.assertEqual(len(prov["kept_edges"]), 4)
        self.assertEqual(prov["excess_edges"], set())

    def test_empty_and_single_node(self):
        prov = sanitise_provenance(nx.Graph())
        self.assertEqual(sum(len(v) for v in prov.values()), 0)

        G = nx.Graph()
        G.add_node(0, pos=np.zeros(3))
        G.graph["root"] = 0
        prov = sanitise_provenance(G)
        self.assertEqual(prov["kept_nodes"], {0})
        self.assertEqual(prov["fragment_nodes"], set())


def _chorded_and_fragmented():
    """Tree + one chord (an excess edge) + one detached pair (two fragment nodes)."""
    G = _tree(seed=0, depth=3)
    leaves = [n for n, d in G.degree() if d == 1]
    G.add_edge(leaves[0], leaves[-1])
    a, b = G.number_of_nodes(), G.number_of_nodes() + 1
    G.add_node(a, pos=np.array([500.0, 0.0, 0.0]))
    G.add_node(b, pos=np.array([510.0, 0.0, 0.0]))
    G.add_edge(a, b)
    return G


class TestOverlayStyleMapping(unittest.TestCase):
    """`_overlay_styles` is where the ordering contract lives, so test it directly.

    Going through matplotlib cannot check ordering: mplot3d re-sorts a Path3DCollection by
    depth during the draw, so post-draw row positions are meaningless (counts survive).
    """

    def test_node_arrays_follow_graph_insertion_order(self):
        from semlaflow.validation.plot import (
            CONTRACTED_SIZE_FRAC, FRAGMENT_NODE_COLOR, NODE_SIZE, _overlay_styles,
        )

        G = nx.Graph()
        # Detached pair inserted FIRST, so insertion order differs from sorted-id order --
        # indexing the colour array by node id would mis-colour this graph.
        G.add_node(99, pos=np.array([500.0, 0.0, 0.0]))
        G.add_node(98, pos=np.array([510.0, 0.0, 0.0]))
        G.add_edge(99, 98)
        for i in range(5):
            G.add_node(i, pos=np.array([float(i) * 10.0, 0.0, 0.0]))
        G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 4)])
        G.graph["root"] = 0

        colors, sizes, _ec, _es = _overlay_styles(G, sanitise_provenance(G), "#8b1e3f")
        order = list(G.nodes())
        self.assertEqual(len(colors), G.number_of_nodes())
        self.assertEqual(order[:2], [99, 98])
        self.assertEqual(colors[0], FRAGMENT_NODE_COLOR)
        self.assertEqual(colors[1], FRAGMENT_NODE_COLOR)
        self.assertNotEqual(colors[2], FRAGMENT_NODE_COLOR)
        for size in sizes:
            self.assertIn(size, (NODE_SIZE, NODE_SIZE * CONTRACTED_SIZE_FRAC))

    def test_contracted_nodes_are_shrunk_not_recoloured(self):
        """Same colour = same tree; smaller = detail the metrics collapsed."""
        from semlaflow.validation.plot import (
            CONTRACTED_SIZE_FRAC, NODE_SIZE, _overlay_styles,
        )

        G = nx.Graph()
        for i in range(5):
            G.add_node(i, pos=np.array([float(i) * 10.0, 0.0, 0.0]))
        G.add_edges_from((i, i + 1) for i in range(4))
        G.graph["root"] = 0

        prov = sanitise_provenance(G)
        colors, sizes, _ec, _es = _overlay_styles(G, prov, "#8b1e3f")
        small = [s for s in sizes if np.isclose(s, NODE_SIZE * CONTRACTED_SIZE_FRAC)]
        self.assertEqual(len(small), len(prov["contracted_nodes"]))
        self.assertEqual(set(colors), {"#8b1e3f"}, "contracted nodes keep the row colour")

    def test_edge_maps_cover_excess_and_fragment_only(self):
        from semlaflow.validation.plot import (
            EXCESS_EDGE_COLOR, FRAGMENT_EDGE_COLOR, _overlay_styles,
        )

        G = _chorded_and_fragmented()
        prov = sanitise_provenance(G)
        _c, _s, edge_colors, edge_styles = _overlay_styles(G, prov, "#8b1e3f")
        self.assertEqual(
            {k for k, v in edge_colors.items() if v == EXCESS_EDGE_COLOR},
            prov["excess_edges"],
        )
        self.assertEqual(
            {k for k, v in edge_colors.items() if v == FRAGMENT_EDGE_COLOR},
            prov["fragment_edges"],
        )
        # Only fragment edges are dashed; kept edges are absent from both maps entirely.
        self.assertEqual(set(edge_styles), prov["fragment_edges"])
        self.assertEqual(len(edge_colors), len(prov["excess_edges"]) + len(prov["fragment_edges"]))


class TestOverlayRendering(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_overlay_renders_fragments_and_excess(self):
        import matplotlib.colors as mcolors
        from semlaflow.validation.plot import EXCESS_EDGE_COLOR, FRAGMENT_NODE_COLOR

        G = _chorded_and_fragmented()
        fig, _ = plot_graph_grid_angles(
            [G], angles=[(20, 30)], max_graphs=1, sanitise_overlay=True
        )
        # mplot3d only populates the full per-point facecolor array during the draw; before
        # it, get_facecolor() returns a truncated view (matplotlib 3.10).
        fig.canvas.draw()
        ax = fig.axes[0]

        facecolors = ax.collections[-1].get_facecolor()
        self.assertEqual(len(facecolors), G.number_of_nodes())
        frag_rgba = mcolors.to_rgba(FRAGMENT_NODE_COLOR)
        # Count, not position: the draw re-sorts the collection by depth.
        n_frag = sum(1 for c in facecolors if np.allclose(c[:3], frag_rgba[:3]))
        self.assertEqual(n_frag, 2, "both fragment nodes should be grey")

        excess_rgba = mcolors.to_rgba(EXCESS_EDGE_COLOR)
        n_excess = sum(
            1 for ln in ax.get_lines()
            if np.allclose(mcolors.to_rgba(ln.get_color()), excess_rgba)
        )
        self.assertEqual(n_excess, 1, "the chord should be the one orange edge")

    def test_fragment_edges_are_dashed(self):
        """Colour alone is not enough -- a solid pale line reads as a real long branch."""
        G = _chorded_and_fragmented()
        fig, _ = plot_graph_grid_angles(
            [G], angles=[(20, 30)], max_graphs=1, sanitise_overlay=True
        )
        dashed = [ln for ln in fig.axes[0].get_lines() if ln.get_linestyle() != "-"]
        self.assertEqual(len(dashed), 1)

    def test_overlay_off_keeps_a_single_colour(self):
        fig, _ = plot_graph_grid_angles([_tree(seed=0)], angles=[(20, 30)], max_graphs=1)
        self.assertEqual(len(fig.axes[0].collections[-1].get_facecolor()), 1)
        self.assertEqual(len(fig.legends), 0)
        for line in fig.axes[0].get_lines():
            self.assertEqual(line.get_linestyle(), "-")

    def test_overlay_adds_a_legend(self):
        fig, _ = plot_graph_grid_angles(
            [_tree(seed=0)], angles=[(20, 30)], max_graphs=1, sanitise_overlay=True
        )
        self.assertEqual(len(fig.legends), 1)
        labels = [t.get_text() for t in fig.legends[0].get_texts()]
        for expected in ("kept (critical tree)", "excess edge (MST-cut)", "fragment edge"):
            self.assertIn(expected, labels)


class TestValidationPlotModes(unittest.TestCase):
    def setUp(self):
        plt.close("all")

    def tearDown(self):
        plt.close("all")

    def test_unconditional_logs_plain_grids_only(self):
        model, logger = _stub_model()
        model._log_validation_plots()
        self.assertEqual(set(logger.images), {"val-plot-examples", "val-plot-examples-gt"})

    def test_class_conditioned_logs_one_row_per_class(self):
        classes = [0, 1, 2, 0, 1, 2]
        model, logger = _stub_model(class_hidden=16, classes=classes)
        model._log_validation_plots()
        self.assertEqual(set(logger.images), {"val-plot-class", "val-plot-class-gt"})
        fig = logger.images["val-plot-class"][0][0]
        titles = [ax.get_title() for ax in fig.axes]
        # 3 distinct classes x 3 default angles, labelled by name not id.
        self.assertEqual(len(fig.axes), 9)
        for name in ("23P", "4P", "5P-IT"):
            self.assertTrue(any(t.startswith(f"Gen {name} ") for t in titles), name)

    def test_tmd_conditioned_logs_interleaved_pairs(self):
        model, logger = _stub_model(tmd_dim=512, max_rows=4)
        model._log_validation_plots()
        self.assertEqual(set(logger.images), {"val-plot-tmd_pairs"})
        fig = logger.images["val-plot-tmd_pairs"][0][0]
        titles = [ax.get_title() for ax in fig.axes]
        # max_rows=4 -> 2 pairs -> 4 rows x 3 angles.
        self.assertEqual(len(fig.axes), 12)
        self.assertTrue(any(t.startswith("Gen #0") for t in titles))
        self.assertTrue(any(t.startswith("GT #0") for t in titles))

    def test_interleaved_pairs_are_colour_coded(self):
        """Gen and GT rows alternate, so title alone is not enough to tell them apart."""
        from semlaflow.validation.plot import GT_COLOR, PRED_COLOR

        model, logger = _stub_model(tmd_dim=512, max_rows=4)
        model._log_validation_plots()
        fig = logger.images["val-plot-tmd_pairs"][0][0]
        colour_by_row = {}
        for ax in fig.axes:
            label = ax.get_title().split(" - ")[0]
            # The node scatter is the last collection added by _plot_graph. With the
            # sanitisation overlay on it carries one RGBA row per node, so take the first
            # rather than the whole array -- rows can differ in node count.
            colour_by_row[label] = ax.collections[-1].get_facecolor()[0]
        for gen_row, gt_row in (("Gen #0", "GT #0"), ("Gen #1", "GT #1")):
            self.assertFalse(
                np.allclose(colour_by_row[gen_row], colour_by_row[gt_row]),
                f"{gen_row} and {gt_row} render in the same colour",
            )
        self.assertNotEqual(PRED_COLOR, GT_COLOR)

    def test_class_and_tmd_conditioning_are_independent(self):
        """A run with both conditioners must get both figures, not just the first."""
        model, logger = _stub_model(class_hidden=16, tmd_dim=512, classes=[0, 1, 2, 0, 1, 2])
        model._log_validation_plots()
        self.assertEqual(
            set(logger.images),
            {"val-plot-class", "val-plot-class-gt", "val-plot-tmd_pairs"},
        )

    def test_caption_carries_the_epoch(self):
        model, logger = _stub_model()
        model._log_validation_plots()
        self.assertEqual(logger.images["val-plot-examples"][1], ["epoch 7"])

    def test_row_cap_is_respected(self):
        model, logger = _stub_model(n_gen=20, max_rows=3)
        model._log_validation_plots()
        fig = logger.images["val-plot-examples"][0][0]
        self.assertEqual(len(fig.axes), 9)  # 3 rows x 3 angles

    def test_no_figures_leak(self):
        model, _ = _stub_model(class_hidden=16, tmd_dim=512, classes=[0, 1, 2, 0, 1, 2])
        model._log_validation_plots()
        self.assertEqual(len(plt.get_fignums()), 0)

    def test_disabled_by_flag(self):
        model, logger = _stub_model(val_plots=False)
        model._log_validation_plots()
        self.assertEqual(logger.images, {})

    def test_non_zero_rank_emits_nothing(self):
        model, logger = _stub_model(is_global_zero=False)
        model._log_validation_plots()
        self.assertEqual(logger.images, {})
        self.assertEqual(len(plt.get_fignums()), 0)

    def test_logger_without_log_image_is_skipped(self):
        """CSVLogger and --trial_run (no logger) must not break validation."""
        model, _ = _stub_model()
        model.loggers = [types.SimpleNamespace()]  # no log_image attribute
        model._log_validation_plots()
        self.assertEqual(len(plt.get_fignums()), 0)

        model, _ = _stub_model()
        model.loggers = []
        model._log_validation_plots()
        self.assertEqual(len(plt.get_fignums()), 0)

    def test_empty_accumulators_are_a_noop(self):
        model, logger = _stub_model(n_gen=0)
        model._log_validation_plots()
        self.assertEqual(logger.images, {})


if __name__ == "__main__":
    unittest.main()
