"""Tests for the TMD conditioning filtration set, its provenance, and the matched-pair eval.

Data-free: every graph here is synthetic, so this suite runs anywhere.

The load-bearing property is rotation invariance. SemlaFlow applies a uniformly random 3D
rotation on every `__getitem__` while the descriptor is computed once at preprocess time,
so a filtration that is not rotation-invariant conditions the model on an axis its input
coordinates no longer agree with. That is why `height`/`rho` are rejected, and why
`test_conditioning_vector_is_rotation_invariant` is the test that must never be relaxed.
"""

import math
import unittest

import networkx as nx
import numpy as np
import torch

from semlaflow.tmd import (
    NEURON_TMD_DIM,
    NEURON_TMD_FILTRATIONS,
    NEURON_TMD_N_BINS,
    NEURON_TMD_SUPPORTED_FILTRATIONS,
    compute_neuron_tmd,
    neuron_tmd_dim,
    validate_filtrations,
)
from semlaflow.util.molrepr import GeometricMol
from semlaflow.validation import tmd_conditional_eval as tce


def _binary_tree_mol(depth: int = 4, seed: int = 0, scale: float = 10.0) -> GeometricMol:
    """A synthetic rooted binary tree as a GeometricMol (node 0 = soma, as swc.py emits)."""
    rng = np.random.default_rng(seed)
    coords = [np.zeros(3)]
    edges: list[tuple[int, int]] = []
    frontier = [0]
    for _ in range(depth):
        new_frontier = []
        for parent in frontier:
            for _ in range(2):
                idx = len(coords)
                coords.append(coords[parent] + rng.normal(0.0, scale, 3))
                edges.append((parent, idx))
                new_frontier.append(idx)
        frontier = new_frontier

    coords_t = torch.tensor(np.stack(coords), dtype=torch.float32)
    edges_t = torch.tensor(edges, dtype=torch.long)
    return GeometricMol(
        coords_t,
        torch.full((coords_t.size(0),), 2, dtype=torch.long),
        bond_indices=edges_t,
        bond_types=torch.ones(edges_t.size(0), dtype=torch.long),
    )


def _binary_tree_graph(depth: int = 4, seed: int = 0, scale: float = 10.0) -> nx.Graph:
    """The same tree as a rooted networkx graph, as the validation suite consumes them."""
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
                G.add_node(idx, pos=np.asarray(G.nodes[parent]["pos"]) + rng.normal(0.0, scale, 3))
                G.add_edge(parent, idx)
                new_frontier.append(idx)
        frontier = new_frontier
    return G


class TestFiltrationSet(unittest.TestCase):
    def test_dim_is_bins_squared_per_filtration(self):
        for filts in (("path",), ("radial_root",), ("path", "radial_root")):
            self.assertEqual(
                neuron_tmd_dim(filts), len(filts) * NEURON_TMD_N_BINS ** 2
            )
        self.assertEqual(NEURON_TMD_DIM, neuron_tmd_dim(NEURON_TMD_FILTRATIONS))

    def test_computed_vector_matches_declared_dim(self):
        mol = _binary_tree_mol()
        for filts in (("path",), ("path", "radial_root")):
            vec = compute_neuron_tmd(mol, filtrations=filts)
            self.assertEqual(vec.shape[0], neuron_tmd_dim(filts))
            self.assertTrue(bool(torch.isfinite(vec).all()))

    def test_only_rotation_invariant_filtrations_are_supported(self):
        self.assertEqual(NEURON_TMD_SUPPORTED_FILTRATIONS, ("path", "radial_root"))
        for bad in (["height"], ["path", "rho"], ["nonsense"]):
            with self.assertRaises(ValueError):
                validate_filtrations(bad)

    def test_duplicate_and_empty_filtration_sets_are_rejected(self):
        # A duplicate would silently double the width with a redundant block.
        with self.assertRaises(ValueError):
            validate_filtrations(["path", "path"])
        with self.assertRaises(ValueError):
            validate_filtrations([])

    def test_filtration_blocks_are_not_redundant(self):
        """path and radial_root must carry different information, or the second is dead weight."""
        vec = compute_neuron_tmd(_binary_tree_mol(), filtrations=("path", "radial_root"))
        block = NEURON_TMD_N_BINS ** 2
        self.assertFalse(torch.allclose(vec[:block], vec[block:]))

    def test_conditioning_vector_is_rotation_invariant(self):
        """The property the whole filtration choice rests on (see module docstring)."""
        mol = _binary_tree_mol()
        base = compute_neuron_tmd(mol)
        for rotation in ((0.7, 1.9, 2.6), (math.pi, 0.0, 0.5), (0.1, 0.2, 6.0)):
            rotated = compute_neuron_tmd(mol.rotate(rotation))
            self.assertTrue(
                torch.allclose(base, rotated, atol=1e-4),
                f"TMD vector changed under rotation {rotation}: "
                f"max |diff| = {float((base - rotated).abs().max())}",
            )

    def test_degenerate_graph_yields_zero_vector(self):
        """A malformed sample must not crash preprocessing."""
        cyclic = GeometricMol(
            torch.zeros(3, 3),
            torch.full((3,), 2, dtype=torch.long),
            bond_indices=torch.tensor([[0, 1], [1, 2], [2, 0]], dtype=torch.long),
            bond_types=torch.ones(3, dtype=torch.long),
        )
        vec = compute_neuron_tmd(cyclic)
        self.assertEqual(vec.shape[0], NEURON_TMD_DIM)
        self.assertEqual(float(vec.abs().sum()), 0.0)


class TestProvenance(unittest.TestCase):
    """Two filtration sets of the same width are otherwise indistinguishable."""

    def test_round_trips_through_smol_bytes(self):
        mol = GeometricMol(
            torch.randn(5, 3), torch.zeros(5, dtype=torch.long),
            tmd=torch.zeros(512), tmd_filtrations=("path", "radial_root"),
        )
        restored = GeometricMol.from_bytes(mol.to_bytes())
        self.assertEqual(restored.tmd_filtrations, ("path", "radial_root"))
        self.assertEqual(restored.tmd.shape[0], 512)

    def test_survives_the_training_transform(self):
        """`neuron_mol_transform` rebuilds the mol on every __getitem__ via _copy_with."""
        mol = GeometricMol(
            torch.randn(5, 3), torch.zeros(5, dtype=torch.long),
            tmd=torch.zeros(512), tmd_filtrations=("path", "radial_root"),
        )
        transformed = mol.scale(0.5).rotate((0.3, 0.4, 0.5)).zero_com()
        self.assertEqual(transformed.tmd_filtrations, ("path", "radial_root"))
        self.assertEqual(transformed.tmd.shape[0], 512)

    def test_absent_on_legacy_smol(self):
        """Pre-provenance .smol files must still load; consumers warn rather than raise."""
        legacy = GeometricMol(
            torch.randn(5, 3), torch.zeros(5, dtype=torch.long), tmd=torch.zeros(256)
        )
        self.assertIsNone(GeometricMol.from_bytes(legacy.to_bytes()).tmd_filtrations)


class TestConditionalPairwiseEval(unittest.TestCase):
    def test_no_pairs_returns_empty_dict(self):
        self.assertEqual(tce.compute_conditional_pairwise_metrics([], []), {})

    def test_identical_pairs_score_zero(self):
        graphs = [_binary_tree_graph(seed=s) for s in range(4)]
        out = tce.compute_conditional_pairwise_metrics(graphs, graphs)
        for key in ("axial_extent_absdiff_mean", "radial_span_absdiff_mean",
                    "total_extent_absdiff_mean", "branch_length_w1_pairwise_mean",
                    "bifurcation_angle_w1_pairwise_mean"):
            self.assertAlmostEqual(out[key], 0.0, places=6, msg=key)
        for filt in NEURON_TMD_FILTRATIONS:
            self.assertAlmostEqual(out[f"pd_wasserstein_{filt}_mean"], 0.0, places=5)
        self.assertEqual(out["n_pairs"], 4.0)
        self.assertEqual(out["n_pairs_skipped"], 0.0)

    def test_mismatched_pairs_score_above_zero(self):
        gt = [_binary_tree_graph(seed=s, scale=10.0) for s in range(4)]
        gen = [_binary_tree_graph(seed=s + 50, scale=40.0) for s in range(4)]
        out = tce.compute_conditional_pairwise_metrics(gen, gt)
        self.assertGreater(out["axial_extent_absdiff_mean"], 0.0)
        self.assertGreater(out["branch_length_w1_pairwise_mean"], 0.0)

    def test_max_pairs_caps_the_work(self):
        graphs = [_binary_tree_graph(seed=s) for s in range(10)]
        out = tce.compute_conditional_pairwise_metrics(graphs, graphs, max_pairs=3)
        self.assertEqual(out["n_pairs"], 3.0)

    def test_keys_match_the_declared_registry(self):
        graphs = [_binary_tree_graph(seed=s) for s in range(2)]
        out = tce.compute_conditional_pairwise_metrics(graphs, graphs)
        self.assertEqual(set(out), set(tce.conditional_metric_keys()))

    def test_degrades_without_persim(self):
        """PD distances go nan; every other metric must still compute."""
        graphs = [_binary_tree_graph(seed=s) for s in range(3)]
        original = tce._persim
        try:
            tce._persim = None
            out = tce.compute_conditional_pairwise_metrics(graphs, graphs)
        finally:
            tce._persim = original
        for filt in NEURON_TMD_FILTRATIONS:
            self.assertTrue(math.isnan(out[f"pd_wasserstein_{filt}_mean"]))
            self.assertEqual(out[f"pd_nan_frac_{filt}"], 1.0)
        self.assertAlmostEqual(out["axial_extent_absdiff_mean"], 0.0, places=6)

    def test_sanitises_a_disconnected_generated_graph(self):
        """Pairing must survive the connected-component step -- sanitise never drops a graph."""
        gt = [_binary_tree_graph(seed=s) for s in range(3)]
        gen = [g.copy() for g in gt]
        # Strand a fragment off the first generated graph.
        gen[0].add_node(999, pos=np.array([500.0, 500.0, 500.0]))
        gen[0].add_node(998, pos=np.array([505.0, 500.0, 500.0]))
        gen[0].add_edge(999, 998)
        out = tce.compute_conditional_pairwise_metrics(gen, gt)
        self.assertEqual(out["n_pairs"], 3.0)
        self.assertEqual(out["n_pairs_skipped"], 0.0)
        self.assertTrue(math.isfinite(out["axial_extent_absdiff_mean"]))


if __name__ == "__main__":
    unittest.main()
