"""NeuronCFM: thin MolecularCFM subclass for non-molecular graph domains.

Strips RDKit-dependent metrics (validity, energy, stability, novelty) and
replaces validation with a pure loss evaluation. Training losses are unchanged.

In addition to the single-step `val-loss` (which drives checkpoint selection), the
validation loop optionally runs a full ODE generation rollout over the val set and
logs Wasserstein-1 structural distribution metrics (branch length, bifurcation
angle/count, leaf/node counts, extents) vs the ground-truth val graphs. These are
logged every validation so their trajectory over training is visible, mirroring the
offline `sample_neurons.py:evaluate_samples` path. They are *not* monitored for
checkpoint selection -- they exist to let a better checkpoint be chosen post-hoc.
"""

from __future__ import annotations

import math

import torch
from torchmetrics import MetricCollection

from semlaflow.models.fm import MolecularCFM


class NeuronCFM(MolecularCFM):
    """Flow matching model for neuron/tree graph data.

    Behaves identically to MolecularCFM during training. At validation time the four
    raw loss components (coord/type/bond/charge) are logged as `val-loss` (no molecules
    are decoded via RDKit), and -- when `val_structural_metrics` is set -- generated
    graphs are scored against the ground-truth val graphs with geometry-only W1 metrics.
    """

    def __init__(self, *args, val_structural_metrics: bool = True, **kwargs):
        # Force-disable molecular post-processing paths.
        kwargs["pairwise_metrics"] = False
        kwargs["train_smiles"] = None
        # Persist the toggle in hparams (saved via **kwargs in the base class) so a
        # reloaded checkpoint reconstructs with the same setting.
        kwargs["val_structural_metrics"] = val_structural_metrics
        super().__init__(*args, **kwargs)

        self.val_structural_metrics = val_structural_metrics

        # Replace the RDKit-driven metric collections with empty ones so any
        # stray .update() calls in base-class code are no-ops.
        self.stability_metrics = MetricCollection({}, compute_groups=False)
        self.gen_metrics = MetricCollection({}, compute_groups=False)

        # Per-epoch accumulators for the structural distribution metrics (plain lists;
        # see on_validation_epoch_end for the single-GPU vs DDP note).
        self._val_gen_graphs = []
        self._val_gt_graphs = []

    def validation_step(self, batch, b_idx):
        prior, data, interpolated, times = batch

        cond_batch = None
        if self.self_condition:
            cond_batch = {
                "coords": torch.zeros_like(interpolated["coords"]),
                "atomics": torch.zeros_like(interpolated["atomics"]),
                "bonds": torch.zeros_like(interpolated["bonds"]),
            }

        with torch.no_grad():
            coords, types, bonds, charges = self(
                interpolated, times, training=False, cond_batch=cond_batch
            )

        predicted = {
            "coords": coords,
            "atomics": types,
            "bonds": bonds,
            "charges": charges,
        }

        losses = self._loss(data, interpolated, predicted)
        loss = sum(losses.values())

        for name, loss_val in losses.items():
            self.log(f"val-{name}", loss_val, on_epoch=True, logger=True)
        self.log("val-loss", loss, on_epoch=True, logger=True, prog_bar=True)

        # Generation-based structural metrics: full ODE rollout from the prior, scored
        # against the GT val graphs in on_validation_epoch_end. Accumulate here.
        if self.val_structural_metrics:
            self._accumulate_structural_graphs(prior, data)

        return loss

    def _accumulate_structural_graphs(self, prior, data):
        """Generate from the prior and stash gen + GT graphs for this val epoch.

        Coord scales: `_generate` rescales its output by `self.coord_scale` -> physical
        microns, so generated mols convert with `coord_scale=1.0`. The GT `data` coords
        are standardised (neuron_mol_transform divides by coord_std), so they convert
        with `coord_scale=self.coord_scale` to recover physical microns. The W1 metrics
        are rotation/translation invariant, so the random rotate/zero-com applied to GT
        during transform does not matter.
        """
        from semlaflow.data.swc import NEURON_EDGE_CLASS_INDEX
        from semlaflow.validation.convert import geometric_mol_to_nx, samples_to_mols

        with torch.no_grad():
            gen = self._generate(prior, self.integrator.steps, self.sampling_strategy)

        gen_mols = samples_to_mols(gen, NEURON_EDGE_CLASS_INDEX)
        gt_mols = samples_to_mols(data, NEURON_EDGE_CLASS_INDEX)

        self._val_gen_graphs.extend(
            geometric_mol_to_nx(m, coord_scale=1.0) for m in gen_mols
        )
        self._val_gt_graphs.extend(
            geometric_mol_to_nx(m, coord_scale=self.coord_scale) for m in gt_mols
        )

    def on_validation_epoch_start(self):
        self._val_gen_graphs = []
        self._val_gt_graphs = []

    def on_validation_epoch_end(self):
        if not self.val_structural_metrics:
            return
        if not self._val_gen_graphs or not self._val_gt_graphs:
            return

        import networkx as nx

        from semlaflow.validation.dist_metrics import METRIC_KEYS, compute_distribution_metrics

        # NOTE: accumulation uses plain per-rank lists (not torchmetrics). Exact on a
        # single GPU; under DDP each rank scores its own shard and sync_dist averages the
        # per-rank W1s (an approximation). This metric is for monitoring only -- checkpoint
        # selection still uses val-loss -- so the approximation is acceptable.
        metrics = compute_distribution_metrics(self._val_gen_graphs, self._val_gt_graphs)

        finite_w1 = []
        for key in METRIC_KEYS:
            val = metrics.get(key)
            if val is not None and math.isfinite(val):
                self.log(f"val-{key}", val, on_epoch=True, logger=True, sync_dist=True)
                finite_w1.append(val)

        if finite_w1:
            self.log(
                "val-structural-w1-mean",
                sum(finite_w1) / len(finite_w1),
                on_epoch=True, logger=True, prog_bar=True, sync_dist=True,
            )

        n_disc = sum(
            1 for g in self._val_gen_graphs
            if g.number_of_nodes() > 0 and not nx.is_connected(g)
        )
        disc_frac = n_disc / len(self._val_gen_graphs)
        self.log("val-disconnected-frac", disc_frac, on_epoch=True, logger=True, sync_dist=True)

        self._val_gen_graphs = []
        self._val_gt_graphs = []
