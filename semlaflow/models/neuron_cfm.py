"""NeuronCFM: thin MolecularCFM subclass for non-molecular graph domains.

Strips RDKit-dependent metrics (validity, energy, stability, novelty) and
replaces validation with a pure loss evaluation. Training losses are unchanged.
"""

from __future__ import annotations

import torch
from torchmetrics import MetricCollection

from semlaflow.models.fm import MolecularCFM


class NeuronCFM(MolecularCFM):
    """Flow matching model for neuron/tree graph data.

    Behaves identically to MolecularCFM during training. At validation time
    only the four raw loss components (coord/type/bond/charge) are logged; no
    molecules are decoded via RDKit, so neuron data with single-token vocab
    and binary edges does not crash the RDKit-based metric stack.
    """

    def __init__(self, *args, **kwargs):
        # Force-disable molecular post-processing paths.
        kwargs["pairwise_metrics"] = False
        kwargs["train_smiles"] = None
        super().__init__(*args, **kwargs)

        # Replace the RDKit-driven metric collections with empty ones so any
        # stray .update() calls in base-class code are no-ops.
        self.stability_metrics = MetricCollection({}, compute_groups=False)
        self.gen_metrics = MetricCollection({}, compute_groups=False)

    def validation_step(self, batch, b_idx):
        _, data, interpolated, times = batch

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
        return loss

    def on_validation_epoch_end(self):
        # No RDKit metrics to finalise.
        return
