"""NeuronCFM: thin MolecularCFM subclass for non-molecular graph domains.

Strips RDKit-dependent metrics (validity, energy, stability, novelty) and
replaces validation with a pure loss evaluation. Training losses are unchanged.

In addition to the single-step `val-loss`, the validation loop optionally runs a full ODE
generation rollout over the val set and scores it against the ground-truth val graphs
with `validation.dist_metrics`: W1 marginals, the joint morphometric block
(`mmd_morpho` / density / coverage), a TMD persistence block, two scale-free normalised
aggregates, and a structural health block. Mirrors the offline
`sample_neurons.py:evaluate_samples` path.

`val-morpho-selection` (mmd_morpho, gated on the health fractions) is available as a
checkpoint monitor; see `train.py`. `val-loss` remains the default.
"""

from __future__ import annotations

import math
import time

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

    def __init__(self, *args, val_structural_metrics: bool = True,
                 per_cell_class: bool = True, per_cell_class_min_count: int = 20,
                 metric_report_level: str = "standard",
                 selection_health_max: dict | None = None,
                 tmd_cond_eval: bool = True, tmd_cond_every: int = 5,
                 tmd_cond_max_pairs: int = 64, **kwargs):
        # Force-disable molecular post-processing paths.
        kwargs["pairwise_metrics"] = False
        kwargs["train_smiles"] = None
        # Persist the toggles in hparams (saved via **kwargs in the base class) so a
        # reloaded checkpoint reconstructs with the same setting.
        kwargs["val_structural_metrics"] = val_structural_metrics
        kwargs["per_cell_class"] = per_cell_class
        kwargs["per_cell_class_min_count"] = per_cell_class_min_count
        kwargs["metric_report_level"] = metric_report_level
        kwargs["selection_health_max"] = selection_health_max
        kwargs["tmd_cond_eval"] = tmd_cond_eval
        kwargs["tmd_cond_every"] = tmd_cond_every
        kwargs["tmd_cond_max_pairs"] = tmd_cond_max_pairs
        super().__init__(*args, **kwargs)

        self.val_structural_metrics = val_structural_metrics
        self.per_cell_class = per_cell_class
        self.per_cell_class_min_count = per_cell_class_min_count
        self.metric_report_level = metric_report_level
        self.selection_health_max = dict(selection_health_max or {})
        # Matched-pair TMD conditioning fidelity. Only meaningful when the model is TMD
        # conditioned, and expensive per pair (a persim Wasserstein per filtration), so it
        # is both gated on `gen.tmd_dim > 0` and run every `tmd_cond_every` epochs.
        self.tmd_cond_eval = tmd_cond_eval
        self.tmd_cond_every = max(1, int(tmd_cond_every))
        self.tmd_cond_max_pairs = int(tmd_cond_max_pairs)

        # Replace the RDKit-driven metric collections with empty ones so any
        # stray .update() calls in base-class code are no-ops.
        self.stability_metrics = MetricCollection({}, compute_groups=False)
        self.gen_metrics = MetricCollection({}, compute_groups=False)

        # Per-epoch accumulators for the structural distribution metrics (plain lists;
        # see on_validation_epoch_end for the single-GPU vs DDP note).
        self._val_gen_graphs = []
        self._val_gt_graphs = []
        # Parallel per-graph class labels (populated only for a class-conditioned run).
        self._val_gen_classes = []
        self._val_gt_classes = []
        # Sampling-time accumulators: seconds spent inside `_generate` this epoch and the
        # number of graphs the ODE integrated. Plain per-rank scalars; reduced via sync_dist
        # in on_validation_epoch_end (same monitoring-only approximation as the W1 metrics).
        self._val_sampling_seconds = 0.0
        self._val_sampling_count = 0
        # GT-derived fit (morpho mean/std, MMD bandwidths, TMD PCA) for the joint metrics.
        # Built ONCE on the first real validation epoch and reused -- see
        # on_validation_epoch_end for why that is valid and why it must skip the
        # sanity-check loop.
        self._gt_cache = None

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
        # Skipped during the sanity check, which scores nothing (see
        # on_validation_epoch_end) -- this just avoids paying for a rollout twice.
        if self.val_structural_metrics and not getattr(self.trainer, "sanity_checking", False):
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

        # Time only the ODE rollout (strict "sampling" cost). CUDA kernels are async, so
        # synchronise on both sides or we'd time kernel launches, not the actual work.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        _t0 = time.perf_counter()
        with torch.no_grad():
            gen = self._generate(prior, self.integrator.steps, self.sampling_strategy)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._val_sampling_seconds += time.perf_counter() - _t0
        self._val_sampling_count += prior["coords"].size(0)

        gen_mols = samples_to_mols(gen, NEURON_EDGE_CLASS_INDEX)
        gt_mols = samples_to_mols(data, NEURON_EDGE_CLASS_INDEX)

        gen_graphs = [geometric_mol_to_nx(m, coord_scale=1.0) for m in gen_mols]
        gt_graphs = [geometric_mol_to_nx(m, coord_scale=self.coord_scale) for m in gt_mols]
        self._val_gen_graphs.extend(gen_graphs)
        self._val_gt_graphs.extend(gt_graphs)

        # Stash class labels for per-cell-class stratification. Each generated graph's class is the
        # paired GT graph's class (gen is conditioned on it), so both lists get the same labels.
        # samples_to_mols drops all-masked graphs; align labels to the graphs actually built.
        if self.per_cell_class and self.gen.class_hidden > 0 and data.get("cell_class") is not None:
            masks = data["mask"].detach().cpu().bool()
            labels = data["cell_class"].detach().cpu().reshape(-1).tolist()
            kept = [int(labels[b]) for b in range(masks.size(0)) if int(masks[b].sum()) > 0]
            self._val_gen_classes.extend(kept)
            self._val_gt_classes.extend(kept)

    def _log_run_constants(self, metrics):
        """Push the per-run constants to logger config, once.

        `mmd_bandwidth_*`, `tmd_eff_rank`, `morpho_gt_nan_frac` and `morpho_version` are
        fixed by the GT fit, so logging them as a time series is pure noise -- but they
        must be recorded somewhere, because `mmd_morpho` is meaningless without the
        bandwidth and `morpho_version` is what tells you whether two runs' `mmd_morpho`
        may share a plot axis.
        """
        from semlaflow.validation.dist_metrics import CONSTANT_KEYS

        if getattr(self, "_logged_run_constants", False):
            return
        payload = {
            f"metrics/{k}": metrics[k]
            for k in CONSTANT_KEYS
            if k in metrics and math.isfinite(metrics[k])
        }
        if not payload:
            return
        for logger in (self.loggers or []):
            try:
                logger.log_hyperparams(payload)
            except Exception:
                # A logger without hyperparameter support must not break validation.
                pass
        self._logged_run_constants = True

    def on_validation_epoch_start(self):
        self._val_gen_graphs = []
        self._val_gt_graphs = []
        self._val_gen_classes = []
        self._val_gt_classes = []
        self._val_sampling_seconds = 0.0
        self._val_sampling_count = 0

    def on_validation_epoch_end(self):
        if not self.val_structural_metrics:
            return
        if not self._val_gen_graphs or not self._val_gt_graphs:
            return

        # Skip the whole block during Lightning's sanity check. It runs only 2 batches, so
        # every number it produced would be computed from a handful of graphs -- and, more
        # importantly, `build_gt_cache` fitted there would derive morpho_mean/std/sigma
        # from those few graphs and poison every subsequent epoch. This is the single
        # easiest thing to get wrong in the whole suite.
        if getattr(self.trainer, "sanity_checking", False):
            return

        from semlaflow.validation.dist_metrics import (
            build_gt_cache,
            compute_distribution_metrics,
            keys_for_level,
            subset_gt_cache,
        )

        # Fit the GT-derived objects ONCE and reuse them for the rest of the run, so the
        # MMD trajectory is comparable across checkpoints (a per-epoch bandwidth would
        # make the curve meaningless). Valid despite the transform's per-epoch random
        # rotation: every morphometric is rotation invariant to ~1e-16 and `choose_root`
        # was measured to flip on 0.0% of graphs under rotation.
        if self._gt_cache is None:
            self._gt_cache = build_gt_cache(self._val_gt_graphs)

        # NOTE: accumulation uses plain per-rank lists (not torchmetrics). Exact on a
        # single GPU; under DDP each rank scores its own shard and sync_dist averages the
        # per-rank values. That is an approximation for the W1 marginals and an outright
        # category error for MMD (the mean of per-rank MMDs is not the MMD of the union),
        # which is why train.py warns when the morpho checkpoint monitor is used with
        # world_size > 1.
        metrics = compute_distribution_metrics(
            self._val_gen_graphs, self._val_gt_graphs, gt_cache=self._gt_cache
        )

        # `headline` keys also go to the progress bar. Each key must be logged exactly
        # once per epoch -- Lightning rejects a second self.log() with different kwargs.
        headline = set(keys_for_level("headline"))
        for key in keys_for_level(self.metric_report_level):
            val = metrics.get(key)
            if val is not None and math.isfinite(val):
                self.log(f"val-{key}", val, on_epoch=True, logger=True,
                         prog_bar=key in headline, sync_dist=True)

        # `val-disconnected-frac` predates the health block and is on existing dashboards,
        # so keep the old key name alive alongside the new `val-disconnected_frac`.
        disc = metrics.get("disconnected_frac")
        if disc is not None and math.isfinite(disc):
            self.log("val-disconnected-frac", disc, on_epoch=True, logger=True, sync_dist=True)

        # Checkpoint-selection signal. Sanitisation deliberately makes mmd_morpho blind to
        # structural failure (it scores the repaired tree), so a model emitting
        # garbage-with-a-plausible-MST would win on mmd_morpho alone. Gate it on the raw
        # health fractions; +inf simply loses to any healthy epoch.
        # (`val-mmd_morpho` itself is already logged by the tier loop above.)
        mmd = metrics.get("mmd_morpho")
        if mmd is not None and math.isfinite(mmd):
            breached = [
                k for k, limit in self.selection_health_max.items()
                if math.isfinite(metrics.get(k, float("nan"))) and metrics[k] > limit
            ]
            self.log(
                "val-morpho-selection",
                float("inf") if breached else mmd,
                on_epoch=True, logger=True, sync_dist=True,
            )

        # Per-run constants go to logger config, not the time series.
        self._log_run_constants(metrics)

        # Matched-pair TMD conditioning fidelity. Everything above is distributional and,
        # now that the evaluation filtration also sits in the conditioning set, partly
        # scores the model echoing its own input. These per-pair numbers are the evidence
        # that the conditioning is actually being followed, so they are worth their cost --
        # but only when the model is conditioned at all.
        if (
            self.tmd_cond_eval
            and getattr(self.gen, "tmd_dim", 0) > 0
            and (self.current_epoch + 1) % self.tmd_cond_every == 0
        ):
            from semlaflow.validation.tmd_conditional_eval import (
                compute_conditional_pairwise_metrics,
            )

            # `gen_graphs[i]` was conditioned on `gt_graphs[i]`: both lists are appended in
            # batch order from the same validation_step, and no permutation is applied.
            cond_metrics = compute_conditional_pairwise_metrics(
                self._val_gen_graphs,
                self._val_gt_graphs,
                max_pairs=self.tmd_cond_max_pairs,
            )
            for key, val in cond_metrics.items():
                if math.isfinite(val):
                    self.log(f"val-tmd_cond-{key}", val, on_epoch=True,
                             logger=True, sync_dist=True)

        # Average ODE-rollout sampling time per generated graph. Per-rank ratio; sync_dist
        # averages across ranks (same monitoring-only approximation as the W1 metrics above).
        if self._val_sampling_count > 0:
            self.log(
                "val-sampling-seconds-per-sample",
                self._val_sampling_seconds / self._val_sampling_count,
                on_epoch=True, logger=True, sync_dist=True,
            )

        # Per-neuron-type stratified metrics (class-conditioned runs only). Group the accumulated
        # gen/GT graphs by class label and score each subset separately; logged as
        # val-class_<name>-<key> (monitoring only, not used for checkpoint selection).
        if (
            self.per_cell_class
            and self._val_gen_classes
            and len(self._val_gen_classes) == len(self._val_gen_graphs)
            and len(self._val_gt_classes) == len(self._val_gt_graphs)
        ):
            from semlaflow.scriptutil import class_label

            # Names are per-corpus (neuron cell types vs tree genera); resolve via the dataset
            # recorded in hparams by train.py. Falls back to `id<N>` for older checkpoints.
            dataset = self.hparams.get("dataset")
            classes = sorted({c for c in self._val_gen_classes})
            for c in classes:
                cname = class_label(dataset, c)
                gen_c = [g for g, gc in zip(self._val_gen_graphs, self._val_gen_classes) if gc == c]
                gt_c = [g for g, gc in zip(self._val_gt_graphs, self._val_gt_classes) if gc == c]
                if len(gen_c) < self.per_cell_class_min_count or len(gt_c) < self.per_cell_class_min_count:
                    continue
                # The GT reference must be THIS class's graphs, but the standardization,
                # bandwidths and TMD PCA stay run-wide -- otherwise the per-class numbers
                # are comparable neither to each other nor to the run-wide number.
                cache_c = (
                    subset_gt_cache(self._gt_cache, gt_c)
                    if self._gt_cache is not None else None
                )
                m_c = compute_distribution_metrics(gen_c, gt_c, gt_cache=cache_c)
                for key in keys_for_level(self.metric_report_level):
                    val = m_c.get(key)
                    if val is not None and math.isfinite(val):
                        self.log(f"val-class_{cname}-{key}", val, on_epoch=True,
                                 logger=True, sync_dist=True)

        self._val_gen_graphs = []
        self._val_gt_graphs = []
        self._val_gen_classes = []
        self._val_gt_classes = []
        self._val_sampling_seconds = 0.0
        self._val_sampling_count = 0
