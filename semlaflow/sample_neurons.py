"""Sample neuron graphs from a trained NeuronCFM checkpoint.

Mirrors `semlaflow/predict.py` but:
  * uses `build_neuron_vocab()` / `neuron_mol_transform` / `NEURON_*` constants,
  * loads `NeuronCFM` (not `MolecularCFM`),
  * skips the RDKit-dependent `_generate_mols` / SDF path,
  * writes a single `.smol` (`GeometricMolBatch`) of raw predicted graphs —
    coords + predicted binary-edge adjacency — with no tree extraction.

Sampling uses a pure-noise prior (`fixed_time=None`), matching `predict.py`'s
eval interpolant. The source dataset (default `val.smol`) is consulted only to
supply the size distribution: the collator replaces each real mol with a
noise-sampled prior of matching shape.
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from time import perf_counter

import lightning as L
import torch
from tqdm import tqdm

import semlaflow.scriptutil as util
from semlaflow.data.datamodules import GeometricInterpolantDM
from semlaflow.data.datasets import GeometricDataset
from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler
from semlaflow.data.swc import NEURON_EDGE_CLASS_INDEX
from semlaflow.models.fm import Integrator
from semlaflow.models.neuron_cfm import NeuronCFM
from semlaflow.models.semla import EquiInvDynamics, SemlaGenerator
from semlaflow.util.molrepr import GeometricMol, GeometricMolBatch


DEFAULT_SAVE_FILE = "neuron_samples.smol"
DEFAULT_DATASET_SPLIT = "val"
DEFAULT_N_MOLECULES = 256
DEFAULT_BATCH_COST = 8192
DEFAULT_BUCKET_COST_SCALE = "linear"
DEFAULT_INTEGRATION_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_ODE_SAMPLING_STRATEGY = "log"
DEFAULT_SEED = 12345
DEFAULT_N_PLOT_EXAMPLES = 8


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(args, vocab):
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    hparams = checkpoint["hyper_parameters"]

    hparams["compile_model"] = False
    hparams["integration-steps"] = args.integration_steps
    hparams["sampling_strategy"] = args.ode_sampling_strategy

    n_bond_types = util.get_n_bond_types(hparams["integration-type-strategy"])

    if hparams.get("architecture") is None:
        hparams["architecture"] = "semla"

    if hparams["architecture"] != "semla":
        raise ValueError(
            f"sample_neurons only supports architecture='semla', got '{hparams['architecture']}'"
        )

    dynamics = EquiInvDynamics(
        hparams["d_model"],
        hparams["d_message"],
        hparams["n_coord_sets"],
        hparams["n_layers"],
        n_attn_heads=hparams["n_attn_heads"],
        d_message_hidden=hparams["d_message_hidden"],
        d_edge=hparams["d_edge"],
        self_cond=hparams["self_cond"],
        coord_norm=hparams["coord_norm"],
    )
    egnn_gen = SemlaGenerator(
        hparams["d_model"],
        dynamics,
        vocab.size,
        hparams["n_atom_feats"],
        d_edge=hparams["d_edge"],
        n_edge_types=n_bond_types,
        self_cond=hparams["self_cond"],
        size_emb=hparams["size_emb"],
        max_atoms=hparams["max_atoms"],
        tmd_dim=hparams.get("tmd_dim", 0),
        tmd_hidden=hparams.get("tmd_hidden", 0),
        n_classes=hparams.get("n_classes", 0),
        class_hidden=hparams.get("class_hidden", 0),
    )

    type_mask_index = (
        vocab.indices_from_tokens(["<MASK>"])[0]
        if hparams["integration-type-strategy"] == "mask"
        else None
    )
    bond_mask_index = (
        util.BOND_MASK_INDEX if hparams["integration-bond-strategy"] == "mask" else None
    )

    integrator = Integrator(
        args.integration_steps,
        type_strategy=hparams["integration-type-strategy"],
        bond_strategy=hparams["integration-bond-strategy"],
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        cat_noise_level=args.cat_sampling_noise_level,
    )

    fm_model = NeuronCFM.load_from_checkpoint(
        args.ckpt_path,
        gen=egnn_gen,
        vocab=vocab,
        integrator=integrator,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        **hparams,
    )
    return fm_model


def _split_path(data_path: str, split: str) -> Path:
    fname = {"train": "train.smol", "val": "val.smol", "test": "test.smol"}.get(split)
    if fname is None:
        raise ValueError(f"Unknown dataset_split '{split}'")
    return Path(data_path) / fname


def resolve_dataset(args, hparams) -> str:
    """The corpus this checkpoint was trained on -- drives coord scale, buckets and class names."""
    dataset = args.dataset or hparams.get("dataset")
    if dataset is None:
        raise SystemExit(
            "This checkpoint records no 'dataset' hyperparameter. Pass --dataset explicitly so "
            "the correct coord scale, bucket limits and class names are used."
        )
    return dataset


def build_dm(args, hparams, vocab):
    # Match the coord scale / buckets the checkpoint was trained with; each corpus has its own.
    cfg = util.get_dataset_config(resolve_dataset(args, hparams))
    coord_std = cfg.coord_std
    bucket_limits = cfg.bucket_limits

    n_bond_types = util.get_n_bond_types(hparams["integration-type-strategy"])
    transform = partial(
        util.neuron_mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std
    )

    dataset = GeometricDataset.load(_split_path(args.data_path, args.dataset_split), transform=transform)
    dataset = dataset.sample(args.n_molecules, replacement=True)

    type_mask_index = (
        vocab.indices_from_tokens(["<MASK>"])[0]
        if hparams["val-prior-type-noise"] == "mask"
        else None
    )
    bond_mask_index = (
        util.BOND_MASK_INDEX if hparams["val-prior-bond-noise"] == "mask" else None
    )

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise=hparams["val-prior-type-noise"],
        bond_noise=hparams["val-prior-bond-noise"],
        scale_ot=hparams["val-prior-noise-scale-ot"],
        zero_com=True,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
    )
    # No fixed_time => pure-noise prior at t=0 (matches predict.py, NOT train.py's
    # val interpolant which uses fixed_time=0.9 for validation loss).
    eval_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=hparams["val-type-interpolation"],
        bond_interpolation=hparams["val-bond-interpolation"],
        equivariant_ot=False,
        batch_ot=False,
    )
    dm = GeometricInterpolantDM(
        None,
        None,
        dataset,
        args.batch_cost,
        test_interpolant=eval_interpolant,
        bucket_limits=bucket_limits,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False,
    )
    return dm


def dm_from_ckpt(args, vocab):
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    hparams = checkpoint["hyper_parameters"]
    return build_dm(args, hparams, vocab), resolve_dataset(args, hparams)


def _add_per_class_metrics(metrics, gen_graphs, gen_classes, gt_graphs, gt_mols,
                           compute_distribution_metrics, min_count,
                           dataset=None, gt_cache=None, subset_fn=None) -> None:
    """Populate metrics[f"class_<name>"] with per-class distribution metrics + print a table.

    Generated graphs are grouped by their conditioning class (`gen_classes`), GT graphs by each
    mol's own `_cell_class`. Classes with fewer than `min_count` graphs on either side are skipped.

    `gt_cache` + `subset_fn` re-target the run-wide GT fit at each class's own GT graphs. The
    reference set must be per-class, but the standardization, MMD bandwidths and TMD PCA must
    stay run-wide or the classes are comparable neither to each other nor to the overall run.
    """
    if len(gen_classes) != len(gen_graphs):
        print(f"[eval] warning: gen_classes ({len(gen_classes)}) != gen_graphs "
              f"({len(gen_graphs)}); skipping per-class metrics.")
        return

    gt_classes = [
        int(m._cell_class) if getattr(m, "_cell_class", None) is not None else None
        for m in gt_mols
    ]
    classes = sorted({c for c in gen_classes if c is not None})

    print(f"\n[eval] per-cell-class metrics (min_count={min_count})")
    print(f"{'Class':<10}{'n_gen':>7}{'n_gt':>7}{'mmd_morpho':>12}{'w1_pooled_n':>13}")
    print("-" * 49)
    for c in classes:
        cname = util.class_label(dataset, c)
        gen_c = [g for g, gc in zip(gen_graphs, gen_classes) if gc == c]
        gt_c = [g for g, gc in zip(gt_graphs, gt_classes) if gc == c]
        if len(gen_c) < min_count or len(gt_c) < min_count:
            print(f"{cname:<10}{len(gen_c):>7}{len(gt_c):>7}{'(skip)':>12}{'':>13}")
            continue
        cache_c = subset_fn(gt_cache, gt_c) if (gt_cache is not None and subset_fn) else None
        m_c = compute_distribution_metrics(gen_c, gt_c, gt_cache=cache_c)
        m_c["n_generated"] = float(len(gen_c))
        m_c["n_ground_truth"] = float(len(gt_c))
        metrics[f"class_{cname}"] = m_c
        mmd = m_c.get("mmd_morpho", float("nan"))
        pooled = m_c.get("w1_pooled_mean_normalized", float("nan"))
        print(f"{cname:<10}{len(gen_c):>7}{len(gt_c):>7}{mmd:>12.5f}{pooled:>13.4f}")
    print()


def evaluate_samples(args, gen_mols: list[GeometricMol], save_dir: Path,
                     gen_classes: list[int] | None = None,
                     timing: dict | None = None, dataset: str | None = None) -> None:
    """Compare generated graphs against the ground-truth split: distribution metrics
    (Wasserstein-1 per structural stat) written to metrics.json + printed, and
    multi-azimuth plot grids of generated and GT samples saved as PNGs.

    When `gen_classes` is given (type-conditioned runs), also computes per-neuron-type
    stratified metrics: generated graphs are grouped by their conditioning class, GT graphs
    by their own `_cell_class` label, and each class is scored distribution-vs-distribution
    against that class's own GT, stored under `metrics["class_<name>"]`.

    Imports are local so the pure sampling path (and `--skip_eval`) pulls in no
    networkx/matplotlib.
    """
    import matplotlib.pyplot as plt

    from semlaflow.validation.convert import geometric_mol_to_nx
    from semlaflow.validation.dist_metrics import (
        build_gt_cache,
        compute_distribution_metrics,
        keys_for_level,
        subset_gt_cache,
    )
    from semlaflow.validation.plot import plot_graph_grid_angles

    gt_path = _split_path(args.data_path, args.dataset_split)
    if not gt_path.exists():
        print(f"[eval] ground-truth split '{gt_path}' not found; skipping metrics/plots.")
        return

    # Ground-truth real graphs (untransformed -> original physical-micron coords).
    gt_dataset = GeometricDataset.load(gt_path, transform=None)
    gt_mols = [gt_dataset[i] for i in range(len(gt_dataset))]
    print(f"[eval] generated={len(gen_mols)} vs ground-truth({args.dataset_split})={len(gt_mols)}")

    # Both sets are already in physical units: the model rescales its output by
    # coord_scale (= the dataset's coord_std, see DATASET_CONFIGS) inside _generate (see fm.py),
    # and GT is loaded untransformed. So no extra scaling here -- coord_scale=1.0 for both.
    gen_graphs = [geometric_mol_to_nx(m, coord_scale=1.0) for m in gen_mols]
    gt_graphs = [geometric_mol_to_nx(m, coord_scale=1.0) for m in gt_mols]

    # Same GT fit and same sanitisation as the in-loop path, so metrics.json lines up
    # key-for-key with the `val-*` series.
    gt_cache = build_gt_cache(gt_graphs)
    metrics = compute_distribution_metrics(gen_graphs, gt_graphs, gt_cache=gt_cache)
    metrics["n_generated"] = float(len(gen_graphs))
    metrics["n_ground_truth"] = float(len(gt_graphs))

    for key in ("disconnected_frac", "multifurcation_frac", "cycle_frac",
                "isolated_node_frac", "non_critical_node_frac"):
        val = metrics.get(key, 0.0)
        if val and val > 0:
            print(f"[eval] structural health: {key} = {val:.1%} "
                  "(ground truth is exactly 0 for both corpora)")

    # Per-neuron-type stratified metrics (type-conditioned runs only): group generated graphs by
    # their conditioning class and GT graphs by their own label, then score each class separately.
    if gen_classes is not None:
        _add_per_class_metrics(
            metrics, gen_graphs, gen_classes, gt_graphs, gt_mols,
            compute_distribution_metrics, args.per_cell_class_min_count,
            dataset=dataset, gt_cache=gt_cache, subset_fn=subset_gt_cache,
        )

    if timing is not None:
        metrics.update(timing)

    metrics_path = save_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[eval] wrote metrics -> {metrics_path}")

    # `*_w1` and `mmd_*` are lower-is-better; `density_*`/`coverage_*` are higher-is-better;
    # the health fractions are 0.0 on ground truth for both corpora.
    print(f"\n[eval] distribution metrics (report level: {args.metric_report_level})")
    print(f"{'Metric':<30}Value")
    print("-" * 42)
    for key in keys_for_level(args.metric_report_level):
        if key in metrics:
            print(f"{key:<30}{metrics[key]:.5f}")
    print()

    # Multi-azimuth plot grids: generated and GT references at the same angles.
    n = min(args.n_plot_examples, len(gen_graphs))
    gen_fig, gen_png = plot_graph_grid_angles(
        gen_graphs[:n], out_dir=save_dir, stem=Path(args.save_file).stem,
        file_tag="gen3d", title_prefix="Gen", max_graphs=n,
    )
    plt.close(gen_fig)
    print(f"[eval] wrote generated plot grid -> {gen_png}")

    n_gt = min(args.n_plot_examples, len(gt_graphs))
    ref_fig, ref_png = plot_graph_grid_angles(
        gt_graphs[:n_gt], out_dir=save_dir, stem=Path(args.save_file).stem,
        file_tag="ref3d", title_prefix="GT", node_color="#1f77b4", max_graphs=n_gt,
    )
    plt.close(ref_fig)
    print(f"[eval] wrote ground-truth plot grid -> {ref_png}")


def main(args):
    print(f"Sampling {args.n_molecules} neuron graphs...")
    print(f"Checkpoint: {args.ckpt_path}")

    L.seed_everything(args.seed)
    util.disable_lib_stdout()
    util.configure_fs()

    print("Building neuron vocab...")
    vocab = util.build_neuron_vocab()
    print(f"Vocab size: {vocab.size}")

    print("Loading datamodule (size-prior source)...")
    dm, dataset = dm_from_ckpt(args, vocab)
    print(f"Dataset: {dataset}")

    print("Loading model...")
    model = load_model(args, vocab)

    # TMD conditioning guards: keep ckpt and CLI flag consistent.
    ckpt_conditional = getattr(model.gen, "tmd_dim", 0) > 0
    if ckpt_conditional and not args.tmd_cond:
        raise SystemExit(
            "This checkpoint was trained with TMD conditioning (tmd_dim > 0). "
            "Pass --tmd_cond to condition generation on the paired val graphs."
        )
    if args.tmd_cond and not ckpt_conditional:
        raise SystemExit(
            "--tmd_cond was set but this checkpoint has no TMD conditioning (tmd_dim = 0)."
        )
    if args.tmd_cond:
        sample_mol = dm.test_dataset[0]
        if getattr(sample_mol, "_tmd", None) is None:
            raise SystemExit(
                "--tmd_cond requires TMD vectors in the dataset, but none were found. "
                "Re-run preprocess_neurons.py with --compute_tmd to regenerate the .smol."
            )
        print("TMD conditioning ON: conditioning on paired val-graph TMD vectors.")

    # Cell-class (neuron type) conditioning guards: keep ckpt and CLI flag consistent.
    ckpt_class_conditional = getattr(model.gen, "class_hidden", 0) > 0
    if ckpt_class_conditional and not args.type_cond:
        raise SystemExit(
            "This checkpoint was trained with cell-class conditioning (class_hidden > 0). "
            "Pass --type_cond to condition generation on the paired val graphs' classes."
        )
    if args.type_cond and not ckpt_class_conditional:
        raise SystemExit(
            "--type_cond was set but this checkpoint has no cell-class conditioning (class_hidden = 0)."
        )
    if args.type_cond:
        sample_mol = dm.test_dataset[0]
        if getattr(sample_mol, "_cell_class", None) is None:
            raise SystemExit(
                "--type_cond requires cell_class labels in the dataset, but none were found. "
                "Preprocess a class-labelled corpus (SWCs with a `# cell_class N` header)."
            )
        print("Cell-class conditioning ON: conditioning on paired val-graph classes.")

    device = _device()
    model.eval().to(device)
    print(f"Model on {device}. Running generation...")

    # Local import keeps the pure-sampling module import free of networkx (convert.py).
    from semlaflow.validation.convert import samples_to_mols

    test_dl = dm.test_dataloader()
    all_mols: list[GeometricMol] = []
    gen_classes: list[int] = []  # conditioning class per emitted sample (type-conditioned runs)
    raw_batches: list[dict] = []
    sampling_seconds_total = 0.0  # wall-clock spent inside _generate (the ODE rollout only)
    n_sampled = 0                 # graphs the ODE integrated (rows fed to _generate)
    for batch in tqdm(test_dl):
        prior = {k: v.to(device) for k, v in batch[0].items()}
        # Time only the _generate rollout. CUDA kernels are async, so synchronise on both
        # sides or we'd measure kernel-launch overhead rather than the actual sampling work.
        if device.type == "cuda":
            torch.cuda.synchronize()
        _t0 = perf_counter()
        with torch.no_grad():
            output = model._generate(prior, args.integration_steps, args.ode_sampling_strategy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        sampling_seconds_total += perf_counter() - _t0
        n_sampled += prior["coords"].size(0)
        batch_mols = samples_to_mols(output, edge_class_index=NEURON_EDGE_CLASS_INDEX)
        all_mols.extend(batch_mols)
        if args.type_cond and "cell_class" in prior:
            # samples_to_mols drops all-masked (empty) graphs; align the class labels the same way
            # so gen_classes stays 1:1 with all_mols. Each sample's class == the paired prior's class.
            masks = output["mask"].detach().cpu().bool()
            cc = prior["cell_class"].detach().cpu().reshape(-1).tolist()
            gen_classes.extend(int(cc[b]) for b in range(masks.size(0)) if int(masks[b].sum()) > 0)
        if args.save_raw:
            raw_batches.append({k: v.detach().cpu() for k, v in output.items()})

    sampling_seconds_per_sample = sampling_seconds_total / max(n_sampled, 1)
    sampling_samples_per_sec = n_sampled / sampling_seconds_total if sampling_seconds_total > 0 else 0.0
    timing = {
        "sampling_seconds_total": sampling_seconds_total,
        "sampling_seconds_per_sample": sampling_seconds_per_sample,
        "sampling_samples_per_sec": sampling_samples_per_sec,
        "n_sampled": n_sampled,
    }
    print(
        f"[timing] sampled {n_sampled} graphs in {sampling_seconds_total:.2f}s "
        f"({sampling_seconds_per_sample * 1e3:.1f} ms/sample, {sampling_samples_per_sec:.1f} samples/s)"
    )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / args.save_file
    out_path.write_bytes(GeometricMolBatch.from_list(all_mols).to_bytes())
    print(f"Wrote {len(all_mols)} samples -> {out_path}")

    if args.save_raw:
        raw_path = out_path.with_suffix(".raw.pt")
        torch.save(raw_batches, raw_path)
        print(f"Wrote raw model outputs (bonds/atomics/coords/mask) -> {raw_path}")

    # Cheap round-trip sanity (not a full stats pass).
    rt = GeometricMolBatch.from_bytes(out_path.read_bytes())
    assert len(rt) == len(all_mols), "round-trip count mismatch"
    print("Round-trip OK.")

    if not args.skip_eval:
        print("Evaluating samples (structural metrics + plots)...")
        evaluate_samples(args, all_mols, save_dir,
                         gen_classes=gen_classes if args.type_cond else None, timing=timing,
                         dataset=dataset)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_file", type=str, default=DEFAULT_SAVE_FILE)

    parser.add_argument("--dataset", type=str, default=None,
                        help="Override the dataset name recorded in the checkpoint. Selects the "
                             "coord scale, bucket limits and class names from "
                             "scriptutil.DATASET_CONFIGS. Normally inferred from the checkpoint.")
    parser.add_argument("--dataset_split", type=str, default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--n_molecules", type=int, default=DEFAULT_N_MOLECULES)
    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--integration_steps", type=int, default=DEFAULT_INTEGRATION_STEPS)
    parser.add_argument("--cat_sampling_noise_level", type=int, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL)
    parser.add_argument("--ode_sampling_strategy", type=str, default=DEFAULT_ODE_SAMPLING_STRATEGY)
    parser.add_argument("--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--save_raw", action="store_true",
                        help="Also dump raw per-batch model outputs (coords/atomics/bonds/mask) "
                             "as <save_file>.raw.pt for diagnostics.")
    parser.add_argument("--tmd_cond", action="store_true",
                        help="Paired conditional generation: condition each sample on the real "
                             "val graph's TMD vector. Requires a TMD-trained checkpoint and a "
                             ".smol built with --compute_tmd.")
    parser.add_argument("--type_cond", action="store_true",
                        help="Paired conditional generation: condition each sample on the real "
                             "val graph's cell-class label. Requires a class-conditioned checkpoint "
                             "and a class-labelled .smol. Enables per-class stratified metrics.")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip post-sampling structural metrics + plots (pure sampling only).")
    parser.add_argument("--n_plot_examples", type=int, default=DEFAULT_N_PLOT_EXAMPLES,
                        help="Number of generated/GT samples per multi-azimuth plot grid.")
    parser.add_argument("--per_cell_class_min_count", type=int, default=20,
                        help="Per-class stratified metrics (with --type_cond): skip classes with "
                             "fewer than this many generated or GT graphs.")
    parser.add_argument("--metric_report_level", type=str, default="standard",
                        choices=["headline", "standard", "full"],
                        help="Which metrics to print. Every metric is computed and written to "
                             "metrics.json regardless -- this only filters the printed table.")

    args = parser.parse_args()
    main(args)
