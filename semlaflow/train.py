import argparse
from functools import partial
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

import semlaflow.scriptutil as util
from semlaflow.data.datamodules import GeometricInterpolantDM
from semlaflow.data.datasets import GeometricDataset
from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler
from semlaflow.models.fm import Integrator, MolecularCFM
from semlaflow.models.neuron_cfm import NeuronCFM
from semlaflow.models.semla import EquiInvDynamics, SemlaGenerator

DEFAULT_DATASET = "geom-drugs"
DEFAULT_ARCH = "semla"

DEFAULT_D_MODEL = 384
DEFAULT_N_LAYERS = 12
DEFAULT_D_MESSAGE = 128
DEFAULT_D_EDGE = 128
DEFAULT_N_COORD_SETS = 32
DEFAULT_N_ATTN_HEADS = 32
DEFAULT_D_MESSAGE_HIDDEN = 128
DEFAULT_COORD_NORM = "length"
DEFAULT_SIZE_EMB = 64

DEFAULT_MAX_ATOMS = 256

DEFAULT_EPOCHS = 300
DEFAULT_LR = 0.0003
DEFAULT_BATCH_COST = 1024
DEFAULT_ACC_BATCHES = 1
DEFAULT_GRADIENT_CLIP_VAL = 1.0
DEFAULT_TYPE_LOSS_WEIGHT = 0.2
DEFAULT_BOND_LOSS_WEIGHT = 1.0
DEFAULT_CHARGE_LOSS_WEIGHT = 1.0
DEFAULT_CATEGORICAL_STRATEGY = "uniform-sample"
DEFAULT_LR_SCHEDULE = "constant"
DEFAULT_WARM_UP_STEPS = 10000
DEFAULT_BUCKET_COST_SCALE = "quadratic"

DEFAULT_N_VALIDATION_MOLS = 1800
DEFAULT_VAL_CHECK_EPOCHS = 20
DEFAULT_NUM_INFERENCE_STEPS = 100
DEFAULT_CAT_SAMPLING_NOISE_LEVEL = 1
DEFAULT_COORD_NOISE_STD_DEV = 0.2
DEFAULT_TYPE_DIST_TEMP = 1.0
DEFAULT_TIME_ALPHA = 2.0
DEFAULT_TIME_BETA = 1.0
DEFAULT_OPTIMAL_TRANSPORT = "equivariant"
DEFAULT_PRECISION = "32"
DEFAULT_METRIC_REPORT_LEVEL = "standard"
DEFAULT_CKPT_MONITOR = "val-loss"

# Health-fraction ceilings for the `val-morpho-selection` checkpoint monitor, keyed by the
# metric name in `validation.dist_metrics`. Defaults are permissive (1.0 = never gates), so
# turning on the morpho monitor changes nothing until a ceiling is set explicitly.
#
# The gate exists because sanitisation deliberately makes `mmd_morpho` blind to structural
# failure -- it scores the repaired tree -- so a model emitting garbage that happens to
# have a plausible minimum spanning tree would win on `mmd_morpho` alone.
SELECTION_HEALTH_FLAGS = {
    "disconnected_frac": "--selection_max_disconnected_frac",
    "multifurcation_frac": "--selection_max_multifurcation_frac",
    "cycle_frac": "--selection_max_cycle_frac",
    "isolated_node_frac": "--selection_max_isolated_node_frac",
}


def selection_health_max(args) -> dict:
    """Collect the `--selection_max_*_frac` ceilings into the dict NeuronCFM expects.

    Ceilings of 1.0 are dropped: a fraction can never exceed 1.0, so keeping them would
    only add noise to the saved hparams.
    """
    out = {}
    for key, flag in SELECTION_HEALTH_FLAGS.items():
        limit = getattr(args, flag.lstrip("-"), 1.0)
        if limit is not None and limit < 1.0:
            out[key] = float(limit)
    return out


def build_model(args, dm, vocab):
    # Get hyperparameeters from the datamodule, pass these into the model to be saved
    hparams = {
        "epochs": args.epochs,
        "gradient_clip_val": args.gradient_clip_val,
        "dataset": args.dataset,
        "precision": args.precision,
        "architecture": args.arch,
        **dm.hparams,
    }

    # Per-dataset constants for the SWC (neuron/tree) pipeline; None for the molecular datasets.
    cfg = util.DATASET_CONFIGS.get(args.dataset)

    # Add 1 for the time (0 <= t <= 1 for flow matching)
    n_atom_feats = vocab.size + 1
    n_bond_types = util.get_n_bond_types(args.categorical_strategy)

    # TMD conditioning: infer the vector width from the (preprocessed) training data.
    tmd_dim = 0
    tmd_hidden = 0
    if getattr(args, "tmd_conditioning", False):
        sample_mol = dm.train_dataset[0]
        if getattr(sample_mol, "_tmd", None) is None:
            raise ValueError(
                "--tmd_conditioning was set but the training data has no TMD vectors. "
                "Re-run preprocess_neurons.py with --compute_tmd."
            )
        tmd_dim = int(sample_mol.tmd.shape[0])
        tmd_hidden = args.tmd_hidden
        print(f"TMD conditioning enabled: tmd_dim={tmd_dim}, tmd_hidden={tmd_hidden}")

    # Cell-class (neuron type) conditioning: a discrete per-graph label embedded like the actual
    # method (one_hot -> Linear). Orthogonal to TMD; both can be on at once.
    n_classes = 0
    class_hidden = 0
    if getattr(args, "type_conditioning", False):
        # Config check first: a dataset with no declared class_names can never be conditioned,
        # which is a clearer failure than "your data has no labels".
        if cfg is None or cfg.class_names is None:
            conditionable = [n for n, c in util.DATASET_CONFIGS.items() if c.class_names]
            raise ValueError(
                f"--type_conditioning was set but dataset '{args.dataset}' declares no class_names "
                f"in scriptutil.DATASET_CONFIGS. Class conditioning is defined for: "
                f"{', '.join(conditionable)}."
            )
        sample_mol = dm.train_dataset[0]
        if getattr(sample_mol, "_cell_class", None) is None:
            raise ValueError(
                "--type_conditioning was set but the training data has no cell_class labels. "
                f"Preprocess a class-labelled corpus (SWCs with a `# cell_class N` header, "
                f"e.g. {args.dataset})."
            )
        n_classes = cfg.n_classes
        # An out-of-range id would otherwise surface as an opaque device-side assert from the
        # one_hot in SemlaGenerator; 2.7k-23k mols is cheap to scan.
        observed = {
            int(m._cell_class) for m in dm.train_dataset if getattr(m, "_cell_class", None) is not None
        }
        if observed and max(observed) >= n_classes:
            raise ValueError(
                f"Dataset '{args.dataset}' declares {n_classes} classes but the training data "
                f"contains class id {max(observed)}. Fix "
                f"DATASET_CONFIGS['{args.dataset}'].class_names."
            )
        class_hidden = args.class_hidden
        print(
            f"Class conditioning enabled: n_classes={n_classes} "
            f"({', '.join(cfg.class_names)}), class_hidden={class_hidden}"
        )

    if args.arch == "semla":
        dynamics = EquiInvDynamics(
            args.d_model,
            args.d_message,
            args.n_coord_sets,
            args.n_layers,
            n_attn_heads=args.n_attn_heads,
            d_message_hidden=args.d_message_hidden,
            d_edge=args.d_edge,
            bond_refine=True,
            self_cond=args.self_condition,
            coord_norm=args.coord_norm,
            grad_checkpointing=args.grad_checkpointing,
        )
        egnn_gen = SemlaGenerator(
            args.d_model,
            dynamics,
            vocab.size,
            n_atom_feats,
            d_edge=args.d_edge,
            n_edge_types=n_bond_types,
            self_cond=args.self_condition,
            size_emb=args.size_emb,
            max_atoms=args.max_atoms,
            tmd_dim=tmd_dim,
            tmd_hidden=tmd_hidden,
            n_classes=n_classes,
            class_hidden=class_hidden,
        )

    elif args.arch == "eqgat":
        from semlaflow.models.eqgat import EqgatGenerator

        # Hardcode for now since we only need one model size
        d_model_eqgat = 256
        n_equi_feats_eqgat = 256
        n_layers_eqgat = 12
        d_edge_eqgat = 128

        egnn_gen = EqgatGenerator(
            d_model_eqgat, n_layers_eqgat, n_equi_feats_eqgat, vocab.size, n_atom_feats, d_edge_eqgat, n_bond_types
        )

    elif args.arch == "egnn":
        from semlaflow.models.egnn import VanillaEgnnGenerator

        egnn_gen = VanillaEgnnGenerator(
            args.d_model, args.n_layers, vocab.size, n_atom_feats, d_edge=args.d_edge, n_edge_types=n_bond_types
        )

    else:
        raise ValueError(f"Unknown architecture '{args.arch}'; known: `semla`, `eqgat` or `egnn`")

    if args.dataset == "qm9":
        coord_scale = util.QM9_COORDS_STD_DEV
    elif args.dataset == "geom-drugs":
        coord_scale = util.GEOM_COORDS_STD_DEV
    elif cfg is not None:
        coord_scale = cfg.coord_std
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    type_mask_index = None
    bond_mask_index = None

    if args.categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        train_strategy = "mask"
        sampling_strategy = "mask"

    elif args.categorical_strategy == "uniform-sample":
        train_strategy = "ce"
        sampling_strategy = "uniform-sample"

    elif args.categorical_strategy == "dirichlet":
        train_strategy = "ce"
        sampling_strategy = "dirichlet"

    else:
        raise ValueError(
            f"Interpolation '{args.categorical_strategy}' is not supported. "
            + "Supported are: `mask`, `uniform-sample` and `dirichlet`"
        )

    train_steps = util.calc_train_steps(dm, args.epochs, args.acc_batches)
    # Neurons have no SMILES — skip the RDKit novelty path entirely.
    if args.trial_run or args.dataset in util.NEURON_DATASETS:
        train_smiles = None
    else:
        train_smiles = [mols.str_id for mols in dm.train_dataset]

    print(f"Total training steps {train_steps}")

    integrator = Integrator(
        args.num_inference_steps,
        type_strategy=sampling_strategy,
        bond_strategy=sampling_strategy,
        cat_noise_level=args.cat_sampling_noise_level,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
    )

    cfm_cls = NeuronCFM if args.dataset in util.NEURON_DATASETS else MolecularCFM
    # NeuronCFM-only: toggle the generation-based structural validation metrics.
    extra_cfm_kwargs = {}
    if args.dataset in util.NEURON_DATASETS:
        extra_cfm_kwargs["val_structural_metrics"] = args.val_structural_metrics
        extra_cfm_kwargs["per_cell_class"] = args.per_cell_class
        extra_cfm_kwargs["per_cell_class_min_count"] = args.per_cell_class_min_count
        extra_cfm_kwargs["metric_report_level"] = args.metric_report_level
        extra_cfm_kwargs["selection_health_max"] = selection_health_max(args)
    fm_model = cfm_cls(
        egnn_gen,
        vocab,
        args.lr,
        integrator,
        coord_scale=coord_scale,
        type_strategy=train_strategy,
        bond_strategy=train_strategy,
        type_loss_weight=args.type_loss_weight,
        bond_loss_weight=args.bond_loss_weight,
        charge_loss_weight=args.charge_loss_weight,
        pairwise_metrics=False,
        use_ema=args.use_ema,
        compile_model=False,
        self_condition=args.self_condition,
        distill=False,
        lr_schedule=args.lr_schedule,
        warm_up_steps=args.warm_up_steps,
        total_steps=train_steps,
        train_smiles=train_smiles,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        **extra_cfm_kwargs,
        **hparams,
    )
    return fm_model


def build_dm(args, vocab):
    if args.dataset == "qm9":
        coord_std = util.QM9_COORDS_STD_DEV
        padded_sizes = util.QM9_BUCKET_LIMITS

    elif args.dataset == "geom-drugs":
        coord_std = util.GEOM_COORDS_STD_DEV
        padded_sizes = util.GEOM_DRUGS_BUCKET_LIMITS

    elif args.dataset in util.DATASET_CONFIGS:
        cfg = util.DATASET_CONFIGS[args.dataset]
        coord_std = cfg.coord_std
        padded_sizes = cfg.bucket_limits

    else:
        raise ValueError(
            f"Unknown dataset {args.dataset}. Available: `qm9`, `geom-drugs`, "
            + ", ".join(f"`{name}`" for name in util.NEURON_DATASETS)
            + "."
        )

    data_path = Path(args.data_path)

    n_bond_types = util.get_n_bond_types(args.categorical_strategy)
    if args.dataset in util.NEURON_DATASETS:
        transform = partial(
            util.neuron_mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std
        )
    else:
        transform = partial(util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std)

    # Load generated dataset with different transform fn if we are distilling a model
    # if args.distill:
    #     distill_transform = partial(util.distill_transform, coord_std=coord_std)
    #     train_dataset = GeometricDataset.load(data_path / "distill.smol", transform=distill_transform)
    # else:
    #     train_dataset = GeometricDataset.load(data_path / "train.smol", transform=transform)

    train_dataset = GeometricDataset.load(data_path / "train.smol", transform=transform)
    val_dataset = GeometricDataset.load(data_path / "val.smol", transform=transform)
    # sample() draws without replacement, so asking for more than the split holds raises. Small
    # corpora (the tree datasets have 337 val graphs vs the 1800 default) would die here.
    n_val = min(args.n_validation_mols, len(val_dataset))
    if n_val < args.n_validation_mols:
        print(f"Val split has {n_val} graphs; using all of them "
              f"(--n_validation_mols={args.n_validation_mols}).")
    val_dataset = val_dataset.sample(n_val)

    type_mask_index = None
    bond_mask_index = None

    if args.categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        categorical_interpolation = "unmask"
        categorical_noise = "mask"

    elif args.categorical_strategy == "uniform-sample":
        categorical_interpolation = "unmask"
        categorical_noise = "uniform-sample"

    elif args.categorical_strategy == "dirichlet":
        categorical_interpolation = "dirichlet"
        categorical_noise = "uniform-dist"

    else:
        raise ValueError(
            f"Interpolation '{args.categorical_strategy}' is not supported. "
            + "Supported are: `mask`, `uniform-sample` and `dirichlet`"
        )

    scale_ot = False
    batch_ot = False
    equivariant_ot = False

    if args.optimal_transport == "batch":
        batch_ot = True
    elif args.optimal_transport == "equivariant":
        equivariant_ot = True
    elif args.optimal_transport == "scale":
        scale_ot = True
        equivariant_ot = True
    elif args.optimal_transport not in ["None", "none", None]:
        raise ValueError(
            f"Unknown value for optimal_transport '{args.optimal_transport}'. "
            + "Acceted values: `batch`, `equivariant` and `scale`."
        )

    # train_fixed_time = 0.5 if args.distill else None
    train_fixed_time = None

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise=categorical_noise,
        bond_noise=categorical_noise,
        scale_ot=scale_ot,
        zero_com=True,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
    )
    train_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=categorical_interpolation,
        bond_interpolation=categorical_interpolation,
        coord_noise_std=args.coord_noise_std_dev,
        type_dist_temp=args.type_dist_temp,
        equivariant_ot=equivariant_ot,
        batch_ot=batch_ot,
        time_alpha=args.time_alpha,
        time_beta=args.time_beta,
        fixed_time=train_fixed_time,
    )
    eval_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=categorical_interpolation,
        bond_interpolation=categorical_interpolation,
        equivariant_ot=False,
        batch_ot=False,
        fixed_time=0.9,
    )

    dm = GeometricInterpolantDM(
        train_dataset,
        val_dataset,
        None,
        args.batch_cost,
        train_interpolant=train_interpolant,
        val_interpolant=eval_interpolant,
        test_interpolant=None,
        bucket_limits=padded_sizes,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False,
    )
    return dm


def build_trainer(args):
    epochs = 1 if args.trial_run else args.epochs
    log_steps = 1 if args.trial_run else 50
    val_check_epochs = 1 if args.trial_run else args.val_check_epochs

    project_name = f"{util.PROJECT_PREFIX}-{args.dataset}"
    print(f"Using precision '{args.precision}'")

    # Construct the logger only when it will actually be used: a trial run discards it
    # below, and building it eagerly makes `--trial_run` require wandb to be installed.
    logger = None if args.trial_run else WandbLogger(
        project=project_name, save_dir="wandb", log_model=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    if args.dataset in util.NEURON_DATASETS:
        # Neurons don't have RDKit validity, so the best checkpoint is chosen either by
        # loss (default) or by the gated morphometric MMD.
        monitor = args.ckpt_monitor
        if monitor == "val-morpho-selection" and not args.val_structural_metrics:
            raise ValueError(
                "--ckpt_monitor val-morpho-selection requires the structural metrics; "
                "drop --no_val_structural_metrics."
            )
        n_devices = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if monitor == "val-morpho-selection" and n_devices > 1:
            # sync_dist averages per-rank scalars, and the mean of per-rank MMDs is not
            # the MMD of the union -- tolerable for monitoring, wrong for selection.
            print(
                "WARNING: --ckpt_monitor val-morpho-selection with "
                f"{n_devices} visible GPUs. Under DDP each rank scores its own shard and "
                "sync_dist averages the result; the mean of per-rank MMDs is NOT the "
                "MMD of the union, so the selected checkpoint may not be the best one. "
                "Restrict to one GPU (CUDA_VISIBLE_DEVICES) for morpho-based selection, "
                "or monitor val-loss."
            )
        best_ckpt = ModelCheckpoint(
            every_n_epochs=val_check_epochs,
            monitor=monitor,
            mode="min",
            save_top_k=1,
            save_last=True,
            filename="best-{epoch:03d}",
        )
        # Periodic weights-only trajectory snapshots: keep all, so a better checkpoint can
        # be picked post-hoc by inspecting the logged structural-metric trajectories.
        # Weights-only (no optimizer/scheduler state) since these are for eval/sampling,
        # not training resumption; the EMA weights used for sampling live in the state_dict.
        periodic_ckpt = ModelCheckpoint(
            every_n_epochs=val_check_epochs,
            save_top_k=-1,
            save_weights_only=True,
            filename="snap-{epoch:03d}",
        )
        checkpointing = [best_ckpt, periodic_ckpt]
    else:
        checkpointing = [
            ModelCheckpoint(
                every_n_epochs=val_check_epochs, monitor="val-validity", mode="max", save_last=True
            )
        ]

    trainer = L.Trainer(
        min_epochs=epochs,
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=log_steps,
        accumulate_grad_batches=args.acc_batches,
        gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=val_check_epochs,
        callbacks=[lr_monitor, *checkpointing],
        precision=args.precision,
    )
    return trainer


def main(args):
    # Set some useful torch properties
    # Float32 precision should only affect computation on A100 and should in theory be a lot faster than the default
    # Increasing the cache size is required since the model will be compiled seperately for each bucket
    torch.set_float32_matmul_precision("high")
    # torch._dynamo.config.cache_size_limit = util.COMPILER_CACHE_SIZE
    # print(f"Set torch compiler cache size to {torch._dynamo.config.cache_size_limit}")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    print("Building model vocab...")
    vocab = util.build_neuron_vocab() if args.dataset in util.NEURON_DATASETS else util.build_vocab()
    print(f"Vocab complete. Size={vocab.size}")

    print("Loading datamodule...")
    dm = build_dm(args, vocab)
    print("Datamodule complete.")

    print("Building equinv model...")
    model = build_model(args, dm, vocab)
    print("Model complete.")

    trainer = build_trainer(args)

    print("Fitting datamodule to model...")
    trainer.fit(model, datamodule=dm)
    print("Training complete.")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # Setup args
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--trial_run", action="store_true")

    # Model args
    parser.add_argument("--d_model", type=int, default=DEFAULT_D_MODEL)
    parser.add_argument("--n_layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--d_message", type=int, default=DEFAULT_D_MESSAGE)
    parser.add_argument("--d_edge", type=int, default=DEFAULT_D_EDGE)
    parser.add_argument("--n_coord_sets", type=int, default=DEFAULT_N_COORD_SETS)
    parser.add_argument("--n_attn_heads", type=int, default=DEFAULT_N_ATTN_HEADS)
    parser.add_argument("--d_message_hidden", type=int, default=DEFAULT_D_MESSAGE_HIDDEN)
    parser.add_argument("--coord_norm", type=str, default=DEFAULT_COORD_NORM)
    parser.add_argument("--size_emb", type=int, default=DEFAULT_SIZE_EMB)
    parser.add_argument("--max_atoms", type=int, default=DEFAULT_MAX_ATOMS)
    parser.add_argument("--arch", type=str, default=DEFAULT_ARCH)
    parser.add_argument("--tmd_conditioning", action="store_true",
                        help="Condition generation on a per-graph TMD vector (requires data "
                             "preprocessed with --compute_tmd). Off => unconditional, unchanged.")
    parser.add_argument("--tmd_hidden", type=int, default=64,
                        help="Hidden/projection dim for the TMD conditioning MLP (when enabled).")
    parser.add_argument("--type_conditioning", action="store_true",
                        help="Condition generation on a per-graph neuron cell-class label (requires a "
                             "class-labelled corpus, e.g. neurons_conditional). Off => unconditional. "
                             "Orthogonal to --tmd_conditioning.")
    parser.add_argument("--class_hidden", type=int, default=16,
                        help="Embedding dim for the cell-class conditioning (one_hot -> Linear).")

    # Training args
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch_cost", type=int, default=DEFAULT_BATCH_COST)
    parser.add_argument("--acc_batches", type=int, default=DEFAULT_ACC_BATCHES)
    parser.add_argument("--gradient_clip_val", type=float, default=DEFAULT_GRADIENT_CLIP_VAL)
    parser.add_argument("--type_loss_weight", type=float, default=DEFAULT_TYPE_LOSS_WEIGHT)
    parser.add_argument("--bond_loss_weight", type=float, default=DEFAULT_BOND_LOSS_WEIGHT)
    parser.add_argument("--charge_loss_weight", type=float, default=DEFAULT_CHARGE_LOSS_WEIGHT)
    parser.add_argument("--categorical_strategy", type=str, default=DEFAULT_CATEGORICAL_STRATEGY)
    parser.add_argument("--lr_schedule", type=str, default=DEFAULT_LR_SCHEDULE)
    parser.add_argument("--warm_up_steps", type=int, default=DEFAULT_WARM_UP_STEPS)
    parser.add_argument("--bucket_cost_scale", type=str, default=DEFAULT_BUCKET_COST_SCALE)
    parser.add_argument("--no_ema", action="store_false", dest="use_ema")
    parser.add_argument("--self_condition", action="store_true")
    parser.add_argument("--precision", type=str, default=DEFAULT_PRECISION,
                        help="Lightning trainer precision. Default '32'. Use 'bf16-mixed' to "
                             "roughly halve activation memory, which large graphs need.")
    parser.add_argument("--grad_checkpointing", action="store_true",
                        help="Recompute each message-passing layer during backward instead of "
                             "storing its activations. Costs ~30%% compute for roughly an order "
                             "of magnitude less memory; required for graphs beyond ~1000 nodes.")
    # Neurons only: disable the generation-based structural validation metrics (skips a
    # full ODE rollout over the val set each validation -- faster, but loses the trajectory).
    parser.add_argument(
        "--no_val_structural_metrics", action="store_false", dest="val_structural_metrics"
    )
    # Neurons only: per-cell-class stratified structural val metrics (only active for a
    # class-conditioned run; harmless otherwise).
    parser.add_argument(
        "--no_per_cell_class", action="store_false", dest="per_cell_class",
        help="Disable per-cell-class stratified structural validation metrics."
    )
    parser.add_argument(
        "--per_cell_class_min_count", type=int, default=20,
        help="Skip per-class metrics for classes with fewer than this many val graphs."
    )
    # Neurons only: which validation metrics reach the logger. Pure logging filter --
    # every metric is computed regardless, so this cannot change a number.
    parser.add_argument(
        "--metric_report_level", type=str, default=DEFAULT_METRIC_REPORT_LEVEL,
        choices=["headline", "standard", "full"],
        help="Which structural metrics to log. 'standard' mirrors dendrite_gen's "
             "dashboard; 'full' adds the redundant/low-power keys. Logging filter only."
    )
    # Neurons only: checkpoint selection.
    parser.add_argument(
        "--ckpt_monitor", type=str, default=DEFAULT_CKPT_MONITOR,
        choices=["val-loss", "val-morpho-selection"],
        help="Metric driving best-checkpoint selection. 'val-morpho-selection' is "
             "mmd_morpho gated on the health fractions (see --selection_max_*). "
             "Single-GPU only -- see the DDP warning."
    )
    for _key, _flag in SELECTION_HEALTH_FLAGS.items():
        parser.add_argument(
            _flag, type=float, default=1.0,
            help=f"Epochs with {_key} above this cannot be selected by "
                 f"--ckpt_monitor val-morpho-selection. Default 1.0 (no gating)."
        )
    # parser.add_argument("--mixed_precision", action="store_true")
    # parser.add_argument("--compile_model", action="store_true")
    # parser.add_argument("--distill", action="store_true")

    # Flow matching and sampling args
    parser.add_argument("--val_check_epochs", type=int, default=DEFAULT_VAL_CHECK_EPOCHS)
    parser.add_argument("--n_validation_mols", type=int, default=DEFAULT_N_VALIDATION_MOLS)
    parser.add_argument("--num_inference_steps", type=int, default=DEFAULT_NUM_INFERENCE_STEPS)
    parser.add_argument("--cat_sampling_noise_level", type=int, default=DEFAULT_CAT_SAMPLING_NOISE_LEVEL)
    parser.add_argument("--coord_noise_std_dev", type=float, default=DEFAULT_COORD_NOISE_STD_DEV)
    parser.add_argument("--type_dist_temp", type=float, default=DEFAULT_TYPE_DIST_TEMP)
    parser.add_argument("--time_alpha", type=float, default=DEFAULT_TIME_ALPHA)
    parser.add_argument("--time_beta", type=float, default=DEFAULT_TIME_BETA)
    parser.add_argument("--optimal_transport", type=str, default=DEFAULT_OPTIMAL_TRANSPORT)

    parser.set_defaults(
        trial_run=False,
        use_ema=True,
        self_condition=True,
        grad_checkpointing=False,
        val_structural_metrics=True,
        per_cell_class=True,
        # compile_model=False,
        # mixed_precision=False,
        # distill=False
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
