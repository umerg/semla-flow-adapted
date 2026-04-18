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
from functools import partial
from pathlib import Path

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


def build_dm(args, hparams, vocab):
    coord_std = util.NEURON_COORDS_STD_DEV
    bucket_limits = util.NEURON_BUCKET_LIMITS

    n_bond_types = util.get_n_bond_types(hparams["integration-type-strategy"])
    transform = partial(
        util.neuron_mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std
    )

    data_dir = Path(args.data_path)
    if args.dataset_split == "train":
        dataset_path = data_dir / "train.smol"
    elif args.dataset_split == "val":
        dataset_path = data_dir / "val.smol"
    elif args.dataset_split == "test":
        dataset_path = data_dir / "test.smol"
    else:
        raise ValueError(f"Unknown dataset_split '{args.dataset_split}'")

    dataset = GeometricDataset.load(dataset_path, transform=transform)
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
    return build_dm(args, hparams, vocab)


def samples_to_mols(output, edge_class_index: int) -> list[GeometricMol]:
    """Extract a list of GeometricMol from a raw `_generate` output dict.

    Skips the RDKit-based builder entirely. For each batch element:
      * crop to real nodes via the mask,
      * argmax over bond distributions,
      * symmetrise (edge iff either direction's argmax equals edge_class),
      * emit one GeometricMol with degenerate atomics and zero charges.
    """
    coords = output["coords"].detach().cpu()
    atomics = output["atomics"].detach().cpu()
    bond_dists = output["bonds"].detach().cpu()
    masks = output["mask"].detach().cpu().bool()

    bond_argmax = bond_dists.argmax(dim=-1)

    mols: list[GeometricMol] = []
    for b in range(coords.size(0)):
        n_real = int(masks[b].sum().item())
        if n_real == 0:
            continue

        coords_b = coords[b, :n_real].float()
        atomics_b = atomics[b, :n_real].argmax(-1).long()
        pair = bond_argmax[b, :n_real, :n_real]

        edge = (pair == edge_class_index) | (pair.T == edge_class_index)
        iu = torch.triu_indices(n_real, n_real, offset=1)
        if iu.size(1) == 0:
            bond_indices = torch.zeros((0, 2), dtype=torch.long)
        else:
            keep = edge[iu[0], iu[1]]
            bond_indices = iu[:, keep].T.long()

        bond_types = torch.full((bond_indices.size(0),), edge_class_index, dtype=torch.long)
        charges = torch.zeros((n_real,), dtype=torch.long)

        mols.append(
            GeometricMol(
                coords=coords_b,
                atomics=atomics_b,
                bond_indices=bond_indices,
                bond_types=bond_types,
                charges=charges,
            )
        )
    return mols


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
    dm = dm_from_ckpt(args, vocab)

    print("Loading model...")
    model = load_model(args, vocab)

    device = _device()
    model.eval().to(device)
    print(f"Model on {device}. Running generation...")

    test_dl = dm.test_dataloader()
    all_mols: list[GeometricMol] = []
    raw_batches: list[dict] = []
    for batch in tqdm(test_dl):
        prior = {k: v.to(device) for k, v in batch[0].items()}
        with torch.no_grad():
            output = model._generate(prior, args.integration_steps, args.ode_sampling_strategy)
        all_mols.extend(samples_to_mols(output, edge_class_index=NEURON_EDGE_CLASS_INDEX))
        if args.save_raw:
            raw_batches.append({k: v.detach().cpu() for k, v in output.items()})

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
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--save_file", type=str, default=DEFAULT_SAVE_FILE)

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

    args = parser.parse_args()
    main(args)
