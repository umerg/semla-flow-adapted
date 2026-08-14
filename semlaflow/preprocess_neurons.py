"""Convert cleaned SWC neuron/tree files into SemlaFlow's .smol binary format.

Expects an input layout produced by dendrite_gen's prepare_neurons_final.py,
prepare_conditional_dataset.py or prepare_tree_dataset.py:

    <input_dir>/train/*.swc
    <input_dir>/val_extended/*.swc   (preferred; fallback: <input_dir>/val)
    <input_dir>/test/*.swc           (optional; skipped when absent)

Writes <output_dir>/{train,val}.smol, plus test.smol when a test split exists, and prints a
measured coord_std (over zero-CoM'd training coordinates) plus per-split max node counts.
Paste those into the dataset's entry in semlaflow.scriptutil.DATASET_CONFIGS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from semlaflow.data.swc import swc_to_geometric_mol
from semlaflow.tmd import (
    NEURON_TMD_FILTRATIONS,
    NEURON_TMD_SUPPORTED_FILTRATIONS,
    neuron_tmd_dim,
    validate_filtrations,
)
from semlaflow.util.molrepr import GeometricMolBatch


MAX_ATOMS = 256  # SemlaFlow default cap; graphs exceeding this are skipped.


def _list_swc(dir_path: Path) -> list[Path]:
    if not dir_path.is_dir():
        return []
    return [
        p for p in sorted(dir_path.iterdir())
        if p.is_file() and p.name.endswith(".swc") and not p.name.startswith("._")
    ]


def _convert(files: list[Path], max_atoms: int, split_name: str, compute_tmd: bool = False,
             tmd_filtrations: tuple = NEURON_TMD_FILTRATIONS):
    mols = []
    dropped = 0
    for f in tqdm(files, desc=f"parsing {split_name}"):
        mol = swc_to_geometric_mol(f)
        if mol.seq_length > max_atoms:
            dropped += 1
            continue
        if compute_tmd:
            from semlaflow.tmd import compute_neuron_tmd

            mol._tmd = compute_neuron_tmd(mol, filtrations=tmd_filtrations)
            # Stamp the provenance alongside the vector so the .smol is self-describing:
            # two filtration sets of the same size are otherwise indistinguishable.
            mol._tmd_filtrations = tuple(tmd_filtrations)
        mols.append(mol)
    if dropped:
        print(f"[{split_name}] dropped {dropped} graphs with > {max_atoms} nodes")
    return mols


def _coord_std(mols) -> float:
    """Compute std over per-tree zero-CoM'd coordinates (matches training transform)."""
    all_centered = []
    for m in mols:
        coords = m.coords.cpu().numpy()
        com = coords.mean(axis=0, keepdims=True)
        all_centered.append(coords - com)
    stacked = np.concatenate(all_centered, axis=0)
    return float(stacked.std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Directory with train/ and val_extended/ (or val/) subfolders of SWC files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to write {train,val,test}.smol.",
    )
    parser.add_argument("--max_atoms", type=int, default=MAX_ATOMS)
    parser.add_argument(
        "--val_dir_name",
        type=str,
        default=None,
        help="Subdir to use for validation. Auto-picks 'val_extended' if present, else 'val'.",
    )
    parser.add_argument(
        "--test_dir_name",
        type=str,
        default=None,
        help="Subdir to use for the test split. Auto-detects 'test' if present; no test.smol "
             "is written when absent (the neuron corpora have no usable test split).",
    )
    parser.add_argument(
        "--compute_tmd",
        action="store_true",
        help="Compute and store a per-graph TMD conditioning vector on each mol (for conditional "
             "training/generation). Off by default => output is identical to the unconditional pipeline.",
    )
    parser.add_argument(
        "--tmd_filtrations",
        type=str,
        nargs="+",
        default=list(NEURON_TMD_FILTRATIONS),
        choices=list(NEURON_TMD_SUPPORTED_FILTRATIONS),
        help="Filtrations concatenated into the TMD vector, one 16x16 persistence image each "
             f"(default: {' '.join(NEURON_TMD_FILTRATIONS)} -> dim {neuron_tmd_dim()}). Only "
             "rotation-invariant filtrations are offered; `height`/`rho` need a fixed anatomical "
             "axis that the random-rotation transform destroys (see COMPATIBLE.md §4.15).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_files = _list_swc(input_dir / "train")
    if args.val_dir_name:
        val_files = _list_swc(input_dir / args.val_dir_name)
    else:
        val_files = _list_swc(input_dir / "val_extended") or _list_swc(input_dir / "val")

    # A missing test split is normal (the neuron corpora have none), so only an explicitly
    # requested --test_dir_name that turns up empty is an error.
    test_files = _list_swc(input_dir / (args.test_dir_name or "test"))
    if args.test_dir_name and not test_files:
        raise SystemExit(f"No test SWC files found under {input_dir / args.test_dir_name}")

    if not train_files:
        raise SystemExit(f"No train SWC files found under {input_dir}/train")
    if not val_files:
        raise SystemExit(f"No val SWC files found under {input_dir}")

    print(f"train files: {len(train_files)}")
    print(f"val   files: {len(val_files)}")
    print(f"test  files: {len(test_files) if test_files else 0}"
          f"{'' if test_files else '  (no test split -- test.smol will not be written)'}")

    # argparse `choices` rejects unknown names; this also catches duplicates, which would
    # silently widen the vector with a repeated block.
    try:
        tmd_filtrations = validate_filtrations(args.tmd_filtrations)
    except ValueError as err:
        raise SystemExit(str(err)) from err
    if args.compute_tmd:
        print(
            f"Computing TMD conditioning vectors "
            f"(filtrations={', '.join(tmd_filtrations)} -> dim {neuron_tmd_dim(tmd_filtrations)})..."
        )
    train_mols = _convert(train_files, args.max_atoms, "train", args.compute_tmd, tmd_filtrations)
    val_mols = _convert(val_files, args.max_atoms, "val", args.compute_tmd, tmd_filtrations)
    test_mols = (
        _convert(test_files, args.max_atoms, "test", args.compute_tmd, tmd_filtrations)
        if test_files else None
    )

    if args.compute_tmd and train_mols:
        tmd_dim = int(train_mols[0]._tmd.shape[0])
        expected = neuron_tmd_dim(tmd_filtrations)
        assert tmd_dim == expected, f"TMD dim {tmd_dim} != expected {expected}"
        print(f"== TMD vector dim: {tmd_dim} ({' + '.join(tmd_filtrations)}) == "
              "(pass --tmd_conditioning to train.py to use it)\n")

    # Cell-class label summary (class-labelled corpora only): confirms every graph carries a
    # `# cell_class N` header and shows the class balance before training with --type_conditioning.
    n_labelled = sum(1 for m in train_mols if m._cell_class is not None)
    if n_labelled:
        from collections import Counter

        class_hist = Counter(int(m._cell_class) for m in train_mols if m._cell_class is not None)
        print(f"== Cell-class labels: {n_labelled}/{len(train_mols)} train graphs labelled ==")
        for cls in sorted(class_hist):
            print(f"   class {cls}: {class_hist[cls]}")
        if n_labelled != len(train_mols):
            print("   WARNING: some graphs are missing a `# cell_class` header "
                  "(--type_conditioning requires all of them labelled).")
        print("   (pass --type_conditioning to train.py to condition on these)\n")

    coord_std = _coord_std(train_mols)
    print(f"\n== Measured coord_std (over train, post zero-CoM): {coord_std:.4f} ==")

    # Bucket limits must cover the largest graph in *every* split, not just train --
    # BucketBatchSampler raises for the val loader too. So report all of them.
    splits = [("train", train_mols), ("val", val_mols)]
    if test_mols is not None:
        splits.append(("test", test_mols))
    max_nodes = 0
    for name, mols in splits:
        sizes = [m.seq_length for m in mols]
        max_nodes = max(max_nodes, max(sizes))
        print(f"   {name:<5} size: min={min(sizes)}, max={max(sizes)}, total={len(mols)}")
    print(f"\nPaste into semlaflow.scriptutil.DATASET_CONFIGS: coord_std={coord_std:.4f}, "
          f"max_nodes={args.max_atoms}.")
    print(f"Top bucket limit must be >= {max_nodes} (largest graph across all splits).\n")

    print("Serialising train.smol...")
    (output_dir / "train.smol").write_bytes(
        GeometricMolBatch.from_list(train_mols).to_bytes()
    )
    print("Serialising val.smol...")
    (output_dir / "val.smol").write_bytes(
        GeometricMolBatch.from_list(val_mols).to_bytes()
    )
    if test_mols is not None:
        print("Serialising test.smol...")
        (output_dir / "test.smol").write_bytes(
            GeometricMolBatch.from_list(test_mols).to_bytes()
        )

    # Round-trip smoke check on the first train mol.
    round_tripped = GeometricMolBatch.from_bytes(
        (output_dir / "train.smol").read_bytes()
    )
    assert len(round_tripped) == len(train_mols)
    m0 = round_tripped[0]
    m0_orig = train_mols[0]
    assert torch.allclose(m0.coords, m0_orig.coords, atol=1e-5), "coord round-trip mismatch"
    assert m0.bond_indices.shape == m0_orig.bond_indices.shape, "bond_indices round-trip mismatch"
    print("Round-trip OK.\n")
    written = ["train.smol", "val.smol"] + (["test.smol"] if test_mols is not None else [])
    print("Wrote:")
    for name in written:
        print(f"  {output_dir / name}")


if __name__ == "__main__":
    main()
