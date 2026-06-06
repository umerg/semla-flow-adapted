"""Convert cleaned SWC neuron files into SemlaFlow's .smol binary format.

Expects an input layout produced by dendrite_gen's prepare_neurons_final.py:

    <input_dir>/train/*.swc
    <input_dir>/val_extended/*.swc   (preferred; fallback: <input_dir>/val)

Writes <output_dir>/{train,val}.smol and prints a measured coord_std
(over zero-CoM'd training coordinates) for NEURON_COORDS_STD_DEV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from semlaflow.data.swc import swc_to_geometric_mol
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
             tmd_filtrations: tuple = ("path",)):
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
        default="/Users/umer/Documents/neurons_final",
        help="Directory with train/ and val_extended/ (or val/) subfolders of SWC files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/Users/umer/Documents/neurons_final/smol",
        help="Where to write {train,val}.smol.",
    )
    parser.add_argument("--max_atoms", type=int, default=MAX_ATOMS)
    parser.add_argument(
        "--val_dir_name",
        type=str,
        default=None,
        help="Subdir to use for validation. Auto-picks 'val_extended' if present, else 'val'.",
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
        default=["path"],
        choices=["path", "height", "rho"],
        help="Filtrations for the TMD vector (default: path-only, rotation-invariant).",
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

    if not train_files:
        raise SystemExit(f"No train SWC files found under {input_dir}/train")
    if not val_files:
        raise SystemExit(f"No val SWC files found under {input_dir}")

    print(f"train files: {len(train_files)}")
    print(f"val   files: {len(val_files)}")

    tmd_filtrations = tuple(args.tmd_filtrations)
    if args.compute_tmd:
        print(f"Computing TMD conditioning vectors (filtrations={tmd_filtrations})...")
    train_mols = _convert(train_files, args.max_atoms, "train", args.compute_tmd, tmd_filtrations)
    val_mols = _convert(val_files, args.max_atoms, "val", args.compute_tmd, tmd_filtrations)

    if args.compute_tmd and train_mols:
        tmd_dim = int(train_mols[0]._tmd.shape[0])
        print(f"== TMD vector dim: {tmd_dim} == (pass --tmd_conditioning to train.py to use it)\n")

    coord_std = _coord_std(train_mols)
    size_hist = np.bincount([m.seq_length for m in train_mols])
    print("\n== Measured neuron coord_std (over train, post zero-CoM): "
          f"{coord_std:.4f} ==")
    print("Paste this into semlaflow.scriptutil as NEURON_COORDS_STD_DEV.\n")
    print(f"train size: min={size_hist.nonzero()[0].min()}, "
          f"max={len(size_hist) - 1}, total={len(train_mols)}")

    print("Serialising train.smol...")
    (output_dir / "train.smol").write_bytes(
        GeometricMolBatch.from_list(train_mols).to_bytes()
    )
    print("Serialising val.smol...")
    (output_dir / "val.smol").write_bytes(
        GeometricMolBatch.from_list(val_mols).to_bytes()
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
    print(f"Wrote:\n  {output_dir / 'train.smol'}\n  {output_dir / 'val.smol'}")


if __name__ == "__main__":
    main()
