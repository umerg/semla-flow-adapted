"""Util file for Equinv scripts"""

import math
import resource
from pathlib import Path

import numpy as np
import torch
from openbabel import pybel
from rdkit import RDLogger
from torchmetrics import MetricCollection
from tqdm import tqdm

import semlaflow.util.functional as smolF
import semlaflow.util.metrics as Metrics
import semlaflow.util.rdkit as smolRD
from semlaflow.data.datasets import GeometricDataset
from semlaflow.util.tokeniser import Vocabulary

# Declarations to be used in scripts
QM9_COORDS_STD_DEV = 1.723299503326416
GEOM_COORDS_STD_DEV = 2.407038688659668

QM9_BUCKET_LIMITS = [12, 16, 18, 20, 22, 24, 30]
GEOM_DRUGS_BUCKET_LIMITS = [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 96, 192]

# Neurons: measured over /Users/umer/Documents/neurons_final/train by
# semlaflow.preprocess_neurons (post per-tree zero-CoM). Re-run that script to
# refresh this value if the training corpus changes.
# Bucket limits cover the observed size distribution (p5=18, median=46,
# p95=108, max=258). Train max observed: 217.
NEURON_COORDS_STD_DEV = 62.6894
# NEURON_COORDS_STD_DEV = 0.0727
NEURON_BUCKET_LIMITS = [24, 40, 56, 72, 96, 128, 160, 200, 220]

# Class-labelled neuron corpus (/Users/umer/Documents/neurons_conditional): soma-rooted,
# binarized, class-labelled SWCs. Same neuron vocab/transform/NeuronCFM as `neurons`, but a
# separate corpus with its own coord scale.
# Measured over the full train split (22740 graphs, post per-tree zero-CoM) by
# preprocess_neurons.py; re-run and refresh if the corpus changes. Train node counts:
# min=8, max=242 (10 graphs > 256 dropped) -> top bucket must cover 242.
NEURON_CONDITIONAL_COORDS_STD_DEV = 66.0298
NEURON_CONDITIONAL_BUCKET_LIMITS = [24, 40, 56, 72, 96, 128, 160, 200, 224, 256]

# Cell-type (class) conditioning: single id<->name source of truth. Integer id == list index,
# matching the `# cell_class N` SWC headers and dendrite_gen's utils.data_loading.CELL_CLASS_NAMES.
NEURON_CELL_CLASS_NAMES = ["23P", "4P", "5P-IT", "5P-ET", "5P-NP", "6P-IT", "6P-CT"]
NEURON_NUM_CLASSES = len(NEURON_CELL_CLASS_NAMES)

# Datasets that use the neuron pipeline (vocab/transform/NeuronCFM/loss-based checkpointing).
NEURON_DATASETS = ("neurons", "neurons_conditional")

PROJECT_PREFIX = "equinv"
BOND_MASK_INDEX = 5
COMPILER_CACHE_SIZE = 128


def disable_lib_stdout():
    pybel.ob.obErrorLog.StopLogging()
    RDLogger.DisableLog("rdApp.*")


# Need to ensure the limits are large enough when using OT since lots of preprocessing needs to be done on the batches
# OT seems to cause a problem when there are not enough allowed open FDs
def configure_fs(limit=4096):
    """
    Try to increase the limit on open file descriptors
    If not possible use a different strategy for sharing files in torch
    """

    n_file_resource = resource.RLIMIT_NOFILE
    soft_limit, hard_limit = resource.getrlimit(n_file_resource)

    print(f"Current limits (soft, hard): {(soft_limit, hard_limit)}")

    if limit > soft_limit:
        try:
            print(f"Attempting to increase open file limit to {limit}...")
            resource.setrlimit(n_file_resource, (limit, hard_limit))
            print("Limit changed successfully!")

        except Exception:
            print("Limit change unsuccessful. Using torch file_system file sharing strategy instead.")

            import torch.multiprocessing

            torch.multiprocessing.set_sharing_strategy("file_system")

    else:
        print("Open file limit already sufficiently large.")


# Applies the following transformations to a molecule:
# 1. Scales coordinate values by 1 / coord_std (so that they are standard normal)
# 2. Applies a random rotation to the coordinates
# 3. Removes the centre of mass of the molecule
# 4. Creates a one-hot vector for the atomic numbers of each atom
# 5. Creates a one-hot vector for the bond type for every possible bond
# 6. Encodes charges as non-negative numbers according to encoding map
def mol_transform(molecule, vocab, n_bonds, coord_std):
    rotation = tuple(np.random.rand(3) * np.pi * 2)
    molecule = molecule.scale(1.0 / coord_std).rotate(rotation).zero_com()

    atomic_nums = [int(atomic) for atomic in molecule.atomics.tolist()]
    tokens = [smolRD.PT.symbol_from_atomic(atomic) for atomic in atomic_nums]
    one_hot_atomics = torch.tensor(vocab.indices_from_tokens(tokens, one_hot=True))

    bond_types = smolF.one_hot_encode_tensor(molecule.bond_types, n_bonds)

    charge_idxs = [smolRD.CHARGE_IDX_MAP[charge] for charge in molecule.charges.tolist()]
    charge_idxs = torch.tensor(charge_idxs)

    transformed = molecule._copy_with(atomics=one_hot_atomics, bond_types=bond_types, charges=charge_idxs)
    return transformed


# When training a distilled model atom types and bonds are already distributions over categoricals
def distill_transform(molecule, coord_std):
    rotation = tuple(np.random.rand(3) * np.pi * 2)
    molecule = molecule.scale(1.0 / coord_std).rotate(rotation).zero_com()

    charge_idxs = [smolRD.CHARGE_IDX_MAP[charge] for charge in molecule.charges.tolist()]
    charge_idxs = torch.tensor(charge_idxs)

    transformed = molecule._copy_with(charges=charge_idxs)
    return transformed


def get_n_bond_types(cat_strategy):
    n_bond_types = len(smolRD.BOND_IDX_MAP.keys()) + 1
    n_bond_types = n_bond_types + 1 if cat_strategy == "mask" else n_bond_types
    return n_bond_types


def build_vocab():
    # Need to make sure PAD has index 0
    special_tokens = ["<PAD>", "<MASK>"]
    core_atoms = ["H", "C", "N", "O", "F", "P", "S", "Cl"]
    other_atoms = ["Br", "B", "Al", "Si", "As", "I", "Hg", "Bi"]
    tokens = special_tokens + core_atoms + other_atoms
    return Vocabulary(tokens)


def build_neuron_vocab():
    # Single real node type; PAD at 0, MASK at 1, NODE at 2.
    # swc.NEURON_NODE_TOKEN_INDEX depends on this ordering.
    return Vocabulary(["<PAD>", "<MASK>", "NODE"])


# Transform for neuron GeometricMol objects. Skips the molecular atomic-number
# -> element-symbol -> vocab-index path and the charge remapping: the adapter
# already stores atomics as vocab indices and charges as zeros.
def neuron_mol_transform(molecule, vocab, n_bonds, coord_std):
    rotation = tuple(np.random.rand(3) * np.pi * 2)
    molecule = molecule.scale(1.0 / coord_std).rotate(rotation).zero_com()

    atomics_one_hot = smolF.one_hot_encode_tensor(molecule.atomics, vocab.size)
    bond_types_one_hot = smolF.one_hot_encode_tensor(molecule.bond_types, n_bonds)

    transformed = molecule._copy_with(atomics=atomics_one_hot, bond_types=bond_types_one_hot)
    return transformed


# TODO support multi gpus
def calc_train_steps(dm, epochs, acc_batches):
    dm.setup("train")
    steps_per_epoch = math.ceil(len(dm.train_dataloader()) / acc_batches)
    return steps_per_epoch * epochs


def init_metrics(data_path, model):
    # Load the train data separately from the DM, just to access the list of train SMILES
    train_path = Path(data_path) / "train.smol"
    train_dataset = GeometricDataset.load(train_path)
    train_smiles = [mol.str_id for mol in train_dataset]

    print("Creating RDKit mols from training SMILES...")
    train_mols = model.builder.mols_from_smiles(train_smiles, explicit_hs=True)
    train_mols = [mol for mol in train_mols if mol is not None]

    metrics = {
        "validity": Metrics.Validity(),
        "connected-validity": Metrics.Validity(connected=True),
        "uniqueness": Metrics.Uniqueness(),
        "novelty": Metrics.Novelty(train_mols),
        "energy-validity": Metrics.EnergyValidity(),
        "opt-energy-validity": Metrics.EnergyValidity(optimise=True),
        "energy": Metrics.AverageEnergy(),
        "energy-per-atom": Metrics.AverageEnergy(per_atom=True),
        "strain": Metrics.AverageStrainEnergy(),
        "strain-per-atom": Metrics.AverageStrainEnergy(per_atom=True),
        "opt-rmsd": Metrics.AverageOptRmsd(),
    }
    stability_metrics = {"atom-stability": Metrics.AtomStability(), "molecule-stability": Metrics.MoleculeStability()}

    metrics = MetricCollection(metrics, compute_groups=False)
    stability_metrics = MetricCollection(stability_metrics, compute_groups=False)

    return metrics, stability_metrics


def generate_molecules(model, dm, steps, strategy, stabilities=False):
    test_dl = dm.test_dataloader()
    model.eval()
    cuda_model = model.to("cuda")

    outputs = []
    for batch in tqdm(test_dl):
        batch = {k: v.cuda() for k, v in batch[0].items()}
        output = cuda_model._generate(batch, steps, strategy)
        outputs.append(output)

    molecules = [cuda_model._generate_mols(output) for output in outputs]
    molecules = [mol for mol_list in molecules for mol in mol_list]

    if not stabilities:
        return molecules, outputs

    stabilities = [cuda_model._generate_stabilities(output) for output in outputs]
    stabilities = [mol_stab for mol_stabs in stabilities for mol_stab in mol_stabs]
    return molecules, outputs, stabilities


def calc_metrics_(rdkit_mols, metrics, stab_metrics=None, mol_stabs=None):
    metrics.reset()
    metrics.update(rdkit_mols)
    results = metrics.compute()

    if stab_metrics is None:
        return results

    stab_metrics.reset()
    stab_metrics.update(mol_stabs)
    stab_results = stab_metrics.compute()

    results = {**results, **stab_results}
    return results


def print_results(results, std_results=None):
    print()
    print(f"{'Metric':<22}Result")
    print("-" * 30)

    for metric, value in results.items():
        result_str = f"{metric:<22}{value:.5f}"
        if std_results is not None:
            std = std_results[metric]
            result_str = f"{result_str} +- {std:.7f}"

        print(result_str)
    print()
