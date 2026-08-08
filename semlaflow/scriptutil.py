"""Util file for Equinv scripts"""

import math
import resource
from dataclasses import dataclass
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

# Cell-type (class) conditioning: single id<->name source of truth. Integer id == list index,
# matching the `# cell_class N` SWC headers and dendrite_gen's utils.data_loading.CELL_CLASS_NAMES.
NEURON_CELL_CLASS_NAMES = ["23P", "4P", "5P-IT", "5P-ET", "5P-NP", "6P-IT", "6P-CT"]

# Botanical-tree genus labels, id == list index, matching the `# cell_class N` SWC headers and
# dendrite_gen's utils.data_loading.TREE_GENUS_NAMES. Ordered by corpus frequency.
TREE_GENUS_NAMES = ["Fagus", "Quercus", "Acer", "Carpinus", "Fraxinus", "Betula"]


@dataclass(frozen=True)
class DatasetConfig:
    """Per-dataset constants for the neuron/tree (non-RDKit) pipeline.

    coord_std     -- measured by preprocess_neurons.py over the train split, post per-tree
                     zero-CoM. Re-run that script and refresh if the corpus changes.
    bucket_limits -- BucketBatchSampler limits. Must cover the largest graph in *every* split,
                     not just train (data/util.py raises otherwise, val loader included).
    max_nodes     -- the `--max_atoms` cap the .smol files were built with. Note train.py's
                     `--max_atoms` must be strictly greater than this, since SemlaGenerator
                     indexes an Embedding(max_atoms) with the raw node count.
    class_names   -- id -> name for cell-class conditioning. None => unconditional corpus,
                     `--type_conditioning` is rejected for it.
    units         -- physical unit of the raw SWC coordinates. Neurons are microns; the
                     botanical-tree corpora are metres.
    """

    coord_std: float
    bucket_limits: list[int]
    max_nodes: int
    class_names: list[str] | None = None
    units: str = "um"
    source: str = ""

    @property
    def n_classes(self) -> int:
        return 0 if self.class_names is None else len(self.class_names)


# Buckets below 200 are shared by every SWC corpus; only the tail differs. Empty buckets cost
# nothing, so reusing the prefix keeps small graphs well-batched across all datasets.
_SWC_BUCKET_PREFIX = [24, 40, 56, 72, 96, 128, 160, 200]

# Every dataset that uses the neuron pipeline (neuron vocab/transform/NeuronCFM/loss-based
# checkpointing). Adding an entry here is all that is needed to register a new SWC corpus.
DATASET_CONFIGS: dict[str, DatasetConfig] = {
    # Original unlabelled neuron corpus. Sizes: p5=18, median=46, p95=108, train max=217.
    "neurons": DatasetConfig(
        coord_std=62.6894,
        bucket_limits=_SWC_BUCKET_PREFIX + [220],
        max_nodes=256,
        class_names=None,
        source="/Users/umer/Documents/neurons_final",
    ),
    # Class-labelled neuron corpus: soma-rooted, binarized, `# cell_class N` headers. Measured
    # over the full train split (22740 graphs); node counts min=8, max=242 (10 graphs > 256
    # dropped) -> top bucket must cover 242.
    "neurons_conditional": DatasetConfig(
        coord_std=66.0298,
        bucket_limits=_SWC_BUCKET_PREFIX + [224, 256],
        max_nodes=256,
        class_names=NEURON_CELL_CLASS_NAMES,
        source="/Users/umer/Documents/neurons_conditional",
    ),
    # Botanical-tree corpora: the same 3368 QSM reconstructions binarized to three depth caps.
    # Base-rooted, binarized, `# cell_class N` genus headers, coords in METRES (not microns).
    # All three keep 100% of their graphs at these caps -- nothing is dropped.
    #
    # These are far larger than any neuron (d20 median 338, max 3056 vs the neuron corpora's
    # max 242), and SemlaFlow's activation memory is O(N^2) at ~57 KB per node-pair in fp32.
    # Training d15/d20 therefore needs `--precision bf16-mixed --grad_checkpointing`; see the
    # memory table in NEURONS.md section 6.
    "trees_genus_d10": DatasetConfig(
        coord_std=1.6417,
        bucket_limits=_SWC_BUCKET_PREFIX + [256, 320, 384],
        max_nodes=384,
        class_names=TREE_GENUS_NAMES,
        units="m",
        source="/Users/umer/Documents/trees_genus_d10",
    ),
    "trees_genus_d15": DatasetConfig(
        coord_std=1.8062,
        bucket_limits=_SWC_BUCKET_PREFIX + [264, 336, 416, 512, 640, 768, 1024, 1280, 1536],
        max_nodes=1536,
        class_names=TREE_GENUS_NAMES,
        units="m",
        source="/Users/umer/Documents/trees_genus_d15",
    ),
    "trees_genus_d20": DatasetConfig(
        coord_std=1.9999,
        bucket_limits=_SWC_BUCKET_PREFIX + [264, 336, 424, 528, 648, 784, 1024, 1408, 1792, 2304, 3072],
        max_nodes=3072,
        class_names=TREE_GENUS_NAMES,
        units="m",
        source="/Users/umer/Documents/trees_genus_d20",
    ),
}


def get_dataset_config(name: str) -> DatasetConfig:
    """Strict lookup for build-time paths. Raises with the list of known names."""
    try:
        return DATASET_CONFIGS[name]
    except KeyError:
        raise ValueError(
            f"Unknown neuron-pipeline dataset '{name}'. Known: {', '.join(DATASET_CONFIGS)}."
        ) from None


def class_names_for_dataset(name) -> list[str] | None:
    """Tolerant lookup for checkpoint-driven paths; never raises.

    Returns None for an unconditional corpus, a missing name, or a checkpoint that predates
    the registry. Callers fall back to `id<N>` labels.
    """
    cfg = DATASET_CONFIGS.get(name) if isinstance(name, str) else None
    return None if cfg is None else cfg.class_names


def class_label(dataset, idx: int) -> str:
    """Human-readable name for class id `idx`, falling back to `id<N>`."""
    names = class_names_for_dataset(dataset)
    return names[idx] if names is not None and 0 <= idx < len(names) else f"id{idx}"


# Back-compat aliases. Derived from DATASET_CONFIGS -- edit the table above, not these.
NEURON_COORDS_STD_DEV = DATASET_CONFIGS["neurons"].coord_std
NEURON_BUCKET_LIMITS = DATASET_CONFIGS["neurons"].bucket_limits
NEURON_CONDITIONAL_COORDS_STD_DEV = DATASET_CONFIGS["neurons_conditional"].coord_std
NEURON_CONDITIONAL_BUCKET_LIMITS = DATASET_CONFIGS["neurons_conditional"].bucket_limits
NEURON_NUM_CLASSES = len(NEURON_CELL_CLASS_NAMES)
NEURON_DATASETS = tuple(DATASET_CONFIGS)

# Catch table typos at import rather than 12 hours into a run.
for _name, _cfg in DATASET_CONFIGS.items():
    assert _cfg.coord_std > 0, f"{_name}: coord_std must be positive"
    assert _cfg.bucket_limits == sorted(_cfg.bucket_limits), f"{_name}: bucket_limits not sorted"
    assert max(_cfg.bucket_limits) <= _cfg.max_nodes, f"{_name}: top bucket exceeds max_nodes"

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
