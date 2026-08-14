"""TMD (Topological Morphology Descriptor) conditioning features for neuron graphs.

Geometry/topology port of dendrite_gen's TMD utilities (numpy + networkx only). The
public entry point is `compute_neuron_tmd`, which turns a `GeometricMol` neuron graph
into a fixed-length conditioning vector by concatenating one 16x16 persistence image per
filtration (`NEURON_TMD_FILTRATIONS` = path + radial_root => 512-dim by default). Only
rotation-invariant filtrations are supported; see `NEURON_TMD_SUPPORTED_FILTRATIONS`.

This vector is a *global* per-graph feature; the model broadcasts it to every node (see
SemlaGenerator), mirroring how `size_emb` is injected.
"""

from __future__ import annotations

import numpy as np
import torch

from .tmd import compute_tmd_mixed

# Persistence image grid: n_bins * n_bins values per filtration.
NEURON_TMD_N_BINS = 16

# Filtrations that may be conditioned on. `compute_tmd_mixed` also implements `height`
# (projection onto a fixed axis) and `rho` (distance from that axis), but neither is
# usable here: both are defined relative to an anatomical axis, while
# `scriptutil.neuron_mol_transform` applies a uniformly random 3D rotation every epoch and
# SemlaGenerator is E(3)-equivariant. The descriptor is computed once at preprocess time in
# the SWC's own frame, so the model would be told "this tree reaches far along +y" while
# seeing coordinates in an arbitrary orientation -- the axis half of the signal is
# unresolvable. dendrite_gen can use them because it is SO(2)-equivariant about a fixed
# `so2_axis`. `path` and `radial_root` are fully rotation-invariant and so survive the
# augmentation intact. See COMPATIBLE.md §4.15.
NEURON_TMD_SUPPORTED_FILTRATIONS = ("path", "radial_root")

# Default conditioning set: geodesic path length from the soma + straight-line distance
# from the soma. The two are complementary -- their ratio is what contraction/tortuosity
# measures -- and both are rotation-invariant.
NEURON_TMD_FILTRATIONS = ("path", "radial_root")
NEURON_TMD_DIM = len(NEURON_TMD_FILTRATIONS) * NEURON_TMD_N_BINS * NEURON_TMD_N_BINS  # 512

# Evaluation filtration for the validation suite.
#
# NOTE: this now *overlaps* the conditioning set. Of the four implemented filtrations only
# `path` and `radial_root` are rotation-invariant, and an E(3)-equivariant model's output
# arrives in an arbitrary global orientation, so no axis-dependent filtration can score it.
# Conditioning on both therefore leaves nothing outside the set, and `mmd_tmd` /
# `tmd_barlen_w1` partly measure the model reproducing its own conditioning input -- the
# same blind spot dendrite_gen documents (VALIDATION_METRICS_SUMMARY.md §8.3). This is a
# deliberate, disclosed trade: read those two metrics as consistency checks, and use the
# matched-pair metrics in `validation.tmd_conditional_eval` (`val-tmd_cond-*`) as the
# evidence that conditioning is actually being followed.
TMD_EVAL_FILTRATION = "radial_root"


def neuron_tmd_dim(filtrations=NEURON_TMD_FILTRATIONS, n_bins: int = NEURON_TMD_N_BINS) -> int:
    """Width of the conditioning vector for a filtration set.

    Single source of truth for the arithmetic, mirroring
    `dendrite_gen/utils/tmd.py:tmd_conditioning_dim`. Every consumer derives the dim from
    this rather than hardcoding it, so a filtration-set change cannot desync the model's
    input projection from the data.
    """
    return len(tuple(filtrations)) * int(n_bins) * int(n_bins)


def validate_filtrations(filtrations) -> tuple:
    """Normalise and check a requested conditioning filtration set.

    Rejects unsupported names (with the reason -- see NEURON_TMD_SUPPORTED_FILTRATIONS) and
    duplicates, which would otherwise silently double the vector width and hand the model
    two identical blocks.
    """
    names = tuple(filtrations)
    if not names:
        raise ValueError("At least one TMD filtration is required.")

    unsupported = [n for n in names if n not in NEURON_TMD_SUPPORTED_FILTRATIONS]
    if unsupported:
        raise ValueError(
            f"Unsupported TMD filtration(s) for conditioning: {', '.join(unsupported)}. "
            f"Supported: {', '.join(NEURON_TMD_SUPPORTED_FILTRATIONS)}. "
            "`height` and `rho` are defined relative to a fixed anatomical axis, but the "
            "neuron training transform applies a random 3D rotation and the model is "
            "E(3)-equivariant, so their axis information cannot be resolved. See "
            "COMPATIBLE.md §4.15."
        )

    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(
            f"Duplicate TMD filtration(s): {', '.join(duplicates)}. Each filtration "
            "contributes one persistence image; repeating one only pads the vector."
        )
    return names


def compute_tmd_embedding(
    G,
    *,
    filtration: str = TMD_EVAL_FILTRATION,
    n_bins: int = NEURON_TMD_N_BINS,
    sigma: float = 0.05,
):
    """Single-filtration TMD persistence-image embedding for evaluation.

    Mirrors `dendrite_gen/utils/tmd.py:compute_tmd_embedding`. Intentionally a single
    filtration -- cheaper than the full conditioning vector, and it keeps the evaluation
    embedding stable when the conditioning set changes.

    Takes a networkx graph that must already be a rooted tree; callers should pass a
    `validation.sanitise.sanitise_graph` output. Raises rather than returning a zero
    vector, so a caller cannot silently drop malformed graphs from a pooled comparison.
    """
    return compute_tmd_mixed(G, filtrations=(filtration,), n_bins=n_bins, sigma=sigma)


def compute_neuron_tmd(
    mol,
    *,
    filtrations=NEURON_TMD_FILTRATIONS,
    n_bins: int = NEURON_TMD_N_BINS,
) -> torch.Tensor:
    """Compute the TMD conditioning vector for a neuron `GeometricMol`.

    Builds a rooted networkx tree (root = node 0, the soma per `swc.py`) and computes the
    mixed-method TMD persistence-image embedding for the requested filtrations. Returns a
    1-D float32 tensor of length `neuron_tmd_dim(filtrations, n_bins)`.

    Degenerate graphs (non-tree, empty, or any computation error) yield a zero vector of
    the expected length, so callers never crash on a malformed sample. An unsupported
    filtration is a caller bug, not a bad sample, so it raises.
    """
    filtrations = validate_filtrations(filtrations)
    dim = neuron_tmd_dim(filtrations, n_bins)
    # Local import avoids a circular dependency (validation.convert is import-light).
    from semlaflow.validation.convert import geometric_mol_to_nx

    try:
        G = geometric_mol_to_nx(mol, coord_scale=1.0, root=0)
        vec = compute_tmd_mixed(G, filtrations=tuple(filtrations), n_bins=n_bins)
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        if vec.shape[0] != dim or not np.all(np.isfinite(vec)):
            return torch.zeros(dim, dtype=torch.float32)
        return torch.from_numpy(vec)
    except Exception:
        return torch.zeros(dim, dtype=torch.float32)
