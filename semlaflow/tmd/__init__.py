"""TMD (Topological Morphology Descriptor) conditioning features for neuron graphs.

Geometry/topology port of dendrite_gen's TMD utilities (numpy + networkx only). The
public entry point is `compute_neuron_tmd`, which turns a `GeometricMol` neuron graph
into a fixed-length conditioning vector using the rotation-invariant path-length-from-root
filtration (a 16x16 persistence image = 256-dim by default).

This vector is a *global* per-graph feature; the model broadcasts it to every node (see
SemlaGenerator), mirroring how `size_emb` is injected.
"""

from __future__ import annotations

import numpy as np
import torch

from .tmd import compute_tmd_mixed

# Path-only persistence image: n_bins * n_bins per filtration.
NEURON_TMD_N_BINS = 16
NEURON_TMD_FILTRATIONS = ("path",)
NEURON_TMD_DIM = len(NEURON_TMD_FILTRATIONS) * NEURON_TMD_N_BINS * NEURON_TMD_N_BINS  # 256

# Evaluation filtration for the validation suite. Deliberately NOT in
# NEURON_TMD_FILTRATIONS: scoring with the same filtration the model is conditioned on
# would partly measure the model echoing its own input (dendrite_gen hits this; see its
# VALIDATION_METRICS_SUMMARY.md blind spot 8.3). `radial_root` is also fully
# rotation-invariant, which `height`/`rho` are not -- required for an E(3)-equivariant
# model whose outputs come out in arbitrary global orientation.
TMD_EVAL_FILTRATION = "radial_root"


def compute_tmd_embedding(
    G,
    *,
    filtration: str = TMD_EVAL_FILTRATION,
    n_bins: int = NEURON_TMD_N_BINS,
    sigma: float = 0.05,
):
    """Single-filtration TMD persistence-image embedding for evaluation.

    Mirrors `dendrite_gen/utils/tmd.py:compute_tmd_embedding`. Intentionally a single
    filtration -- cheaper than `compute_tmd_mixed`'s 3-filtration vector, and one
    filtration outside the conditioning set is the point (see TMD_EVAL_FILTRATION).

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
    1-D float32 tensor of length `len(filtrations) * n_bins * n_bins`.

    Degenerate graphs (non-tree, empty, or any computation error) yield a zero vector of
    the expected length, so callers never crash on a malformed sample.
    """
    dim = len(tuple(filtrations)) * n_bins * n_bins
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
