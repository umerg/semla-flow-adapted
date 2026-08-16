"""SWC → GeometricMol adapter for neuron/tree morphologies.

Bypasses RDKit entirely. Parses SWC parent-pointer format and builds a
GeometricMol suitable for SemlaFlow with a single node token ("NODE") and a
single real edge class (index 1, matching molecular "single bond" slot).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from semlaflow.util.molrepr import GeometricMol

# Node token index. Must match the position of "NODE" in the neuron vocabulary
# built by scriptutil.build_neuron_vocab(): ["<PAD>", "<MASK>", "NODE"] -> 2.
NEURON_NODE_TOKEN_INDEX = 2

# Edge class index for "there is an edge". We reuse molecular "single bond" slot
# (index 1) so the existing BOND_MASK_INDEX=5 constant stays valid and no
# changes to get_n_bond_types are needed. The other molecular bond classes
# (2=double, 3=triple, 4=aromatic) simply never appear as targets.
NEURON_EDGE_CLASS_INDEX = 1


def parse_swc(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int]], int, Optional[int]]:
    """Parse a cleaned SWC file. Returns (coords, edges, root_swc_id, cell_class).

    SWC columns (whitespace separated): id type x y z radius parent_id.
    Root nodes have parent_id <= 0. IDs are 1-indexed in the file; this
    function returns the original 1-indexed IDs and leaves remapping to the
    caller.

    `cell_class` is the integer parsed from a `# cell_class N` header comment
    (class-labelled corpora only); it is None when the header is absent (e.g. the
    original unlabelled neuron corpus).
    """
    coords: list[tuple[float, float, float]] = []
    node_ids: list[int] = []
    edges: list[tuple[int, int]] = []
    root_id: Optional[int] = None
    cell_class: Optional[int] = None

    with Path(path).open("r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                # Capture the class label from a `# cell_class N` header if present.
                if line.startswith("#"):
                    tok = line.lstrip("#").split()
                    if len(tok) == 2 and tok[0] == "cell_class":
                        try:
                            cell_class = int(tok[1])
                        except ValueError:
                            pass
                continue
            parts = line.split()
            if len(parts) < 7:
                raise ValueError(f"{path}: malformed SWC line '{line}'")
            nid = int(parts[0])
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            parent = int(parts[6])

            node_ids.append(nid)
            coords.append((x, y, z))

            if parent <= 0:
                if root_id is not None:
                    raise ValueError(f"{path}: multiple roots (parent<=0) found")
                root_id = nid
            else:
                edges.append((parent, nid))

    if root_id is None:
        raise ValueError(f"{path}: no root node (parent<=0) found")

    # Verify contiguous 1..N ids; this is true for cleaned files but guard anyway.
    expected = list(range(1, len(node_ids) + 1))
    if node_ids != expected:
        # Remap: preserve file order but fix ids to be 1..N.
        id_remap = {old: new for new, old in enumerate(node_ids, start=1)}
        edges = [(id_remap[p], id_remap[c]) for (p, c) in edges]
        root_id = id_remap[root_id]

    return coords, edges, root_id, cell_class


def swc_to_geometric_mol(path: Path, str_id: Optional[str] = None) -> GeometricMol:
    """Build a GeometricMol from a cleaned SWC file.

    Node ordering: root first, then remaining nodes in file order. This matches
    the convention used by dendrite_gen's nx_graph_to_adj_pos.
    """
    path = Path(path)
    coords, edges, root_id, cell_class = parse_swc(path)

    n = len(coords)
    # Reorder so root is at index 0.
    perm = [root_id] + [i for i in range(1, n + 1) if i != root_id]
    old_to_new = {old: new for new, old in enumerate(perm)}

    coords_t = torch.tensor([coords[old - 1] for old in perm], dtype=torch.float32)

    # Edges: use new 0-indexed positions. Store each edge once (undirected);
    # GeometricMol treats bond_indices as symmetric via adj_from_edges(..., symmetric=True).
    bond_indices = torch.tensor(
        [[old_to_new[p], old_to_new[c]] for (p, c) in edges],
        dtype=torch.long,
    ) if edges else torch.zeros((0, 2), dtype=torch.long)

    bond_types = torch.full((bond_indices.size(0),), NEURON_EDGE_CLASS_INDEX, dtype=torch.long)
    atomics = torch.full((n,), NEURON_NODE_TOKEN_INDEX, dtype=torch.long)
    charges = torch.zeros(n, dtype=torch.long)

    return GeometricMol(
        coords=coords_t,
        atomics=atomics,
        bond_indices=bond_indices,
        bond_types=bond_types,
        charges=charges,
        str_id=str_id or path.stem,
        cell_class=cell_class,
    )
