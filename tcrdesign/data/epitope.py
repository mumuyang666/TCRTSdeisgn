#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Epitope definitions on a pMHC structure.

An epitope definition is a JSON list of ``[chain_id, [residue_number, insertion_code]]``
entries, e.g.::

    [["C", [1, " "]], ["C", [2, " "]], ["A", [65, " "]]]

Two ways to produce one:

* :func:`epitope_from_chains` — take whole chains (typically the peptide) and,
  if a reference ligand is unavailable, optionally pad with the MHC residues
  closest to them.
* :func:`epitope_from_interface` — mimic an existing binder: take the receptor
  residues at the interface with a known ligand chain. Use this when the input
  structure already contains a TCR (or any other ligand) whose binding mode you
  want to reproduce.
"""
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .pdb_utils import Protein, VOCAB

# The model consumes at most this many antigen residues (see AgAbComplex).
NUM_EPITOPE_RESIDUES = 48


def _residue_coords(residue) -> Tuple[List, List]:
    """Padded coordinates + presence mask for one residue."""
    coord = {}
    coord.update(residue.get_backbone_coord_map())
    coord.update(residue.get_sidechain_coord_map())
    num_pad = VOCAB.MAX_ATOM_NUMBER - len(coord)
    x = [coord[key] for key in coord] + [[0, 0, 0] for _ in range(num_pad)]
    mask = [1 for _ in coord] + [0 for _ in range(num_pad)]
    return x, mask


def _collect(protein: Protein, chains: Sequence[str]):
    rids, xs, masks = [], [], []
    for chain_name in chains:
        chain = protein.get_chain(chain_name)
        if chain is None:
            raise ValueError(f'chain {chain_name} not found in {protein.get_id()}')
        for i, residue in enumerate(chain):
            x, mask = _residue_coords(residue)
            rids.append((chain_name, i))
            xs.append(x)
            masks.append(mask)
    if not xs:
        raise ValueError(f'no residues collected from chains {list(chains)}')
    return rids, np.array(xs), np.array(masks).astype(bool)


def _min_dist_matrix(a_xs, a_mask, b_xs, b_mask):
    """Per-residue-pair minimum inter-atomic distance, ignoring padding."""
    dist = np.linalg.norm(a_xs[:, None] - b_xs[None, :], axis=-1)  # [Na, Nb, M]
    dist = dist + np.logical_not(a_mask[:, None] * b_mask[None, :]) * 1e6
    return np.min(dist, axis=-1)  # [Na, Nb]


def epitope_from_interface(pdb: str,
                           receptor_chains: Sequence[str],
                           ligand_chains: Sequence[str],
                           k: int = NUM_EPITOPE_RESIDUES) -> List[Tuple]:
    """Receptor residues closest to a known ligand.

    Returns a list of ``(residue, chain_name, index, distance)``, sorted by
    distance. Use when the structure contains a binder whose epitope you want
    to copy — for a TCR-pMHC complex, pass the pMHC chains as receptor and the
    TCR chains as ligand.
    """
    prot = Protein.from_pdb(pdb)
    rec_rids, rec_xs, rec_mask = _collect(prot, receptor_chains)
    lig_rids, lig_xs, lig_mask = _collect(prot, ligand_chains)

    dist_mat = _min_dist_matrix(rec_xs, rec_mask, lig_xs, lig_mask)
    min_dists = np.min(dist_mat, axis=-1)  # [Nrec]

    topk = min(len(min_dists), k)
    ind = np.argpartition(min_dists, topk - 1)[:topk]
    ind = ind[np.argsort(min_dists[ind])]

    out = []
    for idx in ind:
        chain_name, i = rec_rids[idx]
        residue = prot.get_chain(chain_name).get_residue(i)
        out.append((residue, chain_name, i, float(min_dists[idx])))
    return out


def epitope_from_chains(pdb: str,
                        peptide_chains: Sequence[str],
                        mhc_chains: Sequence[str] = (),
                        k: int = NUM_EPITOPE_RESIDUES) -> List[Tuple]:
    """Whole peptide chain(s), padded with the nearest MHC residues.

    The peptide is always included in full. If ``mhc_chains`` is given and the
    peptide alone is shorter than ``k``, the MHC residues closest to the
    peptide are added until ``k`` residues are reached — the model was trained
    on 48 antigen residues, so the MHC groove is normally part of the input.
    """
    prot = Protein.from_pdb(pdb)
    pep_rids, pep_xs, pep_mask = _collect(prot, peptide_chains)

    out = []
    for chain_name, i in pep_rids:
        residue = prot.get_chain(chain_name).get_residue(i)
        out.append((residue, chain_name, i, 0.0))

    remaining = k - len(out)
    if remaining > 0 and mhc_chains:
        mhc_rids, mhc_xs, mhc_mask = _collect(prot, mhc_chains)
        dist_mat = _min_dist_matrix(mhc_xs, mhc_mask, pep_xs, pep_mask)
        min_dists = np.min(dist_mat, axis=-1)  # [Nmhc]
        topk = min(len(min_dists), remaining)
        ind = np.argpartition(min_dists, topk - 1)[:topk]
        ind = ind[np.argsort(min_dists[ind])]
        for idx in ind:
            chain_name, i = mhc_rids[idx]
            residue = prot.get_chain(chain_name).get_residue(i)
            out.append((residue, chain_name, i, float(min_dists[idx])))
    return out


def save_epitope(epitope: List[Tuple], path: str) -> None:
    """Write an epitope definition to JSON."""
    data = [[chain_name, list(residue.get_id())] for residue, chain_name, *_ in epitope]
    with open(path, 'w') as fout:
        json.dump(data, fout)


def load_epitope_def(path: str) -> List[Tuple[str, Tuple]]:
    """Read an epitope definition written by :func:`save_epitope`."""
    with open(path, 'r') as fin:
        data = json.load(fin)
    return [(chain_name, tuple(pos)) for chain_name, pos in data]


def select_epitope_residues(prot: Protein,
                            epitope_def: List[Tuple[str, Tuple]]) -> List:
    """Pick the residues named by an epitope definition, in definition order."""
    wanted: Dict[str, Dict] = {}
    for chain_name, pos in epitope_def:
        wanted.setdefault(chain_name, {})[tuple(pos)] = True

    found, residues = {}, {}
    for chain_name in wanted:
        chain = prot.get_chain(chain_name)
        if chain is None:
            raise ValueError(f'chain {chain_name} of the epitope not found in the structure')
        for residue in chain:
            rid = tuple(residue.get_id())
            if rid in wanted[chain_name]:
                residues[(chain_name, rid)] = residue
                found[(chain_name, rid)] = True

    missing = [(c, p) for c, poss in wanted.items() for p in poss if (c, p) not in found]
    if missing:
        raise ValueError(f'{len(missing)} epitope residue(s) not found in the structure, '
                         f'e.g. {missing[:3]}')
    # preserve the order of the definition
    return [residues[(c, tuple(p))] for c, p in epitope_def]
