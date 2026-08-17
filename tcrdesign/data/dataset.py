#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Inference dataset: turn TCR-pMHC PDB complexes into model inputs.

Chain convention (inherited from the antibody framework the model is built on):

    TCR beta  chain -> "heavy chain"  -> CDR3 is ``H3``
    TCR alpha chain -> "light chain"  -> CDR3 is ``L3``
    peptide (+MHC)  -> "antigen"

Unlike the training dataset this module holds everything in memory and does
not write ``*_processed`` pickle caches.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .pdb_utils import AgAbComplex, VOCAB
from .framework_templates import ConserveTemplateGenerator
from ..utils.logger import print_log


def _generate_chain_data(residues, start):
    """Encode one chain into per-residue coordinates / types / positions."""
    backbone_atoms = VOCAB.backbone_atoms
    # Coords, Sequence, residue positions, mask for loss calculation (exclude missing coordinates)
    X, S, res_pos, xloss_mask = [], [], [], []
    # global node; coordinates will be set to the center of the chain
    X.append([[0, 0, 0] for _ in range(VOCAB.MAX_ATOM_NUMBER)])
    S.append(VOCAB.symbol_to_idx(start))
    res_pos.append(0)
    xloss_mask.append([0 for _ in range(VOCAB.MAX_ATOM_NUMBER)])
    # other nodes
    for residue in residues:
        residue_xloss_mask = [0 for _ in range(VOCAB.MAX_ATOM_NUMBER)]
        bb_atom_coord = residue.get_backbone_coord_map()
        sc_atom_coord = residue.get_sidechain_coord_map()
        if 'CA' not in bb_atom_coord:
            for atom in bb_atom_coord:
                ca_x = bb_atom_coord[atom]
                print_log(f'no ca, use {atom}', level='DEBUG')
                break
        else:
            ca_x = bb_atom_coord['CA']
        x = [ca_x for _ in range(VOCAB.MAX_ATOM_NUMBER)]

        i = 0
        for atom in backbone_atoms:
            if atom in bb_atom_coord:
                x[i] = bb_atom_coord[atom]
                residue_xloss_mask[i] = 1
            i += 1
        for atom in residue.sidechain:
            if atom in sc_atom_coord:
                x[i] = sc_atom_coord[atom]
                residue_xloss_mask[i] = 1
            i += 1

        X.append(x)
        S.append(VOCAB.symbol_to_idx(residue.get_symbol()))
        res_pos.append(residue.get_id()[0])
        xloss_mask.append(residue_xloss_mask)
    X = np.array(X)
    center = np.mean(X[1:].reshape(-1, 3), axis=0)
    X[0] = center  # set center
    if start == VOCAB.BOA:  # epitope does not have position encoding
        res_pos = [0 for _ in res_pos]
    return {'X': X, 'S': S, 'residue_pos': res_pos, 'xloss_mask': xloss_mask}


def encode_complex(cplx: AgAbComplex,
                   cdr: Optional[Sequence[str]] = ('H3',),
                   paratope: Sequence[str] = ('H3',),
                   full_antigen: bool = False) -> Dict:
    """Encode one complex into a single model input item.

    Args:
        cplx: parsed complex.
        cdr: CDRs to redesign, e.g. ``('H3',)``. ``None`` regenerates the
            whole TCR including the framework.
        paratope: CDRs treated as the binding interface (docking anchor).
        full_antigen: use the entire antigen instead of the epitope only.
    """
    # antigen: epitope only by default (48 interface residues)
    ag_residues = []
    if full_antigen:
        ag = cplx.get_antigen()
        for chain_name in ag.get_chain_names():
            chain = ag.get_chain(chain_name)
            for i in range(len(chain)):
                ag_residues.append(chain.get_residue(i))
    else:
        for residue, _chain, _i in cplx.get_epitope():
            ag_residues.append(residue)

    ag_data = _generate_chain_data(ag_residues, VOCAB.BOA)

    hc, lc = cplx.get_heavy_chain(), cplx.get_light_chain()
    hc_data = _generate_chain_data([hc.get_residue(i) for i in range(len(hc))], VOCAB.BOH)
    lc_data = _generate_chain_data([lc.get_residue(i) for i in range(len(lc))], VOCAB.BOL)

    data = {key: np.concatenate([ag_data[key], hc_data[key], lc_data[key]], axis=0)
            for key in hc_data}
    data['pdb_id'] = getattr(cplx, 'pdb_id', 'unknown')

    n_ag, n_hc, n_lc = len(ag_data['S']), len(hc_data['S']), len(lc_data['S'])

    # cmask / smask: 0 for fixed, 1 for generate.
    # coordinates of the global nodes and of the antigen are never generated.
    cmask = [0] * n_ag + [0] + [1] * (n_hc - 1) + [0] + [1] * (n_lc - 1)

    def _mask_for(cdr_names):
        mask = [0] * (n_ag + n_hc + n_lc)
        for name in cdr_names:
            cdr_range = cplx.get_cdr_pos(name)
            if cdr_range is None:
                raise ValueError(
                    f'CDR {name} not found in {data["pdb_id"]}. The complex is '
                    'probably not IMGT-renumbered.')
            offset = n_ag + 1 + (0 if name[0].upper() == 'H' else n_hc)
            for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
                mask[idx] = 1
        return mask

    if cdr is None:
        smask = list(cmask)
    else:
        cdrs = [cdr] if isinstance(cdr, str) else list(cdr)
        smask = _mask_for(cdrs)

    paratope = [paratope] if isinstance(paratope, str) else list(paratope)

    data['cmask'], data['smask'] = cmask, smask
    data['paratope_mask'] = _mask_for(paratope)
    data['template'] = ConserveTemplateGenerator().construct_template(cplx, align=False)
    return data


def collate_fn(batch: List[Dict]) -> Dict:
    """Concatenate items along the residue axis (graph-style batching)."""
    keys = ['X', 'S', 'smask', 'cmask', 'paratope_mask', 'residue_pos', 'template', 'xloss_mask']
    types = [torch.float, torch.long, torch.bool, torch.bool, torch.bool,
             torch.long, torch.float, torch.bool]
    res = {}
    for key, _type in zip(keys, types):
        res[key] = torch.cat([torch.tensor(item[key], dtype=_type) for item in batch], dim=0)
    res['pdb_id'] = [item['pdb_id'] for item in batch]
    res['lengths'] = torch.tensor([len(item['S']) for item in batch], dtype=torch.long)
    return res


class ComplexDataset(torch.utils.data.Dataset):
    """In-memory dataset over a list of parsed complexes."""

    def __init__(self, complexes: List[AgAbComplex],
                 cdr: Optional[Sequence[str]] = ('H3',),
                 paratope: Sequence[str] = ('H3',),
                 full_antigen: bool = False):
        super().__init__()
        self.complexes = complexes
        self.cdr = cdr
        self.paratope = paratope
        self.full_antigen = full_antigen

    def __len__(self):
        return len(self.complexes)

    def __getitem__(self, idx):
        return encode_complex(self.complexes[idx], self.cdr,
                              self.paratope, self.full_antigen)

    collate_fn = staticmethod(collate_fn)
