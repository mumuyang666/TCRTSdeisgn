#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Framework initialisation from conserved IMGT positions.

``ConserveTemplateGenerator`` builds the initial backbone coordinates handed to
the model: conserved framework positions are placed from a stored template
(``template.json``) and the remaining residues are interpolated between them.

The script that regenerated ``template.json`` from a training set is not part of
the inference release.
"""

import os
import json

import numpy as np

from .pdb_utils import VOCAB, AgAbComplex
from ..utils.geometry import kabsch
from ..utils.singleton import singleton


@singleton
class ConserveTemplateGenerator:
    def __init__(self, json_path=None):
        if json_path is None:
            folder = os.path.split(__file__)[0]
            json_path = os.path.join(folder, 'template.json')
        with open(json_path, 'r') as fin:
            self.template_map = json.load(fin)
    
    def _chain_template(self, cplx: AgAbComplex, poses, n_channel, heavy=True):
        chain = cplx.get_heavy_chain() if heavy else cplx.get_light_chain()
        chain_name = 'H' if heavy else 'L'
        hit_map = { pos: False for pos in poses }
        X, hit_index = [], []
        for i, residue in enumerate(chain):
            pos, _ = residue.get_id()
            pos = str(pos)
            if pos in hit_map:
                coord = self.template_map[chain_name][pos]  # N, CA, C, O
                ca, num_sc = coord[1], n_channel - len(coord)
                coord.extend([ca for _ in range(num_sc)])
                hit_index.append(i)
                coord = np.array(coord)
            else:
                coord = [[0, 0, 0] for _ in range(n_channel)]
            X.append(coord)
        # uniform distribution between residues and extension at two ends
        for left_i, right_i in zip(hit_index[:-1], hit_index[1:]):
            left, right = X[left_i], X[right_i]
            span, index_span = right - left, right_i - left_i
            span = span / index_span
            for i in range(left_i + 1, right_i):
                X[i] = X[i - 1] + span
        # start and end
        if hit_index[0] != 0:
            left_i = hit_index[0]
            span = X[left_i] - X[left_i + 1]
            for i in reversed(range(0, left_i)):
                X[i] = X[i + 1] + span
        if hit_index[-1] != len(X) - 1:
            right_i = hit_index[-1]
            span = X[right_i] - X[right_i - 1]
            for i in range(right_i + 1, len(X)):
                X[i] = X[i - 1] + span
        return X, hit_index

    def construct_template(self, cplx: AgAbComplex, n_channel=VOCAB.MAX_ATOM_NUMBER, align=True):
        hc, hc_hit = self._chain_template(cplx, self.template_map['H'], n_channel, heavy=True)
        lc, lc_hit = self._chain_template(cplx, self.template_map['L'], n_channel, heavy=False)
        template = np.array(hc + lc)  # [N, n_channel, 3]
        if align:
            # align (will be dropped in the future)
            true_X_bb, temp_X_bb = [], []
            chains = [cplx.get_heavy_chain(), cplx.get_light_chain()]
            temps, hits = [hc, lc], [hc_hit, lc_hit]
            for chain, temp, hit in zip(chains, temps, hits):
                for i, residue_temp in zip(hit, temp):
                    residue = chain.get_residue(i)
                    bb = residue.get_backbone_coord_map()
                    for ai, atom in enumerate(VOCAB.backbone_atoms):
                        if atom not in bb:
                            continue
                        true_X_bb.append(bb[atom])
                        temp_X_bb.append(residue_temp[ai])
            true_X_bb, temp_X_bb = np.array(true_X_bb), np.array(temp_X_bb)
            _, Q, t = kabsch(temp_X_bb, true_X_bb)
            template = np.dot(template, Q) + t
        return template
