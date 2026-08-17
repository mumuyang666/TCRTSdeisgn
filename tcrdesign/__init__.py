#!/usr/bin/python
# -*- coding:utf-8 -*-
"""TCRDesign — TCR CDR3 sequence-structure co-design (inference).

From a docked TCR-pMHC complex::

    from tcrdesign import load_model, design

    model = load_model('weights/tcrdesign_cdr3.pt', device='cuda:0')
    results = design(model, [{'pdb': 'x.pdb', 'beta_chain': 'B',
                              'alpha_chain': 'A', 'antigen_chains': ['C']}],
                     out_dir='out')

From a pMHC structure and TCR framework sequences, with no TCR structure::

    from tcrdesign import load_model, design_from_sequence

    results = design_from_sequence(model, [{
        'id': 'design_1', 'pmhc_pdb': 'pmhc.pdb',
        'epitope': [['C', [1, ' ']], ['C', [2, ' ']]],
        'beta_seq': '...C' + '-' * 13 + 'FGPG...',
        'alpha_seq': '...',
    }], out_dir='out')
"""
__version__ = '1.1.0'

from .infer import load_model, design, design_from_sequence, load_complexes, to_cplx
from .data.pdb_utils import VOCAB, AgAbComplex
from .data.epitope import (epitope_from_chains, epitope_from_interface,
                           save_epitope, load_epitope_def)
from .utils.random_seed import setup_seed

__all__ = [
    'load_model', 'design', 'design_from_sequence', 'load_complexes', 'to_cplx',
    'epitope_from_chains', 'epitope_from_interface', 'save_epitope',
    'load_epitope_def',
    'VOCAB', 'AgAbComplex', 'setup_seed', '__version__',
]
