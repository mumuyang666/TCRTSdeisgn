#!/usr/bin/python
# -*- coding:utf-8 -*-
"""TCRDesign — TCR CDR3 sequence-structure co-design (inference).

    from tcrdesign import load_model, design

    model = load_model('weights/tcrdesign_cdr3.pt', device='cuda:0')
    results = design(model, [{'pdb': 'x.pdb', 'beta_chain': 'B',
                              'alpha_chain': 'A', 'antigen_chains': ['C']}],
                     out_dir='out')
"""
__version__ = '1.0.0'

from .infer import load_model, design, load_complexes, to_cplx
from .data.pdb_utils import VOCAB, AgAbComplex
from .utils.random_seed import setup_seed

__all__ = [
    'load_model', 'design', 'load_complexes', 'to_cplx',
    'VOCAB', 'AgAbComplex', 'setup_seed', '__version__',
]
