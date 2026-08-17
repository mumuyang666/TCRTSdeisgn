#!/usr/bin/python
# -*- coding:utf-8 -*-
"""IMGT / Chothia numbering definitions.

TCR chains are mapped onto the antibody numbering scheme: the TCR beta chain
plays the role of the heavy chain and the alpha chain that of the light chain.
CDR3 therefore corresponds to ``H3``.

Ranges are ``[start, end]`` residue ids, both ends inclusive.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
RENUMBER = os.path.join(_HERE, 'renumber.py')

# 6.6 A between one pair of atoms means the two residues are interacting
CONTACT_DIST = 6.6


class IMGT:
    # heavy chain (TCR beta)
    HFR1 = (1, 26)
    HFR2 = (39, 55)
    HFR3 = (66, 104)
    HFR4 = (118, 129)

    H1 = (27, 38)
    H2 = (56, 65)
    H3 = (105, 117)

    # light chain (TCR alpha)
    LFR1 = (1, 26)
    LFR2 = (39, 55)
    LFR3 = (66, 104)
    LFR4 = (118, 129)

    L1 = (27, 38)
    L2 = (56, 65)
    L3 = (105, 117)

    Hconserve = {
        23: ['CYS'],
        41: ['TRP'],
        104: ['CYS']
    }

    Lconserve = {
        23: ['CYS'],
        41: ['TRP'],
        104: ['CYS']
    }

    @classmethod
    def renumber(cls, pdb, out_pdb):
        code = os.system(f'python {RENUMBER} {pdb} {out_pdb} imgt 0')
        return code


class Chothia:
    # heavy chain
    HFR1 = (1, 25)
    HFR2 = (33, 51)
    HFR3 = (57, 94)
    HFR4 = (103, 113)

    H1 = (26, 32)
    H2 = (52, 56)
    H3 = (95, 102)

    # light chain
    LFR1 = (1, 23)
    LFR2 = (35, 49)
    LFR3 = (57, 88)
    LFR4 = (98, 107)

    L1 = (24, 34)
    L2 = (50, 56)
    L3 = (89, 97)

    Hconserve = {
        92: ['CYS']
    }

    Lconserve = {
        88: ['CYS']
    }

    @classmethod
    def renumber(cls, pdb, out_pdb):
        code = os.system(f'python {RENUMBER} {pdb} {out_pdb} chothia 0')
        return code
