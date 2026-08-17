#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Write an epitope definition for ``design_seq.py``.

Two modes:

Mimic an existing binder — take the pMHC residues closest to a TCR (or any
other ligand) already present in the structure::

    python scripts/get_epitope.py --pdb 7n2r.pdb \\
        --receptor C A B --ligand F D --out epitope.json

From the peptide alone — take the whole peptide chain and pad with the nearest
MHC residues up to 48::

    python scripts/get_epitope.py --pdb pmhc.pdb \\
        --peptide C --mhc A B --out epitope.json
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcrdesign.data.epitope import (NUM_EPITOPE_RESIDUES, epitope_from_chains,
                                    epitope_from_interface, save_epitope)


def parse():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pdb', required=True, help='structure to read')
    p.add_argument('--out', required=True, help='output JSON path')

    g = p.add_argument_group('mimic an existing binder')
    g.add_argument('--receptor', nargs='+', help='receptor chains (the pMHC)')
    g.add_argument('--ligand', nargs='+', help='ligand chains (e.g. a reference TCR)')

    g = p.add_argument_group('from the peptide')
    g.add_argument('--peptide', nargs='+', help='peptide chain(s), taken in full')
    g.add_argument('--mhc', nargs='+', default=[], help='MHC chains used as padding')

    g = p.add_argument_group('common')
    g.add_argument('-k', '--num_residues', type=int, default=NUM_EPITOPE_RESIDUES,
                   help=f'epitope size (default {NUM_EPITOPE_RESIDUES}, the '
                        'number the model was trained with)')
    args = p.parse_args()

    by_interface = bool(args.receptor and args.ligand)
    by_peptide = bool(args.peptide)
    if by_interface == by_peptide:
        p.error('use either --receptor/--ligand, or --peptide (with optional --mhc)')
    return args, by_interface


def main():
    args, by_interface = parse()

    if by_interface:
        epitope = epitope_from_interface(args.pdb, args.receptor, args.ligand,
                                        k=args.num_residues)
        print(f'{len(epitope)} epitope residues on chains {args.receptor}, '
              f'closest to {args.ligand}:')
    else:
        epitope = epitope_from_chains(args.pdb, args.peptide, args.mhc,
                                      k=args.num_residues)
        n_pep = sum(1 for _r, c, *_ in epitope if c in args.peptide)
        print(f'{len(epitope)} epitope residues: {n_pep} from the peptide '
              f'{args.peptide}, {len(epitope) - n_pep} from the MHC {args.mhc}:')

    print('  chain  position  residue  distance')
    for residue, chain_name, _i, dist in epitope:
        pos, icode = residue.get_id()
        print(f'  {chain_name:5s}  {pos:>4}{icode.strip():1s}      '
              f'{residue.get_symbol()}        {dist:.2f}')

    if len(epitope) < args.num_residues:
        print(f'\nnote: only {len(epitope)} residues found, fewer than the '
              f'{args.num_residues} the model expects. Add MHC chains with '
              '--mhc to fill the epitope out.')

    save_epitope(epitope, args.out)
    print(f'\nsaved to {args.out}')


if __name__ == '__main__':
    main()
