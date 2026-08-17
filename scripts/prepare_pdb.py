#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Renumber a TCR-pMHC structure into IMGT and check it can be parsed.

The model locates CDR3 through IMGT residue numbering, so an input PDB must be
renumbered before use. Structures taken straight from the RCSB are typically
*not* IMGT-numbered.

    python scripts/prepare_pdb.py --pdb 7n2r.pdb --beta F --alpha D \\
        --antigen C A B --out 7n2r_imgt.pdb

Requires ANARCI (``conda install -c bioconda anarci``). If ANARCI is not
available, renumber the file with any other IMGT tool and then run this script
with ``--check_only`` to verify the result.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcrdesign.data.pdb_utils import AgAbComplex, Protein   # noqa: E402


def parse():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pdb', required=True, help='input PDB file')
    p.add_argument('--out', default=None, help='output PDB (default: <input>_imgt.pdb)')
    p.add_argument('--beta', required=True, help='TCR beta chain id')
    p.add_argument('--alpha', required=True, help='TCR alpha chain id')
    p.add_argument('--antigen', nargs='+', required=True, help='peptide / MHC chain ids')
    p.add_argument('--check_only', action='store_true',
                   help='skip renumbering, only validate the input')
    return p.parse_args()


def main():
    args = parse()
    out = args.out or os.path.splitext(args.pdb)[0] + '_imgt.pdb'

    if args.check_only:
        out = args.pdb
    else:
        try:
            from tcrdesign.data.renumber import renumber_pdb
        except ImportError as e:
            raise SystemExit(
                f'cannot import ANARCI ({e}).\n'
                'Install it with:  conda install -c bioconda anarci\n'
                'Or renumber externally and rerun with --check_only.')
        renumber_pdb(args.pdb, out, scheme='imgt')
        print(f'renumbered -> {out}')

    # keep only the chains we need and normalise chain ordering
    prot = Protein.from_pdb(out)
    available = prot.get_chain_names()
    for name, chain in [('beta', args.beta), ('alpha', args.alpha)]:
        if chain not in available:
            raise SystemExit(f'{name} chain {chain!r} not in {available}')
    for chain in args.antigen:
        if chain not in available:
            raise SystemExit(f'antigen chain {chain!r} not in {available}')

    cplx = AgAbComplex.from_pdb(out, args.beta, args.alpha, args.antigen,
                                numbering='imgt')
    cplx.to_pdb(out)

    print(f'parsed OK: {out}')
    print(f'  beta  ({args.beta}): {len(cplx.get_heavy_chain())} residues')
    print(f'  alpha ({args.alpha}): {len(cplx.get_light_chain())} residues')
    print(f'  antigen: {", ".join(args.antigen)}')
    for cdr in ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']:
        seg = cplx.get_cdr(cdr)
        label = 'beta CDR' if cdr[0] == 'H' else 'alpha CDR'
        print(f'  {label}{cdr[1]}: {seg.get_seq() if seg else "NOT FOUND"}')
    print()
    print('spec for design.py:')
    print(f'  --pdb {out} --beta {args.beta} --alpha {args.alpha} '
          f'--antigen {" ".join(args.antigen)}')


if __name__ == '__main__':
    main()
