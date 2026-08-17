#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Command line entry point for TCR CDR3 design.

Single complex::

    python design.py --pdb examples/7n2r.pdb \\
        --beta F --alpha D --antigen C A B \\
        --out_dir out

Batch (JSON list or JSON-lines, see examples/inputs.json)::

    python design.py --inputs examples/inputs.json --out_dir out

Design several candidates for the same input by repeating it with
different seeds::

    python design.py --inputs examples/inputs.json --out_dir out --n_samples 10
"""
import argparse
import json
import os
import sys

# allow running from a checkout without installing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tcrdesign import load_model, design, setup_seed          # noqa: E402
from tcrdesign.utils.logger import print_log                  # noqa: E402

DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'weights', 'tcrdesign_cdr3.pt')


def load_specs(args):
    if args.inputs:
        with open(args.inputs) as fin:
            text = fin.read().strip()
        try:
            specs = json.loads(text)
            if isinstance(specs, dict):
                specs = [specs]
        except json.JSONDecodeError:  # JSON-lines
            specs = [json.loads(line) for line in text.split('\n') if line.strip()]
        # resolve relative pdb paths against the input file's directory
        base = os.path.dirname(os.path.abspath(args.inputs))
        for spec in specs:
            if not os.path.isabs(spec['pdb']):
                cand = os.path.join(base, spec['pdb'])
                if os.path.exists(cand):
                    spec['pdb'] = cand
        return specs

    if not (args.pdb and args.beta and args.alpha and args.antigen):
        raise SystemExit('either --inputs, or all of --pdb/--beta/--alpha/--antigen')
    return [{
        'pdb': args.pdb,
        'beta_chain': args.beta,
        'alpha_chain': args.alpha,
        'antigen_chains': args.antigen,
        'id': args.id,
    }]


def parse():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group('input (single complex)')
    g.add_argument('--pdb', help='path to an IMGT-renumbered TCR-pMHC PDB file')
    g.add_argument('--beta', help='TCR beta chain id')
    g.add_argument('--alpha', help='TCR alpha chain id')
    g.add_argument('--antigen', nargs='+', help='peptide / MHC chain ids')
    g.add_argument('--id', default=None, help='name for the output files')

    g = p.add_argument_group('input (batch)')
    g.add_argument('--inputs', help='JSON list / JSON-lines of complex specs')

    g = p.add_argument_group('model')
    g.add_argument('--ckpt', default=DEFAULT_CKPT, help='checkpoint path')
    g.add_argument('--cdr', nargs='+', default=None,
                   help="CDRs to redesign, e.g. H3 (beta CDR3) or L3 (alpha CDR3). "
                        "Default: whatever the checkpoint was trained for.")

    g = p.add_argument_group('run')
    g.add_argument('--out_dir', default='./results', help='output directory')
    g.add_argument('--n_samples', type=int, default=1,
                   help='number of designs per complex (uses seed, seed+1, ...)')
    g.add_argument('--seed', type=int, default=2023, help='random seed')
    g.add_argument('--batch_size', type=int, default=8)
    g.add_argument('--num_workers', type=int, default=0)
    g.add_argument('--gpu', type=int, default=0, help='GPU id, -1 for CPU')
    g.add_argument('--no_reference', action='store_true',
                   help='do not write the parsed input as *_reference.pdb')
    return p.parse_args()


def main():
    args = parse()
    specs = load_specs(args)
    device = 'cpu' if args.gpu < 0 else f'cuda:{args.gpu}'

    model = load_model(args.ckpt, device=device)
    print_log(f'loaded {args.ckpt} on {device}, cdr_type={model.cdr_type}')

    all_records = []
    for k in range(args.n_samples):
        seed = args.seed + k
        setup_seed(seed)
        out_dir = args.out_dir if args.n_samples == 1 \
            else os.path.join(args.out_dir, f'sample_{k}')
        records = design(model, specs, out_dir=out_dir, cdr=args.cdr,
                         batch_size=args.batch_size, num_workers=args.num_workers,
                         save_reference=not args.no_reference)
        for r in records:
            r['seed'] = seed
        all_records.extend(records)

    if args.n_samples > 1:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as fout:
            for r in all_records:
                fout.write(json.dumps(r) + '\n')

    print()
    width = max([len('id')] + [len(r['id']) for r in all_records])
    print(f'{"id":<{width}}  {"designed":<20} {"native":<20} conf')
    for r in all_records:
        design_seq = r.get('cdr_seq', '-')
        native_seq = r.get('native_seq', '-')
        print(f'{r["id"]:<{width}}  {design_seq:<20} {native_seq:<20} {r["confidence"]:.4f}')
    print(f'\n{len(all_records)} designs -> {args.out_dir}')


if __name__ == '__main__':
    main()
