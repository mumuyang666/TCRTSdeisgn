#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Design CDR3 from a pMHC structure and TCR framework sequences.

No TCR structure is needed: give the pMHC structure, an epitope definition and
the TCR alpha/beta sequences with the CDR3 to design marked by ``-``.

Single design::

    python design_seq.py --pmhc pmhc.pdb --epitope epitope.json \\
        --beta  'GVTQTPKHLI...SALYFC-------------FGPGTRLTVL' \\
        --alpha 'KQEVTQIPAA...SATYLCAVSNFNKFYFGSGTKLNVKP' \\
        --out_dir results_seq

Every ``-`` is generated, so only mask the loop you want designed and supply the
real residues elsewhere.

Batch::

    python design_seq.py --inputs tasks.json --out_dir results_seq

Build the epitope definition first with ``scripts/get_epitope.py``.
"""
import argparse
import json
import os
import sys

from tcrdesign import load_model, design_from_sequence, setup_seed
from tcrdesign.utils.logger import print_log

DEFAULT_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'weights', 'tcrdesign_cdr3.pt')


def load_specs(args):
    """Build the spec list from either --inputs or the single-task flags."""
    if args.inputs:
        with open(args.inputs, 'r') as fin:
            text = fin.read().strip()
        if text.startswith('['):
            specs = json.loads(text)
        else:  # JSON lines
            specs = [json.loads(line) for line in text.split('\n') if line.strip()]
        # resolve relative paths against the input file's directory
        root = os.path.dirname(os.path.abspath(args.inputs))
        for spec in specs:
            for key in ('pmhc_pdb', 'epitope_def'):
                path = spec.get(key)
                if path and not os.path.isabs(path):
                    spec[key] = os.path.join(root, path)
        return specs

    missing = [name for name, val in [('--pmhc', args.pmhc),
                                      ('--epitope', args.epitope),
                                      ('--beta', args.beta),
                                      ('--alpha', args.alpha)] if not val]
    if missing:
        raise SystemExit(f'either --inputs, or all of --pmhc/--epitope/--beta/--alpha '
                         f'(missing: {" ".join(missing)})')
    return [{
        'id': args.id or 'design',
        'pmhc_pdb': args.pmhc,
        'epitope_def': args.epitope,
        'beta_seq': args.beta,
        'alpha_seq': args.alpha,
        'beta_chain_id': args.beta_chain_id,
        'alpha_chain_id': args.alpha_chain_id,
        'remove_chains': args.remove_chains,
    }]


def parse():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group('input (single design)')
    g.add_argument('--pmhc', help='pMHC structure (PDB)')
    g.add_argument('--epitope', help='epitope definition JSON, see scripts/get_epitope.py')
    g.add_argument('--beta', help='TCR beta sequence, CDR3 marked with "-"')
    g.add_argument('--alpha', help='TCR alpha sequence, CDR3 marked with "-"')
    g.add_argument('--id', default=None, help='name for the output files')
    g.add_argument('--beta_chain_id', default='B', help='beta chain id in the output PDB')
    g.add_argument('--alpha_chain_id', default='A', help='alpha chain id in the output PDB')
    g.add_argument('--remove_chains', nargs='+', default=None,
                   help='chains to drop from the structure, e.g. a reference TCR')

    g = p.add_argument_group('input (batch)')
    g.add_argument('--inputs', help='JSON list / JSON-lines of design specs')

    g = p.add_argument_group('model')
    g.add_argument('--ckpt', default=DEFAULT_CKPT, help='checkpoint path')
    g.add_argument('--cdr', nargs='+', default=None,
                   help='CDRs to design, e.g. H3 (beta CDR3) or L3 (alpha CDR3). '
                        'Default: whatever the checkpoint was trained for.')
    g.add_argument('--auto_detect_cdrs', action='store_true',
                   help='take the CDR from the IMGT numbering instead of the "-" mask')

    g = p.add_argument_group('run')
    g.add_argument('--out_dir', default='./results_seq', help='output directory')
    g.add_argument('--n_samples', type=int, default=1,
                   help='number of designs per task (uses seed, seed+1, ...)')
    g.add_argument('--seed', type=int, default=2023, help='random seed')
    g.add_argument('--batch_size', type=int, default=8)
    g.add_argument('--num_workers', type=int, default=0)
    g.add_argument('--gpu', type=int, default=0, help='GPU id, -1 for CPU')
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
        records = design_from_sequence(
            model, specs, out_dir=out_dir, cdr=args.cdr,
            batch_size=args.batch_size, num_workers=args.num_workers,
            auto_detect_cdrs=args.auto_detect_cdrs)
        for r in records:
            r['seed'] = seed
        all_records.extend(records)

    if args.n_samples > 1:
        summary = os.path.join(args.out_dir, 'summary.json')
        with open(summary, 'w') as fout:
            for record in all_records:
                fout.write(json.dumps(record) + '\n')

    if not all_records:
        print_log('nothing was designed', level='WARN')
        return

    width = max(len(r['id']) for r in all_records)
    print()
    print(f'{"id":{width}}  {"designed":20}  conf')
    for r in all_records:
        print(f'{r["id"]:{width}}  {r.get("cdr_seq", ""):20}  {r["confidence"]:.4f}')
    print(f'\n{len(all_records)} designs -> {args.out_dir}')


if __name__ == '__main__':
    main()
