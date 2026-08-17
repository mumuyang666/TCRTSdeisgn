#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Score designs against their native structures.

Reads a ``summary.json`` produced by ``design.py`` (which records both the
generated and the reference PDB for every complex) and reports:

    AAR   amino acid recovery over the designed CDR
    CAAR  recovery restricted to residues contacting the antigen (< 6.6 A)
    RMSD  CDR CA-RMSD, both superposed ("aligned") and in place

    python scripts/evaluate.py --summary results/summary.json

TMscore / LDDT / DockQ are deliberately not included here: they need external
binaries. This script only depends on numpy + biopython.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tcrdesign.data.pdb_utils import AgAbComplex        # noqa: E402
from tcrdesign.data.numbering import CONTACT_DIST       # noqa: E402
from tcrdesign.utils.geometry import compute_rmsd       # noqa: E402


def score_one(record):
    mod_pdb, ref_pdb = record['out_pdb'], record['reference_pdb']
    H, L = record['heavy_chain'], record['light_chain']
    A = record['antigen_chains']
    cdr_type = record['cdr_type']
    if isinstance(cdr_type, str):
        cdr_type = [cdr_type]

    mod = AgAbComplex.from_pdb(mod_pdb, H, L, A, skip_epitope_cal=True)
    ref = AgAbComplex.from_pdb(ref_pdb, H, L, A, skip_epitope_cal=False)
    epitope = ref.get_epitope()

    out = {'id': record['id']}
    hits = chits = total = contacts = 0

    for cdr in cdr_type:
        gt_cdr, pred_cdr = ref.get_cdr(cdr), mod.get_cdr(cdr)
        gt_seq, pred_seq = gt_cdr.get_seq(), pred_cdr.get_seq()

        is_contact = []
        for residue in gt_cdr:
            is_contact.append(int(any(
                residue.dist_to(ag_res) < CONTACT_DIST for ag_res, _, _ in epitope)))

        hit = sum(1 for a, b in zip(gt_seq, pred_seq) if a == b)
        chit = sum(1 for a, b, c in zip(gt_seq, pred_seq, is_contact) if a == b and c)
        out[f'AAR {cdr}'] = hit / max(len(gt_seq), 1)
        out[f'CAAR {cdr}'] = chit / max(sum(is_contact), 1)

        gt_x = np.array([gt_cdr.get_ca_pos(i) for i in range(len(gt_cdr))])
        pred_x = np.array([pred_cdr.get_ca_pos(i) for i in range(len(pred_cdr))])
        out[f'RMSD {cdr} aligned'] = compute_rmsd(gt_x, pred_x, aligned=False)
        out[f'RMSD {cdr}'] = compute_rmsd(gt_x, pred_x, aligned=True)

        hits += hit
        chits += chit
        total += len(gt_seq)
        contacts += sum(is_contact)

    out['AAR'] = hits / max(total, 1)
    out['CAAR'] = chits / max(contacts, 1)
    return out


def parse():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--summary', required=True, help='summary.json written by design.py')
    p.add_argument('--out_csv', default=None, help='also write per-complex scores to CSV')
    return p.parse_args()


def main():
    args = parse()
    with open(args.summary) as fin:
        records = [json.loads(line) for line in fin.read().strip().split('\n') if line.strip()]

    scores = []
    for record in records:
        if 'reference_pdb' not in record or 'out_pdb' not in record:
            print(f'[skip] {record.get("id")}: needs both out_pdb and reference_pdb')
            continue
        try:
            scores.append(score_one(record))
        except Exception as e:
            print(f'[fail] {record.get("id")}: {e}')

    if not scores:
        raise SystemExit('nothing scored')

    keys = [k for k in scores[0] if k != 'id']
    width = max(len('id'), max(len(s['id']) for s in scores))
    print(f'\n{"id":<{width}}  ' + '  '.join(f'{k:>16}' for k in keys))
    for s in scores:
        print(f'{s["id"]:<{width}}  ' + '  '.join(f'{s[k]:>16.4f}' for k in keys))
    print(f'\n{"mean":<{width}}  ' +
          '  '.join(f'{np.mean([s[k] for s in scores]):>16.4f}' for k in keys))
    print(f'({len(scores)} complexes)')

    if args.out_csv:
        import csv
        with open(args.out_csv, 'w', newline='') as fout:
            writer = csv.DictWriter(fout, fieldnames=['id'] + keys)
            writer.writeheader()
            writer.writerows(scores)
        print(f'wrote {args.out_csv}')


if __name__ == '__main__':
    main()
