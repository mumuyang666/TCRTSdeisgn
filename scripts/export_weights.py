#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""Convert a training checkpoint into the portable release format.

The training checkpoints are pickled ``nn.Module`` objects, so loading them
requires the original research repository on ``sys.path`` and is sensitive to
the torch version. This script extracts the architecture config plus the
tensors into a plain dict that :func:`tcrdesign.load_model` can read anywhere.

Run it from the research repository (so that ``models``/``utils``/``data`` are
importable)::

    python export_weights.py \\
        --ckpt .../version_25/checkpoint/epoch171_step3784.ckpt \\
        --out  .../upload/weights/tcrdesign_cdr3.pt
"""
import argparse
import os
import sys

import torch

# Attributes that describe the architecture, mapped onto ctor keyword names.
ATTR_TO_ARG = {
    'mask_id': 'mask_id',
    'num_classes': 'num_classes',
    'k_neighbors': 'k_neighbors',
    'bind_dist_cutoff': 'bind_dist_cutoff',
    'round': 'iter_round',
    'struct_only': 'struct_only',
    'backbone_only': 'backbone_only',
    'fix_channel_weights': 'fix_channel_weights',
    'pred_edge_dist': 'pred_edge_dist',
    'keep_memory': 'keep_memory',
    'cdr_type': 'cdr_type',
    'paratope': 'paratope',
}

# Training-only state that must not leak into the release.
DROP_PREFIXES = ('teacher',)


def infer_sizes(state_dict):
    """Recover embed_size / hidden_size / n_channel / n_layers from tensors."""
    sizes = {}
    emb = state_dict.get('aa_feature.aa_embedding.residue_embedding.weight')
    if emb is not None:
        sizes['embed_size'] = emb.shape[1]
    ffn = state_dict.get('ffn_residue.1.weight')
    if ffn is not None:
        sizes['hidden_size'] = ffn.shape[0]
    atom_type = state_dict.get('aa_feature.residue_atom_type')
    if atom_type is not None:
        sizes['n_channel'] = atom_type.shape[1]
    n_layers = len({k.split('.')[1] for k in state_dict
                    if k.startswith('gnn.ctx_gcl_')})
    if n_layers:
        sizes['n_layers'] = n_layers
    return sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', required=True, help='training checkpoint (pickled nn.Module)')
    parser.add_argument('--out', required=True, help='output .pt path')
    parser.add_argument('--repo', default=None,
                        help='path to the research repository (defaults to cwd)')
    parser.add_argument('--note', default='', help='free-form provenance note')
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.repo or os.getcwd()))

    try:
        model = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    except TypeError:
        model = torch.load(args.ckpt, map_location='cpu')

    if not isinstance(model, torch.nn.Module):
        raise SystemExit(f'expected a pickled nn.Module, got {type(model)}')

    config = {}
    for attr, arg in ATTR_TO_ARG.items():
        if hasattr(model, attr):
            config[arg] = getattr(model, attr)

    state_dict = {k: v for k, v in model.state_dict().items()
                  if not k.startswith(DROP_PREFIXES)}
    config.update(infer_sizes(state_dict))

    # backbone_only checkpoints store 4 channels; keep the two consistent
    if config.get('backbone_only'):
        config['n_channel'] = 4

    payload = {
        'config': config,
        'state_dict': state_dict,
        'source_ckpt': os.path.abspath(args.ckpt),
        'note': args.note,
        'format_version': 1,
    }

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(payload, args.out)

    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f'wrote {args.out} ({size_mb:.1f} MB)')
    print('config:')
    for k in sorted(config):
        print(f'  {k} = {config[k]!r}')
    print(f'{len(state_dict)} tensors')


if __name__ == '__main__':
    main()
