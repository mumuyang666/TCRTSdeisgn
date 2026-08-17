#!/usr/bin/python
# -*- coding:utf-8 -*-
"""High level inference API for TCR CDR3 design.

Typical use::

    from tcrdesign import load_model, design

    model = load_model('weights/tcrdesign_cdr3.pt', device='cuda:0')
    results = design(model, [{
        'pdb': 'examples/7t2c.pdb',
        'beta_chain': 'B',      # -> "heavy"
        'alpha_chain': 'A',     # -> "light"
        'antigen_chains': ['C'],
    }], out_dir='out')

    print(results[0]['cdr_seq'])   # designed CDR3
    print(results[0]['out_pdb'])   # generated structure
"""
import json
import os
from typing import Dict, List, Optional, Sequence, Union

import torch
from torch.utils.data import DataLoader

from .data.dataset import ComplexDataset, collate_fn
from .data.pdb_utils import VOCAB, Residue, Peptide, Protein, AgAbComplex
from .utils.logger import print_log
from .model.dymean import dyMEANModel

# Hyper-parameters of the released checkpoint. Only used as a fallback when a
# checkpoint carries no config (e.g. a bare state_dict).
DEFAULT_CONFIG = {
    'embed_size': 64,
    'hidden_size': 128,
    'n_channel': VOCAB.MAX_ATOM_NUMBER,
    'num_classes': VOCAB.get_num_amino_acid_type(),
    'mask_id': VOCAB.get_mask_idx(),
    'k_neighbors': 9,
    'bind_dist_cutoff': 6.6,
    'n_layers': 3,
    'iter_round': 3,
    'struct_only': False,
    'backbone_only': False,
    'fix_channel_weights': False,
    'pred_edge_dist': True,
    'keep_memory': True,
    'cdr_type': ['H3'],
    'paratope': 'H3',
}


# --------------------------------------------------------------------------- #
# model loading
# --------------------------------------------------------------------------- #
def load_model(ckpt: str, device: Union[str, torch.device] = 'cpu') -> dyMEANModel:
    """Load a released checkpoint.

    Accepts both formats:

    * ``{'config': ..., 'state_dict': ...}`` — the portable format produced by
      ``scripts/export_weights.py`` (recommended, version independent).
    * a pickled ``nn.Module`` — the raw training checkpoint. Requires the
      original module layout to be importable, so it is only a fallback.
    """
    device = torch.device(device)
    try:
        obj = torch.load(ckpt, map_location='cpu', weights_only=False)
    except TypeError:  # torch < 1.13 has no weights_only kwarg
        obj = torch.load(ckpt, map_location='cpu')

    if isinstance(obj, dict) and 'state_dict' in obj:
        config = dict(DEFAULT_CONFIG)
        config.update(obj.get('config', {}))
        model = dyMEANModel(**config)
        missing, unexpected = model.load_state_dict(obj['state_dict'], strict=False)
        # Buffers are rebuilt in __init__ from VOCAB, so a mismatch there is
        # harmless; a mismatch in learned parameters is not.
        params = {name for name, _ in model.named_parameters()}
        bad = [k for k in list(missing) + list(unexpected) if k in params]
        if bad:
            raise RuntimeError(f'checkpoint does not match the architecture: {bad[:8]}')
    elif isinstance(obj, torch.nn.Module):
        model = obj
        # drop training-only attributes if the checkpoint came from a
        # teacher-student run; they are never touched during generation
        for attr in ('teacher_distributions', 'teacher_dist_path', 'epitope_csv_path'):
            if hasattr(model, attr):
                delattr(model, attr)
    else:
        raise ValueError(f'unrecognised checkpoint format: {type(obj)}')

    model.to(device)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# rebuilding a complex from model output
# --------------------------------------------------------------------------- #
def to_cplx(ori_cplx: AgAbComplex, ab_x, ab_s) -> AgAbComplex:
    """Assemble generated coordinates / residue types back into a complex."""
    heavy_chain, light_chain = [], []
    chain = None
    for residue, residue_x in zip(ab_s, ab_x):
        residue = VOCAB.idx_to_symbol(residue)
        if residue == VOCAB.BOA:
            continue
        elif residue == VOCAB.BOH:
            chain = heavy_chain
            continue
        elif residue == VOCAB.BOL:
            chain = light_chain
            continue
        if chain is None:  # still in antigen region
            continue
        coord, atoms = {}, VOCAB.backbone_atoms + VOCAB.get_sidechain_info(residue)
        for atom, x in zip(atoms, residue_x):
            coord[atom] = x
        chain.append(Residue(residue, coord, _id=(len(chain), ' ')))

    heavy_chain = Peptide(ori_cplx.heavy_chain, heavy_chain)
    light_chain = Peptide(ori_cplx.light_chain, light_chain)
    for res, ori_res in zip(heavy_chain, ori_cplx.get_heavy_chain()):
        res.id = ori_res.id
    for res, ori_res in zip(light_chain, ori_cplx.get_light_chain()):
        res.id = ori_res.id

    antibody = Protein(ori_cplx.pdb_id, {
        ori_cplx.heavy_chain: heavy_chain,
        ori_cplx.light_chain: light_chain,
    })
    cplx = AgAbComplex(
        ori_cplx.antigen, antibody, ori_cplx.heavy_chain, ori_cplx.light_chain,
        skip_epitope_cal=True, skip_validity_check=True)
    cplx.cdr_pos = ori_cplx.cdr_pos
    return cplx


# --------------------------------------------------------------------------- #
# input parsing
# --------------------------------------------------------------------------- #
def load_complexes(specs: List[Dict], numbering: str = 'imgt') -> List[AgAbComplex]:
    """Parse complex specifications into ``AgAbComplex`` objects.

    Each spec is a dict with keys:
        ``pdb``            path to the (IMGT-renumbered) PDB file
        ``beta_chain``     TCR beta chain id  (alias: ``heavy_chain``)
        ``alpha_chain``    TCR alpha chain id (alias: ``light_chain``)
        ``antigen_chains`` list of peptide/MHC chain ids
        ``id``             optional name, defaults to the PDB file stem
    """
    complexes = []
    for spec in specs:
        pdb = spec['pdb']
        heavy = spec.get('beta_chain', spec.get('heavy_chain'))
        light = spec.get('alpha_chain', spec.get('light_chain'))
        antigen = spec.get('antigen_chains', [])
        if isinstance(antigen, str):
            antigen = [antigen]
        if not heavy or not light:
            raise ValueError(f'{pdb}: both beta_chain and alpha_chain are required')
        if not antigen:
            raise ValueError(f'{pdb}: antigen_chains is required')

        name = spec.get('id') or os.path.splitext(os.path.basename(pdb))[0]
        try:
            cplx = AgAbComplex.from_pdb(pdb, heavy, light, antigen, numbering=numbering)
        except Exception as e:
            print_log(f'failed to parse {pdb}: {e}', level='ERROR')
            continue
        cplx.pdb_id = name
        complexes.append(cplx)
    return complexes


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def design(model: dyMEANModel,
           specs: List[Dict],
           out_dir: Optional[str] = None,
           cdr: Optional[Sequence[str]] = None,
           batch_size: int = 8,
           num_workers: int = 0,
           save_reference: bool = True,
           numbering: str = 'imgt') -> List[Dict]:
    """Design CDR3 sequence and structure for each input complex.

    Args:
        model: loaded via :func:`load_model`.
        specs: see :func:`load_complexes`.
        out_dir: where to write generated PDBs. ``None`` skips writing.
        cdr: CDRs to redesign; defaults to the model's own ``cdr_type``.
        save_reference: also write the parsed input as ``*_reference.pdb``,
            which is what metric scripts compare against.

    Returns one record per complex with the designed CDR sequence, the native
    one, the perplexity-style confidence and the output paths.
    """
    cdr_type = cdr if cdr is not None else getattr(model, 'cdr_type', ['H3'])
    if isinstance(cdr_type, str):
        cdr_type = [cdr_type]
    paratope = getattr(model, 'paratope', 'H3')
    device = next(model.parameters()).device

    complexes = load_complexes(specs, numbering=numbering)
    if not complexes:
        return []

    dataset = ComplexDataset(complexes, cdr=cdr_type, paratope=paratope)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=collate_fn)

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

    results, idx = [], 0
    for batch in loader:
        batch = {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
        lengths = batch['lengths']
        # sample() does not consume these
        batch.pop('xloss_mask', None)
        batch.pop('pdb_id', None)

        X, S, pmets = model.sample(**batch)
        X, S, pmets = X.tolist(), S.tolist(), pmets.tolist()

        # split the flat residue axis back into individual complexes
        batch_id = torch.zeros(sum(lengths.tolist()), dtype=torch.long)
        batch_id[torch.cumsum(lengths.cpu(), dim=0)[:-1]] = 1
        batch_id = batch_id.cumsum(dim=0).tolist()

        X_list, S_list, cur = [], [], -1
        for i, bid in enumerate(batch_id):
            if bid != cur:
                cur = bid
                X_list.append([])
                S_list.append([])
            X_list[-1].append(X[i])
            S_list[-1].append(S[i])

        for i, (x, s) in enumerate(zip(X_list, S_list)):
            ori_cplx = dataset.complexes[idx]
            cplx = to_cplx(ori_cplx, x, s)
            name = cplx.get_id().split('(')[0]

            record = {
                'id': name,
                'cdr_type': list(cdr_type),
                'confidence': pmets[i],
            }
            for c in cdr_type:
                record[f'cdr_seq_{c}'] = cplx.get_cdr(c).get_seq()
                record[f'native_seq_{c}'] = ori_cplx.get_cdr(c).get_seq()
            # convenience aliases for the common single-CDR case
            if len(cdr_type) == 1:
                record['cdr_seq'] = record[f'cdr_seq_{cdr_type[0]}']
                record['native_seq'] = record[f'native_seq_{cdr_type[0]}']

            if out_dir is not None:
                out_pdb = os.path.join(out_dir, name + '.pdb')
                cplx.to_pdb(out_pdb)
                record['out_pdb'] = out_pdb
                if save_reference:
                    ref_pdb = os.path.join(out_dir, name + '_reference.pdb')
                    ori_cplx.to_pdb(ref_pdb)
                    record['reference_pdb'] = ref_pdb
                record['heavy_chain'] = cplx.heavy_chain
                record['light_chain'] = cplx.light_chain
                record['antigen_chains'] = cplx.antigen.get_chain_names()

            results.append(record)
            idx += 1

    if out_dir is not None:
        summary = os.path.join(out_dir, 'summary.json')
        with open(summary, 'w') as fout:
            for record in results:
                fout.write(json.dumps(record) + '\n')
        print_log(f'{len(results)} designs written to {out_dir}')

    return results
