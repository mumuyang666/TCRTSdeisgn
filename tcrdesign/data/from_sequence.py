#!/usr/bin/python
# -*- coding:utf-8 -*-
"""Build model inputs from a pMHC structure plus TCR framework *sequences*.

The structure-input path (:mod:`tcrdesign.data.dataset`) needs a docked
TCR-pMHC complex. This module removes that requirement: only the pMHC
structure, an epitope definition and the TCR alpha/beta sequences are needed.
The TCR coordinates are irrelevant here because the model overwrites the whole
TCR with a conserved-position framework template before message passing
(``init_mask`` does ``X[cmask] = template``), so placeholder coordinates are
enough.

CDR3 is marked in the input sequence with ``-``, one dash per residue to
design::

    beta:  GVTQTPKHLI...SALYFC-------------FGPGTRLTVL
                              ^^^^^^^^^^^^^ CDR3 to design (13 residues)

The number of dashes sets the loop length, and every dashed position is
generated. Alternatively pass ``auto_detect_cdrs=True`` and a complete sequence,
and the CDR to design is taken from the IMGT numbering instead.

Chain convention, as elsewhere in this package: TCR beta -> heavy (``H``),
TCR alpha -> light (``L``).
"""
import shutil
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .pdb_utils import AgAbComplex, Peptide, Protein, Residue, VOCAB
from .dataset import _generate_chain_data
from .epitope import load_epitope_def, select_epitope_residues
from .framework_templates import ConserveTemplateGenerator
from .numbering import IMGT
from ..utils.logger import print_log

# ANARCI chain types for TCRs: B = beta, A = alpha. Antibody types (H/K/L) are
# accepted too so an antibody Fv still works.
_BETA_TYPES = ('B', 'H')
_ALPHA_TYPES = ('A', 'K', 'L')

MASK_CHAR = '-'


def _renumber(seq: str, scheme: str = 'imgt'):
    """IMGT-number a variable-domain sequence with ANARCI."""
    try:
        from anarci import run_anarci
    except ImportError as e:
        raise ImportError(
            'designing from sequence needs ANARCI to assign IMGT numbering:\n'
            '    conda install -c bioconda anarci\n'
            '(ANARCI also needs the hmmer binaries on PATH)') from e

    # ANARCI shells out to hmmscan; a missing binary otherwise surfaces as a
    # bare FileNotFoundError from deep inside subprocess.
    if shutil.which('hmmscan') is None:
        raise RuntimeError(
            'ANARCI is installed but its "hmmscan" binary is not on PATH.\n'
            '    conda install -c bioconda hmmer\n'
            'If hmmer is already in your environment, activate it (or add its '
            'bin directory to PATH) so hmmscan is visible.')

    _, numbering, details, _ = run_anarci([('A', seq)], scheme=scheme)
    if not numbering[0]:
        raise ValueError(
            f'ANARCI could not number this sequence as a variable domain:\n  {seq}')
    chain_type = details[0][0]['chain_type']
    fv, position = [], []
    for pos, res in numbering[0][0][0]:
        if res == MASK_CHAR:
            continue
        fv.append(res)
        position.append(pos)
    return ''.join(fv), position, chain_type


def _placeholder_residues(seq: str, positions: List[Tuple]) -> List[Residue]:
    """Residues carrying IMGT ids but no meaningful coordinates.

    The template generator only reads residue ids, and the model replaces every
    TCR coordinate with the template, so zeros are never used as real geometry.
    """
    fake_coord = {atom: [0.0, 0.0, 0.0] for atom in VOCAB.backbone_atoms}
    return [Residue(symbol, dict(fake_coord), pos)
            for symbol, pos in zip(seq, positions)]


def encode_chain_sequence(chain_seq: str,
                          cdr: Sequence[str],
                          auto_detect_cdrs: bool = False,
                          scheme: str = 'imgt'):
    """Number one TCR chain and derive its design mask.

    Returns ``(residues, smask, chain_type)``. Masked (``-``) positions are
    filled with a placeholder residue; the model overwrites them anyway since
    ``smask`` sets them to ``mask_id``.
    """
    smask_full = [1 if s == MASK_CHAR else 0 for s in chain_seq]
    # ANARCI needs a real sequence: put a placeholder residue at masked spots.
    # Which residue does not matter, the model masks them before use.
    probe = ''.join('G' if s == MASK_CHAR else s for s in chain_seq)

    fv, positions, chain_type = _renumber(probe, scheme)
    start = probe.find(fv)
    if start == -1:
        raise ValueError('the numbered variable domain does not match the input sequence')
    smask = smask_full[start:start + len(fv)]
    seq = fv

    prefix = 'H' if chain_type in _BETA_TYPES else 'L'
    wanted = [name for name in cdr if name.upper().startswith(prefix)]

    if auto_detect_cdrs:
        # take the loop from the numbering and ignore any dashes
        smask = [0] * len(seq)
        for name in wanted:
            lo, hi = getattr(IMGT, name.upper())
            for i, (pos_num, _icode) in enumerate(positions):
                if lo <= pos_num <= hi:
                    smask[i] = 1
    else:
        # Every dash is generated: a dash means no residue was supplied, so it
        # cannot be held fixed. Dashes outside the CDRs named in `cdr` are still
        # generated, but they are off-distribution for a CDR3-trained
        # checkpoint, so say so.
        allowed = set()
        for name in wanted:
            lo, hi = getattr(IMGT, name.upper())
            allowed.update(range(lo, hi + 1))
        outside = [pos for i, pos in enumerate(positions)
                   if smask[i] and pos[0] not in allowed]
        if outside:
            shown = ', '.join(f'{p}{c.strip()}' for p, c in outside[:6])
            print_log(
                f'chain type {chain_type}: {len(outside)} masked position(s) lie '
                f'outside {wanted or "the requested CDRs"} and will be generated '
                f'anyway ({shown}{", ..." if len(outside) > 6 else ""}). Supply '
                f'those residues instead of "-" to keep them fixed.',
                level='WARN')

    # A chain with no mask is fine when nothing on it was requested: designing
    # only H3 with a complete alpha sequence is the normal case.
    if wanted and not any(smask):
        raise ValueError(
            f'nothing to design for {wanted} in this chain (ANARCI chain type '
            f'{chain_type}). Mark the loop with "-", one per residue, or pass '
            'auto_detect_cdrs=True.')

    return _placeholder_residues(seq, positions), smask, chain_type


def encode_from_sequence(pmhc_pdb: str,
                         epitope: List[Tuple[str, Tuple]],
                         beta_seq: str,
                         alpha_seq: str,
                         cdr: Sequence[str] = ('H3',),
                         paratope: Sequence[str] = ('H3',),
                         beta_chain_id: str = 'B',
                         alpha_chain_id: str = 'A',
                         pdb_id: str = 'design',
                         remove_chains: Optional[Sequence[str]] = None,
                         auto_detect_cdrs: bool = False) -> Dict:
    """Encode one design task into a model input item.

    Args:
        pmhc_pdb: pMHC structure. Any TCR present should be listed in
            ``remove_chains`` so it does not leak into the antigen.
        epitope: epitope definition, see :mod:`tcrdesign.data.epitope`.
        beta_seq / alpha_seq: TCR chain sequences with CDR3 marked by ``-``.
        cdr: CDRs to design, e.g. ``('H3',)``.
        paratope: CDRs used as the docking anchor.
        beta_chain_id / alpha_chain_id: chain ids for the output PDB.
        remove_chains: chains to drop from the structure before use.
    """
    antigen = Protein.from_pdb(pmhc_pdb)
    if remove_chains:
        for chain_name in remove_chains:
            if chain_name in antigen.peptides:
                del antigen.peptides[chain_name]
    if not antigen.peptides:
        raise ValueError('no chain left in the structure after remove_chains')

    epitope_residues = select_epitope_residues(antigen, epitope)
    ag_data = _generate_chain_data(epitope_residues, VOCAB.BOA)

    hc_residues, hc_smask, hc_type = encode_chain_sequence(
        beta_seq, cdr, auto_detect_cdrs)
    lc_residues, lc_smask, lc_type = encode_chain_sequence(
        alpha_seq, cdr, auto_detect_cdrs)

    if hc_type in _ALPHA_TYPES and lc_type in _BETA_TYPES:
        raise ValueError(
            f'beta_seq was numbered as chain type {hc_type} and alpha_seq as '
            f'{lc_type}; the two sequences look swapped')
    for tag, ctype, expect in [('beta_seq', hc_type, _BETA_TYPES),
                               ('alpha_seq', lc_type, _ALPHA_TYPES)]:
        if ctype not in expect:
            print_log(f'{tag}: unexpected ANARCI chain type {ctype}', level='WARN')

    hc_data = _generate_chain_data(hc_residues, VOCAB.BOH)
    lc_data = _generate_chain_data(lc_residues, VOCAB.BOL)

    data = {key: np.concatenate([ag_data[key], hc_data[key], lc_data[key]], axis=0)
            for key in hc_data}
    data['pdb_id'] = pdb_id

    n_ag, n_hc, n_lc = len(ag_data['S']), len(hc_data['S']), len(lc_data['S'])
    # global nodes and the antigen are never generated
    data['cmask'] = [0] * n_ag + [0] + [1] * (n_hc - 1) + [0] + [1] * (n_lc - 1)
    data['smask'] = [0] * n_ag + [0] + hc_smask + [0] + lc_smask

    antibody = Protein(pdb_id, {
        beta_chain_id: Peptide(beta_chain_id, hc_residues),
        alpha_chain_id: Peptide(alpha_chain_id, lc_residues),
    })
    # skip_epitope_cal: the epitope is given, and the TCR has no real
    # coordinates to compute contacts from.
    cplx = AgAbComplex(antigen=antigen, antibody=antibody,
                       heavy_chain=beta_chain_id, light_chain=alpha_chain_id,
                       numbering='imgt', skip_epitope_cal=True)
    cplx.pdb_id = pdb_id

    paratope = [paratope] if isinstance(paratope, str) else list(paratope)
    paratope_mask = [0] * (n_ag + n_hc + n_lc)
    for name in paratope:
        cdr_range = cplx.get_cdr_pos(name)
        if cdr_range is None:
            raise ValueError(f'CDR {name} not found after IMGT numbering')
        offset = n_ag + 1 + (0 if name[0].upper() == 'H' else n_hc)
        for idx in range(offset + cdr_range[0], offset + cdr_range[1] + 1):
            paratope_mask[idx] = 1
    data['paratope_mask'] = paratope_mask

    data['template'] = ConserveTemplateGenerator().construct_template(cplx, align=False)
    data['cplx'] = cplx
    return data


def collate_fn(batch: List[Dict]) -> Dict:
    """Batch items along the residue axis, carrying the complexes alongside."""
    keys = ['X', 'S', 'smask', 'cmask', 'paratope_mask', 'residue_pos', 'template']
    types = [torch.float, torch.long, torch.bool, torch.bool, torch.bool,
             torch.long, torch.float]
    res = {}
    for key, _type in zip(keys, types):
        res[key] = torch.cat([torch.tensor(item[key], dtype=_type) for item in batch], dim=0)
    res['lengths'] = torch.tensor([len(item['S']) for item in batch], dtype=torch.long)
    res['pdb_id'] = [item['pdb_id'] for item in batch]
    res['cplxes'] = [item['cplx'] for item in batch]
    return res


class SequenceDesignDataset(torch.utils.data.Dataset):
    """Dataset over design tasks specified by sequence, not structure.

    Each spec is a dict with keys:
        ``pmhc_pdb``       path to the pMHC structure
        ``epitope``        epitope definition, or ``epitope_def`` for a JSON path
        ``beta_seq``       TCR beta sequence, CDR3 marked with ``-``
        ``alpha_seq``      TCR alpha sequence
        ``id``             optional name
        ``remove_chains``  optional chains to drop (e.g. a reference TCR)
        ``beta_chain_id`` / ``alpha_chain_id``  optional output chain ids
    """

    def __init__(self, specs: List[Dict],
                 cdr: Sequence[str] = ('H3',),
                 paratope: Sequence[str] = ('H3',),
                 auto_detect_cdrs: bool = False):
        super().__init__()
        self.specs = specs
        self.cdr = [cdr] if isinstance(cdr, str) else list(cdr)
        self.paratope = [paratope] if isinstance(paratope, str) else list(paratope)
        self.auto_detect_cdrs = auto_detect_cdrs

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        spec = self.specs[idx]
        epitope = spec.get('epitope')
        if epitope is None:
            epitope_def = spec.get('epitope_def')
            if epitope_def is None:
                raise ValueError(f'{spec.get("id")}: either epitope or epitope_def is required')
            epitope = load_epitope_def(epitope_def)
        else:
            epitope = [(c, tuple(p)) for c, p in epitope]

        return encode_from_sequence(
            pmhc_pdb=spec['pmhc_pdb'],
            epitope=epitope,
            beta_seq=spec['beta_seq'],
            alpha_seq=spec['alpha_seq'],
            cdr=self.cdr,
            paratope=self.paratope,
            beta_chain_id=spec.get('beta_chain_id', 'B'),
            alpha_chain_id=spec.get('alpha_chain_id', 'A'),
            pdb_id=spec.get('id', f'design_{idx}'),
            remove_chains=spec.get('remove_chains'),
            auto_detect_cdrs=self.auto_detect_cdrs,
        )

    collate_fn = staticmethod(collate_fn)
