# TCRDesign

Sequence-structure co-design of TCR CDR3 loops against a given peptide-MHC target.

Given a TCR-pMHC complex structure, the model masks out the CDR3 loop and
regenerates both its **amino acid sequence** and its **full-atom 3D structure**
in one pass, iteratively refining the loop and its docking pose against the
epitope.

> **Coming soon.** A script that builds the complex directly from a pMHC
> sequence and a TCR framework sequence — no input structure required — is in
> preparation and will be released shortly.

---

## Citation

> TCRTSdesign: End-to-End Co-Design of Antigen-Specific TCR Sequences and
> Structures. In: *Proceedings of the 32nd ACM SIGKDD Conference on Knowledge
> Discovery and Data Mining V.2* (KDD 2026). Jeju Island, Republic of Korea;
> 2026:12468-12479.
> [[paper]](https://dl.acm.org/doi/pdf/10.1145/3770855.3818991)

---

## Install

```bash
git clone <this-repo> tcrdesign && cd tcrdesign

conda create -n tcrdesign python=3.9 -y
conda activate tcrdesign

# torch first, matching your CUDA version (see https://pytorch.org)
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121
# torch-scatter must match the torch build exactly
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

pip install -r requirements.txt
```

Verify:

```bash
python design.py --inputs examples/inputs.json --out_dir results
```

Expected output on GPU (seed 2023):

```
id    designed             native               conf
7n5c  ASSLGGEQY            ASSFGREQY            0.4776
7n2r  ASSLRRRSTDTQY        ASSVATYSTDTQY        0.6157
7nme  ASSAEYEQY            ASSLHHEQY            0.3652
```

CPU works (`--gpu -1`), roughly 15 s per complex versus under 1 s on a GPU.
Confidence scores shift slightly on CPU (float accumulation order differs
between backends) and can also shift if you change `--batch_size`, since the
neighbour graph is built per batch. Results are reproducible for a fixed seed,
input, device and batch size.

---

## Input requirements

The model finds CDR3 through **IMGT residue numbering**, so input PDBs must be
IMGT-renumbered. Files downloaded straight from the RCSB usually are not.

```bash
# needs ANARCI:  conda install -c bioconda anarci
python scripts/prepare_pdb.py --pdb 7n2r.pdb \
    --beta F --alpha D --antigen C A B --out 7n2r_imgt.pdb
```

This renumbers the file, prints all six detected CDRs so you can sanity check
the chain assignment, and echoes the exact flags to pass to `design.py`. If you
renumber with another tool, validate the result with `--check_only`.

A complex needs three things: the TCR **beta** chain, the TCR **alpha** chain,
and at least one **antigen** chain (the peptide, plus MHC chains if present).
The antigen is reduced internally to the 48 residues nearest the TCR, so
including the MHC is cheap and generally helps.

---

## Usage

### Single complex

```bash
python design.py --pdb examples/7n2r.pdb \
    --beta F --alpha D --antigen C A B \
    --out_dir results
```

### Batch

```bash
python design.py --inputs examples/inputs.json --out_dir results
```

`--inputs` accepts a JSON list or JSON-lines file:

```json
[
  {
    "id": "7n2r",
    "pdb": "examples/7n2r.pdb",
    "beta_chain": "F",
    "alpha_chain": "D",
    "antigen_chains": ["C", "A", "B"]
  }
]
```

Relative `pdb` paths resolve against the input file's own directory.

### Python API

```python
from tcrdesign import load_model, design, setup_seed

setup_seed(2023)
model = load_model('weights/tcrdesign_cdr3.pt', device='cuda:0')

results = design(model, [{
    'pdb': 'examples/7n2r.pdb',
    'beta_chain': 'F',
    'alpha_chain': 'D',
    'antigen_chains': ['C', 'A', 'B'],
}], out_dir='results')

print(results[0]['cdr_seq'])     # designed CDR3
print(results[0]['native_seq'])  # native CDR3, for reference
print(results[0]['out_pdb'])     # generated full-atom structure
```

### Designing the alpha chain CDR3

The released checkpoint was trained on the **beta** chain CDR3 (`H3` in the
internal antibody-style naming). You can point it at other loops with `--cdr`,
but note this is off-distribution for these weights:

```bash
python design.py --inputs examples/inputs.json --cdr L3 --out_dir results_alpha
```

---

## Outputs

`--out_dir` receives, per complex:

| file | contents |
|---|---|
| `<id>.pdb` | generated complex, full-atom, designed CDR3 |
| `<id>_reference.pdb` | parsed input, for like-for-like comparison |
| `summary.json` | one JSON record per design |

Each record carries the designed sequence, the native sequence, a confidence
score and the output paths:

```json
{"id": "7n2r", "cdr_type": ["H3"], "confidence": 0.6157,
 "cdr_seq_H3": "ASSLRRRSTDTQY", "native_seq_H3": "ASSVATYSTDTQY",
 "cdr_seq": "ASSLRRRSTDTQY", "native_seq": "ASSVATYSTDTQY",
 "out_pdb": "results/7n2r.pdb", "reference_pdb": "results/7n2r_reference.pdb",
 "heavy_chain": "F", "light_chain": "D", "antigen_chains": ["C", "A", "B"]}
```

`confidence` is the mean per-residue maximum softmax probability over the
designed loop — useful for ranking designs, not calibrated as a probability.

Compare `<id>.pdb` against `<id>_reference.pdb` rather than against the
original input file: chain order and atom naming are normalised during parsing,
so the reference is the correct baseline.

### Scoring designs

When the native CDR3 is known (retrospective benchmarking):

```bash
python scripts/evaluate.py --summary results/summary.json
```

Reports amino acid recovery (AAR), recovery restricted to antigen-contacting
positions (CAAR), and CDR CA-RMSD both superposed and in place. On the
18-complex held-out test set the released weights give AAR 0.512, CAAR 0.308,
superposed CDR3 RMSD 1.62 A.

The three bundled examples are among the better cases in that test set (mean
AAR 0.712), so treat them as a smoke test rather than a benchmark.

Only numpy and biopython are required. TMscore, LDDT and DockQ need external
binaries and are out of scope here.

---

## Layout

```
design.py                 CLI entry point
weights/
  tcrdesign_cdr3.pt       released weights (beta CDR3, 7.7 MB)
tcrdesign/
  infer.py                load_model / design
  model/
    network.py            the network (inference only)
    am_egnn.py            equivariant graph layers
    am_enc.py             encoder over the complex + interface
  data/
    pdb_utils.py          PDB parsing, complex/CDR handling, vocabulary
    dataset.py            complex -> model tensors
    numbering.py          IMGT / Chothia CDR definitions
    framework_templates.py  conserved-position framework initialisation
    renumber.py           ANARCI wrapper
  utils/
    nn_utils.py           residue/atom features, graph construction
    geometry.py           Kabsch superposition, RMSD
scripts/
  prepare_pdb.py          IMGT renumbering + validation
  evaluate.py             AAR / CAAR / RMSD
  export_weights.py       convert a training checkpoint to the release format
examples/                 three ready-to-run test complexes
```

### Chain naming

The model is built on an antibody design framework, so internally the TCR is
mapped onto antibody conventions. This surfaces in some field names:

| TCR | internal name | CDR3 label |
|---|---|---|
| beta chain | heavy chain (`H`) | `H3` |
| alpha chain | light chain (`L`) | `L3` |
| peptide + MHC | antigen | — |

The CLI uses `--beta` / `--alpha`; `summary.json` reports `heavy_chain` /
`light_chain`. They refer to the same chains.

---

## Notes and limitations

- **Weights.** `weights/tcrdesign_cdr3.pt` stores a config plus a plain
  `state_dict`, so it loads without needing the training repository and is not
  tied to a torch version. `scripts/export_weights.py` regenerates it from a
  raw training checkpoint. `load_model` also accepts a pickled `nn.Module`
  checkpoint as a fallback.
- **Held-out performance.** AAR around 0.51 means roughly half the CDR3
  positions are recovered exactly. Treat outputs as hypotheses for
  experimental or computational screening, not as confident predictions.
- **Binding is not predicted.** The model designs a loop compatible with the
  given interface geometry. It returns no affinity estimate, and a design is
  not evidence that the TCR will bind.
- **Determinism is per-device.** Fixed seed, fixed input and fixed device
  reproduce results exactly. CPU versus GPU, different GPU architectures, or
  different torch builds can shift confidences and occasionally flip a
  low-confidence residue.
