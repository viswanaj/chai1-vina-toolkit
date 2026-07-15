# chai1-vina-toolkit

A protein-agnostic toolkit for structure-based ligand docking and validation,
built around [Chai-1](https://github.com/chaidiscovery/chai-lab) co-folding
and [AutoDock Vina](https://vina.scripps.edu/) scoring, with an optional
[Boltz-2](https://github.com/jwohlwend/boltz) affinity module. Every tool
takes a plain sequence/CSV interface — there is no coupling to any specific
project's database, target set, or results.

## What's here

| Module | What it does |
|---|---|
| `predict_structures` | Chai-1 co-fold + AutoDock Vina `score_only` ΔG for a target + a CSV of ligands |
| `dock_triage` | AlphaFold fetch + fpocket pocket detection + full Vina docking + concordance triage against an upstream model score |
| `dock_compare_structures` | Dock one ligand set into an AlphaFold model AND experimental structures; compares rank agreement + native-ligand redock RMSD |
| `pose_recovery` | Validate a co-folding model against known crystal/cryo-EM poses (TM-align + symmetric RMSD + pocket-residue Jaccard) |
| `pocket_engagement` | Pose-consistency / pocket-overlap readout across a multi-pose ensemble — a ΔG-orthogonal discrimination signal |
| `make_matched_decoys` | Build a property-matched, topologically-dissimilar decoy set (DUD-E logic) from a candidate pool |
| `chemical_similarity` | Tanimoto / Dice / Cosine / MCS pairwise similarity + adjacency matrices for a CSV of SMILES |
| `affinity_boltz` | Boltz-2 learned-affinity scoring, run in parallel over a whole ligand panel |
| `prep_dock_input` | Coerce an arbitrary candidate CSV into the docking schema, with optional active/decoy controls |
| `backfill_delta_g` | Fill in missing Vina ΔG values in an existing summary.csv without re-folding |
| `plotting` | Generic ΔG/score comparison plots (across targets, or across groups within a target) |

## Why these modules exist together

The core idea running through this toolkit: **no single score should be
trusted alone.**

- A physics-based ΔG score (Vina `score_only`) rewards any well-sized
  lipophile that fits a pocket — it can't reliably separate real binders from
  property-matched decoys on its own. `make_matched_decoys` gives you a decoy
  floor that actually controls for size/lipophilicity, so `predict_structures`
  output means something.
- `pocket_engagement` gives a ΔG-orthogonal readout (does the ligand seat
  reproducibly, in the right sub-pocket?) computed for free from poses you
  already generated.
- `affinity_boltz` gives a second, *learned* affinity axis — where a physics
  score and a learned score agree, trust the call more; where they disagree,
  flag it.
- Before trusting any of this on a target with no ground truth, run
  `pose_recovery` on a target that DOES have known co-crystal structures. If
  the model can't recover known poses there, its output on an unknown target
  is not informative — that's a tool-competence check, not a target-specific
  result.

## Install

```bash
pip install -e ".[all]"
```

Or install pieces as you need them — see `pyproject.toml`'s optional-dependency
groups (`chai1`, `vina`, `boltz`, `posecheck`, `plotting`). Two CLI tools are
required by some modules but aren't pip-installable:

```bash
# obabel (openbabel) — used by predict_structures, dock_triage, dock_compare_structures
brew install openbabel      # or: apt install openbabel / conda install -c conda-forge openbabel

# fpocket — used by dock_triage for pocket detection
brew install fpocket        # or: conda install -c conda-forge fpocket
```

Chai-1, Vina docking (not just scoring), and Boltz-2 all expect a CUDA GPU;
`predict_structures --dry-run` and `affinity_boltz --dry-run` let you smoke-test
the rest of a pipeline without one.

## Quick example

```bash
python -m chai1_vina_toolkit.predict_structures \
    --uniprot P00520 \
    --target-id example_target \
    --hits-csv examples/example_hits.csv \
    --output-dir /tmp/example_run \
    --no-esm --dry-run
```

(`--uniprot` here is just an example accession to demonstrate sequence
resolution — swap in any accession, or use `--sequence`/`--fasta` for a
sequence you already have.) This writes FASTA inputs and validates the
pipeline without needing a GPU; drop `--dry-run` (and `--no-esm` if you want
ESM embeddings) to actually run Chai-1 + Vina.

## Sequence resolution

Every module that needs a target sequence accepts the same three options,
tried in priority order:

```
--sequence MSEQUENCE...      # pass it directly
--fasta path/to/target.fasta # a single-record FASTA file
--uniprot P00000              # fetched from the public UniProt REST API
```

## Common CSV schema

Most modules read/write a shared ligand-list schema:

```
target_id, molecule_id, smiles, predicted_score, rank, group
```

`target_id` is any string identifier for a run's target (a UniProt accession
is a natural choice but not required — the sequence itself is resolved
separately, see above). `group` is a free-form label (`"binder"`,
`"decoy"`, `"matched_decoy"`, ...) used to organize output directories and
downstream reports; it doesn't need to mean anything to the tools themselves.

`dock_triage` and `dock_compare_structures` need *real* UniProt accessions /
RCSB PDB IDs, since they fetch AlphaFold/experimental structures directly.

## License

MIT — see `LICENSE`.
