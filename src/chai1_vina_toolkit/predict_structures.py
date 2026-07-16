#!/usr/bin/env python3
"""
predict_structures.py — Chai-1 co-fold structure prediction + AutoDock Vina
ΔG scoring for protein-ligand complexes, with optional ligand-free (apo)
folding.

For a given target (any protein sequence), loads a ligand list from a CSV,
runs Chai-1 co-folding on each (target, ligand) pair -- or on the target
alone, for rows with a blank SMILES -- then scores the predicted pose with
AutoDock Vina in score_only mode (skipped automatically for apo rows, since
there's no ligand to dock).

Input CSV schema
-----------------
  target_id, molecule_id, smiles, group[, predicted_score, rank]

  `target_id` is any string identifier for the row's target — a UniProt
  accession is a natural choice but not required. Rows are filtered to
  --target-id. `group` is a free-form label (e.g. "binder"/"decoy"/"apo")
  used only for organizing output directories and downstream reports.
  Leave `smiles` blank for an apo (ligand-free) fold of the target alone.

Outputs
-------
  <output-dir>/<group>/<molecule_id>/pred.model_idx_{0..4}.cif   -- all 5
    diffusion samples chai-lab generates per complex (not just the top one)
  <output-dir>/<group>/<molecule_id>/scores.model_idx_{0..4}.npz -- chai-lab's
    own per-sample scores (aggregate_score, ptm, iptm, per_chain_ptm,
    has_inter_chain_clashes, chain_chain_clashes)
  <output-dir>/summary.csv -- one row per complex, written incrementally,
    with the full metrics panel described below

Metrics panel
-------------
Beyond the top-ranked sample's aggregate_score/ptm/iptm (chai-lab's own
ranking), this also surfaces fields chai-lab computes but the original
version of this script discarded, plus two derived cross-sample metrics:

  per_chain_ptm            -- pTM for each chain, ';'-joined (top sample)
  chain_chain_clashes      -- flattened inter/intra-chain clash-count matrix
  has_inter_chain_clashes  -- True if ANY of the 5 samples has a clash
  ptm_mean/std_across_samples, iptm_mean/std_across_samples
                            -- spread across all 5 diffusion samples; large
                               spread = low reproducibility, independent of
                               the top sample's absolute score
  ca_rmsd_mean_across_samples -- mean pairwise CA RMSD (Kabsch-superposed)
                               across the 5 samples' receptor chain -- a
                               structural (not just score-based) ensemble
                               reproducibility signal
  pocket_mean_plddt        -- mean per-atom pLDDT (top sample) restricted to
                               --pocket-positions, if given

For apo rows: iptm and aggregate_score are set blank, not zero. Chai-lab's
ipTM/aggregate_score formulas are defined over chain-chain interfaces; with
only one chain, ipTM degenerates to a meaningless 0 (there's no second chain
for the "interface" term to be computed against), and aggregate_score
inherits that degeneracy (0.2*ptm + 0.8*0 = 0.2*ptm). Reporting that as if
it were a real score would misrepresent an apo fold as a bad complex. Use
`ptm` (and pocket_mean_plddt) for apo fold-confidence instead.

Dependencies (beyond the toolkit's own requirements.txt)
---------------------------------------------------------
  pip install chai-lab vina meeko gemmi

Compute
-------
  Requires a CUDA GPU for Chai-1 (A10/A100 class). Not runnable on CPU/MPS.
  ~10-15 min per complex on an A10. Use --no-esm for a quick smoke test.

Usage
-----
  python -m chai1_vina_toolkit.predict_structures \\
      --uniprot O00144 --hits-csv my_ligands.csv --output-dir out/O00144
  python -m chai1_vina_toolkit.predict_structures \\
      --sequence MSEQUENCE... --target-id my_target --hits-csv my_ligands.csv \\
      --output-dir out/my_target --no-esm --dry-run
  # apo fold: leave smiles blank for that row in --hits-csv
  python -m chai1_vina_toolkit.predict_structures \\
      --target-id my_target --fasta my_target.fasta --hits-csv apo_row.csv \\
      --output-dir out/my_target_apo --pocket-positions 151,155,339

NOTE ON TESTING: the apo-mode + full-metrics code below was written directly
against chai-lab's `chai_lab/ranking/{rank,ptm,plddt,clashes}.py` source (field
names/shapes confirmed by reading it), and the CIF-parsing/RMSD helpers were
unit-tested standalone with synthetic coordinates. The GPU-dependent path as
a whole (actually calling run_inference and reading its real output) has NOT
been exercised end-to-end -- this machine has no CUDA GPU. Sanity-check the
first real run's summary.csv (column values not just presence) before trusting
a full batch.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import tempfile
import warnings
from pathlib import Path

import gemmi
import numpy as np
import torch
from rdkit import Chem

from .sequences import add_sequence_args, resolve_sequence

SUMMARY_FIELDS = [
    "target_id", "molecule_id", "smiles", "predicted_score", "rank", "group",
    "is_apo", "aggregate_score", "ptm", "iptm", "delta_g_kcal_mol",
    "per_chain_ptm", "has_inter_chain_clashes", "chain_chain_clashes",
    "ptm_mean_across_samples", "ptm_std_across_samples",
    "iptm_mean_across_samples", "iptm_std_across_samples",
    "ca_rmsd_mean_across_samples", "pocket_mean_plddt", "n_samples",
]


def desalt_smiles(smiles: str) -> str:
    """Strip counterions/solvate from a salt SMILES, keeping the largest fragment."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) <= 1:
        return smiles
    largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(largest)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target-id", default=None,
                   help="Value to filter --hits-csv's target_id column by. "
                        "Defaults to --uniprot if not given.")
    add_sequence_args(p, required=False)
    p.add_argument("--hits-csv", required=True, help="Ligand list CSV (see module docstring for schema).")
    p.add_argument("--output-dir", required=True, help="Output root dir.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-recycles", type=int, default=3,
                   help="Chai-1 trunk recycle count (default 3)")
    p.add_argument("--no-esm", action="store_true",
                   help="Skip ESM embeddings (faster; use for smoke tests)")
    p.add_argument("--no-delta-g", action="store_true",
                   help="Skip AutoDock Vina ΔG scoring")
    p.add_argument("--pocket-positions", default=None,
                   help="Comma-separated 1-indexed receptor residue positions "
                        "(matching the input sequence's own numbering) to "
                        "average pLDDT over, e.g. --pocket-positions 151,155,339")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs and generate FASTAs without running "
                        "Chai-1 or Vina. Use to smoke-test the pipeline locally.")
    args = p.parse_args()
    if not args.target_id:
        if not args.uniprot:
            sys.exit("ERROR: pass --target-id (or --uniprot, used as the default target-id)")
        args.target_id = args.uniprot
    return args


def parse_pocket_positions(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(x) for x in raw.split(",") if x.strip()]


# ── Hits loading ──────────────────────────────────────────────────────────────

def load_hits(hits_csv: Path, target_id: str) -> dict[str, list[dict]]:
    rows: list[dict] = []
    with hits_csv.open() as f:
        for r in csv.DictReader(f):
            if r["target_id"] == target_id:
                rows.append(r)

    if not rows:
        sys.exit(f"ERROR: target_id {target_id} not found in {hits_csv.name}.")

    grouped: dict[str, list[dict]] = {}
    for r in rows:
        g = r.get("group", "ungrouped")
        grouped.setdefault(g, []).append(r)
    for g in grouped:
        if all("predicted_score" in r and r["predicted_score"] for r in grouped[g]):
            grouped[g].sort(key=lambda r: float(r["predicted_score"]))
    total = sum(len(v) for v in grouped.values())
    labels = ", ".join(f"{g}={len(v)}" for g, v in grouped.items())
    print(f"  {total} hits for {target_id} ({labels})")
    return grouped


# ── Chai-1 inference ──────────────────────────────────────────────────────────

def run_chai1(
    target_name: str,
    sequence: str,
    molecule_id: str,
    smiles: str,
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Run Chai-1 inference, writing all 5 samples' CIF+npz to out_dir.
    Doesn't return scores itself -- collect_full_metrics reads them back from
    disk afterward, so cached and freshly-run complexes go through the same
    metrics-extraction path."""
    from chai_lab.chai1 import run_inference

    out_dir.mkdir(parents=True, exist_ok=True)
    is_apo = not smiles
    if is_apo:
        fasta_content = f">protein|name={target_name}\n{sequence}\n"
    else:
        fasta_content = (
            f">protein|name={target_name}\n{sequence}\n\n"
            f">ligand|name={molecule_id}\n{smiles}\n"
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fasta", delete=False
    ) as fh:
        fh.write(fasta_content)
        fasta_path = Path(fh.name)

    try:
        run_inference(
            fasta_file=fasta_path,
            output_dir=out_dir,
            num_trunk_recycles=args.num_recycles,
            num_diffn_timesteps=200,
            seed=args.seed,
            device=torch.device("cuda:0"),
            use_esm_embeddings=not args.no_esm,
        )
    finally:
        fasta_path.unlink(missing_ok=True)


# ── Cross-sample structural metrics ──────────────────────────────────────────

def _find_chain(model: gemmi.Model, chain_id: str = "A") -> gemmi.Chain:
    for chain in model:
        if chain.name == chain_id:
            return chain
    return model[0]


def extract_ca_coords(cif_path: Path, chain_id: str = "A") -> np.ndarray:
    st = gemmi.read_structure(str(cif_path))
    chain = _find_chain(st[0], chain_id)
    coords = []
    for residue in chain:
        for atom in residue:
            if atom.name == "CA":
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                break
    return np.array(coords)


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> float:
    """RMSD between two (N,3) coordinate sets after optimal superposition."""
    pc = p - p.mean(axis=0)
    qc = q - q.mean(axis=0)
    h = pc.T @ qc
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    corr = np.diag([1.0, 1.0, d])
    r = vt.T @ corr @ u.T
    p_rot = (r @ pc.T).T
    diff = p_rot - qc
    return float(np.sqrt((diff**2).sum() / len(p)))


def mean_pairwise_ca_rmsd(cif_paths: list[Path], chain_id: str = "A") -> float | None:
    if len(cif_paths) < 2:
        return None
    coord_sets = [extract_ca_coords(p, chain_id) for p in cif_paths]
    lengths = {len(c) for c in coord_sets}
    if len(lengths) != 1:
        warnings.warn(
            f"CA count mismatch across samples ({lengths}) -- skipping cross-sample RMSD"
        )
        return None
    rmsds = [
        kabsch_rmsd(coord_sets[i], coord_sets[j])
        for i in range(len(coord_sets))
        for j in range(i + 1, len(coord_sets))
    ]
    return statistics.mean(rmsds)


def pocket_mean_bfactor(
    cif_path: Path, positions: list[int] | None, chain_id: str = "A"
) -> float | None:
    if not positions:
        return None
    st = gemmi.read_structure(str(cif_path))
    chain = _find_chain(st[0], chain_id)
    position_set = set(positions)
    vals = [
        atom.b_iso
        for residue in chain
        if residue.seqid.num in position_set
        for atom in residue
    ]
    if not vals:
        warnings.warn(f"None of --pocket-positions found in {cif_path.name} chain {chain_id}")
        return None
    return statistics.mean(vals)


def collect_full_metrics(
    group_dir: Path, is_apo: bool, pocket_positions: list[int] | None
) -> dict:
    """Read every pred.model_idx_*.cif / scores.model_idx_*.npz already on
    disk in group_dir (cached from a prior run, or just written) and compute
    the full metrics panel. Returns {} if nothing is on disk yet."""
    cif_paths = sorted(
        group_dir.glob("pred.model_idx_*.cif"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    npz_paths = sorted(
        group_dir.glob("scores.model_idx_*.npz"),
        key=lambda p: int(p.stem.rsplit("_", 1)[-1]),
    )
    if not cif_paths or not npz_paths:
        return {}

    per_sample = [dict(np.load(p, allow_pickle=False)) for p in npz_paths]
    ptm_vals = [float(np.atleast_1d(s["ptm"])[0]) for s in per_sample]
    iptm_vals = [float(np.atleast_1d(s["iptm"])[0]) for s in per_sample]
    top = per_sample[0]

    return {
        "is_apo": is_apo,
        "ptm": ptm_vals[0],
        "iptm": "" if is_apo else iptm_vals[0],
        "aggregate_score": "" if is_apo else float(np.atleast_1d(top["aggregate_score"])[0]),
        "per_chain_ptm": ";".join(f"{v:.4f}" for v in np.atleast_1d(top["per_chain_ptm"])),
        "has_inter_chain_clashes": bool(
            any(np.atleast_1d(s["has_inter_chain_clashes"]).reshape(-1).any() for s in per_sample)
        ),
        "chain_chain_clashes": ";".join(
            str(int(v)) for v in np.atleast_1d(top["chain_chain_clashes"]).reshape(-1)
        ),
        "ptm_mean_across_samples": statistics.mean(ptm_vals),
        "ptm_std_across_samples": statistics.pstdev(ptm_vals),
        "iptm_mean_across_samples": "" if is_apo else statistics.mean(iptm_vals),
        "iptm_std_across_samples": "" if is_apo else statistics.pstdev(iptm_vals),
        "ca_rmsd_mean_across_samples": mean_pairwise_ca_rmsd(cif_paths),
        "pocket_mean_plddt": pocket_mean_bfactor(cif_paths[0], pocket_positions),
        "n_samples": len(per_sample),
    }


# ── ΔG scoring ────────────────────────────────────────────────────────────────

def compute_delta_g(cif_path: Path, smiles: str) -> float | None:
    try:
        import subprocess

        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from rdkit.Chem import AllChem
        from vina import Vina
    except ImportError as e:
        warnings.warn(f"ΔG scoring skipped — missing dependency: {e}")
        return None

    prot_pdb = rec_pdbqt = lig_pdb = lig_pdbqt = None

    try:
        # Write full complex as PDB, then split by ATOM (protein) / HETATM (ligand)
        struct = gemmi.read_structure(str(cif_path))
        with tempfile.NamedTemporaryFile(suffix="_full.pdb", delete=False, mode="w") as fh:
            full_pdb = Path(fh.name)
        struct.write_pdb(str(full_pdb))
        full_text = full_pdb.read_text()
        full_pdb.unlink(missing_ok=True)

        # ── Receptor: ATOM records → PDB → obabel → PDBQT ────────────────────
        prot_lines = [l for l in full_text.splitlines()
                      if l.startswith(("ATOM", "TER", "END"))]
        with tempfile.NamedTemporaryFile(suffix="_prot.pdb", delete=False, mode="w") as fh:
            prot_pdb = Path(fh.name)
            fh.write("\n".join(prot_lines) + "\n")

        with tempfile.NamedTemporaryFile(suffix="_rec.pdbqt", delete=False) as fh:
            rec_pdbqt = Path(fh.name)
        r = subprocess.run(
            ["obabel", str(prot_pdb), "-O", str(rec_pdbqt), "-xr"],
            capture_output=True, timeout=120,
        )
        if not rec_pdbqt.exists() or rec_pdbqt.stat().st_size == 0:
            raise RuntimeError(f"obabel receptor prep failed: {r.stderr.decode().strip()}")

        # ── Ligand: HETATM records → PDB → RDKit + SMILES template → PDBQT ──
        hetatm_lines = [l for l in full_text.splitlines() if l.startswith("HETATM")]
        if not hetatm_lines:
            raise RuntimeError("No HETATM records found — ligand missing from PDB")
        with tempfile.NamedTemporaryFile(suffix="_lig.pdb", delete=False, mode="w") as fh:
            lig_pdb = Path(fh.name)
            fh.write("\n".join(hetatm_lines) + "\nEND\n")

        # Load 3D pose from PDB, then impose correct bond orders from SMILES
        template = Chem.MolFromSmiles(smiles)
        lig_raw = Chem.MolFromPDBBlock(lig_pdb.read_text(), removeHs=False, sanitize=False)
        if lig_raw is None or template is None:
            return None
        try:
            lig_3d = AllChem.AssignBondOrdersFromTemplate(template, lig_raw)
            lig_3d = AllChem.AddHs(lig_3d, addCoords=True)
        except Exception:
            # Fall back to fresh embed if template matching fails
            lig_3d = AllChem.AddHs(template)
            if AllChem.EmbedMolecule(lig_3d, AllChem.ETKDGv3()) != 0:
                return None
            AllChem.MMFFOptimizeMolecule(lig_3d)

        mk = MoleculePreparation()
        mols = mk.prepare(lig_3d)
        pdbqt_str, ok, err = PDBQTWriterLegacy.write_string(mols[0])
        if not ok:
            raise RuntimeError(f"Meeko PDBQT prep failed: {err}")

        with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False, mode="w") as fh:
            lig_pdbqt = Path(fh.name)
            fh.write(pdbqt_str)

        # ── Score with Vina (score_only — no docking search) ─────────────────
        # Vina requires affinity maps even for score_only; center on ligand centroid
        positions = lig_3d.GetConformer().GetPositions()
        center = positions.mean(axis=0).tolist()

        v = Vina(sf_name="vina", verbosity=0)
        v.set_receptor(str(rec_pdbqt))
        v.set_ligand_from_file(str(lig_pdbqt))
        v.compute_vina_maps(center=center, box_size=[30.0, 30.0, 30.0])
        energy = v.score()
        return float(energy[0])

    except Exception as exc:
        warnings.warn(f"ΔG scoring failed for {cif_path.parent.name}: {exc}")
        return None
    finally:
        for p in [prot_pdb, rec_pdbqt, lig_pdb, lig_pdbqt]:
            if p is not None:
                try:
                    p.unlink()
                except Exception:
                    pass


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    pocket_positions = parse_pocket_positions(args.pocket_positions)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root / "summary.csv"

    print(f"── Target: {args.target_id} ──")
    print(f"── Output: {output_root} ──\n")

    print("── Phase 1: Loading hits ──")
    hits_csv = Path(args.hits_csv)
    if not hits_csv.exists():
        sys.exit(f"ERROR: hits CSV not found at {hits_csv}")
    hits = load_hits(hits_csv, args.target_id)

    print("\n── Phase 2: Resolving target sequence ──")
    sequence = resolve_sequence(args.sequence, args.fasta, args.uniprot)
    target_name = args.target_id

    print("\n── Phase 3+4: Structure prediction + ΔG scoring ──")
    completed: list[dict] = []

    summary_exists = summary_csv.exists()
    already_recorded: set[str] = set()
    if summary_exists:
        with summary_csv.open(newline="") as rf:
            for existing in csv.DictReader(rf):
                already_recorded.add(existing["molecule_id"])
    summary_fh = summary_csv.open("a", newline="")
    writer = csv.DictWriter(summary_fh, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
    if not summary_exists:
        writer.writeheader()

    seen_mol_ids: set[str] = set()
    for group_name, group_rows in hits.items():
        for row in group_rows:
            mol_id = row["molecule_id"]
            smiles_raw = row.get("smiles", "") or ""
            is_apo = not smiles_raw.strip()
            smiles = "" if is_apo else desalt_smiles(smiles_raw)
            if not is_apo and smiles != smiles_raw:
                print(f"\n  [{group_name}] {mol_id}  desalted: {smiles_raw[:40]}… → {smiles[:40]}")

            if mol_id in seen_mol_ids:
                print(f"\n  [{group_name}] {mol_id} — duplicate, skipping")
                continue
            seen_mol_ids.add(mol_id)

            if mol_id in already_recorded:
                print(f"\n  [{group_name}] {mol_id} — already in summary.csv, skipping")
                continue

            group_dir = output_root / group_name / mol_id
            group_dir.mkdir(parents=True, exist_ok=True)

            score_display = row.get("predicted_score", "")
            label = "apo (no ligand)" if is_apo else smiles
            print(f"\n  [{group_name}] {mol_id}  score={score_display}  {label}")

            metrics: dict = {}
            delta_g = None
            if args.dry_run:
                fasta_content = (
                    f">protein|name={target_name}\n{sequence}\n"
                    if is_apo
                    else f">protein|name={target_name}\n{sequence}\n\n>ligand|name={mol_id}\n{smiles}\n"
                )
                fasta_path = group_dir / "input.fasta"
                fasta_path.write_text(fasta_content)
                print(f"    [dry-run] FASTA written to {fasta_path}")
                print(f"    [dry-run] {'apo, no ligand' if is_apo else f'SMILES: {smiles}'}")
            else:
                metrics = collect_full_metrics(group_dir, is_apo, pocket_positions)
                cache_label = "cached"
                if not metrics:
                    cache_label = "fresh"
                    try:
                        run_chai1(target_name, sequence, mol_id, smiles, group_dir, args)
                        metrics = collect_full_metrics(group_dir, is_apo, pocket_positions)
                    except Exception as exc:
                        warnings.warn(f"Chai-1 failed for {mol_id}: {exc}")
                        metrics = {}

                if metrics:
                    iptm_display = "n/a (apo)" if is_apo else f"{metrics['iptm']:.3f}"
                    print(f"    Chai-1 ({cache_label}) ptm={metrics['ptm']:.3f}  iptm={iptm_display}")

                if not is_apo and not args.no_delta_g and metrics:
                    cif0 = group_dir / "pred.model_idx_0.cif"
                    if cif0.exists():
                        delta_g = compute_delta_g(cif0, smiles)
                        if delta_g is not None:
                            print(f"    ΔG = {delta_g:.2f} kcal/mol")
                        else:
                            print("    ΔG scoring failed (see warning above)")
                    else:
                        warnings.warn(f"No CIF found in {group_dir} — skipping ΔG")
                metrics["delta_g_kcal_mol"] = delta_g if delta_g is not None else ""

            result = {
                "target_id":       args.target_id,
                "molecule_id":     mol_id,
                "smiles":          smiles,
                "predicted_score": row.get("predicted_score", ""),
                "rank":            row.get("rank", ""),
                "group":           group_name,
                **{k: metrics.get(k, "") for k in SUMMARY_FIELDS if k not in
                   ("target_id", "molecule_id", "smiles", "predicted_score", "rank", "group")},
            }
            writer.writerow(result)
            summary_fh.flush()
            completed.append(result)

    summary_fh.close()
    print(f"\n── Done. {len(completed)} complexes scored ──")
    print(f"  Summary: {summary_csv}\n")

    has_dg = any(r.get("delta_g_kcal_mol") not in ("", None) for r in completed)
    sort_key = "delta_g_kcal_mol" if has_dg else "ptm"
    rev = not has_dg
    ranked = sorted(
        completed,
        key=lambda r: float(r[sort_key]) if r.get(sort_key) not in ("", None) else float("inf"),
        reverse=rev,
    )
    hdr = f"{'Mol':16} {'Group':8} {'apo':>5} {'pTM':>6} {'ipTM':>6} {'ΔG (kcal/mol)':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in ranked:
        dg = f"{float(r['delta_g_kcal_mol']):>14.2f}" if r.get("delta_g_kcal_mol") not in ("", None) else f"{'—':>14}"
        ptm = f"{float(r['ptm']):>6.3f}" if r.get("ptm") not in ("", None) else f"{'—':>6}"
        iptm = f"{float(r['iptm']):>6.3f}" if r.get("iptm") not in ("", None) else f"{'—':>6}"
        apo_flag = str(r.get("is_apo", ""))[:5]
        print(f"{r['molecule_id']:16} {r['group']:8} {apo_flag:>5} {ptm} {iptm} {dg}")


if __name__ == "__main__":
    main()
