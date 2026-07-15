#!/usr/bin/env python3
"""
predict_structures.py — Chai-1 co-fold structure prediction + AutoDock Vina
ΔG scoring for protein-ligand complexes.

For a given target (any protein sequence), loads a ligand list from a CSV,
runs Chai-1 protein-ligand co-folding on each (target, ligand) pair, then
scores the predicted pose with AutoDock Vina in score_only mode.

Input CSV schema
-----------------
  target_id, molecule_id, smiles, group[, predicted_score, rank]

  `target_id` is any string identifier for the row's target — a UniProt
  accession is a natural choice but not required. Rows are filtered to
  --target-id. `group` is a free-form label (e.g. "binder"/"decoy") used only
  for organizing output directories and downstream reports.

Outputs
-------
  <output-dir>/<group>/<molecule_id>/pred.model_idx_0.cif
  <output-dir>/summary.csv   — confidence scores + ΔG per complex, written incrementally

Dependencies (beyond the toolkit's own requirements.txt)
---------------------------------------------------------
  pip install chai-lab vina meeko gemmi

Compute
-------
  Requires a CUDA GPU for Chai-1 (A10/A100 class). Not runnable on CPU/MPS.
  ~10-15 min per complex on an A10. Use --no-esm for a quick smoke test.

Usage
-----
  python -m chai1_vina_toolkit.predict_structures \
      --uniprot O00144 --hits-csv my_ligands.csv --output-dir out/O00144
  python -m chai1_vina_toolkit.predict_structures \
      --sequence MSEQUENCE... --target-id my_target --hits-csv my_ligands.csv \
      --output-dir out/my_target --no-esm --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
import tempfile
import warnings
from pathlib import Path

import torch
from rdkit import Chem

from .sequences import add_sequence_args, resolve_sequence

SUMMARY_FIELDS = [
    "target_id", "molecule_id", "smiles", "predicted_score", "rank", "group",
    "aggregate_score", "ptm", "iptm", "delta_g_kcal_mol",
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
    p.add_argument("--dry-run", action="store_true",
                   help="Validate inputs and generate FASTAs without running "
                        "Chai-1 or Vina. Use to smoke-test the pipeline locally.")
    args = p.parse_args()
    if not args.target_id:
        if not args.uniprot:
            sys.exit("ERROR: pass --target-id (or --uniprot, used as the default target-id)")
        args.target_id = args.uniprot
    return args


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
) -> dict:
    from chai_lab.chai1 import run_inference

    out_dir.mkdir(parents=True, exist_ok=True)
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
        candidates = run_inference(
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

    # chai-lab >=0.5 returns StructureCandidates (not a list); scores live in
    # ranking_data[i] (SampleRanking) and ptm_scores (PTMScores).
    rd = candidates.ranking_data[0]
    return {
        "aggregate_score": float(rd.aggregate_score.item()),
        "ptm":             float(rd.ptm_scores.complex_ptm.item()),
        "iptm":            float(rd.ptm_scores.interface_ptm.item()),
    }


# ── ΔG scoring ────────────────────────────────────────────────────────────────

def compute_delta_g(cif_path: Path, smiles: str) -> float | None:
    try:
        import subprocess
        import gemmi
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        from rdkit import Chem
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
            smiles_raw = row["smiles"]
            smiles = desalt_smiles(smiles_raw)
            if smiles != smiles_raw:
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
            print(f"\n  [{group_name}] {mol_id}  score={score_display}")

            fasta_content = (
                f">protein|name={target_name}\n{sequence}\n\n"
                f">ligand|name={mol_id}\n{smiles}\n"
            )

            if args.dry_run:
                fasta_path = group_dir / "input.fasta"
                fasta_path.write_text(fasta_content)
                print(f"    [dry-run] FASTA written to {fasta_path}")
                print(f"    [dry-run] SMILES: {smiles}")
                chai_scores = {"aggregate_score": None, "ptm": None, "iptm": None}
                delta_g = None
            else:
                cif_done = group_dir / "pred.model_idx_0.cif"
                npz_done = group_dir / "scores.model_idx_0.npz"
                if cif_done.exists() and npz_done.exists():
                    import numpy as np
                    d = np.load(npz_done)
                    chai_scores = {
                        "aggregate_score": float(d["aggregate_score"][0]),
                        "ptm":             float(d["ptm"][0]),
                        "iptm":            float(d["iptm"][0]),
                    }
                    print(f"    Chai-1 (cached) iptm={chai_scores['iptm']:.3f}  "
                          f"ptm={chai_scores['ptm']:.3f}")
                else:
                    try:
                        chai_scores = run_chai1(
                            target_name, sequence, mol_id, smiles, group_dir, args
                        )
                        print(f"    Chai-1 iptm={chai_scores['iptm']:.3f}  "
                              f"ptm={chai_scores['ptm']:.3f}")
                    except Exception as exc:
                        warnings.warn(f"Chai-1 failed for {mol_id}: {exc}")
                        chai_scores = {"aggregate_score": None, "ptm": None, "iptm": None}

                delta_g = None
                if not args.no_delta_g:
                    cif_candidates = list(group_dir.glob("pred.model_idx_0.cif"))
                    if cif_candidates:
                        delta_g = compute_delta_g(cif_candidates[0], smiles)
                        if delta_g is not None:
                            print(f"    ΔG = {delta_g:.2f} kcal/mol")
                        else:
                            print("    ΔG scoring failed (see warning above)")
                    else:
                        warnings.warn(f"No CIF found in {group_dir} — skipping ΔG")

            result = {
                "target_id":        args.target_id,
                "molecule_id":      mol_id,
                "smiles":           smiles,
                "predicted_score":  row.get("predicted_score", ""),
                "rank":             row.get("rank", ""),
                "group":            group_name,
                "aggregate_score":  chai_scores["aggregate_score"] or "",
                "ptm":              chai_scores["ptm"] or "",
                "iptm":             chai_scores["iptm"] or "",
                "delta_g_kcal_mol": delta_g if delta_g is not None else "",
            }
            writer.writerow(result)
            summary_fh.flush()
            completed.append(result)

    summary_fh.close()
    print(f"\n── Done. {len(completed)} complexes scored ──")
    print(f"  Summary: {summary_csv}\n")

    has_dg = any(r["delta_g_kcal_mol"] != "" for r in completed)
    sort_key = "delta_g_kcal_mol" if has_dg else "iptm"
    rev = not has_dg
    ranked = sorted(
        completed,
        key=lambda r: float(r[sort_key]) if r[sort_key] != "" else float("inf"),
        reverse=rev,
    )
    hdr = f"{'Mol':16} {'Group':12} {'Score':>7} {'ipTM':>6} {'ΔG (kcal/mol)':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in ranked:
        dg = f"{float(r['delta_g_kcal_mol']):>14.2f}" if r["delta_g_kcal_mol"] != "" else f"{'—':>14}"
        iptm = f"{float(r['iptm']):>6.3f}" if r["iptm"] != "" else f"{'—':>6}"
        score = float(r["predicted_score"]) if r["predicted_score"] not in ("", None) else float("nan")
        print(f"{r['molecule_id']:16} {r['group']:12} {score:>7.1f} "
              f"{iptm} {dg}")


if __name__ == "__main__":
    main()
