#!/usr/bin/env python3
"""
chemical_similarity.py — chemical-similarity module for a CSV of SMILES.

Takes a CSV of SMILES and does three things:

  1. 3D models (obabel): generate a 3D `.sdf` for each molecule
     (`obabel -:"<SMILES>" -osdf --gen3d`), written per-molecule under
     `<out>/sdf/<id>.sdf` plus a combined `<out>/all_3d.sdf`.

  2. Pairwise similarity for every unordered pair of molecules, four metrics:
       * Tanimoto, Dice, Cosine  — on RDKit Morgan/ECFP4 fingerprints
       * Maximum Common Substructure (MCS) — rdFMCS, scored Tanimoto-style as
         mcs_atoms / (atomsA + atomsB - mcs_atoms)
     Emitted long-format as `<out>/pairwise_similarity.csv`.

  3. Adjacency matrix per metric: symmetric N×N CSVs
     `<out>/{tanimoto,dice,cosine,mcs}_adjacency.csv`, diagonal = 1.0
     (self-similarity, doubles as a sanity check).

Every SMILES is RDKit-validated first; invalid rows are skipped and logged.
Local-only dependencies: rdkit + obabel (no GPU).

Usage
-----
  python -m chai1_vina_toolkit.chemical_similarity --input molecules.csv [--out out_dir]
      [--id-col molecule_id] [--smiles-col smiles] [--radius 2] [--nbits 2048] [--no-sdf]
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from itertools import combinations
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

RDLogger.DisableLog("rdApp.*")

_ID_CANDIDATES     = ("molecule_id", "mol_id", "id", "name", "compound_id", "chembl_id")
_SMILES_CANDIDATES = ("smiles", "canonical_smiles", "smi", "structure")


def detect_col(header: list[str], preferred: str | None, candidates: tuple[str, ...], kind: str) -> str:
    if preferred:
        if preferred not in header:
            sys.exit(f"ERROR: --{kind}-col '{preferred}' not in CSV header {header}")
        return preferred
    low = {h.lower(): h for h in header}
    for c in candidates:
        if c in low:
            return low[c]
    sys.exit(f"ERROR: could not auto-detect {kind} column in {header}; pass --{kind}-col.")


def load_molecules(path: Path, id_col: str | None, smiles_col: str | None) -> list[dict]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        idc = detect_col(header, id_col, _ID_CANDIDATES, "id")
        smc = detect_col(header, smiles_col, _SMILES_CANDIDATES, "smiles")
        print(f"Using id column '{idc}', smiles column '{smc}'")
        mols, seen = [], set()
        for row in reader:
            mid, smi = (row.get(idc) or "").strip(), (row.get(smc) or "").strip()
            if not mid or not smi:
                continue
            m = Chem.MolFromSmiles(smi)
            if m is None:
                print(f"  WARNING: invalid SMILES, skipping {mid}: {smi}")
                continue
            if mid in seen:
                print(f"  WARNING: duplicate id '{mid}', skipping the second occurrence")
                continue
            seen.add(mid)
            mols.append({"id": mid, "smiles": smi, "mol": m})
    return mols


# ── Step 1: 3D SDF via obabel ──────────────────────────────────────────────────

def have_obabel() -> bool:
    try:
        subprocess.run(["obabel", "-V"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def build_3d_sdf(mols: list[dict], out: Path) -> None:
    sdf_dir = out / "sdf"
    sdf_dir.mkdir(parents=True, exist_ok=True)
    combined = out / "all_3d.sdf"
    ok = 0
    with combined.open("w") as comb:
        for m in mols:
            per = sdf_dir / f"{m['id']}.sdf"
            r = subprocess.run(
                ["obabel", f"-:{m['smiles']}", "-osdf", "--gen3d", "--title", m["id"]],
                capture_output=True, text=True,
            )
            if r.returncode != 0 or "$$$$" not in r.stdout:
                print(f"  WARNING: obabel --gen3d failed for {m['id']}")
                continue
            per.write_text(r.stdout)
            comb.write(r.stdout)
            ok += 1
    print(f"Wrote {ok}/{len(mols)} 3D SDFs to {sdf_dir}/ (combined: {combined.name})")


# ── Step 2: pairwise similarity ────────────────────────────────────────────────

def mcs_similarity(a: Chem.Mol, b: Chem.Mol) -> float:
    """Tanimoto-style MCS score: mcs_atoms / (atomsA + atomsB - mcs_atoms)."""
    res = rdFMCS.FindMCS(
        [a, b], timeout=10,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
        bondCompare=rdFMCS.BondCompare.CompareOrder,
        ringMatchesRingOnly=True, completeRingsOnly=True,
    )
    if res.canceled or res.numAtoms == 0:
        return 0.0
    na, nb = a.GetNumAtoms(), b.GetNumAtoms()
    denom = na + nb - res.numAtoms
    return res.numAtoms / denom if denom > 0 else 0.0


def compute_pairwise(mols: list[dict], radius: int, nbits: int):
    gen = GetMorganGenerator(radius=radius, fpSize=nbits)
    for m in mols:
        m["fp"] = gen.GetFingerprint(m["mol"])
    metrics = ("tanimoto", "dice", "cosine", "mcs")
    pairs = []
    for a, b in combinations(mols, 2):
        pairs.append({
            "id_a": a["id"], "id_b": b["id"],
            "tanimoto": DataStructs.TanimotoSimilarity(a["fp"], b["fp"]),
            "dice":     DataStructs.DiceSimilarity(a["fp"], b["fp"]),
            "cosine":   DataStructs.CosineSimilarity(a["fp"], b["fp"]),
            "mcs":      mcs_similarity(a["mol"], b["mol"]),
        })
    return metrics, pairs


# ── Step 3: adjacency matrices ─────────────────────────────────────────────────

def write_outputs(mols: list[dict], metrics, pairs, out: Path) -> None:
    pw = out / "pairwise_similarity.csv"
    with pw.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id_a", "id_b", *metrics])
        for p in pairs:
            w.writerow([p["id_a"], p["id_b"], *(f"{p[m]:.4f}" for m in metrics)])
    print(f"Wrote {pw.name} ({len(pairs)} pairs)")

    ids = [m["id"] for m in mols]
    lookup = {(p["id_a"], p["id_b"]): p for p in pairs}
    for metric in metrics:
        mat = out / f"{metric}_adjacency.csv"
        with mat.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["", *ids])
            for ri in ids:
                row = [ri]
                for ci in ids:
                    if ri == ci:
                        row.append("1.0000")
                    else:
                        p = lookup.get((ri, ci)) or lookup.get((ci, ri))
                        row.append(f"{p[metric]:.4f}")
                w.writerow(row)
        print(f"Wrote {mat.name} ({len(ids)}x{len(ids)})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--input", required=True, help="CSV of SMILES.")
    p.add_argument("--out", default=None, help="Output dir (default: <input_stem>_similarity/ alongside input).")
    p.add_argument("--id-col", default=None)
    p.add_argument("--smiles-col", default=None)
    p.add_argument("--radius", type=int, default=2, help="Morgan radius (ECFP4 = 2).")
    p.add_argument("--nbits", type=int, default=2048, help="Fingerprint bit size.")
    p.add_argument("--no-sdf", action="store_true", help="Skip 3D SDF generation (similarity only).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"ERROR: input not found: {in_path}")
    out = Path(args.out) if args.out else in_path.parent / f"{in_path.stem}_similarity"
    out.mkdir(parents=True, exist_ok=True)

    mols = load_molecules(in_path, args.id_col, args.smiles_col)
    if len(mols) < 2:
        sys.exit(f"ERROR: need >=2 valid molecules, got {len(mols)}.")
    print(f"Loaded {len(mols)} valid molecules")

    if not args.no_sdf:
        if have_obabel():
            build_3d_sdf(mols, out)
        else:
            print("WARNING: obabel not found — skipping 3D SDF generation (use --no-sdf to silence).")

    metrics, pairs = compute_pairwise(mols, args.radius, args.nbits)
    write_outputs(mols, metrics, pairs, out)
    print(f"\nDone. Outputs in {out}/")


if __name__ == "__main__":
    main()
