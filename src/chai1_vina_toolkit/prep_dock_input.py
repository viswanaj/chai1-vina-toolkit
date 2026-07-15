#!/usr/bin/env python3
"""
prep_dock_input.py — coerce an externally-generated candidate CSV (e.g. from
an LLM-proposed scaffold list, a virtual screen export, etc.) into the
predict_structures.py docking schema, and optionally append your own
positive/negative control sets so the candidate set is read against a known
good/bad band on the same docking run.

Auto-detects the SMILES and ID columns, validates + canonicalises with RDKit
(desalts to largest fragment).

Output schema: target_id, gene_name, molecule_id, smiles, predicted_score, group

Usage
-----
  python -m chai1_vina_toolkit.prep_dock_input \
      --in  candidates.csv \
      --out candidates.dock_input.csv \
      --target-id my_target \
      --group my_candidates \
      --actives  known_actives.csv \
      --decoys   known_decoys.csv

--actives/--decoys are optional CSVs with columns molecule_id,smiles; if
given, their rows are appended with group="active"/"decoy" respectively so
the run always carries a positive/negative control.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def pick_col(cols, wanted):
    low = {c.lower(): c for c in cols}
    for w in wanted:
        if w in low:
            return low[w]
    return None


def desalt(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) > 1:
        mol = max(frags, key=lambda m: m.GetNumHeavyAtoms())
    return Chem.MolToSmiles(mol)


def load_control_set(path: str | None) -> list[tuple[str, str]]:
    if not path:
        return []
    out = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out.append((row["molecule_id"], row["smiles"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-id", required=True)
    ap.add_argument("--gene", default="", help="optional display label")
    ap.add_argument("--group", default="candidates")
    ap.add_argument("--actives", default=None,
                    help="optional CSV (molecule_id,smiles) appended with group=active")
    ap.add_argument("--decoys", default=None,
                    help="optional CSV (molecule_id,smiles) appended with group=decoy")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    smi_col = pick_col(df.columns, ["smiles", "smi", "canonical_smiles"])
    id_col = pick_col(df.columns, ["id", "molecule_id", "name", "compound_id"])
    if smi_col is None:
        sys.exit(f"ERROR: no SMILES column found (columns: {list(df.columns)})")

    rows, seen, bad = [], set(), 0
    for i, r in df.iterrows():
        canon = desalt(str(r[smi_col]))
        if canon is None:
            bad += 1
            continue
        if canon in seen:
            continue
        seen.add(canon)
        mid = str(r[id_col]) if id_col else f"{args.group}_{i:03d}"
        rows.append((args.target_id, args.gene, mid, canon, i + 1, args.group))

    n_cand = len(rows)
    base = n_cand + 1
    for mid, smi in load_control_set(args.actives):
        c = desalt(smi)
        if c is None:
            continue
        rows.append((args.target_id, args.gene, mid, c, base, "active"))
        base += 1
    for mid, smi in load_control_set(args.decoys):
        c = desalt(smi)
        if c is None:
            continue
        rows.append((args.target_id, args.gene, mid, c, base, "decoy"))
        base += 1

    out = pd.DataFrame(rows, columns=[
        "target_id", "gene_name", "molecule_id", "smiles", "predicted_score", "group"])
    # predict_structures.py records a per-group rank in its summary.
    out["rank"] = out.groupby("group").cumcount() + 1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"read {len(df)} rows from {Path(args.inp).name} "
          f"(smiles='{smi_col}', id='{id_col}'), {bad} invalid dropped")
    print(f"wrote {args.out}")
    print(out["group"].value_counts().to_string())


if __name__ == "__main__":
    main()
