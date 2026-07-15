#!/usr/bin/env python3
"""
backfill_delta_g.py — fill in missing AutoDock Vina ΔG values in a
predict_structures.py summary.csv without re-folding.

Use when the co-folding step succeeded (ipTM present) but the ΔG column is
empty — e.g. `obabel` was absent at run time. The folded
`pred.model_idx_0.cif` for each complex is already on disk, so this reuses
predict_structures.compute_delta_g to score just the rows whose
`delta_g_kcal_mol` is blank, then rewrites summary.csv in place (atomically).

Vina score_only is CPU-only, so this is safe to run alongside a GPU job — but
do NOT run it while the same summary.csv is still being appended to (write
race); wait until the docking run has finished.

Usage
-----
  python -m chai1_vina_toolkit.backfill_delta_g --output-dir out/my_target
  # summary defaults to <output-dir>/summary.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

from .predict_structures import compute_delta_g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output-dir", required=True,
                    help="structures dir holding <group>/<molecule_id>/ and summary.csv")
    ap.add_argument("--summary", default=None,
                    help="summary.csv path (default: <output-dir>/summary.csv)")
    ap.add_argument("--model", default="pred.model_idx_0.cif",
                    help="which folded CIF to score (matches predict_structures.py)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    summary = Path(args.summary) if args.summary else out_dir / "summary.csv"
    if not summary.exists():
        sys.exit(f"ERROR: no summary at {summary}")

    with summary.open() as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if "delta_g_kcal_mol" not in (fieldnames or []):
        sys.exit("ERROR: summary has no delta_g_kcal_mol column")

    todo = [r for r in rows if not str(r.get("delta_g_kcal_mol", "")).strip()]
    print(f"{len(rows)} rows, {len(todo)} missing ΔG")

    filled = 0
    for r in todo:
        mol_id, group = r["molecule_id"], r["group"]
        cif = out_dir / group / mol_id / args.model
        if not cif.exists():
            print(f"  {mol_id}: no {args.model} — skipping")
            continue
        dg = compute_delta_g(cif, r["smiles"])
        if dg is None:
            print(f"  {mol_id}: ΔG scoring returned None")
            continue
        r["delta_g_kcal_mol"] = dg
        filled += 1
        print(f"  {mol_id}: ΔG = {dg:.2f} kcal/mol")

    if filled == 0:
        print("nothing filled — leaving summary unchanged")
        return

    fd, tmp = tempfile.mkstemp(dir=str(summary.parent), suffix=".csv")
    with os.fdopen(fd, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, summary)
    print(f"backfilled {filled} ΔG values → {summary}")


if __name__ == "__main__":
    main()
