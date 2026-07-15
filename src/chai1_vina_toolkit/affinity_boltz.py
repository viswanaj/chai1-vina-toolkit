#!/usr/bin/env python3
"""
affinity_boltz.py — Boltz-2 learned binding-affinity prediction, run in
PARALLEL over a whole ligand panel, as an orthogonal axis to a physics-based
ΔG assay (e.g. predict_structures.py's Chai-1 + Vina score_only pipeline).

Why this module exists
------------------------
A rigid physics score (Vina `score_only`) rewards any well-sized lipophile
dropped into a cavity, so it can fail to separate real binders from
property-matched decoys. Boltz-2 is a *learned* co-folding + affinity model
with a dedicated affinity head that outputs (a) a continuous predicted
affinity and (b) a binder-vs-decoy probability — a fundamentally different
signal from a physics ΔG. Running it across the same panel gives a second,
independent readout: where the two agree, trust the call more; where they
disagree, flag it (the same two-independent-models concordance logic used
throughout this toolkit).

"Parallel"
----------
Boltz predicts an entire directory of inputs in a single `boltz predict`
call: one model load, every complex processed as a batch, and `--devices N`
fans the batch across multiple GPUs. This module writes one Boltz YAML per
ligand into an inputs dir, launches ONE boltz run over the dir, then parses
every `affinity_*.json` back into a tidy summary.csv.

Input schema (identical to predict_structures.py)
--------------------------------------------------
  CSV with columns: target_id, gene_name, molecule_id, smiles, predicted_score,
  rank, group   (order irrelevant; only these names are read).
  Rows are filtered to --target-id and grouped by the `group` column.

Outputs
-------
  <output-dir>/
    boltz_inputs/<molecule_id>.yaml         one co-fold+affinity spec per ligand
    boltz_results_boltz_inputs/predictions/<molecule_id>/
        affinity_<molecule_id>.json         Boltz-2 affinity head output
        confidence_<molecule_id>_model_0.json
    manifest.json                           name -> (group, molecule_id, smiles)
    summary.csv                             one row per ligand, parsed + derived

summary.csv columns
-------------------
  target_id, gene_name, molecule_id, smiles, predicted_score, rank, group,
  boltz_affinity_pred_value   Boltz-2 raw affinity (log10 IC50 [µM]; lower = stronger)
  boltz_affinity_prob_binary  P(binder) in [0,1] from the binary head
  boltz_pic50_est             6 - affinity_pred_value  (pIC50, per Boltz-2's µM convention)
  boltz_iptm, boltz_ptm       Boltz-2 confidence (if a confidence json is present)

Dependencies
------------
  pip install boltz          # pulls torch; needs a CUDA GPU
  First run downloads the Boltz-2 weights (~cached under --cache, default ~/.boltz).
  MSA: --use-msa-server (default) uses the ColabFold MMseqs2 server; or pass a
  precomputed --msa <a3m> (shared by every ligand — efficient for one target,
  many ligands); or --single-sequence for no MSA.

Usage
-----
  python -m chai1_vina_toolkit.affinity_boltz \
      --uniprot <UNIPROT_ACCESSION> \
      --hits-csv panel.csv \
      --output-dir out/boltz_affinity \
      --devices 1
  # stage YAMLs only, no GPU:
  python -m chai1_vina_toolkit.affinity_boltz --uniprot <ACCESSION> \
      --hits-csv panel.csv --output-dir /tmp/boltz_smoke --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

from rdkit import Chem

from .sequences import add_sequence_args, resolve_sequence

SUMMARY_FIELDS = [
    "target_id", "gene_name", "molecule_id", "smiles", "predicted_score", "rank",
    "group", "boltz_affinity_pred_value", "boltz_affinity_prob_binary",
    "boltz_pic50_est", "boltz_iptm", "boltz_ptm",
]


def desalt_smiles(smiles: str) -> str:
    """Keep the largest fragment of a salt/solvate SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(frags) <= 1:
        return smiles
    return Chem.MolToSmiles(max(frags, key=lambda m: m.GetNumHeavyAtoms()))


def load_hits(hits_csv: Path, target_id: str) -> list[dict]:
    rows = []
    with hits_csv.open() as f:
        for r in csv.DictReader(f):
            if r.get("target_id") == target_id:
                rows.append(r)
    if not rows:
        sys.exit(f"ERROR: target_id {target_id} not found in {hits_csv.name}")
    labels: dict[str, int] = {}
    for r in rows:
        labels[r.get("group", "?")] = labels.get(r.get("group", "?"), 0) + 1
    print(f"  {len(rows)} hits for {target_id} "
          f"({', '.join(f'{g}={n}' for g, n in labels.items())})")
    return rows


def safe_name(molecule_id: str) -> str:
    """Boltz uses the YAML filename stem as the record name; sanitise it."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in molecule_id)


# ── Boltz YAML authoring (hand-written to avoid a pyyaml dependency) ───────────

def write_boltz_yaml(path: Path, sequence: str, smiles: str,
                     msa: str | None, single_sequence: bool) -> None:
    """One protein + one ligand + an affinity property on the ligand chain."""
    lines = [
        "version: 1",
        "sequences:",
        "  - protein:",
        "      id: A",
        f"      sequence: {sequence}",
    ]
    if msa:
        lines.append(f"      msa: {msa}")
    elif single_sequence:
        lines.append("      msa: empty")
    lines += [
        "  - ligand:",
        "      id: B",
        f"      smiles: '{smiles}'",
        "properties:",
        "  - affinity:",
        "      binder: B",
        "",
    ]
    path.write_text("\n".join(lines))


# ── Output parsing ────────────────────────────────────────────────────────────

def find_json(root: Path, prefix: str, name: str) -> Path | None:
    hits = sorted(root.rglob(f"{prefix}_{name}*.json"))
    return hits[0] if hits else None


def parse_affinity(results_root: Path, name: str) -> dict:
    out = {"boltz_affinity_pred_value": "", "boltz_affinity_prob_binary": "",
           "boltz_pic50_est": "", "boltz_iptm": "", "boltz_ptm": ""}
    aff = find_json(results_root, "affinity", name)
    if aff is not None:
        d = json.loads(aff.read_text())
        val = d.get("affinity_pred_value", d.get("affinity_pred_value1"))
        prob = d.get("affinity_probability_binary",
                     d.get("affinity_probability_binary1"))
        if val is not None:
            out["boltz_affinity_pred_value"] = f"{float(val):.4f}"
            out["boltz_pic50_est"] = f"{6.0 - float(val):.4f}"
        if prob is not None:
            out["boltz_affinity_prob_binary"] = f"{float(prob):.4f}"
    conf = find_json(results_root, "confidence", name)
    if conf is not None:
        d = json.loads(conf.read_text())
        for src, dst in (("iptm", "boltz_iptm"), ("complex_iptm", "boltz_iptm"),
                         ("ptm", "boltz_ptm"), ("complex_ptm", "boltz_ptm")):
            if src in d and out[dst] == "":
                try:
                    out[dst] = f"{float(d[src]):.4f}"
                except (TypeError, ValueError):
                    pass
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--target-id", default=None,
                   help="Value to filter --hits-csv's target_id column by. "
                        "Defaults to --uniprot if not given.")
    add_sequence_args(p, required=False)
    p.add_argument("--hits-csv", required=True, help="input panel CSV")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--devices", type=int, default=1,
                   help="GPUs for data-parallel prediction (boltz --devices)")
    p.add_argument("--diffusion-samples", type=int, default=1)
    p.add_argument("--recycling-steps", type=int, default=None)
    p.add_argument("--cache", default=None, help="Boltz weight/cache dir (~/.boltz)")
    p.add_argument("--boltz-bin", default="boltz", help="boltz executable")
    msa = p.add_mutually_exclusive_group()
    msa.add_argument("--use-msa-server", action="store_true", default=True,
                     help="generate MSA via ColabFold server (default)")
    msa.add_argument("--single-sequence", action="store_true",
                     help="no MSA (msa: empty) — faster, less accurate")
    p.add_argument("--msa", default=None,
                   help="precomputed .a3m shared by every ligand (most efficient "
                        "for one target / many ligands)")
    p.add_argument("--override", action="store_true",
                   help="re-predict ligands even if outputs already exist")
    p.add_argument("--dry-run", action="store_true",
                   help="write YAMLs + manifest and print the boltz command; no GPU")
    args = p.parse_args()
    if not args.target_id:
        if not args.uniprot:
            sys.exit("ERROR: pass --target-id (or --uniprot, used as the default target-id)")
        args.target_id = args.uniprot
    return args


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_dir)
    inputs_dir = out_root / "boltz_inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_root / "summary.csv"
    manifest_path = out_root / "manifest.json"

    print(f"── Boltz-2 affinity · target {args.target_id} ──")
    print("\n── Phase 1: Loading panel ──")
    rows = load_hits(Path(args.hits_csv), args.target_id)

    print("\n── Phase 2: Target sequence ──")
    sequence = resolve_sequence(args.sequence, args.fasta, args.uniprot)
    gene_name = next((r.get("gene_name") for r in rows if r.get("gene_name")),
                     args.target_id)

    msa_ref = None
    if args.msa:
        msa_ref = str(Path(args.msa).resolve())

    print("\n── Phase 3: Writing Boltz inputs ──")
    manifest: dict[str, dict] = {}
    already = set()
    if summary_csv.exists() and not args.override:
        with summary_csv.open() as f:
            for r in csv.DictReader(f):
                already.add(r["molecule_id"])
    seen = set()
    n_written = 0
    for r in rows:
        mol_id = r["molecule_id"]
        if mol_id in seen:
            continue
        seen.add(mol_id)
        smiles = desalt_smiles(r["smiles"])
        name = safe_name(mol_id)
        manifest[name] = {
            "molecule_id": mol_id, "group": r.get("group", ""),
            "smiles": smiles, "predicted_score": r.get("predicted_score", ""),
            "rank": r.get("rank", ""),
        }
        if mol_id in already:
            continue
        results_dir = out_root / "boltz_results_boltz_inputs" / "predictions" / name
        if not args.override and find_json(results_dir, "affinity", name):
            continue
        write_boltz_yaml(inputs_dir / f"{name}.yaml", sequence, smiles,
                         msa_ref, args.single_sequence)
        n_written += 1
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  {n_written} YAML(s) written to {inputs_dir} "
          f"({len(already)} already done, {len(manifest)} total)")

    cmd = [args.boltz_bin, "predict", str(inputs_dir),
           "--out_dir", str(out_root),
           "--devices", str(args.devices),
           "--diffusion_samples", str(args.diffusion_samples),
           "--output_format", "pdb"]
    if args.recycling_steps is not None:
        cmd += ["--recycling_steps", str(args.recycling_steps)]
    if args.cache:
        cmd += ["--cache", args.cache]
    if args.msa is None and not args.single_sequence and args.use_msa_server:
        cmd += ["--use_msa_server"]
    if args.override:
        cmd += ["--override"]

    print("\n── Phase 4: Boltz-2 prediction ──")
    print("  " + " ".join(cmd))
    if args.dry_run:
        print("  [dry-run] skipping boltz execution")
    elif n_written == 0:
        print("  nothing new to predict — parsing existing outputs")
    else:
        if shutil.which(args.boltz_bin) is None:
            sys.exit(f"ERROR: '{args.boltz_bin}' not found on PATH — pip install boltz")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(f"ERROR: boltz exited {r.returncode}")

    print("\n── Phase 5: Parsing affinity outputs ──")
    results_root = out_root / "boltz_results_boltz_inputs"
    existing_rows: dict[str, dict] = {}
    if summary_csv.exists():
        with summary_csv.open() as f:
            for r in csv.DictReader(f):
                existing_rows[r["molecule_id"]] = r

    records = []
    for name, meta in manifest.items():
        mol_id = meta["molecule_id"]
        parsed = parse_affinity(results_root, name) if results_root.exists() else {}
        if not parsed.get("boltz_affinity_pred_value") and mol_id in existing_rows:
            prev = existing_rows[mol_id]
            parsed = {k: prev.get(k, "") for k in SUMMARY_FIELDS
                      if k.startswith("boltz_")}
        rec = {
            "target_id": args.target_id, "gene_name": gene_name,
            "molecule_id": mol_id, "smiles": meta["smiles"],
            "predicted_score": meta["predicted_score"], "rank": meta["rank"],
            "group": meta["group"],
        }
        rec.update({k: parsed.get(k, "") for k in SUMMARY_FIELDS if k.startswith("boltz_")})
        records.append(rec)

    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            w.writerow(rec)
    n_scored = sum(1 for r in records if r["boltz_affinity_pred_value"] != "")
    print(f"  wrote {summary_csv}  ({n_scored}/{len(records)} with affinity)")

    scored = [r for r in records if r["boltz_affinity_pred_value"] != ""]
    if scored:
        scored.sort(key=lambda r: float(r["boltz_affinity_pred_value"]))
        hdr = f"{'Mol':18} {'Group':16} {'aff(log10 µM)':>13} {'P(bind)':>8} {'pIC50':>7}"
        print("\n" + hdr)
        print("-" * len(hdr))
        for r in scored:
            print(f"{r['molecule_id']:18} {r['group']:16} "
                  f"{float(r['boltz_affinity_pred_value']):>13.3f} "
                  f"{float(r['boltz_affinity_prob_binary'] or 'nan'):>8.3f} "
                  f"{float(r['boltz_pic50_est']):>7.2f}")


if __name__ == "__main__":
    main()
