#!/usr/bin/env python3
"""
dock_triage.py — Docking-based triage of an external model's predicted hits.

Takes a CSV of (target, ligand, model_score) rows from any upstream scoring
model, downloads AlphaFold structures, detects binding pockets with fpocket,
runs full AutoDock Vina docking on the top-K ligands per target, and flags
hits where the upstream model score and Vina ΔG are concordant.

Rationale: an upstream sequence/LM-based scorer and Vina are independent
models (statistical vs physics-based). Concordant hits are much harder to
dismiss as model artefacts than either signal alone. Only concordant hits are
carried forward.

Phases
------
  1  Load top-N targets + top-K ligands from --hits-csv
  2  Download AlphaFold structures (AF DB v4; skip if already cached)
  3  Prepare receptors  (obabel: PDB → PDBQT, strip HETATM, add H)
  4  Detect binding pockets (fpocket; select top pocket by druggability)
  5  Prepare ligands  (RDKit ETKDG 3D → meeko PDBQT)
  6  Dock  (AutoDock Vina, exhaustiveness 8, 9 poses)
  7  Concordance filter + report

Input CSV schema
-----------------
  uniprot, molecule_id, smiles, model_score[, gene_name, protein_name]

  `uniprot` must be a real UniProt accession here (used to fetch the
  AlphaFold structure). `model_score`: lower = predicted stronger binder.

Outputs (--out-dir)
--------------------
  dock_results.csv          all (target, ligand, model_score, vina_dg) rows
  dock_triage_report.txt    human-readable ranked concordant hits
  dock_concordance.html     interactive scatter: model score vs Vina ΔG

Dependencies
------------
  pip install vina meeko rdkit plotly requests
  conda install -c conda-forge fpocket      # or: brew install fpocket
  apt/brew install openbabel                # obabel CLI

Usage
-----
  python -m chai1_vina_toolkit.dock_triage --hits-csv hits.csv --out-dir out/
  python -m chai1_vina_toolkit.dock_triage --hits-csv hits.csv --out-dir out/ \
      --top-targets 5 --top-ligands 20
  python -m chai1_vina_toolkit.dock_triage --out-dir out/ --skip-docking
  python -m chai1_vina_toolkit.dock_triage --hits-csv hits.csv --out-dir out/ \
      --target P46095 --top-ligands 50
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import requests
from rdkit import Chem
from rdkit.Chem import AllChem
import plotly.graph_objects as go

AF_URL = "https://alphafold.ebi.ac.uk/files/AF-{uniprot}-F1-model_v4.pdb"

DOCK_RESULT_FIELDS = [
    "uniprot", "gene_name", "protein_name",
    "molecule_id", "smiles", "model_score", "vina_dg",
    "pocket_score", "concordant",
]


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])

    g = p.add_argument_group("scope")
    g.add_argument("--hits-csv", default=None, help="CSV with the schema in the module docstring.")
    g.add_argument("--out-dir", required=True, help="Output directory.")
    g.add_argument("--top-targets", type=int, default=10,
                   help="Number of top targets to triage (by best model_score).")
    g.add_argument("--top-ligands", type=int, default=50,
                   help="Number of top ligands per target to dock.")
    g.add_argument("--target", default=None,
                   help="Dock a single target UniProt accession (overrides --top-targets).")

    g = p.add_argument_group("docking")
    g.add_argument("--exhaustiveness", type=int, default=8,
                   help="Vina exhaustiveness (default 8; raise to 16 for final hits).")
    g.add_argument("--n-poses", type=int, default=9)
    g.add_argument("--box-size", type=float, default=25.0,
                   help="Docking box edge length in Å (cubic).")
    g.add_argument("--vina-threshold", type=float, default=-7.0,
                   help="Vina ΔG cutoff for concordance (kcal/mol; more negative = tighter).")
    g.add_argument("--model-score-threshold", type=float, default=500.0,
                   help="Upstream model score cutoff for concordance (lower = stronger binder).")

    p.add_argument("--skip-docking", action="store_true",
                   help="Load existing dock_results.csv and regenerate reports only.")
    args = p.parse_args()
    if not args.skip_docking and not args.hits_csv:
        sys.exit("ERROR: --hits-csv is required unless --skip-docking is set.")
    return args


# ── Phase 1: Load targets + ligands ────────────────────────────────────────────

def load_hits(
    csv_path: Path,
    top_targets: int,
    top_ligands: int,
    target_filter: str | None,
) -> dict[str, list[dict]]:
    """Return {uniprot: [rows]} for the top-N targets."""
    all_rows: list[dict] = []
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            r["model_score"] = float(r["model_score"])
            all_rows.append(r)

    if not all_rows:
        sys.exit(f"ERROR: {csv_path} is empty.")

    if target_filter:
        all_rows = [r for r in all_rows if r["uniprot"] == target_filter]
        if not all_rows:
            sys.exit(f"ERROR: target {target_filter} not found in {csv_path.name}.")

    best: dict[str, float] = {}
    for r in all_rows:
        u = r["uniprot"]
        best[u] = min(best.get(u, float("inf")), r["model_score"])

    top_uniprots = sorted(best, key=lambda u: best[u])
    if not target_filter:
        top_uniprots = top_uniprots[:top_targets]

    bucket: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        if r["uniprot"] in top_uniprots:
            bucket[r["uniprot"]].append(r)

    result: dict[str, list[dict]] = {}
    for u in top_uniprots:
        rows = sorted(bucket[u], key=lambda r: r["model_score"])[:top_ligands]
        result[u] = rows
        print(f"  {u}  {rows[0].get('gene_name', '—'):10s}  "
              f"best={rows[0]['model_score']:.0f}  n={len(rows)}")

    return result


# ── Phase 2: AlphaFold structure download ──────────────────────────────────────

def fetch_alphafold_structure(uniprot: str, out_dir: Path) -> Path | None:
    pdb_path = out_dir / f"{uniprot}_af.pdb"
    if pdb_path.exists() and pdb_path.stat().st_size > 0:
        return pdb_path

    url = AF_URL.format(uniprot=uniprot)
    print(f"    Downloading AlphaFold structure: {url}")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            pdb_path.write_bytes(r.content)
            print(f"    Saved: {pdb_path.name} ({pdb_path.stat().st_size // 1024} KB)")
            return pdb_path
        print(f"    WARNING: AlphaFold returned {r.status_code} for {uniprot}; skipping.")
        return None
    except Exception as e:
        print(f"    WARNING: AlphaFold download failed ({e}); skipping {uniprot}.")
        return None


# ── Phase 3: Receptor prep (obabel PDB → PDBQT) ───────────────────────────────

def _strip_hetatm(pdb_path: Path, out_path: Path) -> None:
    lines = [
        ln for ln in pdb_path.read_text().splitlines(keepends=True)
        if ln.startswith(("ATOM", "TER", "END"))
    ]
    out_path.write_text("".join(lines))


def prepare_receptor(pdb_path: Path, rec_dir: Path) -> Path | None:
    stripped = rec_dir / "receptor_stripped.pdb"
    pdbqt    = rec_dir / "receptor.pdbqt"

    if pdbqt.exists() and pdbqt.stat().st_size > 0:
        return pdbqt

    _strip_hetatm(pdb_path, stripped)

    r = subprocess.run(
        ["obabel", str(stripped), "-O", str(pdbqt), "-xr", "--partialcharge", "gasteiger"],
        capture_output=True,
    )
    if not pdbqt.exists() or pdbqt.stat().st_size == 0:
        print(f"    WARNING: obabel receptor prep failed: {r.stderr.decode().strip()}")
        return None
    return pdbqt


# ── Phase 4: Binding pocket detection (fpocket) ───────────────────────────────

def detect_pocket(pdb_path: Path, rec_dir: Path) -> tuple[list[float], float] | None:
    """Run fpocket and return (center_xyz, druggability_score) of top pocket."""
    pocket_cache = rec_dir / "pocket_center.txt"
    if pocket_cache.exists():
        parts = pocket_cache.read_text().split()
        return [float(parts[0]), float(parts[1]), float(parts[2])], float(parts[3])

    fpocket_out = rec_dir / "fpocket_out"
    if fpocket_out.exists():
        shutil.rmtree(fpocket_out)

    r = subprocess.run(
        ["fpocket", "-f", str(pdb_path), "-o", str(fpocket_out)],
        capture_output=True,
    )
    if r.returncode != 0:
        print(f"    WARNING: fpocket failed: {r.stderr.decode().strip()[:200]}")
        return None

    info_file = fpocket_out / f"{pdb_path.stem}_info.txt"
    if not info_file.exists():
        matches = list(fpocket_out.rglob("*_info.txt"))
        if not matches:
            print("    WARNING: fpocket info file not found.")
            return None
        info_file = matches[0]

    best_score = -1.0
    best_center: list[float] = []
    pocket_block: list[str] = []

    for line in info_file.read_text().splitlines():
        if line.strip().startswith("Pocket"):
            pocket_block = []
        pocket_block.append(line)
        if "Druggability Score" in line:
            m = re.search(r":\s*([\d.]+)", line)
            if m and float(m.group(1)) > best_score:
                best_score = float(m.group(1))
                cx = cy = cz = None
                for bl in pocket_block:
                    if "x_barycenter" in bl.lower():
                        cx = float(re.search(r":\s*([\d.-]+)", bl).group(1))
                    if "y_barycenter" in bl.lower():
                        cy = float(re.search(r":\s*([\d.-]+)", bl).group(1))
                    if "z_barycenter" in bl.lower():
                        cz = float(re.search(r":\s*([\d.-]+)", bl).group(1))
                if cx is not None:
                    best_center = [cx, cy, cz]

    if not best_center:
        print("    WARNING: could not parse pocket center from fpocket output.")
        return None

    pocket_cache.write_text(f"{best_center[0]} {best_center[1]} {best_center[2]} {best_score}")
    print(f"    Pocket center: ({best_center[0]:.1f}, {best_center[1]:.1f}, "
          f"{best_center[2]:.1f})  druggability={best_score:.2f}")
    return best_center, best_score


# ── Phase 5: Ligand prep (RDKit ETKDG → meeko PDBQT) ─────────────────────────

def prepare_ligand(smiles: str, mol_id: str, lig_dir: Path) -> Path | None:
    pdbqt_path = lig_dir / f"{mol_id}.pdbqt"
    if pdbqt_path.exists() and pdbqt_path.stat().st_size > 0:
        return pdbqt_path

    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except ImportError:
        sys.exit("ERROR: meeko not installed. Run: pip install meeko")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)

    preparator = MoleculePreparation()
    mol_setup_list = preparator.prepare(mol)
    if not mol_setup_list:
        return None
    pdbqt_str, ok, err = PDBQTWriterLegacy.write_string(mol_setup_list[0])
    if not ok:
        return None

    pdbqt_path.write_text(pdbqt_str)
    return pdbqt_path


# ── Phase 6: Docking (AutoDock Vina) ─────────────────────────────────────────

def dock_ligand(
    rec_pdbqt: Path,
    lig_pdbqt: Path,
    center: list[float],
    box_size: float,
    exhaustiveness: int,
    n_poses: int,
    out_pdbqt: Path,
) -> float | None:
    """Return best Vina ΔG (kcal/mol) or None on failure."""
    if out_pdbqt.exists() and out_pdbqt.stat().st_size > 0:
        for line in out_pdbqt.read_text().splitlines():
            if line.startswith("REMARK VINA RESULT"):
                return float(line.split()[3])
        return None

    try:
        from vina import Vina
    except ImportError:
        sys.exit("ERROR: vina not installed. Run: pip install vina")

    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(str(rec_pdbqt))
    v.set_ligand_from_file(str(lig_pdbqt))
    v.compute_vina_maps(center=center, box_size=[box_size] * 3)

    try:
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    except Exception as e:
        print(f"      Docking failed: {e}")
        return None

    energies = v.energies(n_poses=1)
    if not energies or not energies[0]:
        return None
    best_dg = float(energies[0][0])
    v.write_poses(str(out_pdbqt), n_poses=1, overwrite=True)
    return best_dg


# ── Phase 7: Concordance filter + reports ─────────────────────────────────────

def write_triage_report(
    rows: list[dict],
    vina_thr: float,
    model_thr: float,
    out_path: Path,
) -> None:
    concordant = [r for r in rows if r["concordant"] == "1"]
    lines = [
        "# Docking triage: concordant hits",
        f"# Concordance criteria: Vina ΔG <= {vina_thr} kcal/mol  AND  "
        f"model_score <= {model_thr}",
        f"# {len(concordant)} concordant hits from "
        f"{len({r['uniprot'] for r in concordant})} targets "
        f"({len(rows)} total docked)",
        "",
    ]

    by_target: dict[str, list[dict]] = defaultdict(list)
    for r in concordant:
        by_target[r["uniprot"]].append(r)

    for u in sorted(by_target, key=lambda u: float(by_target[u][0]["vina_dg"])):
        hits = sorted(by_target[u], key=lambda r: float(r["vina_dg"]))
        meta = hits[0]
        lines.append(f"## {u}  {meta['gene_name']}  —  {meta['protein_name']}")
        lines.append(f"   {'Mol ID':<15} {'model_score':>11} {'Vina ΔG':>10}")
        lines.append("   " + "-" * 40)
        for h in hits:
            lines.append(
                f"   {h['molecule_id']:<15} "
                f"{float(h['model_score']):>11.0f} "
                f"{float(h['vina_dg']):>9.2f} kcal/mol"
            )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n")
    print(f"  Wrote: {out_path}")


def write_concordance_html(rows: list[dict], out_path: Path) -> None:
    concordant = [r for r in rows if r["concordant"] == "1"]
    other      = [r for r in rows if r["concordant"] != "1"]

    def trace(subset: list[dict], name: str, color: str) -> go.Scatter:
        return go.Scatter(
            x=[float(r["model_score"]) for r in subset],
            y=[float(r["vina_dg"])     for r in subset],
            mode="markers",
            name=name,
            marker=dict(color=color, size=7, opacity=0.7,
                        line=dict(width=0.5, color="white")),
            customdata=[
                (f"<b>{r['gene_name'] or r['uniprot']}</b> ({r['uniprot']})<br>"
                 f"{r['molecule_id']}<br>"
                 f"model score: {float(r['model_score']):.0f}<br>"
                 f"Vina ΔG: {float(r['vina_dg']):.2f} kcal/mol")
                for r in subset
            ],
            hovertemplate="%{customdata}<extra></extra>",
        )

    fig = go.Figure()
    fig.add_trace(trace(other,      "Not concordant", "#aec6e8"))
    fig.add_trace(trace(concordant, "Concordant",     "#e74c3c"))
    fig.update_layout(
        title=dict(
            text=(
                "Docking concordance: upstream model score vs Vina ΔG<br>"
                "<sup>red = concordant (both thresholds passed) · "
                "lower model score = stronger predicted binder · "
                "more negative Vina ΔG = stronger docked binder</sup>"
            ),
            x=0.5,
        ),
        xaxis_title="Upstream model score (lower = stronger binder)",
        yaxis_title="Vina ΔG (kcal/mol, more negative = stronger binder)",
        height=650, width=1100,
        template="plotly_white",
    )
    fig.write_html(str(out_path))
    print(f"  Report: {out_path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dock_csv = out_dir / "dock_results.csv"
    report_txt = out_dir / "dock_triage_report.txt"
    concord_html = out_dir / "dock_concordance.html"

    if args.skip_docking:
        if not dock_csv.exists():
            sys.exit(f"--skip-docking set but {dock_csv} not found; run without flag first.")
        print(f"Loading existing results from {dock_csv.name}…")
        rows: list[dict] = []
        with dock_csv.open() as f:
            for r in csv.DictReader(f):
                rows.append(r)
        print(f"  {len(rows)} rows")
        write_triage_report(rows, args.vina_threshold, args.model_score_threshold, report_txt)
        write_concordance_html(rows, concord_html)
        print("\nDone.")
        return

    print("── Phase 1: Loading top hits ──")
    hits_csv = Path(args.hits_csv)
    if not hits_csv.exists():
        sys.exit(f"ERROR: {hits_csv} not found.")
    bucket = load_hits(hits_csv, args.top_targets, args.top_ligands, args.target)
    total_pairs = sum(len(v) for v in bucket.values())
    print(f"  {len(bucket)} targets · {total_pairs} target-ligand pairs to dock")

    all_results: list[dict] = []

    for rec_idx, (uniprot, lig_rows) in enumerate(bucket.items(), 1):
        gene = lig_rows[0].get("gene_name", "")
        protein = lig_rows[0].get("protein_name", "")
        print(f"\n── [{rec_idx}/{len(bucket)}] {uniprot}  {gene or '—'} ──")
        rec_dir = out_dir / uniprot
        rec_dir.mkdir(exist_ok=True)
        lig_dir = rec_dir / "ligands"
        lig_dir.mkdir(exist_ok=True)

        print("  Phase 2: AlphaFold structure")
        pdb_path = fetch_alphafold_structure(uniprot, rec_dir)
        if pdb_path is None:
            print(f"  Skipping {uniprot} — no AlphaFold structure available.")
            continue

        print("  Phase 3: Receptor prep (obabel)")
        rec_pdbqt = prepare_receptor(pdb_path, rec_dir)
        if rec_pdbqt is None:
            print(f"  Skipping {uniprot} — receptor prep failed.")
            continue

        print("  Phase 4: fpocket")
        pocket_result = detect_pocket(pdb_path, rec_dir)
        if pocket_result is None:
            print(f"  Skipping {uniprot} — pocket detection failed.")
            continue
        pocket_center, pocket_score = pocket_result

        n_docked = n_failed = 0
        for lig in lig_rows:
            mol_id = lig.get("molecule_id", "")
            smiles = lig.get("smiles", "")
            model_score = float(lig["model_score"])

            lig_pdbqt = prepare_ligand(smiles, mol_id, lig_dir)
            if lig_pdbqt is None:
                n_failed += 1
                continue

            out_pdbqt = rec_dir / f"{mol_id}_docked.pdbqt"
            dg = dock_ligand(
                rec_pdbqt, lig_pdbqt,
                pocket_center, args.box_size,
                args.exhaustiveness, args.n_poses,
                out_pdbqt,
            )
            if dg is None:
                n_failed += 1
                continue

            concordant = (
                "1" if dg <= args.vina_threshold and model_score <= args.model_score_threshold
                else "0"
            )
            all_results.append({
                "uniprot":      uniprot,
                "gene_name":    gene,
                "protein_name": protein,
                "molecule_id":  mol_id,
                "smiles":       smiles,
                "model_score":  model_score,
                "vina_dg":      round(dg, 3),
                "pocket_score": round(pocket_score, 3),
                "concordant":   concordant,
            })
            n_docked += 1

        n_conc = sum(1 for r in all_results
                     if r["uniprot"] == uniprot and r["concordant"] == "1")
        print(f"  Docked: {n_docked}  failed prep: {n_failed}  concordant: {n_conc}")

    if not all_results:
        sys.exit("No docking results to report.")

    with dock_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DOCK_RESULT_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_results)
    print(f"\n  Saved: {dock_csv}")

    print("\n── Phase 7: Concordance reports ──")
    n_conc = sum(1 for r in all_results if r["concordant"] == "1")
    print(f"  Concordant hits: {n_conc} / {len(all_results)}")
    write_triage_report(all_results, args.vina_threshold, args.model_score_threshold, report_txt)
    write_concordance_html(all_results, concord_html)
    print("\nDone.")


if __name__ == "__main__":
    main()
