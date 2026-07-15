#!/usr/bin/env python3
"""
dock_compare_structures.py — Dock one ligand set into MULTIPLE receptor
structures (AlphaFold model vs experimental PDBs) and compare the results.

Motivation
----------
A co-folding model (e.g. Chai-1, via predict_structures.py) folds the ligand
together with the receptor and ignores any input coordinates, so it cannot
answer "does the receptor source change the docking result?". This script
uses rigid-receptor AutoDock Vina, where the receptor structure is a fixed
input, to dock the SAME ligands into:
  * an AlphaFold model (AF DB v4, by UniProt), and
  * one or more experimental structures (RCSB PDB IDs, with a co-crystal ligand)
and reports a per-(structure, ligand) ΔG matrix, the AF-vs-experimental rank
agreement, and the native-ligand redock RMSD (docking-setup validation).

Pocket definition (kept identical across structures)
----------------------------------------------------
For each experimental structure the box is centred on the centroid of its
co-crystal ligand, and the orthosteric "pocket residues" (receptor residues
with any atom within --pocket-cutoff Å of that ligand) are recorded. The AF box
is then centred on the centroid of those SAME residue numbers in the AF model —
so the only variable between structures is receptor conformation, not the pocket
location. (Experimental and AF receptors must share residue numbering for this
transfer to be valid — sanity-check the logged residue numbers/counts.)

Dependencies (CPU-only)
-------------------------
  pip install vina meeko rdkit requests
  apt/brew/conda install openbabel        # obabel CLI

Usage
-----
  python -m chai1_vina_toolkit.dock_compare_structures \
      --ligands  my_ligands.csv \
      --uniprot  <UNIPROT_ACCESSION> \
      --experimental <PDB_ID_1> <PDB_ID_2> \
      --native <PDB_ID_1>:<RESN> <PDB_ID_2>:<RESN> \
      --out-dir  out/

Notes
-----
* --native maps each experimental PDB to its co-crystal ligand 3-letter code for
  box centring / redock. If omitted the largest non-solvent HETATM residue is
  auto-selected (and logged); override if the auto-pick is a lipid/buffer.
* Resume-safe: cached receptor PDBQTs, ligand PDBQTs and docked poses are reused.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import requests
from rdkit import Chem
from rdkit.Chem import AllChem

AF_API    = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
RCSB_URL  = "https://files.rcsb.org/download/{pdb}.pdb"


def af_pdb_url(uniprot: str) -> str | None:
    """Resolve the current AlphaFold DB pdbUrl (model version isn't fixed across time)."""
    try:
        r = requests.get(AF_API.format(uniprot=uniprot), timeout=60)
        r.raise_for_status()
        return r.json()[0]["pdbUrl"]
    except Exception as e:
        print(f"    WARNING: AF API lookup failed for {uniprot} ({e})")
        return None

# HETATM residue codes that are never the ligand of interest (solvent/ions/lipids/sugars/cryo)
_NON_LIGAND = {
    "HOH", "WAT", "DOD", "NA", "CL", "K", "MG", "CA", "ZN", "SO4", "PO4", "GOL",
    "EDO", "PEG", "PG4", "PGE", "ACT", "FMT", "DMS", "MES", "TRS", "BOG", "OLC",
    "OLA", "OLB", "PLM", "STE", "CLR", "Y01", "HEM", "NAG", "BMA", "MAN", "BGC",
    "CIT", "EPE", "IOD", "BR", "FLC", "MLI", "1PE", "P6G", "LMT", "LDA",
}


# ── PDB fetch ──────────────────────────────────────────────────────────────────

def fetch(url: str, out_path: Path) -> Path | None:
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    print(f"    downloading {url}")
    try:
        r = requests.get(url, timeout=60)
    except Exception as e:
        print(f"    WARNING: download failed ({e})")
        return None
    if r.status_code != 200:
        print(f"    WARNING: HTTP {r.status_code} for {url}")
        return None
    out_path.write_bytes(r.content)
    return out_path


# ── PDB parsing (fixed-column ATOM/HETATM) ─────────────────────────────────────

def parse_atoms(pdb_path: Path) -> list[dict]:
    atoms: list[dict] = []
    for ln in pdb_path.read_text().splitlines():
        rec = ln[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        try:
            atoms.append({
                "record":  rec,
                "name":    ln[12:16].strip(),
                "altloc":  ln[16].strip(),
                "resname": ln[17:20].strip(),
                "chain":   ln[21].strip(),
                "resseq":  int(ln[22:26]),
                "x": float(ln[30:38]), "y": float(ln[38:46]), "z": float(ln[46:54]),
                "element": ln[76:78].strip() or ln[12:16].strip()[:1],
                "raw":     ln,
            })
        except (ValueError, IndexError):
            continue
    return atoms


def centroid(atoms: list[dict]) -> list[float]:
    n = len(atoms)
    return [sum(a[k] for a in atoms) / n for k in ("x", "y", "z")]


def pick_native_ligand(atoms: list[dict], forced_resn: str | None) -> tuple[str, list[dict]]:
    """Return (resname, ligand_atoms) for the co-crystal ligand."""
    hets: dict[tuple, list[dict]] = {}
    for a in atoms:
        if a["record"] != "HETATM" or a["resname"] in _NON_LIGAND:
            continue
        hets.setdefault((a["chain"], a["resseq"], a["resname"]), []).append(a)
    if forced_resn:
        match = [v for k, v in hets.items() if k[2] == forced_resn]
        if not match:
            sys.exit(f"ERROR: forced native ligand {forced_resn} not found among HETATM.")
        return forced_resn, max(match, key=len)
    if not hets:
        sys.exit("ERROR: no candidate co-crystal ligand HETATM found; pass --native PDB:RESN.")
    key, lig = max(hets.items(), key=lambda kv: len(kv[1]))
    return key[2], lig


def pocket_resseqs(receptor_atoms: list[dict], ligand_atoms: list[dict], cutoff: float) -> set[int]:
    c2 = cutoff * cutoff
    keep: set[int] = set()
    for ra in receptor_atoms:
        for la in ligand_atoms:
            if (ra["x"]-la["x"])**2 + (ra["y"]-la["y"])**2 + (ra["z"]-la["z"])**2 <= c2:
                keep.add(ra["resseq"])
                break
    return keep


# ── Receptor prep ──────────────────────────────────────────────────────────────

def write_protein_pdb(atoms: list[dict], out_path: Path, chain: str | None) -> None:
    """Write protein ATOM records (optionally one chain), dropping HETATM/waters & altloc B+."""
    lines = []
    for a in atoms:
        if a["record"] != "ATOM":
            continue
        if chain and a["chain"] != chain:
            continue
        if a["altloc"] not in ("", "A"):
            continue
        lines.append(a["raw"])
    lines.append("END")
    out_path.write_text("\n".join(lines) + "\n")


def receptor_to_pdbqt(protein_pdb: Path, pdbqt: Path) -> Path | None:
    if pdbqt.exists() and pdbqt.stat().st_size > 0:
        return pdbqt
    subprocess.run(
        ["obabel", str(protein_pdb), "-O", str(pdbqt), "-xr", "--partialcharge", "gasteiger"],
        capture_output=True,
    )
    if not pdbqt.exists() or pdbqt.stat().st_size == 0:
        print("    WARNING: obabel receptor prep failed")
        return None
    return pdbqt


# ── Ligand prep (RDKit ETKDG → meeko PDBQT) ────────────────────────────────────

def prepare_ligand(smiles: str, mol_id: str, lig_dir: Path) -> Path | None:
    pdbqt = lig_dir / f"{mol_id}.pdbqt"
    if pdbqt.exists() and pdbqt.stat().st_size > 0:
        return pdbqt
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
    except ImportError:
        sys.exit("ERROR: meeko not installed. Run: pip install meeko")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    setups = MoleculePreparation().prepare(mol)
    if not setups:
        return None
    pdbqt_str, ok, _ = PDBQTWriterLegacy.write_string(setups[0])
    if not ok:
        return None
    pdbqt.write_text(pdbqt_str)
    return pdbqt


# ── Docking (AutoDock Vina) ────────────────────────────────────────────────────

def dock(rec_pdbqt: Path, lig_pdbqt: Path, center: list[float], box: float,
         exhaustiveness: int, n_poses: int, out_pdbqt: Path) -> float | None:
    if out_pdbqt.exists() and out_pdbqt.stat().st_size > 0:
        for ln in out_pdbqt.read_text().splitlines():
            if ln.startswith("REMARK VINA RESULT"):
                return float(ln.split()[3])
        return None
    try:
        from vina import Vina
    except ImportError:
        sys.exit("ERROR: vina not installed. Run: pip install vina")
    v = Vina(sf_name="vina", verbosity=0)
    v.set_receptor(str(rec_pdbqt))
    v.set_ligand_from_file(str(lig_pdbqt))
    v.compute_vina_maps(center=center, box_size=[box] * 3)
    try:
        v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)
    except Exception as e:
        print(f"      docking failed: {e}")
        return None
    e = v.energies(n_poses=1)
    if e is None or len(e) == 0 or len(e[0]) == 0:
        return None
    v.write_poses(str(out_pdbqt), n_poses=1, overwrite=True)
    return float(e[0][0])


# ── Native redock RMSD (best-effort validation) ────────────────────────────────

def native_redock_rmsd(native_atoms: list[dict], native_smiles: str | None,
                       docked_pdbqt: Path) -> float | None:
    """Symmetry-corrected heavy-atom RMSD between crystal pose and docked pose."""
    if native_smiles is None or not docked_pdbqt.exists():
        return None
    try:
        block = "\n".join(a["raw"] for a in native_atoms) + "\nEND\n"
        cryst = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=True)
        tmpl  = Chem.MolFromSmiles(native_smiles)
        if cryst is None or tmpl is None:
            return None
        cryst = AllChem.AssignBondOrdersFromTemplate(tmpl, cryst)
        docked_pdb = docked_pdbqt.with_suffix(".pdb")
        subprocess.run(["obabel", str(docked_pdbqt), "-O", str(docked_pdb)], capture_output=True)
        dock = Chem.MolFromPDBFile(str(docked_pdb), sanitize=False, removeHs=True)
        if dock is None:
            return None
        dock = AllChem.AssignBondOrdersFromTemplate(tmpl, dock)
        return AllChem.GetBestRMS(Chem.RemoveHs(dock), Chem.RemoveHs(cryst))
    except Exception as e:
        print(f"      RMSD calc skipped: {e}")
        return None


# ── Rank correlation (Spearman, no scipy) ──────────────────────────────────────

def spearman(a: list[float], b: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        rk = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j+1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    ra, rb = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    n = len(pairs)
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
    return 1 - 6 * d2 / (n * (n * n - 1))


# ── Structure assembly ─────────────────────────────────────────────────────────

def build_structure(label: str, pdb_path: Path, work: Path, chain: str | None,
                    forced_resn: str | None, pocket_cutoff: float,
                    af_pocket_resseqs: set[int] | None):
    """Return dict(label, pdbqt, center, native_resn, native_atoms, pocket_resseqs)."""
    sdir = work / label
    sdir.mkdir(parents=True, exist_ok=True)
    atoms = parse_atoms(pdb_path)
    protein_pdb = sdir / "receptor.pdb"
    write_protein_pdb(atoms, protein_pdb, chain)
    pdbqt = receptor_to_pdbqt(protein_pdb, sdir / "receptor.pdbqt")

    native_resn, native_atoms, prs = None, None, None
    if af_pocket_resseqs is None:
        native_resn, native_atoms = pick_native_ligand(atoms, forced_resn)
        center = centroid(native_atoms)
        rec_atoms = [a for a in atoms if a["record"] == "ATOM" and (not chain or a["chain"] == chain)]
        prs = pocket_resseqs(rec_atoms, native_atoms, pocket_cutoff)
        print(f"    {label}: native ligand={native_resn} ({len(native_atoms)} atoms), "
              f"box centre=({center[0]:.1f},{center[1]:.1f},{center[2]:.1f}), "
              f"{len(prs)} pocket residues")
    else:
        pa = [a for a in atoms if a["record"] == "ATOM" and a["resseq"] in af_pocket_resseqs]
        if not pa:
            sys.exit(f"ERROR: none of the pocket residues {sorted(af_pocket_resseqs)} found in {label}.")
        center = centroid(pa)
        print(f"    {label}: box centre from {len(af_pocket_resseqs)} transferred pocket residues "
              f"=({center[0]:.1f},{center[1]:.1f},{center[2]:.1f})")
    return {"label": label, "pdbqt": pdbqt, "center": center,
            "native_resn": native_resn, "native_atoms": native_atoms, "pocket_resseqs": prs}


# ── Main ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ligands", required=True, help="CSV with molecule_id,smiles[,native_smiles].")
    p.add_argument("--uniprot", required=True, help="UniProt accession for the AlphaFold model.")
    p.add_argument("--experimental", nargs="+", required=True,
                   help="Experimental PDB IDs to dock into.")
    p.add_argument("--native", nargs="*", default=[],
                   help="PDB:RESN overrides for co-crystal ligand identity, e.g. 1ABC:LIG.")
    p.add_argument("--native-smiles", nargs="*", default=[],
                   help="PDB:SMILES for native-ligand redock RMSD, e.g. 1ABC:'C...'.")
    p.add_argument("--chain", default=None, help="Receptor chain to keep (default: all).")
    p.add_argument("--pocket-cutoff", type=float, default=5.0)
    p.add_argument("--box-size", type=float, default=22.0)
    p.add_argument("--exhaustiveness", type=int, default=16)
    p.add_argument("--n-poses", type=int, default=9)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    struct_dir = out / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)
    forced = dict(kv.split(":", 1) for kv in args.native)
    nat_smiles = dict(kv.split(":", 1) for kv in args.native_smiles)

    ligs: list[dict] = []
    with open(args.ligands) as f:
        for r in csv.DictReader(f):
            if Chem.MolFromSmiles(r["smiles"]) is None:
                print(f"  WARNING: skipping invalid SMILES {r['molecule_id']}")
                continue
            ligs.append(r)
    print(f"Ligands: {len(ligs)} valid")

    print("\n── Building experimental structures ──")
    structures = []
    primary_pocket: set[int] | None = None
    for pdb_id in args.experimental:
        path = fetch(RCSB_URL.format(pdb=pdb_id), struct_dir / f"{pdb_id}.pdb")
        if path is None:
            continue
        s = build_structure(pdb_id, path, struct_dir, args.chain,
                            forced.get(pdb_id), args.pocket_cutoff, None)
        structures.append(s)
        if primary_pocket is None:
            primary_pocket = s["pocket_resseqs"]

    print("\n── Building AlphaFold model ──")
    af_dest = struct_dir / f"AF_{args.uniprot}.pdb"
    af_path = af_dest if (af_dest.exists() and af_dest.stat().st_size > 0) else None
    if af_path is None:
        url = af_pdb_url(args.uniprot)
        af_path = fetch(url, af_dest) if url else None
    if af_path and primary_pocket:
        structures.insert(0, build_structure(f"AF_{args.uniprot}", af_path, struct_dir,
                                             "A", None, args.pocket_cutoff, primary_pocket))

    if not structures:
        sys.exit("ERROR: no receptor structures could be built.")

    print("\n── Docking ──")
    matrix: dict[str, dict[str, float | None]] = {}
    for s in structures:
        if s["pdbqt"] is None:
            continue
        lig_dir = out / s["label"] / "ligands"
        lig_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [{s['label']}]")
        for lg in ligs:
            mid = lg["molecule_id"]
            lp = prepare_ligand(lg["smiles"], mid, lig_dir)
            if lp is None:
                matrix.setdefault(mid, {})[s["label"]] = None
                continue
            dg = dock(s["pdbqt"], lp, s["center"], args.box_size,
                      args.exhaustiveness, args.n_poses,
                      out / s["label"] / f"{mid}_docked.pdbqt")
            matrix.setdefault(mid, {})[s["label"]] = dg
            print(f"    {mid:<22} ΔG={dg if dg is None else round(dg,2)}")

    labels = [s["label"] for s in structures if s["pdbqt"]]
    mtx_csv = out / "dock_matrix.csv"
    with mtx_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["molecule_id", "smiles"] + [f"dG_{l}" for l in labels])
        for lg in ligs:
            mid = lg["molecule_id"]
            w.writerow([mid, lg["smiles"]] + [matrix.get(mid, {}).get(l) for l in labels])
    print(f"\nWrote {mtx_csv}")

    rmsd_rows = []
    for s in structures:
        if s["native_atoms"] is None:
            continue
        sm = nat_smiles.get(s["label"])
        docked = out / s["label"] / f"NATIVE_{s['native_resn']}_docked.pdbqt"
        if sm and s["pdbqt"]:
            lp = prepare_ligand(sm, f"NATIVE_{s['native_resn']}", out / s["label"] / "ligands")
            if lp:
                dock(s["pdbqt"], lp, s["center"], args.box_size,
                     args.exhaustiveness, args.n_poses, docked)
        rmsd = native_redock_rmsd(s["native_atoms"], sm, docked)
        rmsd_rows.append((s["label"], s["native_resn"], rmsd))
        print(f"  redock RMSD [{s['label']} / {s['native_resn']}]: "
              f"{'NA' if rmsd is None else round(rmsd,2)} Å")
    if rmsd_rows:
        with (out / "redock_rmsd.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["structure", "native_ligand", "rmsd_angstrom"])
            w.writerows(rmsd_rows)

    af_label = next((l for l in labels if l.startswith("AF_")), None)
    if af_label:
        af_vals = [matrix.get(lg["molecule_id"], {}).get(af_label) for lg in ligs]
        print("\n── AF-vs-experimental Spearman ρ (ΔG ranking) ──")
        for l in labels:
            if l == af_label:
                continue
            ev = [matrix.get(lg["molecule_id"], {}).get(l) for lg in ligs]
            rho = spearman(af_vals, ev)
            print(f"  {af_label} vs {l}: ρ = {'NA' if rho is None else round(rho,3)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
