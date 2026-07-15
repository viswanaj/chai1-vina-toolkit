#!/usr/bin/env python3
"""
pose_recovery.py — validate a co-folding model (e.g. Chai-1) against known
experimental (crystal / cryo-EM) ligand poses.

Question this answers
----------------------
Given only sequence + SMILES, does the co-folding model place a known ligand
where the crystallographers found it? This is the standard way to separate
"the model is bad at this pocket" from "there's no information to recover" —
run it on a target with real co-crystal ground truth before trusting the
model's output on a target that has none.

Method (all in a single, receptor-superposed coordinate frame)
----------------------------------------------------------------
For each predicted pose we TM-align the *predicted receptor* Cα trace onto the
*experimental receptor* Cα trace, then apply that same rigid transform to the
predicted ligand. In the experimental frame we then measure, against the crystal:

  rmsd          : symmetry-corrected heavy-atom RMSD to the crystal ligand
                  (no further ligand fitting — pure pose recovery). < 2 Å is a
                  common "recapitulated" threshold.
  centroid      : distance between predicted- and crystal-ligand centroids
                  (robust: needs no atom correspondence).
  pocket_jaccard: Jaccard overlap of the EXPERIMENTAL receptor residues contacted
                  (<= cutoff) by the crystal ligand vs. by the aligned predicted
                  ligand. Measured entirely on the experimental structure, so no
                  cross-numbering between predicted/experimental is needed.

Reference mapping
------------------
Pass --ref-map pointing at a CSV with columns:
  molecule_id, experimental_pdb, ligand_resname, chain
one row per (ligand, ground-truth structure) pair you want to validate.

Dependencies
------------
  pip install gemmi rdkit numpy tmtools

Usage
-----
  python -m chai1_vina_toolkit.pose_recovery \
      --predicted-dir  my_predictions/ \
      --exp-dir        my_experimental_structures/ \
      --ref-map        ref_map.csv \
      --out-csv        pose_recovery.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import gemmi
import numpy as np
from rdkit import Chem

AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}
WATER = {"HOH", "WAT", "DOD"}
IONS = {"NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "CU", "NI", "CO",
        "BR", "IOD", "SO4", "PO4"}
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "C", "PYL": "K",
}


def _heavy(atom) -> bool:
    return atom.element.name not in ("H", "D")


def _first_altloc(residue):
    """Yield one atom per atom-name (first altloc), heavy atoms only."""
    seen = set()
    for a in residue:
        if not _heavy(a):
            continue
        nm = a.name
        if nm in seen:
            continue
        seen.add(nm)
        yield a


def parse_experimental(pdb_path: Path, lig_resn: str, chain_id: str):
    """Return dict with receptor Cα (N,3)+seq, receptor heavy atoms per residue,
    and native-ligand (elements, coords)."""
    st = gemmi.read_structure(str(pdb_path))
    model = st[0]
    ca_xyz, seq = [], []
    prot_res = {}
    lig_elems, lig_xyz = [], []
    for chain in model:
        if chain.name != chain_id:
            continue
        for res in chain:
            name = res.name.strip().upper()
            if name in AA3:
                key = (chain.name, res.seqid.num)
                coords = [[a.pos.x, a.pos.y, a.pos.z] for a in _first_altloc(res)]
                if not coords:
                    continue
                prot_res[key] = np.asarray(coords, float)
                ca = next((a for a in res if a.name == "CA"), None)
                if ca is not None:
                    ca_xyz.append([ca.pos.x, ca.pos.y, ca.pos.z])
                    seq.append(THREE_TO_ONE.get(name, "X"))
            elif name == lig_resn:
                for a in _first_altloc(res):
                    lig_elems.append(a.element.name)
                    lig_xyz.append([a.pos.x, a.pos.y, a.pos.z])
    return {
        "ca": np.asarray(ca_xyz, float),
        "seq": "".join(seq),
        "prot_res": prot_res,
        "lig_elems": lig_elems,
        "lig_xyz": np.asarray(lig_xyz, float),
    }


def parse_predicted(cif_path: Path):
    """Return receptor Cα (N,3)+seq and ligand (elements, coords) from a
    predicted-structure CIF (e.g. a Chai-1 output)."""
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    model = st[0]
    ca_xyz, seq = [], []
    lig_elems, lig_xyz = [], []
    for chain in model:
        for res in chain:
            name = res.name.strip().upper()
            if name in AA3:
                ca = next((a for a in res if a.name == "CA"), None)
                if ca is not None:
                    ca_xyz.append([ca.pos.x, ca.pos.y, ca.pos.z])
                    seq.append(THREE_TO_ONE.get(name, "X"))
            elif name in WATER or name in IONS:
                continue
            else:
                for a in res:
                    if _heavy(a):
                        lig_elems.append(a.element.name)
                        lig_xyz.append([a.pos.x, a.pos.y, a.pos.z])
    return {
        "ca": np.asarray(ca_xyz, float),
        "seq": "".join(seq),
        "lig_elems": lig_elems,
        "lig_xyz": np.asarray(lig_xyz, float),
    }


def superpose(pred_ca, pred_seq, exp_ca, exp_seq):
    """TM-align predicted receptor onto experimental. Return (R, t) such that
    x_in_exp_frame = x_pred @ R.T + t, plus the TM-score (norm on experimental)."""
    from tmtools import tm_align
    res = tm_align(pred_ca, exp_ca, pred_seq, exp_seq)
    return np.asarray(res.u, float), np.asarray(res.t, float), float(res.tm_norm_chain2)


def apply_transform(xyz, R, t):
    return xyz @ R.T + t


def _pdb_block(elems, xyz):
    lines = []
    for i, (el, (x, y, z)) in enumerate(zip(elems, xyz), start=1):
        el = el.capitalize()
        name = f"{el}{i}"[:4]
        lines.append(
            f"HETATM{i:>5d} {name:<4s} LIG A   1    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}"
        )
    lines.append("END")
    return "\n".join(lines)


def symmetric_rmsd(exp_elems, exp_xyz, pred_elems, pred_xyz):
    """Symmetry-corrected heavy-atom RMSD in the (already receptor-aligned) frame.

    No rigid fitting: atom labels are permuted via graph automorphism so
    chemically-equivalent atoms are matched optimally, using proximity-bonded
    RDKit graphs (no bond-order perception required). Falls back to greedy
    element-matched nearest-neighbour if the graphs don't match atom-for-atom.
    """
    exp_mol = Chem.MolFromPDBBlock(_pdb_block(exp_elems, exp_xyz),
                                   removeHs=False, sanitize=False, proximityBonding=True)
    pred_mol = Chem.MolFromPDBBlock(_pdb_block(pred_elems, pred_xyz),
                                    removeHs=False, sanitize=False, proximityBonding=True)
    exp_xyz = np.asarray(exp_xyz, float)
    pred_xyz = np.asarray(pred_xyz, float)
    method = "graph"
    if exp_mol is not None and pred_mol is not None \
            and exp_mol.GetNumAtoms() == pred_mol.GetNumAtoms():
        matches = pred_mol.GetSubstructMatches(exp_mol, uniquify=False, maxMatches=100000)
        if matches:
            best = min(
                float(np.sqrt(np.mean(np.sum(
                    (exp_xyz - pred_xyz[list(m)]) ** 2, axis=1))))
                for m in matches
            )
            return best, method, len(matches)
    method = "greedy"
    used = set()
    sq = []
    for el, p in zip(exp_elems, exp_xyz):
        cands = [(j, q) for j, (e2, q) in enumerate(zip(pred_elems, pred_xyz))
                 if e2.capitalize() == el.capitalize() and j not in used]
        if not cands:
            continue
        j, q = min(cands, key=lambda c: float(np.sum((p - c[1]) ** 2)))
        used.add(j)
        sq.append(float(np.sum((p - q) ** 2)))
    rmsd = float(np.sqrt(np.mean(sq))) if sq else float("nan")
    return rmsd, method, 1


def pocket_residues(prot_res, lig_xyz, cutoff):
    if len(lig_xyz) == 0:
        return set()
    c2 = cutoff * cutoff
    out = set()
    for key, xyz in prot_res.items():
        d2 = ((xyz[:, None, :] - lig_xyz[None, :, :]) ** 2).sum(-1)
        if d2.min() <= c2:
            out.add(key)
    return out


def jaccard(a, b):
    if not a and not b:
        return float("nan")
    u = len(a | b)
    return len(a & b) / u if u else float("nan")


def load_scores(summary_csv):
    scores = {}
    if summary_csv and Path(summary_csv).exists():
        with open(summary_csv) as f:
            for row in csv.DictReader(f):
                mid = row.get("molecule_id")
                if mid:
                    scores[mid] = (row.get("iptm", ""), row.get("ptm", ""))
    return scores


def load_ref_map(path: Path) -> dict[str, tuple[str, str, str]]:
    """molecule_id -> (experimental_pdb, ligand_resname, chain)."""
    ref = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            ref[row["molecule_id"]] = (
                row["experimental_pdb"], row["ligand_resname"], row["chain"],
            )
    return ref


def find_predicted(predicted_dir: Path, molecule_id: str):
    """Locate pred*.cif for a molecule under predicted_dir (searched recursively;
    matches any subdirectory layout, e.g. group/molecule_id/pred.model_idx_*.cif)."""
    hits = sorted(predicted_dir.rglob(f"{molecule_id}/*.cif"))
    if not hits:
        hits = sorted(predicted_dir.rglob(f"*{molecule_id}*.cif"))
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--predicted-dir", required=True,
                    help="Directory containing predicted-structure CIFs.")
    ap.add_argument("--exp-dir", required=True,
                    help="Directory containing experimental PDB files.")
    ap.add_argument("--ref-map", required=True,
                    help="CSV: molecule_id,experimental_pdb,ligand_resname,chain")
    ap.add_argument("--summary", default=None,
                    help="Optional summary.csv with molecule_id,iptm,ptm columns.")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--cutoff", type=float, default=4.5)
    args = ap.parse_args()

    predicted_dir = Path(args.predicted_dir)
    exp_dir = Path(args.exp_dir)
    ref = load_ref_map(Path(args.ref_map))
    scores = load_scores(args.summary)

    rows = []
    for mol, (pdb, resn, chain) in ref.items():
        exp = parse_experimental(exp_dir / pdb, resn, chain)
        if len(exp["lig_xyz"]) == 0:
            print(f"  WARN: no native ligand {resn} in {pdb}")
            continue
        exp_pocket = pocket_residues(exp["prot_res"], exp["lig_xyz"], args.cutoff)
        cifs = find_predicted(predicted_dir, mol)
        if not cifs:
            print(f"  WARN: no predicted structures for {mol} under {predicted_dir}")
            continue
        per_pose = []
        for cif in cifs:
            ch = parse_predicted(cif)
            if len(ch["lig_xyz"]) == 0:
                continue
            R, t, tm = superpose(ch["ca"], ch["seq"], exp["ca"], exp["seq"])
            lig_al = apply_transform(ch["lig_xyz"], R, t)
            rmsd, method, nmatch = symmetric_rmsd(
                exp["lig_elems"], exp["lig_xyz"], ch["lig_elems"], lig_al)
            cent = float(np.linalg.norm(lig_al.mean(0) - exp["lig_xyz"].mean(0)))
            pred_pocket = pocket_residues(exp["prot_res"], lig_al, args.cutoff)
            jac = jaccard(exp_pocket, pred_pocket)
            idx_parts = [p for p in cif.stem.split("_") if p.isdigit()]
            idx = int(idx_parts[-1]) if idx_parts else 0
            per_pose.append({"idx": idx, "rmsd": rmsd, "centroid": cent,
                             "pocket_jaccard": jac, "tm": tm,
                             "method": method, "n_lig_atoms": len(exp["lig_xyz"])})
        if not per_pose:
            print(f"  WARN: no ligand parsed in any pose for {mol}")
            continue
        per_pose.sort(key=lambda r: r["idx"])
        p0 = next((p for p in per_pose if p["idx"] == 0), per_pose[0])
        best = min(per_pose, key=lambda r: r["rmsd"])
        ipt, pt = scores.get(mol, ("", ""))
        rows.append({
            "molecule_id": mol, "pdb": pdb, "ligand_resname": resn,
            "n_poses": len(per_pose), "n_lig_atoms": p0["n_lig_atoms"],
            "rmsd_idx0": p0["rmsd"], "centroid_idx0": p0["centroid"],
            "pocket_jaccard_idx0": p0["pocket_jaccard"],
            "rmsd_best": best["rmsd"], "best_idx": best["idx"],
            "tm_score": p0["tm"], "rmsd_method": p0["method"],
            "iptm": ipt, "ptm": pt, "exp_pocket_n": len(exp_pocket),
        })
        print(f"{mol:<20} idx0 RMSD={p0['rmsd']:.2f}A  best-of-{len(per_pose)}="
              f"{best['rmsd']:.2f}A (idx{best['idx']})  centroid={p0['centroid']:.2f}A  "
              f"pocketJ={p0['pocket_jaccard']:.2f}  TM={p0['tm']:.2f}  "
              f"ipTM={ipt}  [{p0['method']}]")

    cols = ["molecule_id", "pdb", "ligand_resname", "n_poses", "n_lig_atoms",
            "rmsd_idx0", "centroid_idx0", "pocket_jaccard_idx0",
            "rmsd_best", "best_idx", "tm_score", "rmsd_method",
            "iptm", "ptm", "exp_pocket_n"]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = dict(r)
            for k in ("rmsd_idx0", "centroid_idx0", "pocket_jaccard_idx0",
                      "rmsd_best", "tm_score"):
                if isinstance(row.get(k), float):
                    row[k] = "" if math.isnan(row[k]) else f"{row[k]:.3f}"
            w.writerow(row)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
