#!/usr/bin/env python3
"""
pocket_engagement.py — a discrimination readout for docked ligands that is
orthogonal to binding-affinity scores (ΔG or similar), computed on a
multi-pose co-folding ensemble already on disk (e.g. 5 Chai-1 diffusion
samples per ligand). No new GPU work — pure geometry.

Rationale
---------
A raw affinity/ΔG score rewards any well-sized lipophile that fits a pocket's
cavity, so it can fail to separate hypothesised binders from property-matched
decoys. A real binder should instead (a) seat REPRODUCIBLY in the same
sub-pocket across independent samples, and (b) engage the same pocket that
known actives use. Decoys that merely fit geometrically should do neither.

Per-ligand metrics (over its up-to-N poses)
--------------------------------------------
  pose_consistency : mean pairwise Jaccard of the per-pose receptor contact sets
                     (receptor residues with any heavy atom within --cutoff of
                     the ligand). 1.0 = identical pocket every pose; ~0 = drifts.
  n_contacts       : mean number of contacted receptor residues (buriedness proxy)
  consensus        : residues contacted in >= --consensus-frac of the ligand's poses
  pocket_overlap   : Jaccard(this ligand's consensus, REFERENCE pocket), where the
                     reference pocket is built from the --ref-group ligands —
                     "does it bind where the reference actives bind?"
  engagement_score : pose_consistency * pocket_overlap  (both in [0,1])

Reference pocket
----------------
A receptor residue is in the reference pocket if it appears in the consensus
set of at least --ref-frac of the reference-group ligands. This only works
when the receptor sequence is identical across every co-fold in a panel (so
residue numbering is directly comparable between complexes).

Usage
-----
  python -m chai1_vina_toolkit.pocket_engagement \
      --panels  panel_a/ panel_b/ panel_c/ \
      --ref-group known_actives \
      --out-csv pocket_engagement.csv \
      --out-png pocket_engagement.png
"""
from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import gemmi
import numpy as np

AA3 = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    "MSE", "SEC", "PYL",
}
WATER = {"HOH", "WAT", "DOD"}
IONS = {"NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "CU", "NI", "CO",
        "BR", "IOD", "SO4", "PO4"}


def heavy_atoms(residue):
    for atom in residue:
        el = atom.element.name
        if el != "H" and el != "D":
            yield atom


def parse_pose(cif_path: Path):
    """Return (protein_res_xyz: dict[int -> (n,3) array], ligand_xyz: (m,3) array).

    Protein residues keyed by seqid (comparable across complexes since the input
    sequence is identical). Ligand = all non-AA, non-water, non-ion heavy atoms.
    """
    st = gemmi.read_structure(str(cif_path))
    st.setup_entities()
    model = st[0]

    prot: dict[int, list] = {}
    lig: list = []
    for chain in model:
        for res in chain:
            name = res.name.strip().upper()
            if name in AA3:
                key = res.seqid.num
                coords = [[a.pos.x, a.pos.y, a.pos.z] for a in heavy_atoms(res)]
                if coords:
                    prot.setdefault(key, []).extend(coords)
            elif name in WATER or name in IONS:
                continue
            else:
                lig.extend([[a.pos.x, a.pos.y, a.pos.z] for a in heavy_atoms(res)])

    prot_arr = {k: np.asarray(v, dtype=float) for k, v in prot.items()}
    lig_arr = np.asarray(lig, dtype=float) if lig else np.empty((0, 3))
    return prot_arr, lig_arr


def contact_set(prot: dict[int, np.ndarray], lig: np.ndarray, cutoff: float) -> set[int]:
    if lig.shape[0] == 0:
        return set()
    contacts = set()
    c2 = cutoff * cutoff
    for seqid, xyz in prot.items():
        d2 = ((xyz[:, None, :] - lig[None, :, :]) ** 2).sum(-1)
        if d2.min() <= c2:
            contacts.add(seqid)
    return contacts


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    u = len(a | b)
    return len(a & b) / u if u else float("nan")


def ligand_dirs(panel: Path):
    """Yield (group, molecule_id, [cif paths]) for a structures panel dir laid
    out as <panel>/<group>/<molecule_id>/pred*.cif."""
    for group_dir in sorted(p for p in panel.iterdir() if p.is_dir()):
        for mol_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            cifs = sorted(mol_dir.glob("*.cif"))
            if cifs:
                yield group_dir.name, mol_dir.name, cifs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--panels", nargs="+", required=True,
                    help="Structure panel dirs (group/molecule_id/*.cif)")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-png", default=None)
    ap.add_argument("--cutoff", type=float, default=4.5,
                    help="heavy-atom contact cutoff in Å (default 4.5)")
    ap.add_argument("--consensus-frac", type=float, default=0.6,
                    help="residue is in a ligand's consensus pocket if contacted "
                         "in >= this fraction of its poses (default 0.6)")
    ap.add_argument("--ref-group", required=True,
                    help="group whose consensus pockets define the reference pocket "
                         "(e.g. a set of known/literature actives)")
    ap.add_argument("--ref-frac", type=float, default=0.5,
                    help="residue is in the reference pocket if in >= this fraction "
                         "of reference-group ligands' consensus sets (default 0.5)")
    args = ap.parse_args()

    records = []
    for panel in args.panels:
        panel = Path(panel)
        if not panel.exists():
            print(f"  WARN: panel not found: {panel}")
            continue
        for group, mol_id, cifs in ligand_dirs(panel):
            pose_contacts = []
            for cif in cifs:
                prot, lig = parse_pose(cif)
                pose_contacts.append(contact_set(prot, lig, args.cutoff))
            pose_contacts = [c for c in pose_contacts if c]
            n_poses = len(pose_contacts)
            if n_poses == 0:
                print(f"  WARN: no ligand contacts for {group}/{mol_id}")
                continue

            if n_poses >= 2:
                pcons = float(np.nanmean(
                    [jaccard(a, b) for a, b in itertools.combinations(pose_contacts, 2)]
                ))
            else:
                pcons = float("nan")
            n_contacts = float(np.mean([len(c) for c in pose_contacts]))

            need = math.ceil(args.consensus_frac * n_poses)
            counts: dict[int, int] = {}
            for c in pose_contacts:
                for r in c:
                    counts[r] = counts.get(r, 0) + 1
            consensus = {r for r, n in counts.items() if n >= need}

            records.append({
                "group": group, "molecule_id": mol_id, "n_poses": n_poses,
                "pose_consistency": pcons, "n_contacts": n_contacts,
                "consensus": consensus,
            })

    if not records:
        raise SystemExit("ERROR: no ligands parsed from any panel")

    ref_consensus = [r["consensus"] for r in records if r["group"] == args.ref_group]
    if not ref_consensus:
        raise SystemExit(f"ERROR: no ligands in reference group '{args.ref_group}'")
    ref_need = math.ceil(args.ref_frac * len(ref_consensus))
    rc: dict[int, int] = {}
    for cons in ref_consensus:
        for r in cons:
            rc[r] = rc.get(r, 0) + 1
    reference_pocket = {r for r, n in rc.items() if n >= ref_need}
    print(f"reference pocket ({args.ref_group}, {len(ref_consensus)} ligands, "
          f">= {ref_need} agree): {len(reference_pocket)} residues")
    print(f"  residues: {sorted(reference_pocket)}")

    for r in records:
        r["pocket_overlap"] = jaccard(r["consensus"], reference_pocket)
        pc = r["pose_consistency"]
        po = r["pocket_overlap"]
        r["engagement_score"] = (pc * po) if not (math.isnan(pc) or math.isnan(po)) else float("nan")

    import csv
    cols = ["group", "molecule_id", "n_poses", "pose_consistency", "n_contacts",
            "pocket_overlap", "engagement_score"]
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(records, key=lambda x: (x["group"], -(x["engagement_score"]
                        if not math.isnan(x["engagement_score"]) else -1))):
            row = {k: r[k] for k in cols}
            for k in ("pose_consistency", "n_contacts", "pocket_overlap", "engagement_score"):
                row[k] = f"{r[k]:.3f}" if not (isinstance(r[k], float) and math.isnan(r[k])) else ""
            w.writerow(row)
    print(f"wrote {out_csv}")

    groups = {}
    for r in records:
        groups.setdefault(r["group"], []).append(r)
    print("\ngroup                  n   pose_cons  n_contacts  pocket_ovlp  engage")
    order = [args.ref_group] + [g for g in groups if g != args.ref_group]
    for g in order:
        if g not in groups:
            continue
        rs = groups[g]
        def m(key):
            vals = [r[key] for r in rs if not math.isnan(r[key])]
            return np.mean(vals) if vals else float("nan")
        print(f"{g:<20}{len(rs):>4}   {m('pose_consistency'):>8.3f}  "
              f"{m('n_contacts'):>9.1f}   {m('pocket_overlap'):>9.3f}   "
              f"{m('engagement_score'):>6.3f}")

    if args.out_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib unavailable — skipping plot (CSV written)")
            return
        palette = plt.get_cmap("tab10").colors
        colours = {g: palette[i % len(palette)] for i, g in enumerate(order)}
        fig, ax = plt.subplots(figsize=(8.5, 7))
        for g, rs in groups.items():
            xs = [r["pose_consistency"] for r in rs]
            ys = [r["pocket_overlap"] for r in rs]
            sz = [20 + 6 * r["n_contacts"] for r in rs]
            ax.scatter(xs, ys, s=sz, color=colours.get(g, "#000000"),
                       edgecolor="black", linewidth=0.6, alpha=0.85, label=g)
        ax.set_xlabel("pose consistency  (mean pairwise Jaccard across poses)", fontsize=11)
        ax.set_ylabel(f"pocket overlap with '{args.ref_group}' reference (Jaccard)", fontsize=11)
        ax.set_title("Pose-consistency × pocket-engagement\n"
                     "(point size ∝ mean # contacted residues)", fontsize=12)
        ax.grid(ls=":", alpha=0.4)
        ax.legend(fontsize=9, frameon=False)
        fig.tight_layout()
        Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_png, dpi=150)
        print(f"wrote {args.out_png}")


if __name__ == "__main__":
    main()
