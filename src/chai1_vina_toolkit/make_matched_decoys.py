#!/usr/bin/env python3
"""
make_matched_decoys.py — build a property-matched, topologically-dissimilar
decoy set for docking benchmarks (DUD-E logic).

Raw "drug-like vs tiny-polar" decoys make separation trivial for the wrong
reason. This picks, for each candidate, a decoy from a pool that MATCHES its
physicochemistry (MW/logP/HBD/HBA/TPSA/rotB) but is structurally UNRELATED
(ECFP4 Tanimoto below a cutoff to every candidate and to any molecule you
name in --avoid-smiles) — so it is a presumed non-binder that controls for
size/lipophilicity rather than being trivially distinguishable by it.

Inputs
------
  --candidates  CSV with a smiles column (a docking summary.csv works; rows
                are filtered to --group-filter when a group column exists).
  --pool        a large candidate pool CSV (e.g. de novo generator samples)
                with a SMILES column.

Output
------
  <out>.dock_input.csv  predict_structures.py schema, group="matched_decoy"
  <out>.match.csv       each decoy paired to its candidate w/ property deltas

Usage
-----
  python -m chai1_vina_toolkit.make_matched_decoys \
      --candidates candidates.csv \
      --pool       pool.csv \
      --out        out/matched_decoys \
      --target-id  my_target
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, DataStructs
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")

# Physicochemical descriptors used for matching.
DESCRIPTORS = {
    "mw":   Descriptors.MolWt,
    "logp": Descriptors.MolLogP,
    "hbd":  Descriptors.NumHDonors,
    "hba":  Descriptors.NumHAcceptors,
    "tpsa": Descriptors.TPSA,
    "rotb": Descriptors.NumRotatableBonds,
}


def fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def props(mol) -> dict:
    return {k: f(mol) for k, f in DESCRIPTORS.items()}


def find_smiles_col(df) -> str:
    for c in ["SMILES", "Smiles", "smiles", "canonical_smiles"]:
        if c in df.columns:
            return c
    sys.exit(f"ERROR: no SMILES column (cols: {list(df.columns)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim-cutoff", type=float, default=0.35,
                    help="max ECFP4 Tanimoto a decoy may have to any active-space mol")
    ap.add_argument("--qed-min", type=float, default=0.50)
    ap.add_argument("--per-candidate", type=int, default=1,
                    help="decoys to match to each candidate")
    ap.add_argument("--target-id", required=True,
                    help="target identifier written to the dock_input CSV")
    ap.add_argument("--gene", default="",
                    help="optional display label written to the dock_input CSV")
    ap.add_argument("--group-filter", default=None,
                    help="if the candidates CSV has a 'group' column, keep only this "
                         "group; omit to keep all rows")
    ap.add_argument("--avoid-smiles", default=None,
                    help="comma-separated extra SMILES (or a path to a file with one "
                         "SMILES per line) the decoys must NOT resemble, in addition "
                         "to the candidates themselves")
    args = ap.parse_args()

    target_id, gene = args.target_id, args.gene

    avoid_smiles: list[str] = []
    if args.avoid_smiles is not None:
        p = Path(args.avoid_smiles)
        if p.exists():
            avoid_smiles = [s.strip() for s in p.read_text().splitlines() if s.strip()]
        else:
            avoid_smiles = [s.strip() for s in args.avoid_smiles.split(",") if s.strip()]

    # ── Candidates ──────────────────────────────────────────────────────────
    cdf = pd.read_csv(args.candidates)
    if "group" in cdf.columns and args.group_filter:
        cdf = cdf[cdf["group"] == args.group_filter]
    csmi = find_smiles_col(cdf)
    cand = []
    for _, r in cdf.iterrows():
        m = Chem.MolFromSmiles(str(r[csmi]))
        if m is None:
            continue
        mid = str(r["molecule_id"]) if "molecule_id" in cdf.columns else f"cand_{len(cand)}"
        cand.append({"id": mid, "mol": m, "fp": fp(m), "props": props(m)})
    if not cand:
        sys.exit("ERROR: no valid candidates")

    # Avoid-set fingerprints (candidates themselves are also in the avoid set).
    avoid_fps = [c["fp"] for c in cand]
    for smi in avoid_smiles:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            avoid_fps.append(fp(m))

    scale = {}
    for k in DESCRIPTORS:
        vals = np.array([c["props"][k] for c in cand], dtype=float)
        s = vals.std()
        scale[k] = s if s > 1e-6 else (abs(vals.mean()) * 0.1 + 1e-6)

    # ── Pool → drug-like, dissimilar candidates ─────────────────────────────
    pdf = pd.read_csv(args.pool)
    psmi = find_smiles_col(pdf)
    pains = FilterCatalog(FilterCatalogParams(FilterCatalogParams.FilterCatalogs.PAINS))

    seen, pool = set(), []
    n_raw = 0
    for smi in pdf[psmi].dropna():
        n_raw += 1
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            continue
        canon = Chem.MolToSmiles(m)
        if canon in seen:
            continue
        seen.add(canon)
        if QED.qed(m) < args.qed_min or pains.HasMatch(m):
            continue
        f = fp(m)
        max_sim = max(DataStructs.TanimotoSimilarity(f, a) for a in avoid_fps) if avoid_fps else 0.0
        if max_sim >= args.sim_cutoff:
            continue
        pool.append({"smiles": canon, "mol": m, "fp": f,
                     "props": props(m), "max_sim": max_sim})

    if len(pool) < len(cand) * args.per_candidate:
        sys.exit(f"ERROR: only {len(pool)} pool mols passed filters — "
                 f"need {len(cand)*args.per_candidate}; loosen filters / grow pool")

    def pdist(a, b) -> float:
        return float(np.sqrt(sum(((a[k] - b[k]) / scale[k]) ** 2 for k in DESCRIPTORS)))

    # ── Greedy per-candidate nearest match, without reuse ────────────────────
    used, rows = set(), []
    for c in cand:
        ranked = sorted(
            (p for i, p in enumerate(pool) if i not in used),
            key=lambda p: pdist(c["props"], p["props"]),
        )
        picks = 0
        for p in ranked:
            idx = pool.index(p)
            if idx in used:
                continue
            used.add(idx)
            did = f"mdecoy_{len(rows):02d}"
            rows.append({
                "molecule_id": did, "smiles": p["smiles"],
                "matched_candidate": c["id"],
                "prop_dist": round(pdist(c["props"], p["props"]), 3),
                "max_tanimoto_to_active_space": round(p["max_sim"], 3),
                **{f"decoy_{k}": round(p["props"][k], 1) for k in DESCRIPTORS},
                **{f"cand_{k}": round(c["props"][k], 1) for k in DESCRIPTORS},
            })
            picks += 1
            if picks >= args.per_candidate:
                break

    match = pd.DataFrame(rows)
    match_path = Path(f"{args.out}.match.csv")
    match.to_csv(match_path, index=False)

    dock = pd.DataFrame({
        "target_id": target_id,
        "gene_name": gene,
        "molecule_id": match["molecule_id"],
        "smiles": match["smiles"],
        "predicted_score": range(1, len(match) + 1),
        "rank": range(1, len(match) + 1),
        "group": "matched_decoy",
    })
    dock_path = Path(f"{args.out}.dock_input.csv")
    dock.to_csv(dock_path, index=False)

    print(f"candidates: {len(cand)} | pool raw: {n_raw} | pool passed filters: {len(pool)}")
    print(f"picked {len(match)} matched decoys "
          f"(sim<{args.sim_cutoff}, QED>={args.qed_min})")
    print(f"  mean prop_dist {match['prop_dist'].mean():.2f} | "
          f"mean max_Tanimoto {match['max_tanimoto_to_active_space'].mean():.2f}")
    print(f"wrote {match_path}")
    print(f"wrote {dock_path}")


if __name__ == "__main__":
    main()
