#!/usr/bin/env python3
"""
plotting.py — generic ΔG (or any docking-score) comparison plots across
predict_structures.py-style summary.csv files.

Two subcommands:

  across-targets     Violin + box + strip plot of a score column across
                     MULTIPLE targets (one summary.csv per target), split
                     into two classes (e.g. binder / non-binder) by group name.

  group-comparison   Box + strip plot of a score column across arbitrary
                     GROUPS within one or more summary.csv files (e.g.
                     candidates vs several decoy floors vs a positive
                     anchor), with a reference line at the best value in a
                     chosen floor group.

Usage
-----
  python -m chai1_vina_toolkit.plotting across-targets \
      --summary target_a=out/target_a/summary.csv \
                target_b=out/target_b/summary.csv \
      --binder-groups binder predicted \
      --decoy-groups non_binder decoy \
      --score-col delta_g_kcal_mol \
      --out comparison.png

  python -m chai1_vina_toolkit.plotting group-comparison \
      --summary out/panel_a/summary.csv out/panel_b/summary.csv \
      --score-col delta_g_kcal_mol \
      --floor-group matched_decoy \
      --out group_comparison.png
"""

from __future__ import annotations

import argparse
import statistics as stats
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _lower_is_better_axis(ax) -> None:
    ax.invert_yaxis()  # stronger (more negative) binding at top


def cmd_across_targets(args: argparse.Namespace) -> None:
    import seaborn as sns

    pairs = []
    for item in args.summary:
        if "=" not in item:
            sys.exit(f"ERROR: --summary entries must be label=path, got '{item}'")
        label, path = item.split("=", 1)
        pairs.append((label, path))

    binder_groups = set(args.binder_groups)
    decoy_groups = set(args.decoy_groups)

    frames = []
    for label, path in pairs:
        df = pd.read_csv(path)
        df = df[df["group"].isin(binder_groups | decoy_groups)].copy()
        df["target"] = label
        df["klass"] = df["group"].apply(
            lambda g: "binder" if g in binder_groups else "non-binder"
        )
        frames.append(df[["target", "klass", args.score_col, "molecule_id"]])

    data = pd.concat(frames, ignore_index=True)
    dropped = data[args.score_col].isna().sum()
    data = data.dropna(subset=[args.score_col])

    order = [p[0] for p in pairs]
    palette = {"binder": "black", "non-binder": "red"}

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(9, 6))

    if args.highlight_targets:
        highlight = set(args.highlight_targets)
        for i, t in enumerate(order):
            color = "#f2c9a0" if t in highlight else "#bcd4e6"
            ax.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.30, zorder=0)

    sns.violinplot(
        data=data[data["klass"] == "binder"], x="target", y=args.score_col,
        order=order, color="0.85", inner=None, cut=0, linewidth=1, ax=ax,
    )
    for art in ax.collections:
        art.set_alpha(0.45)

    sns.boxplot(
        data=data[data["klass"] == "binder"], x="target", y=args.score_col,
        order=order, width=0.18, showfliers=False, ax=ax,
        boxprops=dict(facecolor="white", edgecolor="0.3", zorder=2),
        medianprops=dict(color="0.3"), whiskerprops=dict(color="0.3"),
        capprops=dict(color="0.3"),
    )

    sns.stripplot(
        data=data, x="target", y=args.score_col, order=order,
        hue="klass", hue_order=["binder", "non-binder"], palette=palette,
        dodge=False, jitter=0.18, size=7, alpha=0.9, edgecolor="white",
        linewidth=0.4, ax=ax, zorder=3,
    )

    ax.axhline(0, color="0.6", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("Target", fontsize=12)
    ax.set_ylabel(args.score_label or args.score_col, fontsize=12)
    ax.set_title(args.title or "Predicted binders vs. decoys across targets", fontsize=13)
    if args.invert_y:
        _lower_is_better_axis(ax)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-2:], labels[-2:], title="", loc="lower right", frameon=True)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    print(f"wrote {out}  (n={len(data)} compounds, dropped {dropped} with no score)")


def cmd_group_comparison(args: argparse.Namespace) -> None:
    frames = []
    for p in args.summary:
        df = pd.read_csv(p)
        df = df[df[args.score_col].astype(str).str.strip() != ""]
        df[args.score_col] = df[args.score_col].astype(float)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    groups_present = [g for g in df["group"].unique()]
    if args.group_order:
        order = [g for g in args.group_order if g in groups_present]
        order += [g for g in groups_present if g not in order]
    else:
        order = groups_present

    palette = plt.get_cmap("tab10").colors
    colours = {g: palette[i % len(palette)] for i, g in enumerate(order)}

    fig, ax = plt.subplots(figsize=(10.5, 6))
    rng = np.random.default_rng(0)

    data_by_group = {g: df.loc[df["group"] == g, args.score_col].tolist() for g in order}
    positions = list(range(1, len(order) + 1))
    ax.boxplot([data_by_group[g] for g in order], positions=positions,
               widths=0.5, showfliers=False,
               boxprops=dict(color="#555555"), medianprops=dict(color="#222222"),
               whiskerprops=dict(color="#999999"), capprops=dict(color="#999999"))
    for pos, g in zip(positions, order):
        ys = data_by_group[g]
        xs = rng.normal(pos, 0.06, size=len(ys))
        ax.scatter(xs, ys, s=70, color=colours[g], edgecolor="black",
                   linewidth=0.6, zorder=3, alpha=0.9)

    if args.floor_group and args.floor_group in data_by_group:
        floor = min(data_by_group[args.floor_group]) if args.invert_y else max(data_by_group[args.floor_group])
        ax.axhline(floor, ls="--", lw=1.2, color="#f46d43", alpha=0.8,
                   label=f"best {args.floor_group} ({floor:.1f})")
        ax.legend(loc="lower right", fontsize=9, frameon=False)

    ax.set_xticks(positions)
    ax.set_xticklabels(order, fontsize=9)
    ax.set_ylabel(args.score_label or args.score_col, fontsize=11)
    if args.invert_y:
        _lower_is_better_axis(ax)
    ax.set_title(args.title or "Score comparison across groups", fontsize=12, pad=12)
    ax.grid(axis="y", ls=":", alpha=0.4)
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    print(f"\n{'group':<22} {'n':>4} {'mean':>7} {'best':>7} {'worst':>7}")
    for g in order:
        v = data_by_group[g]
        best = min(v) if args.invert_y else max(v)
        worst = max(v) if args.invert_y else min(v)
        print(f"{g:<22} {len(v):>4} {stats.mean(v):>7.2f} {best:>7.2f} {worst:>7.2f}")

    if args.floor_group and args.floor_group in data_by_group and args.candidate_groups:
        floor_vals = data_by_group[args.floor_group]
        floor = min(floor_vals) if args.invert_y else max(floor_vals)
        for g in args.candidate_groups:
            if g not in data_by_group:
                continue
            cand = data_by_group[g]
            n_beats = sum(1 for c in cand if (c < floor if args.invert_y else c > floor))
            print(f"\n[{g}] beating best {args.floor_group} ({floor:.2f}): {n_beats}/{len(cand)}")
            gap = stats.mean(floor_vals) - stats.mean(cand)
            print(f"  mean gap vs {args.floor_group} floor: {gap:+.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("across-targets")
    p1.add_argument("--summary", nargs="+", required=True,
                     help="label=path pairs, one per target")
    p1.add_argument("--binder-groups", nargs="+", default=["binder"])
    p1.add_argument("--decoy-groups", nargs="+", default=["non_binder", "decoy"])
    p1.add_argument("--score-col", default="delta_g_kcal_mol")
    p1.add_argument("--score-label", default=None)
    p1.add_argument("--title", default=None)
    p1.add_argument("--highlight-targets", nargs="*", default=None,
                     help="target labels to visually highlight (e.g. a seen-in-training control)")
    p1.add_argument("--invert-y", action="store_true", default=True)
    p1.add_argument("--out", required=True)

    p2 = sub.add_parser("group-comparison")
    p2.add_argument("--summary", nargs="+", required=True, help="one or more summary.csv paths")
    p2.add_argument("--score-col", default="delta_g_kcal_mol")
    p2.add_argument("--score-label", default=None)
    p2.add_argument("--title", default=None)
    p2.add_argument("--group-order", nargs="*", default=None)
    p2.add_argument("--floor-group", default=None,
                     help="group treated as the non-binder floor for the reference line")
    p2.add_argument("--candidate-groups", nargs="*", default=None,
                     help="groups to report a 'beats the floor' stat for")
    p2.add_argument("--invert-y", action="store_true", default=True)
    p2.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.cmd == "across-targets":
        cmd_across_targets(args)
    elif args.cmd == "group-comparison":
        cmd_group_comparison(args)


if __name__ == "__main__":
    main()
