#!/usr/bin/env python3
"""
plot_dssp.py

Generates a multi-panel visualization of DSSP secondary structure composition
for the 102 synthetic proteins from Hýsková et al. (2026).

Panels:
    1. Stacked bar chart: per-structure SS composition (H/E/L fractions),
       sorted by helix fraction — highlights helical vs mixed vs sheet-rich proteins
    2. Ternary-style scatter (helix vs strand vs loop fractions)
    3. Violin plots: distribution of H, E, L fractions across all structures
    4. Heatmap: DSSP strings for all proteins (residue x protein), sorted by length

Usage:
    python plot_dssp.py \
        --summary ./dssp_summary.tsv \
        --per_residue ./dssp_per_residue.tsv \
        --outfile ./dssp_summary_plot.pdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from matplotlib.gridspec import GridSpec


# ── color scheme ────────────────────────────────────────────────────────────
SS_COLORS = {
    "H": "#E05A5A",   # red  — helix
    "E": "#5A8AE0",   # blue — strand
    "L": "#AAAAAA",   # grey — loop
}


def load_data(summary_path: str, per_residue_path: str):
    summary = pd.read_csv(summary_path, sep="\t")
    per_res = pd.read_csv(per_residue_path, sep="\t")
    return summary, per_res


# ── Panel 1: stacked bar chart ───────────────────────────────────────────────
def plot_stacked_bars(ax, summary: pd.DataFrame):
    df = summary.sort_values("frac_helix", ascending=False).reset_index(drop=True)
    x = np.arange(len(df))
    width = 1.0

    ax.bar(x, df["frac_helix"], width=width, color=SS_COLORS["H"],
           label="α-helix", linewidth=0)
    ax.bar(x, df["frac_strand"], width=width, bottom=df["frac_helix"],
           color=SS_COLORS["E"], label="β-strand", linewidth=0)
    ax.bar(x, df["frac_loop"], width=width,
           bottom=df["frac_helix"] + df["frac_strand"],
           color=SS_COLORS["L"], label="Loop/coil", linewidth=0)

    ax.set_xlim(-0.5, len(df) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Structures (sorted by helix fraction)", fontsize=9)
    ax.set_ylabel("Fraction of residues", fontsize=9)
    ax.set_title("Secondary structure composition\n(102 synthetic proteins)", fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.tick_params(axis="y", labelsize=8)

    legend = ax.legend(loc="upper right", fontsize=8, framealpha=0.9,
                       handles=[
                           mpatches.Patch(color=SS_COLORS["H"], label="α-helix"),
                           mpatches.Patch(color=SS_COLORS["E"], label="β-strand"),
                           mpatches.Patch(color=SS_COLORS["L"], label="Loop/coil"),
                       ])


# ── Panel 2: helix vs strand scatter (coloured by loop) ─────────────────────
def plot_hs_scatter(ax, summary: pd.DataFrame):
    sc = ax.scatter(
        summary["frac_helix"],
        summary["frac_strand"],
        c=summary["frac_loop"],
        cmap="viridis_r",
        s=35,
        alpha=0.8,
        edgecolors="none",
        vmin=0, vmax=0.7
    )
    cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Loop fraction", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # diagonal line: helix + strand = 1 (no loop)
    t = np.linspace(0, 1, 100)
    ax.plot(t, 1 - t, "--", color="#cccccc", lw=0.8, zorder=0)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.82)
    ax.set_xlabel("Helix fraction", fontsize=9)
    ax.set_ylabel("Strand fraction", fontsize=9)
    ax.set_title("Helix vs strand content\n(colour = loop fraction)", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

    # add median crosshairs
    mh = summary["frac_helix"].median()
    ms = summary["frac_strand"].median()
    ax.axvline(mh, color=SS_COLORS["H"], lw=0.8, ls=":", alpha=0.6)
    ax.axhline(ms, color=SS_COLORS["E"], lw=0.8, ls=":", alpha=0.6)
    ax.text(mh + 0.01, 0.78, f"median H={mh:.2f}", color=SS_COLORS["H"], fontsize=7)
    ax.text(0.01, ms + 0.01, f"median E={ms:.2f}", color=SS_COLORS["E"], fontsize=7)


# ── Panel 3: violin plots ────────────────────────────────────────────────────
def plot_violins(ax, summary: pd.DataFrame):
    data = [
        summary["frac_helix"].values,
        summary["frac_strand"].values,
        summary["frac_loop"].values,
    ]
    labels = ["α-helix (H)", "β-strand (E)", "Loop/coil (L)"]
    colors = [SS_COLORS["H"], SS_COLORS["E"], SS_COLORS["L"]]

    parts = ax.violinplot(data, positions=[1, 2, 3], showmedians=True,
                          showextrema=True, widths=0.6)

    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
        pc.set_edgecolor("none")

    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.5)
    parts["cmaxes"].set_color("#555555")
    parts["cmins"].set_color("#555555")
    parts["cbars"].set_color("#555555")

    # Overlay individual points
    for i, (d, color) in enumerate(zip(data, colors), start=1):
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=len(d))
        ax.scatter(i + jitter, d, color=color, s=8, alpha=0.5, zorder=3,
                   edgecolors="none")

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Fraction of residues", fontsize=9)
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Distribution of SS fractions", fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", labelsize=8)

    # Add median text
    for i, d in enumerate(data, start=1):
        med = np.median(d)
        ax.text(i, med + 0.03, f"{med:.2f}", ha="center", va="bottom",
                fontsize=7, color="black", fontweight="bold")


# ── Panel 4: DSSP string heatmap ────────────────────────────────────────────
def plot_heatmap(ax, summary: pd.DataFrame):
    # Sort by protein length so shorter proteins align left
    df = summary.sort_values("n_residues").reset_index(drop=True)
    max_len = df["n_residues"].max()

    # Encode: H=1, E=2, L=0, gap=-1
    encode = {"H": 1, "E": 2, "L": 0}
    matrix = np.full((len(df), max_len), np.nan)

    for i, row in df.iterrows():
        ss = row["ss_string"]
        for j, c in enumerate(ss):
            matrix[i, j] = encode.get(c, 0)

    # Colormap: NaN=white, L=grey, H=red, E=blue
    cmap = ListedColormap(["#AAAAAA", SS_COLORS["H"], SS_COLORS["E"]])
    masked = np.ma.masked_invalid(matrix)
    cmap.set_bad("white")

    ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0, vmax=2,
              interpolation="nearest", origin="lower")

    ax.set_xlabel("Residue position", fontsize=9)
    ax.set_ylabel("Protein (sorted by length)", fontsize=9)
    ax.set_title("DSSP per-residue assignments\n(grey=loop, red=helix, blue=strand)",
                 fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)

    # Legend patches
    legend_handles = [
        mpatches.Patch(color=SS_COLORS["H"], label="H (helix)"),
        mpatches.Patch(color=SS_COLORS["E"], label="E (strand)"),
        mpatches.Patch(color=SS_COLORS["L"], label="L (loop)"),
        mpatches.Patch(color="white", label="(gap)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7,
              framealpha=0.8)


# ── Print summary statistics ─────────────────────────────────────────────────
def print_summary_stats(summary: pd.DataFrame):
    print("\n=== DSSP Summary Statistics ===")
    print(f"N structures: {len(summary)}")
    for col, label in [
        ("frac_helix", "Helix fraction"),
        ("frac_strand", "Strand fraction"),
        ("frac_loop", "Loop fraction"),
        ("n_residues", "Chain length"),
        ("n_helix_segments", "# helix segments"),
        ("n_strand_segments", "# strand segments"),
    ]:
        vals = summary[col]
        print(f"  {label:25s}  median={vals.median():.3f}  "
              f"mean={vals.mean():.3f}  "
              f"std={vals.std():.3f}  "
              f"min={vals.min():.3f}  max={vals.max():.3f}")

    # Classify structures
    all_helix = (summary["frac_helix"] >= 0.5).sum()
    all_sheet = (summary["frac_strand"] >= 0.5).sum()
    mixed = ((summary["frac_helix"] < 0.5) & (summary["frac_strand"] < 0.5)).sum()
    print(f"\n  Predominantly helical (H≥50%): {all_helix} ({100*all_helix/len(summary):.1f}%)")
    print(f"  Predominantly strand (E≥50%): {all_sheet} ({100*all_sheet/len(summary):.1f}%)")
    print(f"  Mixed/loop-dominant:           {mixed} ({100*mixed/len(summary):.1f}%)")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Plot DSSP results for synthetic proteins")
    parser.add_argument("--summary", default="./dssp_summary.tsv")
    parser.add_argument("--per_residue", default="./dssp_per_residue.tsv")
    parser.add_argument("--outfile", default="./dssp_summary_plot.pdf")
    args = parser.parse_args()

    print("Loading data...")
    summary, per_res = load_data(args.summary, args.per_residue)
    print(f"  {len(summary)} structures loaded")

    print_summary_stats(summary)

    print("\nGenerating figure...")
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Secondary structure composition of 102 synthetic proteins\n"
        "(Hýsková et al. 2026 benchmark dataset, experimental structures)",
        fontsize=12, fontweight="bold", y=0.98
    )

    gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                  left=0.07, right=0.97, top=0.92, bottom=0.06)

    ax1 = fig.add_subplot(gs[0, :])   # stacked bars — full width top
    ax2 = fig.add_subplot(gs[1, 0])   # scatter
    ax3 = fig.add_subplot(gs[1, 1])   # violins

    plot_stacked_bars(ax1, summary)
    plot_hs_scatter(ax2, summary)
    plot_violins(ax3, summary)

    plt.savefig(args.outfile, dpi=180, bbox_inches="tight")
    print(f"Saved: {args.outfile}")

    # Also save heatmap separately (tall figure)
    heatmap_path = args.outfile.replace(".pdf", "_heatmap.pdf")
    fig2, ax_hm = plt.subplots(figsize=(14, 10))
    plot_heatmap(ax_hm, summary)
    fig2.suptitle("Per-residue DSSP assignments — synthetic proteins",
                  fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=180, bbox_inches="tight")
    print(f"Saved heatmap: {heatmap_path}")


if __name__ == "__main__":
    main()
