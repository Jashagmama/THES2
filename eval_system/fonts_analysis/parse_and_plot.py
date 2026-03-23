"""
parse_and_plot.py

Parses a wide-format indiv xlsx file and generates charts split into
uppercase (A-Z) and lowercase (a-z) sets:
  1. Grouped bar chart — per-letter, one bar per font/rep
  2. Heatmap          — all letters × reps, coloured by score

Usage:
    python parse_and_plot.py --file path/to/indiv_4_-_Grade_1.xlsx
    python parse_and_plot.py --file path/to/indiv_4_-_Grade_1.xlsx --metric Form
    python parse_and_plot.py --file path/to/indiv_4_-_Grade_1.xlsx --out-dir ./charts
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


BLOCK_COLS  = ["Letter", "Rep", "Form", "Size", "Align", "Avg"]
BLOCK_WIDTH = len(BLOCK_COLS)
SEPARATOR   = 1
METRICS     = ["Form", "Size", "Align", "Avg"]
FONT_COLORS = ["#378ADD", "#1D9E75", "#D85A30", "#BA7517", "#993356"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_indiv(filepath: str) -> pd.DataFrame:
    raw = pd.read_excel(filepath, header=None)
    blocks = []
    col = 0
    while col < raw.shape[1]:
        header_val = str(raw.iloc[0, col]).strip() if pd.notna(raw.iloc[0, col]) else ""
        if header_val == "Letter":
            block = raw.iloc[1:, col: col + BLOCK_WIDTH].copy()
            block.columns = BLOCK_COLS
            block = block.dropna(subset=["Letter", "Rep"])
            block["Letter"] = block["Letter"].astype(str).str.strip()
            block["Rep"]    = block["Rep"].astype(int)
            for c in ["Form", "Size", "Align", "Avg"]:
                block[c] = pd.to_numeric(block[c], errors="coerce")
            blocks.append(block)
            col += BLOCK_WIDTH + SEPARATOR
        else:
            col += 1

    if not blocks:
        raise ValueError("No valid data blocks found. Check the file format.")

    df = pd.concat(blocks, ignore_index=True)
    df = df.sort_values(["Letter", "Rep"]).reset_index(drop=True)
    return df


def split_by_case(df: pd.DataFrame):
    """Return (uppercase_df, lowercase_df) subsets."""
    upper = df[df["Letter"].str.isupper()].copy()
    lower = df[df["Letter"].str.islower()].copy()
    return upper, lower


# ---------------------------------------------------------------------------
# Chart 1 — grouped bar
# ---------------------------------------------------------------------------

def plot_grouped_bar(df: pd.DataFrame, metric: str, title_suffix: str, out_path: str):
    letters = sorted(df["Letter"].unique())
    reps    = sorted(df["Rep"].unique())
    n_letters = len(letters)
    n_reps    = len(reps)

    font_style = [
        "Times New Roman", "Roboto", "Arial", "Poppins", "Calibri"
    ]
    x     = np.arange(n_letters)
    width = 0.8 / n_reps

    fig, ax = plt.subplots(figsize=(max(8, n_letters * 0.55), 5))

    for i, rep in enumerate(reps):
        vals = []
        for letter in letters:
            row = df[(df["Letter"] == letter) & (df["Rep"] == rep)]
            vals.append(row[metric].values[0] if not row.empty else np.nan)
        offset = (i - n_reps / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.9,
               color=FONT_COLORS[i % len(FONT_COLORS)],
               label=f"{font_style[rep - 1]}", zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(letters, fontsize=10, fontfamily="monospace")
    ax.set_ylabel(metric, fontsize=11)
    ax.set_ylim(50, 105)
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, zorder=0)
    ax.grid(axis="y", which="minor", color="#f0f0f0", linewidth=0.4, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{metric} score by letter and font — {title_suffix}", fontsize=13, pad=12)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Font / Rep", fontsize=9,
              title_fontsize=9, loc="lower right", framealpha=0.7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Chart 2 — heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(df: pd.DataFrame, metric: str, title_suffix: str, out_path: str):
    letters = sorted(df["Letter"].unique())
    reps    = sorted(df["Rep"].unique())

    matrix = np.full((len(reps), len(letters)), np.nan)
    for j, letter in enumerate(letters):
        for i, rep in enumerate(reps):
            row = df[(df["Letter"] == letter) & (df["Rep"] == rep)]
            if not row.empty:
                matrix[i, j] = row[metric].values[0]

    fig_w = max(8, len(letters) * 0.45)
    fig_h = max(3, len(reps)   * 0.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                   vmin=60, vmax=100, interpolation="nearest")

    ax.set_xticks(range(len(letters)))
    ax.set_xticklabels(letters, fontsize=10, fontfamily="monospace")
    ax.set_yticks(range(len(reps)))
    font_style = [
        "Times New Roman", "Roboto", "Arial", "Poppins", "Calibri"
    ]
    ax.set_yticklabels([f"{font_style[r-1]}" for r in reps], fontsize=9)
    ax.set_title(f"{metric} heatmap — {title_suffix}", fontsize=13, pad=12)

    for i in range(len(reps)):
        for j in range(len(letters)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=8, color="white" if val < 72 else "black")

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label(metric, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse indiv xlsx and generate charts (uppercase / lowercase split).")
    parser.add_argument("--file",    required=True, help="Path to the indiv xlsx file")
    parser.add_argument("--metric",  default="Avg", choices=METRICS,
                        help="Metric to visualise (default: Avg)")
    parser.add_argument("--out-dir", default=".",   help="Output directory for charts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.file))[0]

    print(f"Parsing {args.file} ...")
    df = parse_indiv(args.file)
    print(f"  {len(df)} rows | {df['Letter'].nunique()} letters | {df['Rep'].nunique()} fonts\n")

    upper_df, lower_df = split_by_case(df)

    for subset_df, label in [(upper_df, "uppercase"), (lower_df, "lowercase")]:
        if subset_df.empty:
            print(f"No {label} letters found, skipping.")
            continue

        print(f"Generating {label} charts ...")
        plot_grouped_bar(
            subset_df, args.metric, label,
            os.path.join(args.out_dir, f"{base}_{args.metric}_{label}_bar.png")
        )
        plot_heatmap(
            subset_df, args.metric, label,
            os.path.join(args.out_dir, f"{base}_{args.metric}_{label}_heatmap.png")
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
