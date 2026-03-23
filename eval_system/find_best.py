"""
find_best.py

Finds the highest Overall Avg for a given letter by crawling all subfolders
(and optionally flat files) under --base-dir. No assumptions are made about
folder names. Handles both filename formats:
    avg_4 - Grade 1.xlsx
    avg_4_-_Grade_1.xlsx

Usage:
    python find_best.py --base-dir /path/to/data --letter A
    python find_best.py --base-dir /path/to/data --letter a --top 3
"""

import argparse
import os
import re
import pandas as pd


# Handles both "avg_4 - Grade 1.xlsx" and "avg_4_-_Grade_1.xlsx"
FILENAME_RE = re.compile(r"avg[_ ](\d+)[_ ]-[_ ](Grade[_ ]\d+)\.xlsx", re.IGNORECASE)


def extract_id_and_grade(filename):
    m = FILENAME_RE.match(filename)
    if m:
        return int(m.group(1)), m.group(2).replace("_", " ")
    return None, None


def load_overall_avg(filepath):
    df = pd.read_excel(filepath, usecols=["Letter", "Overall Avg"])
    df = df.dropna(subset=["Letter", "Overall Avg"])
    df["Letter"] = df["Letter"].astype(str).str.strip()
    return df.set_index("Letter")["Overall Avg"]


def crawl(base_dir):
    """
    Yield (folder_label, file_id, grade, filepath) for every avg_ xlsx found.
    Searches both directly in base_dir and inside any subfolders (one level deep).
    """
    for entry in sorted(os.scandir(base_dir), key=lambda e: e.name):
        if entry.is_dir():
            for fname in sorted(os.listdir(entry.path)):
                if not fname.startswith("avg_") or not fname.endswith(".xlsx"):
                    continue
                file_id, grade = extract_id_and_grade(fname)
                if file_id is None:
                    continue
                yield entry.name, file_id, grade, os.path.join(entry.path, fname)
        elif entry.name.startswith("avg_") and entry.name.endswith(".xlsx"):
            file_id, grade = extract_id_and_grade(entry.name)
            if file_id is not None:
                yield "(root)", file_id, grade, entry.path


def find_best(base_dir, letter, top=None):
    results = []

    for folder, file_id, grade, filepath in crawl(base_dir):
        try:
            series = load_overall_avg(filepath)
        except Exception as e:
            print(f"  WARNING: Could not read {filepath}: {e}")
            continue

        if letter in series.index:
            results.append({
                "Folder":      folder,
                "File ID":     file_id,
                "Grade":       grade,
                "Letter":      letter,
                "Overall Avg": round(series[letter], 2),
            })

    if not results:
        print(f"\nNo results found for letter '{letter}'.")
        print("Tips:")
        print("  - Letter is case-sensitive ('A' and 'a' are different)")
        print("  - Check that your avg_ files contain a 'Letter' and 'Overall Avg' column")
        return

    df = (pd.DataFrame(results)
            .sort_values("Overall Avg", ascending=False)
            .reset_index(drop=True))

    if top:
        df = df.head(top)

    print(f"\nResults for letter '{letter}' — sorted by Overall Avg (highest first):\n")
    print(df.to_string(index=False))
    print(f"\nBest: File ID {df.iloc[0]['File ID']} | {df.iloc[0]['Folder']} "
          f"| {df.iloc[0]['Grade']} | Overall Avg = {df.iloc[0]['Overall Avg']}")


def main():
    parser = argparse.ArgumentParser(
        description="Find highest Overall Avg for a letter by crawling all subfolders.")
    parser.add_argument("--base-dir", required=True,
                        help="Parent folder to crawl (searches root + all subfolders)")
    parser.add_argument("--letter",   required=True,
                        help="Letter to search for (case-sensitive, e.g. A or a)")
    parser.add_argument("--top",      type=int, default=None,
                        help="Show only top N results")
    args = parser.parse_args()

    find_best(args.base_dir, args.letter, args.top)


if __name__ == "__main__":
    main()
