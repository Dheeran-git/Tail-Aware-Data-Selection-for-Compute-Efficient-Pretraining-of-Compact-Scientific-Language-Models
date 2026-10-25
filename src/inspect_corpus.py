import json
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config


def main():
    config.ensure_dirs()
    df = pd.read_parquet(config.RAW_DIR / "arxiv_cs_ml.parquet")
    df["n_words"] = df["abstract"].str.split().str.len()
    df["year"] = df["v1_created"].dt.year

    print(f"Rows: {len(df):,}")
    print("\nMissing/empty:")
    print("  empty abstract:", int((df["n_words"] == 0).sum()))
    print("  missing v1 date:", int(df["v1_created"].isna().sum()))

    print("\nPrimary category (top 25):")
    print(df["primary_category"].value_counts().head(25).to_string())

    print("\nPapers per year (v1):")
    print(df["year"].value_counts().sort_index().to_string())

    pcts = [0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999]
    print("\nAbstract word-count percentiles:")
    print(df["n_words"].quantile(pcts).to_string())

    key = df["abstract"].str.lower().map(lambda s: re.sub(r"[^a-z0-9]", "", s))
    n_dup = int(key.duplicated().sum())
    withdrawn = int(
        df["abstract"].str.contains(r"(?i)\bwithdrawn\b", regex=True).sum()
    )
    print(f"\nExact duplicate abstracts (normalized): {n_dup:,}")
    print(f"Abstracts mentioning 'withdrawn': {withdrawn:,}")

    # Figures
    cats = df["primary_category"].value_counts().head(30)
    fig, ax = plt.subplots(figsize=(8, 8))
    cats[::-1].plot.barh(ax=ax)
    ax.set_title("Primary category (top 30)")
    ax.set_xlabel("Papers")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "primary_category_top30.png", dpi=150)
    plt.close(fig)

    all_cats = df.explode("categories")["categories"].value_counts()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(len(all_cats)), all_cats.values)
    ax.set_yscale("log")
    ax.set_title("Category frequency, rank order (long-tail view)")
    ax.set_xlabel("Category rank")
    ax.set_ylabel("Papers (log)")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "category_long_tail.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    df["year"].value_counts().sort_index().plot.bar(ax=ax)
    ax.set_title("Papers per year (v1 date)")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "papers_per_year.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["n_words"].clip(upper=df["n_words"].quantile(0.999)), bins=80)
    ax.set_title("Abstract length (words)")
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "abstract_length.png", dpi=150)
    plt.close(fig)

    stats = {
        "rows": len(df),
        "n_primary_categories": int(df["primary_category"].nunique()),
        "n_all_categories": int(len(all_cats)),
        "word_count_percentiles": {str(p): float(v) for p, v in df["n_words"].quantile(pcts).items()},
        "exact_duplicates": n_dup,
        "withdrawn_mentions": withdrawn,
    }
    (config.REPORTS_DIR / "raw_stats.json").write_text(json.dumps(stats, indent=2))
    print(f"\nFigures saved to {config.FIG_DIR}")


if __name__ == "__main__":
    main()
