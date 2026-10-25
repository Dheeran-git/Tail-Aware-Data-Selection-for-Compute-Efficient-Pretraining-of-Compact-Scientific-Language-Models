import hashlib
import json

import pandas as pd

import config


def assign_split(paper_id: str) -> str:
    h = int(hashlib.md5(f"{config.SEED}:{paper_id}".encode()).hexdigest(), 16) % 10000
    f = config.SPLIT_FRACTIONS
    cut1 = round(f["train"] * 10000)
    cut2 = round((f["train"] + f["val"]) * 10000)
    if h < cut1:
        return "train"
    if h < cut2:
        return "val"
    return "test"


def main():
    config.ensure_dirs()
    df = pd.read_parquet(config.INTERIM_DIR / "cleaned.parquet")
    df["split"] = df["id"].map(assign_split)

    report = {"seed": config.SEED, "fractions": config.SPLIT_FRACTIONS, "splits": {}}
    for name in ("train", "val", "test"):
        part = df[df["split"] == name].reset_index(drop=True)
        part.to_parquet(config.PROCESSED_DIR / f"{name}.parquet", index=False)
        report["splits"][name] = {
            "rows": len(part),
            "approx_words": int(part["text"].str.split().str.len().sum()),
            "primary_categories": int(part["primary_category"].nunique()),
        }

    df[["id", "primary_category", "split"]].to_parquet(
        config.PROCESSED_DIR / "split_manifest.parquet", index=False
    )

    # Sanity: do category proportions match across splits?
    prop = (
        df.groupby("split")["primary_category"]
        .value_counts(normalize=True)
        .unstack(0)
        .fillna(0)
    )
    top = prop.sort_values("train", ascending=False).head(10)
    print("Top-10 primary category proportions per split:")
    print(top.round(4).to_string())

    test_counts = df[df["split"] == "test"]["primary_category"].value_counts()
    report["test_categories_with_100plus_docs"] = int((test_counts >= 100).sum())
    report["test_categories_with_under_30_docs"] = int((test_counts < 30).sum())

    (config.REPORTS_DIR / "split_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
