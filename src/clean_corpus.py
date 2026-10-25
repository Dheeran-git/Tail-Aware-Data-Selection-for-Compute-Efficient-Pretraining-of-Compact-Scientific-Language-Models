import hashlib
import json
import re

import pandas as pd

import config

WITHDRAWN_RE = re.compile(
    r"(?i)\b(this|the)\s+(paper|article|submission|manuscript|preprint)\s+"
    r"(has\s+been|is|was)\s+withdrawn"
)


def norm_key(s: str) -> str:
    return hashlib.md5(re.sub(r"[^a-z0-9]", "", s.lower()).encode()).hexdigest()


def main():
    config.ensure_dirs()
    df = pd.read_parquet(config.RAW_DIR / "arxiv_cs_ml.parquet")
    df["title"] = df["title"].fillna("").str.strip()
    df["abstract"] = df["abstract"].fillna("").str.strip()
    df["n_words"] = df["abstract"].str.split().str.len()

    log = {"input_rows": len(df)}
    removed_parts = []

    def drop(frame, mask, reason):
        removed_parts.append(frame.loc[mask, ["id"]].assign(reason=reason))
        log[reason] = int(mask.sum())
        return frame.loc[~mask]

    df = drop(df, df["n_words"] == 0, "empty_abstract")
    df = drop(df, (df["n_words"] > 0) & (df["n_words"] < config.MIN_WORDS), "too_short")
    df = drop(df, df["n_words"] > config.MAX_WORDS, "too_long")
    df = drop(df, df["abstract"].map(lambda s: bool(WITHDRAWN_RE.search(s))), "withdrawn")

    # Exact duplicates (after normalization): keep the lowest id, deterministic.
    df = df.sort_values("id").reset_index(drop=True)
    df["_key"] = df["abstract"].map(norm_key)
    df = drop(df, df["_key"].duplicated(keep="first"), "exact_duplicate")
    df = df.drop(columns="_key").reset_index(drop=True)

    if config.INCLUDE_TITLE:
        df["text"] = df["title"] + "\n\n" + df["abstract"]
    else:
        df["text"] = df["abstract"]

    log["output_rows"] = len(df)
    log["settings"] = {
        "MIN_WORDS": config.MIN_WORDS,
        "MAX_WORDS": config.MAX_WORDS,
        "INCLUDE_TITLE": config.INCLUDE_TITLE,
    }

    df.to_parquet(config.INTERIM_DIR / "cleaned.parquet", index=False)
    pd.concat(removed_parts).to_parquet(config.INTERIM_DIR / "removed.parquet", index=False)
    (config.REPORTS_DIR / "cleaning_report.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
