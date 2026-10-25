import ast
import json
from datetime import datetime, timezone

import pandas as pd
from datasets import load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm

import config


def clean_none(x):
    if x is None:
        return None
    s = str(x).strip()
    return None if s in ("", "None", "nan") else s


def parse_versions(v):
    if v is None:
        return []
    if isinstance(v, str):
        try:
            v = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return []
    return list(v)


def main():
    config.ensure_dirs()
    info_path = config.RAW_DIR / "source_info.json"

    # Pin the dataset revision so the corpus is reproducible.
    if info_path.exists():
        revision = json.loads(info_path.read_text())["revision"]
        print(f"Reusing pinned revision: {revision}")
    else:
        revision = HfApi().dataset_info(config.HF_DATASET).sha
        print(f"Pinning current revision: {revision}")

    ds = load_dataset(
        config.HF_DATASET, split="train", streaming=True, revision=revision
    )

    rows, scanned = [], 0
    for r in tqdm(ds, desc="Scanning snapshot"):
        scanned += 1
        cats = (r.get("categories") or "").split()
        if not config.TARGET_CATEGORIES.intersection(cats):
            continue
        vers = parse_versions(r.get("versions"))
        v1 = vers[0].get("created") if vers and isinstance(vers[0], dict) else None
        rows.append(
            {
                "id": r["id"],
                "title": " ".join((r.get("title") or "").split()),
                "abstract": " ".join((r.get("abstract") or "").split()),
                "categories": cats,
                "primary_category": cats[0],
                "v1_created": v1,
                "n_versions": len(vers),
                "update_date": str(r.get("update_date")),
                "journal_ref": clean_none(r.get("journal-ref")),
                "license": clean_none(r.get("license")),
            }
        )

    df = pd.DataFrame(rows)
    df["v1_created"] = pd.to_datetime(df["v1_created"], errors="coerce", utc=True)
    out = config.RAW_DIR / "arxiv_cs_ml.parquet"
    df.to_parquet(out, index=False)

    info = {
        "dataset": config.HF_DATASET,
        "revision": revision,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "target_categories": sorted(config.TARGET_CATEGORIES),
        "records_scanned": scanned,
        "records_kept": len(df),
    }
    info_path.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
