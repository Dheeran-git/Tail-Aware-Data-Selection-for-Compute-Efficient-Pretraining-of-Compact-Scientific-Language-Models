# Tail-Aware Data Selection for Compact Scientific LMs

Part 1: corpus collection. Source: librarian-bots/arxiv-metadata-snapshot
(pinned revision in data/raw/source_info.json).

Run order:
  uv run python src/build_raw.py
  uv run python src/inspect_corpus.py
  uv run python src/clean_corpus.py
  uv run python src/split_corpus.py
