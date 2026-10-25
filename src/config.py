from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
INTERIM_DIR = ROOT / "data" / "interim"
PROCESSED_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

HF_DATASET = "librarian-bots/arxiv-metadata-snapshot"
TARGET_CATEGORIES = {"cs.CL", "cs.LG", "cs.AI", "stat.ML"}

SEED = 42
SPLIT_FRACTIONS = {"train": 0.90, "val": 0.05, "test": 0.05}

# Text format used for training: "title\n\nabstract" or abstract only.
# Decide once and keep it identical across all selection strategies.
INCLUDE_TITLE = True

# Starting thresholds. Tune them after looking at the inspect_corpus.py output.
MIN_WORDS = 20
MAX_WORDS = 800


def ensure_dirs():
    for d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, REPORTS_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
