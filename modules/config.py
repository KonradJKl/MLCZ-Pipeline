import os
from pathlib import Path
from dotenv import load_dotenv

# Paths can be overridden in .env, relative paths are resolved from the repo root
ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / os.getenv("DATA_DIR", "data")
OUTPUT_DIR = ROOT / os.getenv("OUTPUT_DIR", "processed")
LOGS_DIR = ROOT / os.getenv("LOGS_DIR", "logs")
DOCS_DIR = ROOT / os.getenv("DOCS_DIR", "docs")
LMDB_PATH = OUTPUT_DIR / "MLCZ.lmdb"
PARQUET_PATH = OUTPUT_DIR / "MLCZ.parquet"
