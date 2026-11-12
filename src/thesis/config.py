"""Paths and run settings. Every setting can be overridden with an environment variable.

THESIS_DATA     folder holding csv_data/ (the raw NetEase CSVs) and parquet/ (derived data).
                Default: <repo>/data if it exists, otherwise the folder that contains the repository.
THESIS_OUTPUTS  folder for results (default <repo>/outputs)
THESIS_MEM      DuckDB memory limit (default 8GB);  THESIS_THREADS  DuckDB threads (default 8)
THESIS_SAMPLE   10 = run the panel notebooks on the 10% user sample (quick tests); default 100 = all users
"""
import os
from pathlib import Path
import duckdb

REPO = Path(__file__).resolve().parents[2]
_default_data = REPO / "data" if (REPO / "data").exists() else REPO.parent
DATA = Path(os.environ.get("THESIS_DATA", _default_data))
RAW, P = DATA / "csv_data", DATA / "parquet"
O = Path(os.environ.get("THESIS_OUTPUTS", REPO / "outputs"))
MEM = os.environ.get("THESIS_MEM", "8GB")
THREADS = int(os.environ.get("THESIS_THREADS", "8"))
TEMP = str(P / "_duckdb_tmp")                     # DuckDB spills here when a query exceeds MEM
SAMPLE = int(os.environ.get("THESIS_SAMPLE", "100"))
TAG = "" if SAMPLE == 100 else f"_s{SAMPLE}"      # suffix of every panel file and result on the sample
SEED = 42
P.mkdir(parents=True, exist_ok=True); O.mkdir(parents=True, exist_ok=True)

def connect(database=":memory:", keep_order=True):
    """DuckDB connection with the memory, thread and spill settings above. keep_order=False lets DuckDB return rows
    in any order when a query has no ORDER BY (less memory on large exports; used only where the order is irrelevant)."""
    con = duckdb.connect(str(database))
    con.execute(f"SET memory_limit='{MEM}'; SET threads={THREADS}; SET temp_directory='{TEMP}'")
    if not keep_order: con.execute("SET preserve_insertion_order=false")
    return con
