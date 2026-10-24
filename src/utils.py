"""Small shared helpers: logging, hashing, JSON, Parquet writing, plan capture."""
import contextlib
import hashlib
import io
import json
import logging
from datetime import date, datetime
from pathlib import Path

from src.config import OUTPUT_FILES_PER_TABLE


def get_logger(name: str) -> logging.Logger:
    """Return a logger that prints timestamped messages to the console."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def sha256_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file in 1 MB chunks so large files never load fully into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")


def write_parquet(df, path: Path, sort_cols=None,
                  num_files: int = OUTPUT_FILES_PER_TABLE) -> None:
    """Write a DataFrame as a small number of Parquet files.

    Spark writes one file per partition. repartition(n) fixes the file count
    (it costs one shuffle, but keeps upstream stages parallel, unlike
    coalesce(n), which would also shrink the upstream stage to n tasks).
    Sorting within the partition by (city, date) stores similar rows together,
    which gives Parquet tighter min/max statistics for filtering by city.
    """
    out = df.repartition(num_files)
    if sort_cols:
        out = out.sortWithinPartitions(*sort_cols)
    out.write.mode("overwrite").parquet(path.as_posix())


def parquet_file_stats(path: Path) -> dict:
    """Count the data files and bytes Spark wrote for one Parquet table."""
    files = [p for p in path.rglob("*.parquet")]
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def explain_to_text(df, mode: str = "formatted") -> str:
    """Capture df.explain() output (PySpark prints it) as a string."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        df.explain(mode=mode)
    return buffer.getvalue()
