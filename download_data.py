"""Download the three dataset configs into data/ at a fixed source revision.

The source dataset is updated daily, so downloads are pinned to the revision
used for this project (verified in Phase 1: the local CSVs are value-identical
to this revision). Existing CSVs are never overwritten unless --force is given.

Usage:
    python download_data.py            # download missing files only
    python download_data.py --force    # re-download and overwrite
"""
import argparse
import hashlib
from pathlib import Path

from datasets import load_dataset

REPO = "AkshayKarthick/india-weather-air-quality"
REVISION = "339a4f37b76b9d2bae2f4b34cb189244019b4ed1"
CONFIGS = ["weather", "air_quality", "cities"]
DATA_DIR = Path(__file__).resolve().parent / "data"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="overwrite existing CSVs")
    args = parser.parse_args()
    DATA_DIR.mkdir(exist_ok=True)

    for config in CONFIGS:
        target = DATA_DIR / f"{config}.csv"
        if target.exists() and not args.force:
            print(f"Skipping {config}: {target.name} already exists (use --force to replace)")
            continue
        print(f"Downloading {config} @ {REVISION[:8]} ...")
        dataset = load_dataset(REPO, config, split="train", revision=REVISION)
        dataset.to_csv(str(target))
        print(f"Saved {len(dataset)} rows to {target} (sha256 {sha256(target)[:16]}...)")

    print("Done. The SHA-256 of each analysed file is recorded in "
          "outputs/reports/validation_report.json. A fresh download can differ byte-wise "
          "if the `datasets` library writes CSVs differently, even at the same revision.")


if __name__ == "__main__":
    main()
