from datasets import load_dataset
from pathlib import Path

Path("data").mkdir(exist_ok=True)

configs = ["weather", "air_quality", "cities"]

for config in configs:
    print(f"Downloading {config}...")

    dataset = load_dataset(
        "AkshayKarthick/india-weather-air-quality",
        config,
        split="train"
    )

    dataset.to_csv(f"data/{config}.csv")

    print(f"Saved {len(dataset)} rows to data/{config}.csv")

print("All datasets downloaded.")