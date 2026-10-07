from pathlib import Path

import pandas as pd


CSV_FILES = {
    "train": "../train_model_ready.csv",
    "val": "../val_model_ready.csv",
    "test": "../test_model_ready.csv",
}

MEL_ROOT = Path("../processed/mel_spectrograms")


def check_split(split):
    df = pd.read_csv(CSV_FILES[split])

    folder = MEL_ROOT / split

    found = 0
    missing = []

    for track_id in df["track_id"]:
        filename = f"{int(track_id):06d}.npy"
        path = folder / filename

        if path.exists():
            found += 1
        else:
            missing.append(filename)

    print(f"\n{split.upper()}")
    print("-" * 40)
    print("CSV rows:", len(df))
    print("Mel files found:", found)
    print("Missing:", len(missing))

    if missing:
        print("Missing files:")
        for name in missing:
            print(" ", name)


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        check_split(split)