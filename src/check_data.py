from pathlib import Path
from collections import Counter

import numpy as np


DATA_ROOT = Path("../processed/mel_spectrograms")


def check_split(split_name):
    folder = DATA_ROOT / split_name
    files = sorted(folder.glob("*.npy"))

    print(f"\n{split_name.upper()}")
    print("-" * 40)
    print("Number of files:", len(files))

    shapes = Counter()
    dtypes = Counter()

    for file in files:
        try:
            arr = np.load(file)

            shapes[arr.shape] += 1
            dtypes[str(arr.dtype)] += 1

        except Exception as e:
            print("ERROR:", file)
            print(e)

    print("Shapes:")
    for shape, count in shapes.items():
        print(f"  {shape}: {count}")

    print("Dtypes:")
    for dtype, count in dtypes.items():
        print(f"  {dtype}: {count}")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        check_split(split)