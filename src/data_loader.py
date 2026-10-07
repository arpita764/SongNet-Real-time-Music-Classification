from pathlib import Path

import numpy as np
import pandas as pd


GENRES = [
    "Electronic",
    "Experimental",
    "Folk",
    "Instrumental",
    "International",
    "Pop",
    "Hip-Hop",
    "Rock",
]

NUM_CLASSES = len(GENRES)


def load_metadata(csv_path):
    df = pd.read_csv(csv_path)

    required_columns = {
        "track_id",
        "genre",
        "label",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    return df


def get_mel_path(track_id, split, mel_root):
    return (
        Path(mel_root)
        / split
        / f"{int(track_id):06d}.npy"
    )


def load_dataset(
    csv_path,
    split,
    mel_root="../processed/mel_spectrograms"
):
    """
    Load all available mel-spectrograms for a split.

    Returns:
        X: shape (samples, 128, 1292)
        y: integer labels
        metadata: rows corresponding to successfully loaded files
    """

    df = load_metadata(csv_path)

    X = []
    y = []
    valid_rows = []

    missing_files = []

    for _, row in df.iterrows():

        path = get_mel_path(
            row["track_id"],
            split,
            mel_root
        )

        if not path.exists():
            missing_files.append(
                int(row["track_id"])
            )
            continue

        mel = np.load(path)

        if mel.ndim != 2:
            raise ValueError(
                f"Unexpected shape {mel.shape} "
                f"for {path}"
            )

        X.append(mel.astype(np.float32))
        y.append(int(row["label"]))
        valid_rows.append(row)

    if not X:
        raise RuntimeError(
            f"No mel-spectrograms found for {split}"
        )

    X = np.stack(X)
    y = np.array(y, dtype=np.int64)

    valid_metadata = pd.DataFrame(
        valid_rows
    ).reset_index(drop=True)

    print(f"\nLoaded {split} dataset")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Missing:", len(missing_files))

    if missing_files:
        print(
            "Missing track IDs:",
            missing_files
        )

    return X, y, valid_metadata