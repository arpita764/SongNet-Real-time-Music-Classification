import tensorflow as tf

from data_loader import load_dataset
from model import build_songnet


if __name__ == "__main__":

    X_train, y_train, _ = load_dataset(
        "../train_model_ready.csv",
        "train"
    )

    # Convert:
    # (samples, 128, 1292)
    # to:
    # (samples, 1292, 128)

    X_train = X_train.transpose(
        0, 2, 1
    )

    print("\nAfter transpose:")
    print("X_train:", X_train.shape)

    input_shape = X_train.shape[1:]

    print(
        "Model input shape:",
        input_shape
    )

    model = build_songnet(
        input_shape
    )

    model.summary()

    # Test one real sample
    sample = X_train[:1]

    output = model.predict(
        sample,
        verbose=0
    )

    print("\nReal-data test")
    print("=" * 50)
    print("Input:", sample.shape)
    print("Output:", output.shape)

    print(
        "Probabilities at first timestep:",
        output[0, 0]
    )

    print(
        "Sum:",
        output[0, 0].sum()
    )