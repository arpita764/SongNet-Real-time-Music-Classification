from data_loader import load_dataset


if __name__ == "__main__":

    X_train, y_train, train_meta = load_dataset(
        "../train_model_ready.csv",
        "train"
    )

    X_val, y_val, val_meta = load_dataset(
        "../val_model_ready.csv",
        "val"
    )

    X_test, y_test, test_meta = load_dataset(
        "../test_model_ready.csv",
        "test"
    )

    print("\nFINAL CHECK")
    print("=" * 50)

    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)