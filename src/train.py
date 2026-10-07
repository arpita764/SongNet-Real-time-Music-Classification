import os
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from data_loader import load_dataset
from model import build_songnet


# -----------------------------
# Paths
# -----------------------------

TRAIN_CSV = "../train_model_ready.csv"
VAL_CSV = "../val_model_ready.csv"

MODEL_DIR = "../models"
RESULTS_DIR = "../results"


# -----------------------------
# Configuration
# -----------------------------

EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0005


# -----------------------------
# Main
# -----------------------------

def main():

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )

    print("=" * 60)
    print("SONGNET TRAINING - 2D CNN")
    print("=" * 60)

    # -------------------------
    # Load training data
    # -------------------------

    print("\nLoading training data...")

    X_train, y_train, _ = load_dataset(
        TRAIN_CSV,
        "train"
    )

    # -------------------------
    # Load validation data
    # -------------------------

    print("\nLoading validation data...")

    X_val, y_val, _ = load_dataset(
        VAL_CSV,
        "val"
    )

    # -------------------------
    # Transpose
    # -------------------------

    print("\nTransposing data...")

    X_train = X_train.transpose(
        0, 2, 1
    )

    X_val = X_val.transpose(
        0, 2, 1
    )

    print(
        "Training shape:",
        X_train.shape
    )

    print(
        "Validation shape:",
        X_val.shape
    )

    # -------------------------
    # Build model
    # -------------------------

    print("\nBuilding SongNet model...")

    model = build_songnet(
        input_shape=(
            X_train.shape[1],
            X_train.shape[2]
        )
    )

    model.summary()

    # -------------------------
    # Optimizer
    # -------------------------

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    # -------------------------
    # Compile
    # -------------------------

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # -------------------------
    # Callbacks
    # -------------------------

    checkpoint_path = (
        MODEL_DIR +
        "/songnet_best.keras"
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=6,
        restore_best_weights=True,
        verbose=1
    )

    # -------------------------
    # Train
    # -------------------------

    print("\nStarting training...\n")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_val,
            y_val
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[
            checkpoint,
            reduce_lr,
            early_stopping
        ],
        verbose=1
    )

    # -------------------------
    # Save final model
    # -------------------------

    final_model_path = (
        MODEL_DIR +
        "/songnet_final.keras"
    )

    model.save(
        final_model_path
    )

    print(
        "\nFinal model saved:",
        final_model_path
    )

    print(
        "Best model saved:",
        checkpoint_path
    )

    # -------------------------
    # Save training history
    # -------------------------

    np.save(
        RESULTS_DIR + "/history.npy",
        history.history
    )

    # -------------------------
    # Plot loss
    # -------------------------

    plt.figure()

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SongNet Training and Validation Loss")
    plt.legend()

    plt.savefig(
        RESULTS_DIR + "/loss.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # -------------------------
    # Plot accuracy
    # -------------------------

    plt.figure()

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("SongNet Training and Validation Accuracy")
    plt.legend()

    plt.savefig(
        RESULTS_DIR + "/accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nTraining complete.")
    print("Results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()