import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from tensorflow.keras.models import load_model

from data_loader import load_dataset


TEST_CSV = "../test_model_ready.csv"
MODEL_PATH = "../models/songnet_best.keras"

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


def main():

    print("Loading test data...")

    X_test, y_test, _ = load_dataset(
        TEST_CSV,
        "test"
    )

    X_test = X_test.transpose(
        0, 2, 1
    )

    print(
        "Test shape:",
        X_test.shape
    )

    print("Loading trained model...")

    model = load_model(
        MODEL_PATH
    )

    print("Generating predictions...")

    predictions = model.predict(
        X_test
    )

    print(
        "Prediction shape:",
        predictions.shape
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(
        "\nTest Accuracy:",
        accuracy
    )

    print(
        "\nClassification Report:"
    )

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=GENRES,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=GENRES
    )

    fig, ax = plt.subplots(figsize=(9, 8))

    display.plot(
        ax=ax,
        xticks_rotation=45,
        values_format="d"
    )

    plt.title("SongNet Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        "../results/confusion_matrix.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nConfusion matrix saved to ../results/confusion_matrix.png")


if __name__ == "__main__":
    main()