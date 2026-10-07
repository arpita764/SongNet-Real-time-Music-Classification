import numpy as np

from data_loader import load_dataset


# Load training data
X_train, y_train, metadata = load_dataset(
    "../train_model_ready.csv",
    "train"
)

print("Before transpose:")
print("X_train shape:", X_train.shape)


# Transpose:
# (samples, 128, 1292)
#        ↓
# (samples, 1292, 128)

X_train = X_train.transpose(0, 2, 1)


print("\nAfter transpose:")
print("X_train shape:", X_train.shape)