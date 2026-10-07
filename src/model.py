import tensorflow as tf
from tensorflow.keras import layers, Model


NUM_CLASSES = 8


def build_songnet(input_shape):

    inputs = layers.Input(
        shape=input_shape,
        name="mel_spectrogram"
    )

    # Add channel dimension: (time, frequency) -> (time, frequency, 1)
    x = layers.Reshape(
        (input_shape[0], input_shape[1], 1)
    )(inputs)

    # CNN block 1
    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(0.25)(x)

    # CNN block 2
    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(0.25)(x)

    # CNN block 3
    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(
        (2, 2)
    )(x)

    x = layers.Dropout(0.30)(x)

    # CNN block 4
    x = layers.Conv2D(
        256,
        (3, 3),
        padding="same",
        activation="relu"
    )(x)

    x = layers.BatchNormalization()(x)

    # Convert feature maps into one vector
    x = layers.GlobalAveragePooling2D()(x)

    # Fully connected classifier
    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.40)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="genre_classifier"
    )(x)

    return Model(
        inputs=inputs,
        outputs=outputs,
        name="SongNet"
    )