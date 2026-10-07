import numpy as np
import librosa
from tensorflow.keras.models import load_model


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


def create_mel_spectrogram(audio_path):

    SAMPLE_RATE = 22050
    DURATION = 30
    N_MELS = 128
    N_FFT = 2048
    HOP_LENGTH = 512

    audio, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        duration=DURATION
    )

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    return mel_db.astype(np.float32)


def predict_song(audio_path):

    model = load_model(MODEL_PATH)

    mel = create_mel_spectrogram(audio_path)

    # Convert:
    # (128, time)
    # -> (time, 128)

    mel = mel.T

    # Add batch dimension
    mel = np.expand_dims(
        mel,
        axis=0
    )

    predictions = model.predict(mel, verbose=0)
    print("Prediction shape:", predictions.shape)
    probabilities = predictions[0]
    predicted_index = np.argmax(probabilities)
    predicted_genre = GENRES[predicted_index]
    return predicted_genre, probabilities

if __name__ == "__main__":

    audio_path = input("Enter path to audio file: ").strip()

    genre, probabilities = predict_song(audio_path)

    print("\n==============================")
    print("SONGNET PREDICTION")
    print("==============================")

    print("Predicted Genre:", genre)

    print("\nGenre Probabilities:")

    for genre_name, probability in zip(GENRES, probabilities):
        print(f"{genre_name:15s}: {probability:.4f}")

    print("\nConfidence:",
          f"{probabilities.max() * 100:.2f}%")