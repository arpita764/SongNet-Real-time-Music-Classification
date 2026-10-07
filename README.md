# SongNet - Real-Time Music Genre Classification

SongNet is a deep learning project that classifies music into different genres using audio features and a Convolutional Neural Network (CNN).

## Genres

The model classifies songs into 8 genres:

- Electronic
- Experimental
- Folk
- Instrumental
- International
- Pop
- Hip-Hop
- Rock

## Features

- Audio preprocessing using mel-spectrograms
- CNN-based music genre classification
- Model evaluation using accuracy and confusion matrix
- Single-song genre prediction
- Genre probability and confidence output

## Model

The model takes a mel-spectrogram as input with shape:

```text
(1292, 128)