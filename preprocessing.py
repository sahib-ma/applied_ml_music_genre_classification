import numpy as np
import librosa
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA


def split_audio_into_clips(file_path, clip_duration=10):
    """Split audio file into 10-second clips."""
    y, sr = librosa.load(file_path, sr=None)
    clip_samples = clip_duration * sr
    return [y[i:i+clip_samples] for i in range(0, len(y), clip_samples) if len(y[i:i+clip_samples]) == clip_samples]

def extract_mel_spectrogram(audio_clip, sr=22050, n_mels=128):
    """Convert audio clip to Mel spectrogram."""
    S = librosa.feature.melspectrogram(y=audio_clip, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB.flatten()

def preprocess_audio_dataset(audio_files, labels, sr=22050, n_mels=128, pca_components=100):
    X_features = []
    y_expanded = []

    for file_path, label in zip(audio_files, labels):
        clips = split_audio_into_clips(file_path)
        for clip in clips:
            mel_spec = extract_mel_spectrogram(clip, sr=sr, n_mels=n_mels)
            X_features.append(mel_spec)
            y_expanded.append(label)

    # Normalize between 0 and 1
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_features)

    # Apply PCA
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca, y_expanded, scaler, pca
