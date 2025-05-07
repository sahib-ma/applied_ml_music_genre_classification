import os
import random
from preprocessing import preprocess_audio_dataset

# Adjust this path if needed
SPLIT_DIR = 'audio_split/train'
GENRES = os.listdir(SPLIT_DIR)
SAMPLES_PER_GENRE = 1  # Adjust based on how many samples you want to test

audio_files = []
labels = []

for genre in GENRES:
    genre_dir = os.path.join(SPLIT_DIR, genre)
    files = [f for f in os.listdir(genre_dir) if f.endswith('.wav')]
    selected = random.sample(files, min(SAMPLES_PER_GENRE, len(files)))
    
    for f in selected:
        audio_files.append(os.path.join(genre_dir, f))
        labels.append(genre)

# Run preprocessing
X, y, scaler, pca = preprocess_audio_dataset(audio_files, labels)

# Output some info about the result
print(f"Number of total audio clips processed: {len(y)}")
print(f"Shape of processed feature matrix: {X.shape}")
print(f"Labels (first 10): {y[:10]}")
