import os
import random
import shutil
from sklearn.model_selection import train_test_split

DATASET_DIR = 'music_genre_classification/data/genres_original'
OUTPUT_DIR = 'audio_split'

splits = ['train', 'test', 'val']
genres = os.listdir(DATASET_DIR)

for split in splits:
    for genre in genres:
        os.makedirs(os.path.join(OUTPUT_DIR, split, genre), exist_ok=True)

data = []
for genre in genres:
    genre_path = os.path.join(DATASET_DIR, genre)
    for filename in os.listdir(genre_path):
        if filename.endswith('.wav'):
            data.append((os.path.join(genre_path, filename), genre))
            
random.shuffle(data)

files_by_genre = {genre: [] for genre in genres}
for filepath, genre in data:
    files_by_genre[genre].append(filepath)
    
for genre, files in files_by_genre.items():
    train_files, temp_files = train_test_split(files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    
    for f in train_files:
        shutil.copy(f, os.path.join(OUTPUT_DIR, 'train', genre))
    for f in val_files:
        shutil.copy(f, os.path.join(OUTPUT_DIR, 'val', genre))
    for f in test_files:
        shutil.copy(f, os.path.join(OUTPUT_DIR, 'test', genre))