import os
import librosa
import numpy as np

DATA_DIR = "./Data"  
SR = 16000
WINDOW_SEC = 5
STRIDE_SEC = 1
MIN_SEC = 3

WINDOW_SAMPLES = WINDOW_SEC * SR
STRIDE_SAMPLES = STRIDE_SEC * SR
MIN_SAMPLES = MIN_SEC * SR

def pad_with_silence(y, target_len):
    if len(y) >= target_len:
        return y[:target_len]
    return np.pad(y, (0, target_len - len(y)), mode="constant")

dataset = []

for class_name in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_name)
    if not os.path.isdir(class_path):
        continue

    for fname in os.listdir(class_path):
        if not fname.endswith(".wav"):
            continue

        path = os.path.join(class_path, fname)
        y, sr = librosa.load(path, sr=SR, mono=True)

        if len(y) < MIN_SAMPLES:
            continue 

        
        for start in range(0, len(y), STRIDE_SAMPLES):
            clip = y[start:start + WINDOW_SAMPLES]

            if len(clip) < MIN_SAMPLES:
                continue

            clip = pad_with_silence(clip, WINDOW_SAMPLES)

            dataset.append({
                "audio": clip,
                "label": class_name,
                "source": fname
            })

print("Total clips:", len(dataset))


from collections import Counter
counts = Counter([x["label"] for x in dataset])
print(counts)
print("min:", min(counts.values()), "max:", max(counts.values()))