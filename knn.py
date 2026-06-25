import os
import numpy as np
import librosa
from collections import Counter, defaultdict

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# ===== CONFIG =====
DATA_DIR = "data"
SR = 16000

WINDOW_SEC = 5
STRIDE_SEC = 1
MIN_SEC = 3

LOUD_SEG_SEC = 1.2

ENERGY_FILTER = True
RMS_THRESHOLD = 0.01

N_SPLITS = 5
RANDOM_SEED = 42

K = 7
# ==================

WINDOW_SAMPLES = WINDOW_SEC * SR
STRIDE_SAMPLES = STRIDE_SEC * SR
MIN_SAMPLES = MIN_SEC * SR


def pad_with_silence(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) >= target_len:
        return y[:target_len]
    return np.pad(y, (0, target_len - len(y)), mode="constant")


def rms_energy(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(y * y) + 1e-12))


def loudest_segment(y: np.ndarray, sr: int, seg_sec: float) -> np.ndarray:
    seg_len = int(seg_sec * sr)
    if len(y) <= seg_len:
        return pad_with_silence(y, seg_len)

    frame_length = int(0.05 * sr)
    hop_length = int(0.01 * sr)
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    if len(rms) == 0:
        return pad_with_silence(y[:seg_len], seg_len)

    frames_per_seg = max(1, seg_len // hop_length)
    if len(rms) <= frames_per_seg:
        start_sample = 0
    else:
        csum = np.cumsum(rms)
        seg_sums = csum[frames_per_seg:] - csum[:-frames_per_seg]
        best_frame = int(np.argmax(seg_sums))
        start_sample = best_frame * hop_length

    seg = y[start_sample:start_sample + seg_len]
    return pad_with_silence(seg, seg_len)


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    y = loudest_segment(y, sr, LOUD_SEG_SEC)

    feats = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    def stats(mat):
        return np.concatenate([np.mean(mat, axis=1), np.std(mat, axis=1)], axis=0)

    feats.append(stats(mfcc))
    feats.append(stats(d1))
    feats.append(stats(d2))

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y)

    def stats1(x):
        return np.array([np.mean(x), np.std(x)], dtype=np.float32)

    feats.append(stats1(centroid))
    feats.append(stats1(bandwidth))
    feats.append(stats1(rolloff))
    feats.append(stats1(zcr))

    flat = librosa.feature.spectral_flatness(y=y)
    feats.append(stats1(flat))

    rms = librosa.feature.rms(y=y)
    feats.append(np.array([np.std(rms)], dtype=np.float32))

    try:
        f0 = librosa.yin(y, fmin=50, fmax=2000, sr=sr)
        f0 = f0[np.isfinite(f0)]
        feats.append(np.array([np.mean(f0), np.std(f0)], dtype=np.float32) if len(f0) else np.array([0.0, 0.0], dtype=np.float32))
    except Exception:
        feats.append(np.array([0.0, 0.0], dtype=np.float32))

    return np.concatenate(feats).astype(np.float32)


def build_dataset(data_dir: str):
    X_list, y_list, g_list = [], [], []

    for label in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, label)
        if not os.path.isdir(class_path):
            continue

        for fname in sorted(os.listdir(class_path)):
            if not fname.lower().endswith(".wav"):
                continue

            fpath = os.path.join(class_path, fname)
            y, _ = librosa.load(fpath, sr=SR, mono=True)
            if len(y) < MIN_SAMPLES:
                continue

            max_start = max(1, len(y) - MIN_SAMPLES + 1)
            for start in range(0, max_start, STRIDE_SAMPLES):
                clip = y[start:start + WINDOW_SAMPLES]
                if len(clip) < MIN_SAMPLES:
                    continue
                clip = pad_with_silence(clip, WINDOW_SAMPLES)

                if ENERGY_FILTER and rms_energy(clip) < RMS_THRESHOLD:
                    continue

                X_list.append(extract_features(clip, SR))
                y_list.append(label)
                g_list.append(fpath)  # group by source wav

    X = np.vstack(X_list)
    y = np.array(y_list)
    groups = np.array(g_list)
    return X, y, groups


def main():
    print("Building dataset...")
    X, y, groups = build_dataset(DATA_DIR)
    print("Total clips:", len(y))
    print("Class counts:", Counter(y.tolist()))

    gkf = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=K))
    ])

    scores = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        acc = accuracy_score(y[te], pred)
        scores.append(acc)
        print(f"Fold {fold} acc: {acc:.4f}")

    scores = np.array(scores)
    print(f"\nKNN(k={K}) GroupCV mean={scores.mean():.4f} std={scores.std():.4f}")


if __name__ == "__main__":
    main()