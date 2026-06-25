import os
import numpy as np
import librosa
from collections import Counter, defaultdict

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupKFold, GridSearchCV

from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC


# =========================
# CONFIG
# =========================
DATA_DIR = "data"
SR = 16000

WINDOW_SEC = 5
STRIDE_SEC = 1
MIN_SEC = 3

# Focus on the event inside each window
LOUD_SEG_SEC = 1.2

ENERGY_FILTER = True
RMS_THRESHOLD = 0.01

# Downsampling (preserve source diversity)
DOWNSAMPLE_MODE = "cap_idle"   # "cap_idle" or "cap_all" or "none"
IDLE_LABEL_NAME = "Idle wav"
IDLE_CAP = 70
CAP_ALL_MAX = 70

RANDOM_SEED = 42
N_SPLITS = 5
# =========================

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
    """
    Return the loudest contiguous segment of length seg_sec in y,
    based on frame RMS energy. Pads with silence if needed.
    """
    seg_len = int(seg_sec * sr)
    if len(y) <= seg_len:
        return pad_with_silence(y, seg_len)

    frame_length = int(0.05 * sr)  # 50 ms
    hop_length = int(0.01 * sr)    # 10 ms

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
    """
    Extract features from the loudest part of y (reduces fluff/idle/talking).
    Features:
      - MFCC(13) + deltas + delta-deltas: mean & std
      - spectral centroid/bandwidth/rolloff: mean & std
      - ZCR: mean & std
      - spectral flatness: mean & std
      - RMS modulation: std(rms)
      - pitch proxy (YIN f0): mean & std
    """
    # Focus on event region
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
        if len(f0) > 0:
            feats.append(np.array([np.mean(f0), np.std(f0)], dtype=np.float32))
        else:
            feats.append(np.array([0.0, 0.0], dtype=np.float32))
    except Exception:
        feats.append(np.array([0.0, 0.0], dtype=np.float32))

    return np.concatenate(feats).astype(np.float32)


def build_dataset_features(data_dir: str):
    """
    Walk folders -> load wav -> sliding windows -> pad -> energy filter -> feature extraction.
    groups = source filepath to prevent leakage.
    """
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
                g_list.append(fpath)

    X = np.vstack(X_list) if X_list else np.zeros((0, 1), dtype=np.float32)
    y = np.array(y_list)
    groups = np.array(g_list)
    return X, y, groups


def print_counts(y, title):
    c = Counter(y.tolist())
    print(title, c)
    print("min:", min(c.values()), "max:", max(c.values()))


def print_unique_sources(y, groups, title="Unique SOURCE wav files per class:"):
    label_to_sources = defaultdict(set)
    for lab, grp in zip(y, groups):
        label_to_sources[lab].add(grp)

    print("\n" + title)
    for lab in sorted(label_to_sources):
        print(f"{lab}: {len(label_to_sources[lab])}")


def downsample_preserve_sources(X, y, groups, mode="none"):
    """
    Downsample while preserving ALL sources for each label.
    For capped labels, cap clips PER SOURCE (keeps source diversity).
    """
    if mode == "none":
        return X, y, groups

    rng = np.random.default_rng(RANDOM_SEED)

    # label -> source -> indices
    label_source_to_indices = defaultdict(lambda: defaultdict(list))
    for i, (lab, grp) in enumerate(zip(y, groups)):
        label_source_to_indices[lab][grp].append(i)

    keep = []

    for lab, src_map in label_source_to_indices.items():
        sources = list(src_map.keys())

        if mode == "cap_idle" and lab == IDLE_LABEL_NAME:
            cap_total = IDLE_CAP
        elif mode == "cap_all":
            cap_total = CAP_ALL_MAX
        else:
            cap_total = None

        if cap_total is None:
            for s in sources:
                keep.extend(src_map[s])
            continue

        # cap clips per source (so we don't lose sources)
        n_sources = len(sources)
        per_source_cap = max(1, int(np.ceil(cap_total / n_sources)))

        kept_lab = []
        for s in sources:
            idxs = src_map[s]
            if len(idxs) <= per_source_cap:
                kept_lab.extend(idxs)
            else:
                kept_lab.extend(rng.choice(idxs, size=per_source_cap, replace=False).tolist())

        # if still too many, trim randomly but keep sources already represented
        if len(kept_lab) > cap_total:
            kept_lab = rng.choice(kept_lab, size=cap_total, replace=False).tolist()

        keep.extend(kept_lab)

    keep = np.array(sorted(set(keep)), dtype=int)
    return X[keep], y[keep], groups[keep]


def grouped_cv_scores(model, X, y, groups, n_splits=5):
    unique_groups = np.unique(groups)
    n_splits = min(n_splits, len(unique_groups))
    if n_splits < 2:
        raise RuntimeError("Not enough unique source recordings (groups) to do CV.")

    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])
        scores.append(accuracy_score(y[test_idx], pred))
    return np.array(scores)


def main():
    print("Building dataset + extracting features...")
    X, y, groups = build_dataset_features(DATA_DIR)
    print("Total clips:", len(y))
    print_counts(y, "Class counts:")
    print_unique_sources(y, groups)

    X, y, groups = downsample_preserve_sources(X, y, groups, mode=DOWNSAMPLE_MODE)
    print("\nAfter downsampling:", len(y))
    print_counts(y, "Class counts:")
    print_unique_sources(y, groups, title="Unique SOURCE wav files per class (after downsampling):")

    # Baselines
    knn = Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=7))])
    lda = Pipeline([("scaler", StandardScaler()), ("clf", LinearDiscriminantAnalysis())])

    print("\nGrouped CV (no leakage) baseline scores:")
    for name, model in [("kNN(k=7)", knn), ("LDA", lda)]:
        scores = grouped_cv_scores(model, X, y, groups, n_splits=N_SPLITS)
        print(f"{name}: mean={scores.mean():.3f}, std={scores.std():.3f}, folds={len(scores)} -> {np.round(scores,3)}")

    # SVM tuning (grouped CV)
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", class_weight="balanced"))
    ])

    param_grid = {
        "clf__C": [0.25, 0.5, 1, 2, 5, 10, 20, 50],
        "clf__gamma": [0.002, 0.005, 0.01, 0.02, 0.05, "scale"]
    }

    unique_groups = np.unique(groups)
    n_splits = min(N_SPLITS, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)

    print("\nTuning SVM with grouped CV...")
    gs = GridSearchCV(
        svm_pipe,
        param_grid=param_grid,
        cv=gkf.split(X, y, groups=groups),
        scoring="accuracy",
        n_jobs=-1
    )
    gs.fit(X, y)

    print("Best SVM params:", gs.best_params_)
    print("Best CV accuracy:", round(gs.best_score_, 4))


if __name__ == "__main__":
    main()