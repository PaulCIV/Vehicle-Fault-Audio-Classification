import os
import numpy as np
import librosa
from collections import Counter

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# ===== CONFIG =====
DATA_DIR = "./data"
SR = 16000
WINDOW_SEC = 5
STRIDE_SEC = 1
MIN_SEC = 3
LOUD_SEG_SEC = 1.2
ENERGY_FILTER = True
RMS_THRESHOLD = 0.01

WINDOW_SAMPLES = WINDOW_SEC * SR
STRIDE_SAMPLES = STRIDE_SEC * SR
MIN_SAMPLES = MIN_SEC * SR


def pad_with_silence(y, target_len):
    return y[:target_len] if len(y) >= target_len else np.pad(y, (0, target_len - len(y)))


def rms_energy(y):
    return float(np.sqrt(np.mean(y * y) + 1e-12))


def loudest_segment(y, sr, seg_sec):
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
        start_sample = int(np.argmax(seg_sums)) * hop_length

    return pad_with_silence(y[start_sample:start_sample + seg_len], seg_len)


def extract_features(y, sr):
    y = loudest_segment(y, sr, LOUD_SEG_SEC)
    feats = []

    # MFCC + deltas
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    d1 = librosa.feature.delta(mfcc)
    d2 = librosa.feature.delta(mfcc, order=2)

    def stats(x):
        return np.array([np.mean(x), np.std(x)], dtype=np.float32)

    feats += [stats(mfcc), stats(d1), stats(d2)]

    # Spectral features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)

    def stats1(x):
        return np.array([np.mean(x), np.std(x)], dtype=np.float32)

    feats += [stats1(centroid), stats1(bandwidth), stats1(rolloff), stats1(zcr)]

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

    return np.concatenate(feats, axis=0)


# ---------- synthetic (cVAE) helpers ----------
def synth_spec_to_audio(spec_db: np.ndarray) -> np.ndarray:
    """Convert a generated log-mel spectrogram (dB) back to waveform using librosa inversion."""
    mel_power = librosa.db_to_power(spec_db)
    try:
        y = librosa.feature.inverse.mel_to_audio(
            mel_power,
            sr=SR,
            n_fft=2048,
            hop_length=256,
            power=2.0,
        )
    except Exception:
        y = np.zeros(int(WINDOW_SEC * SR), dtype=np.float32)
    return y.astype(np.float32)


def load_synthetic_features(synth_dir: str, valid_labels: set):
    """Load synthetic spectrogram .npy, invert to audio, extract_features -> Xsyn, ysyn."""
    if synth_dir is None or not os.path.isdir(synth_dir):
        return np.zeros((0, 1), dtype=np.float32), np.array([], dtype=object)

    X_list, y_list = [], []
    for cls_name in sorted(os.listdir(synth_dir)):
        cls_dir = os.path.join(synth_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        if cls_name not in valid_labels:
            continue
        for fn in os.listdir(cls_dir):
            if not fn.endswith(".npy"):
                continue
            spec = np.load(os.path.join(cls_dir, fn)).astype(np.float32)
            if spec.ndim != 2:
                continue
            y_audio = synth_spec_to_audio(spec)
            X_list.append(extract_features(y_audio, SR))
            y_list.append(cls_name)

    if not X_list:
        return np.zeros((0, 1), dtype=np.float32), np.array([], dtype=object)

    return np.vstack(X_list), np.array(y_list, dtype=object)


def build_dataset(data_dir):
    X_list, y_list, g_list = [], [], []

    for label in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, label)
        if not os.path.isdir(class_path):
            continue

        for fname in sorted(os.listdir(class_path)):
            if not fname.lower().endswith(".wav"):
                continue

            fpath = os.path.join(class_path, fname)
            try:
                y, _ = librosa.load(fpath, sr=SR, mono=True)
            except Exception:
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

                # group by SOURCE FILE to avoid leakage
                g_list.append(fpath)

    return np.vstack(X_list), np.array(y_list), np.array(g_list)


def train_and_eval(use_synth: bool = False, synth_dir: str = "synthetic_specs", data_dir: str = "data"):
    print("Building dataset...")
    X, y, groups = build_dataset(data_dir)
    print("Total clips:", len(y))
    print("Class counts:", Counter(y.tolist()))

    n_splits = min(5, len(np.unique(groups)))
    gkf = GroupKFold(n_splits=n_splits)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC())
    ])

    param_grid = {
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", "auto"],
        "clf__kernel": ["rbf"]
    }

    print("Tuning SVM with grouped CV...")
    gs = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=gkf,
        scoring="accuracy",
        n_jobs=-1
    )
    gs.fit(X, y, groups=groups)
    best_model = gs.best_estimator_
    print("Best grouped-CV accuracy:", round(gs.best_score_, 4))
    print("Best params:", gs.best_params_)

    print("Computing per-fold accuracies with best params...")
    accs, f1s = [], []

    Xsyn_all, ysyn_all = (None, None)
    if use_synth:
        Xsyn_all, ysyn_all = load_synthetic_features(synth_dir, set(np.unique(y).tolist()))
        print(f"Loaded synthetic features: {len(ysyn_all)}")

    for tr, te in gkf.split(X, y, groups):
        Xtr, ytr = X[tr], y[tr]
        if use_synth and Xsyn_all is not None and len(ysyn_all) > 0:
            Xtr = np.vstack([Xtr, Xsyn_all])
            ytr = np.concatenate([ytr, ysyn_all])

        best_model.fit(Xtr, ytr)
        pred = best_model.predict(X[te])
        accs.append(accuracy_score(y[te], pred))
        f1s.append(f1_score(y[te], pred, average="macro"))

    metrics = {"accuracy": float(np.mean(accs)), "macro_f1": float(np.mean(f1s))}
    print("ACC_FOLDS=", [round(a, 4) for a in accs])
    print("F1_FOLDS=", [round(a, 4) for a in f1s])
    print("SVM mean accuracy:", round(metrics["accuracy"], 4), "macro_f1:", round(metrics["macro_f1"], 4))
    return metrics


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_synth", type=int, default=0)
    parser.add_argument("--metrics_out", type=str, default="")
    parser.add_argument("--synth_dir", type=str, default="synthetic_specs")
    parser.add_argument("--data_dir", type=str, default="data")
    args = parser.parse_args()

    metrics = train_and_eval(
        use_synth=bool(args.use_synth),
        synth_dir=args.synth_dir,
        data_dir=args.data_dir,
    )

    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print("Wrote metrics:", args.metrics_out)