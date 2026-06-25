import os
import numpy as np
import librosa
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score

# ================= CONFIG =================
DATA_DIR = "data"
SR = 16000

WINDOW_SEC = 5
STRIDE_SEC = 1
MIN_SEC = 3

LOUD_SEG_SEC = 1.5   # CNN benefits from slightly longer context

ENERGY_FILTER = True
RMS_THRESHOLD = 0.01

N_MELS = 64
HOP = 256

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

N_SPLITS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================

WINDOW_SAMPLES = WINDOW_SEC * SR
STRIDE_SAMPLES = STRIDE_SEC * SR
MIN_SAMPLES = MIN_SEC * SR


# ---------- audio utils ----------
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
        start = 0
    else:
        csum = np.cumsum(rms)
        seg_sums = csum[frames_per_seg:] - csum[:-frames_per_seg]
        start = int(np.argmax(seg_sums)) * hop_length

    return pad_with_silence(y[start:start + seg_len], seg_len)


def mel_spectrogram(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, hop_length=HOP
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)


# ---------- dataset ----------
class MelDataset(Dataset):
    def __init__(self, specs, labels):
        self.X = specs
        self.y = labels

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]).unsqueeze(0),  # [1, mel, time]
            torch.tensor(self.y[idx], dtype=torch.long)
        )


# ---------- model ----------
class SmallCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------- data builder ----------
def build_dataset(data_dir):
    specs, labels, groups = [], [], []
    label_to_idx = {}

    for label in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, label)
        if not os.path.isdir(class_path):
            continue

        label_to_idx[label] = len(label_to_idx)

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

                clip = loudest_segment(clip, SR, LOUD_SEG_SEC)
                mel = mel_spectrogram(clip)

                specs.append(mel)
                labels.append(label_to_idx[label])
                groups.append(fpath)

    return np.array(specs), np.array(labels), np.array(groups), label_to_idx


# ---------- NEW: load synthetic specs ----------
def load_synth_specs(synth_dir, label_map):
    Xs, ys = [], []
    if not os.path.isdir(synth_dir):
        return None, None

    inv = {v: k for k, v in label_map.items()}
    for cls_idx, cls_name in inv.items():
        cls_path = os.path.join(synth_dir, cls_name)
        if not os.path.isdir(cls_path):
            continue

        for fn in os.listdir(cls_path):
            if fn.endswith(".npy"):
                spec = np.load(os.path.join(cls_path, fn)).astype(np.float32)
                # expect [N_MELS, T]
                Xs.append(spec)
                ys.append(cls_idx)

    if len(Xs) == 0:
        return None, None

    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)


# ---------- training ----------
def train_epoch(model, loader, opt, loss_fn):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()


def eval_model(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            p = model(x).argmax(dim=1).cpu().numpy()
            preds.extend(p)
            trues.extend(y.numpy())
    return accuracy_score(trues, preds)


# ---------- main ----------
def main(use_synth=False, synth_dir="synthetic_specs"):
    print("Building dataset...")
    X, y, groups, label_map = build_dataset(DATA_DIR)
    print("Total clips:", len(y))
    print("Class counts:", Counter(y.tolist()))

    Xsyn, ysyn = (None, None)
    if use_synth:
        Xsyn, ysyn = load_synth_specs(synth_dir, label_map)
        if Xsyn is None:
            print(f"[WARN] use_synth=1 but no .npy found in {synth_dir}/<class>/")
        else:
            print(f"[OK] Loaded synthetic specs: {len(ysyn)} from {synth_dir}/")

    gkf = GroupKFold(n_splits=min(N_SPLITS, len(np.unique(groups))))
    scores = []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        print(f"\nFold {fold}")

        Xtr, ytr = X[tr], y[tr]

        # ✅ only training gets synthetic
        if Xsyn is not None:
            Xtr = np.concatenate([Xtr, Xsyn], axis=0)
            ytr = np.concatenate([ytr, ysyn], axis=0)
            print(f"[Fold {fold}] Train size after synth: {len(ytr)}")

        train_ds = MelDataset(Xtr, ytr)
        test_ds = MelDataset(X[te], y[te])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        model = SmallCNN(n_classes=len(label_map)).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        loss_fn = nn.CrossEntropyLoss()

        for ep in range(EPOCHS):
            train_epoch(model, train_loader, opt, loss_fn)

        acc = eval_model(model, test_loader)
        scores.append(acc)
        print(f"Fold {fold} acc: {acc:.4f}")

    scores = np.array(scores)
    print(f"\nCNN GroupCV mean={scores.mean():.4f} std={scores.std():.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--use_synth", type=int, default=0)
    p.add_argument("--synth_dir", type=str, default="synthetic_specs")
    args = p.parse_args()
    main(bool(args.use_synth), args.synth_dir)