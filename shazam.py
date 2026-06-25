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
LOUD_SEG_SEC = 1.5

ENERGY_FILTER = True
RMS_THRESHOLD = 0.01

N_MELS = 64
HOP = 256

BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3

N_SPLITS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transformer size (keep small)
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 2
D_FF = 256
DROPOUT = 0.1
# =========================================

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
        start = 0
    else:
        csum = np.cumsum(rms)
        seg_sums = csum[frames_per_seg:] - csum[:-frames_per_seg]
        start = int(np.argmax(seg_sums)) * hop_length

    return pad_with_silence(y[start:start + seg_len], seg_len)


def logmel(y):
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)  # [M, T]
    # Normalize per-clip (helps deep models)
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_db


def build_dataset(data_dir):
    X, y, groups = [], [], []
    label_to_idx = {}

    for label in sorted(os.listdir(data_dir)):
        p = os.path.join(data_dir, label)
        if not os.path.isdir(p):
            continue
        if label not in label_to_idx:
            label_to_idx[label] = len(label_to_idx)

        for fn in sorted(os.listdir(p)):
            if not fn.lower().endswith(".wav"):
                continue
            fpath = os.path.join(p, fn)
            wav, _ = librosa.load(fpath, sr=SR, mono=True)
            if len(wav) < MIN_SAMPLES:
                continue

            max_start = max(1, len(wav) - MIN_SAMPLES + 1)
            for start in range(0, max_start, STRIDE_SAMPLES):
                clip = wav[start:start + WINDOW_SAMPLES]
                if len(clip) < MIN_SAMPLES:
                    continue
                clip = pad_with_silence(clip, WINDOW_SAMPLES)

                if ENERGY_FILTER and rms_energy(clip) < RMS_THRESHOLD:
                    continue

                clip = loudest_segment(clip, SR, LOUD_SEG_SEC)
                mel = logmel(clip)  # [M, T]
                X.append(mel)
                y.append(label_to_idx[label])
                groups.append(fpath)

    X = np.array(X, dtype=np.float32)   # [N, M, T]
    y = np.array(y, dtype=np.int64)
    groups = np.array(groups)
    return X, y, groups, label_to_idx


# ---------- NEW: load synthetic specs ----------
def load_synth_specs(synth_dir, label_map):
    """
    Loads synthetic spectrograms saved as .npy in:
      synth_dir/<class_name>/*.npy

    They must be shape [N_MELS, T] just like the real ones in X.
    """
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
                # expect [M, T]
                Xs.append(spec)
                ys.append(cls_idx)

    if len(Xs) == 0:
        return None, None

    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int64)


class MelSeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        mel = self.X[idx]        # [M, T]
        mel = mel.T              # [T, M] tokens over time
        return torch.tensor(mel), torch.tensor(self.y[idx])


def collate_pad(batch):
    """
    Pad sequences in time dimension so transformer can batch them.
    """
    xs, ys = zip(*batch)
    lengths = [x.shape[0] for x in xs]
    T_max = max(lengths)
    M = xs[0].shape[1]

    x_pad = torch.zeros(len(xs), T_max, M, dtype=torch.float32)
    attn_mask = torch.ones(len(xs), T_max, dtype=torch.bool)  # True = pad

    for i, x in enumerate(xs):
        T = x.shape[0]
        x_pad[i, :T] = x
        attn_mask[i, :T] = False  # False = keep

    y = torch.stack(ys)
    return x_pad, attn_mask, y


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:, :T]


class AudioTransformer(nn.Module):
    def __init__(self, n_mels, n_classes):
        super().__init__()
        self.in_proj = nn.Linear(n_mels, D_MODEL)
        self.pos = PositionalEncoding(D_MODEL)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_FF,
            dropout=DROPOUT,
            batch_first=True,
            activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.head = nn.Linear(D_MODEL, n_classes)

    def forward(self, x, pad_mask):
        # x: [B, T, M]
        x = self.in_proj(x)          # [B, T, D]
        x = self.pos(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask)  # [B, T, D]

        # masked mean pooling
        keep = ~pad_mask  # [B, T]
        keep = keep.unsqueeze(-1).float()
        x_sum = (x * keep).sum(dim=1)
        denom = keep.sum(dim=1).clamp(min=1.0)
        pooled = x_sum / denom

        return self.head(pooled)


def train_epoch(model, loader, opt, loss_fn):
    model.train()
    for x, mask, y in loader:
        x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        logits = model(x, mask)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()


def eval_acc(model, loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for x, mask, y in loader:
            x, mask = x.to(DEVICE), mask.to(DEVICE)
            p = model(x, mask).argmax(dim=1).cpu().numpy()
            preds.extend(p)
            trues.extend(y.numpy())
    return accuracy_score(trues, preds)


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

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), 1):
        print(f"\nFold {fold}")

        Xtr, ytr = X[tr], y[tr]

        # ✅ only training gets synthetic
        if Xsyn is not None:
            Xtr = np.concatenate([Xtr, Xsyn], axis=0)
            ytr = np.concatenate([ytr, ysyn], axis=0)
            print(f"[Fold {fold}] Train size after synth: {len(ytr)}")

        train_ds = MelSeqDataset(Xtr, ytr)
        test_ds = MelSeqDataset(X[te], y[te])

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_pad)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_pad)

        model = AudioTransformer(n_mels=N_MELS, n_classes=len(label_map)).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(EPOCHS):
            train_epoch(model, train_loader, opt, loss_fn)

        acc = eval_acc(model, test_loader)
        scores.append(acc)
        print(f"Fold {fold} acc: {acc:.4f}")

    scores = np.array(scores)
    print(f"\nTransformer GroupCV mean={scores.mean():.4f} std={scores.std():.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--use_synth", type=int, default=0)
    p.add_argument("--synth_dir", type=str, default="synthetic_specs")
    args = p.parse_args()
    main(bool(args.use_synth), args.synth_dir)