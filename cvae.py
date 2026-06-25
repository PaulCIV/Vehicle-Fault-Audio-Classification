import os
import numpy as np
import librosa
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ================= CONFIG =================
DATA_DIR = "data"
OUT_DIR = "synthetic_specs"

SR = 16000
WINDOW_SEC = 5
STRIDE_SEC = 1
MIN_SEC = 3
LOUD_SEG_SEC = 1.5

ENERGY_FILTER = True
RMS_THRESHOLD = 0.01

N_MELS = 64
HOP = 256

LATENT_DIM = 64
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================

WINDOW_SAMPLES = WINDOW_SEC * SR
STRIDE_SAMPLES = STRIDE_SEC * SR
MIN_SAMPLES = MIN_SEC * SR


# ---------- EXACT SAME AUDIO UTILS ----------
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


# ---------- DATASET ----------
class SpecDataset(Dataset):
    def __init__(self, specs, labels):
        self.X = specs
        self.y = labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def build_spectrogram_dataset():
    specs, labels = [], []
    label_map = {}

    print("Extracting spectrograms for cVAE training...")

    for label in sorted(os.listdir(DATA_DIR)):
        class_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(class_path):
            continue

        label_map[label] = len(label_map)

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
                labels.append(label_map[label])

    X = np.array(specs, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)

    print("Total spectrograms for cVAE:", len(y))
    return X, y, label_map


# ---------- CONDITIONAL VAE ----------
class CVAE(nn.Module):
    def __init__(self, n_classes, time_dim):
        super().__init__()
        self.time_dim = time_dim
        in_dim = N_MELS * time_dim

        self.embed = nn.Embedding(n_classes, 32)

        self.enc = nn.Sequential(
            nn.Linear(in_dim + 32, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.mu = nn.Linear(256, LATENT_DIM)
        self.logvar = nn.Linear(256, LATENT_DIM)

        self.dec = nn.Sequential(
            nn.Linear(LATENT_DIM + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, in_dim)
        )

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        y_emb = self.embed(y)
        x = x.view(x.size(0), -1)
        h = self.enc(torch.cat([x, y_emb], dim=1))
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparam(mu, logvar)
        out = self.dec(torch.cat([z, y_emb], dim=1))
        return out, mu, logvar


# ---------- TRAIN ----------
def train():
    X, y, label_map = build_spectrogram_dataset()
    time_dim = X.shape[2]

    ds = SpecDataset(torch.tensor(X), torch.tensor(y))
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    model = CVAE(len(label_map), time_dim).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    for ep in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            recon, mu, logvar = model(xb, yb)
            recon = recon.view_as(xb)

            recon_loss = ((recon - xb) ** 2).mean()
            kl = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
            loss = recon_loss + 0.1 * kl

            loss.backward()
            opt.step()
            total += loss.item()

        print(f"Epoch {ep:03d} | loss={total / len(dl):.4f}")

    return model, label_map, time_dim


# ---------- GENERATE ----------
def generate(model, label_map, time_dim, per_class=15):
    os.makedirs(OUT_DIR, exist_ok=True)
    inv_map = {v: k for k, v in label_map.items()}

    model.eval()
    with torch.no_grad():
        for cls_idx, cls_name in inv_map.items():
            out_dir = os.path.join(OUT_DIR, cls_name)
            os.makedirs(out_dir, exist_ok=True)

            y = torch.full((per_class,), cls_idx, dtype=torch.long).to(DEVICE)
            z = torch.randn(per_class, LATENT_DIM).to(DEVICE)
            y_emb = model.embed(y)
            out = model.dec(torch.cat([z, y_emb], dim=1))
            out = out.view(per_class, N_MELS, time_dim).cpu().numpy()

            for i, spec in enumerate(out):
                np.save(os.path.join(out_dir, f"gen_{i:04d}.npy"), spec)


if __name__ == "__main__":
    model, label_map, time_dim = train()
    generate(model, label_map, time_dim)