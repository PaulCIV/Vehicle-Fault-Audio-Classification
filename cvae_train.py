# cvae_train.py
import os
import math
import json
import random
from collections import Counter

import numpy as np
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ================= CONFIG =================
DATA_DIR = "./data"
SR = 16000

WINDOW_SEC = 5        # must match your pipeline if you use it elsewhere
STRIDE_SEC = 1
MIN_SEC = 3

LOUD_SEG_SEC = 1.5    # same as your cnn.py
ENERGY_FILTER = True
RMS_THRESHOLD = 0.01

N_MELS = 64
HOP = 256

BATCH_SIZE = 32
EPOCHS = 40
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# VAE
LATENT_DIM = 64
LABEL_EMB_DIM = 16

# KL settings
BETA_TARGET = 1.0
KL_WARMUP_EPOCHS = 10

# Generation
SYNTH_PER_CLASS = 300   # change this (ex: to balance rare classes)
OUT_DIR = "synthetic_specs"
MODEL_OUT = "cvae.pt"

SEED = 7
# =========================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)


# ---------- audio helpers ----------
def pad_with_silence(y, target_len):
    if len(y) >= target_len:
        return y[:target_len]
    out = np.zeros(target_len, dtype=np.float32)
    out[:len(y)] = y
    return out

def rms_energy(y):
    rms = librosa.feature.rms(y=y)[0]
    return float(np.mean(rms))

def loudest_segment(y, sr, seg_sec):
    seg_len = int(seg_sec * sr)
    if len(y) < seg_len:
        return pad_with_silence(y, seg_len)

    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    frames_per_seg = max(1, int(seg_len / hop_length))

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
    return mel_db.astype(np.float32)  # shape [N_MELS, time]


# ---------- dataset builder (matches your cnn.py structure) ----------
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
            try:
                y, sr = librosa.load(fpath, sr=SR, mono=True)
            except Exception as e:
                print("Failed load:", fpath, e)
                continue

            # simple length filter (optional)
            if len(y) < int(MIN_SEC * SR):
                continue

            clip = loudest_segment(y, SR, LOUD_SEG_SEC)

            if ENERGY_FILTER and rms_energy(clip) < RMS_THRESHOLD:
                continue

            mel = mel_spectrogram(clip)
            specs.append(mel)
            labels.append(label_to_idx[label])
            groups.append(fpath)

    X = np.array(specs, dtype=np.float32)     # [N, mel, time]
    y = np.array(labels, dtype=np.int64)      # [N]
    groups = np.array(groups)
    return X, y, groups, label_to_idx


class SpecDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # Return [1, mel, time]
        return (
            torch.tensor(self.X[idx]).unsqueeze(0),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# ---------- cVAE model ----------
class CVAE(nn.Module):
    def __init__(self, n_classes, in_shape_hw, latent_dim=64, label_emb_dim=16):
        """
        in_shape_hw: (H, W) = (n_mels, time)
        """
        super().__init__()
        self.n_classes = n_classes
        self.in_h, self.in_w = in_shape_hw
        self.latent_dim = latent_dim

        # label embedding
        self.label_emb = nn.Embedding(n_classes, label_emb_dim)

        # encoder conv
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # /2
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.Conv2d(16, 32, 3, stride=2, padding=1), # /2
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, stride=2, padding=1), # /2
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # figure out flattened size after convs (dynamic)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.in_h, self.in_w)
            h = self.enc(dummy)
            self.enc_out_shape = h.shape[1:]  # (C, H', W')
            self.enc_flat_dim = int(np.prod(self.enc_out_shape))

        # condition injected at the MLP stage
        self.fc_mu = nn.Linear(self.enc_flat_dim + label_emb_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_flat_dim + label_emb_dim, latent_dim)

        # decoder
        self.fc_dec = nn.Linear(latent_dim + label_emb_dim, self.enc_flat_dim)

        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # *2
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),  # *2
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),   # *2
        )

    def encode(self, x, y):
        # x: [B,1,H,W], y:[B]
        h = self.enc(x)
        h = h.view(h.size(0), -1)
        yemb = self.label_emb(y)
        hcat = torch.cat([h, yemb], dim=1)
        mu = self.fc_mu(hcat)
        logvar = self.fc_logvar(hcat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        yemb = self.label_emb(y)
        zcat = torch.cat([z, yemb], dim=1)
        h = self.fc_dec(zcat)
        h = h.view(h.size(0), *self.enc_out_shape)  # [B,64,H',W']
        xhat = self.dec(h)

        # xhat might be slightly larger due to transpose conv rounding; crop to original
        xhat = xhat[:, :, :self.in_h, :self.in_w]
        return xhat

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z, y)
        return xhat, mu, logvar

    @torch.no_grad()
    def generate(self, y, n=1):
        # y can be int or tensor
        if isinstance(y, int):
            y = torch.tensor([y]*n, device=DEVICE, dtype=torch.long)
        else:
            y = y.to(DEVICE)
        z = torch.randn(y.size(0), self.latent_dim, device=DEVICE)
        xhat = self.decode(z, y)
        return xhat


def kl_divergence(mu, logvar):
    # mean KL per batch
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def main():
    print("Building dataset...")
    X, y, groups, label_map = build_dataset(DATA_DIR)
    print("Total specs:", len(y))
    print("Class counts:", Counter(y.tolist()))
    if len(y) == 0:
        raise RuntimeError("No data found. Check DATA_DIR structure: data/<class>/*.wav")

    # simple train/val split (group-aware would be better, but this is clean + easy)
    # We'll do a file-level split by unique group paths to reduce leakage.
    uniq = np.unique(groups)
    rng = np.random.default_rng(SEED)
    rng.shuffle(uniq)
    cut = int(0.85 * len(uniq))
    train_files = set(uniq[:cut])
    train_idx = np.array([g in train_files for g in groups], dtype=bool)

    Xtr, ytr = X[train_idx], y[train_idx]
    Xva, yva = X[~train_idx], y[~train_idx]

    print("Train:", len(ytr), "Val:", len(yva))

    # shapes
    H, W = Xtr.shape[1], Xtr.shape[2]
    n_classes = len(label_map)

    ds_tr = SpecDataset(Xtr, ytr)
    ds_va = SpecDataset(Xva, yva)

    dl_tr = DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH_SIZE, shuffle=False)

    model = CVAE(
        n_classes=n_classes,
        in_shape_hw=(H, W),
        latent_dim=LATENT_DIM,
        label_emb_dim=LABEL_EMB_DIM
    ).to(DEVICE)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # track best val recon
    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        # KL warmup
        if KL_WARMUP_EPOCHS > 0:
            beta = BETA_TARGET * min(1.0, epoch / KL_WARMUP_EPOCHS)
        else:
            beta = BETA_TARGET

        running = 0.0
        for x, yy in dl_tr:
            x = x.to(DEVICE)       # [B,1,H,W]
            yy = yy.to(DEVICE)

            xhat, mu, logvar = model(x, yy)

            # recon loss (L1 works nicely for spectrogram "images")
            recon = F.l1_loss(xhat, x, reduction="mean")
            kl = kl_divergence(mu, logvar)

            loss = recon + beta * kl

            opt.zero_grad()
            loss.backward()
            opt.step()

            running += loss.item()

        # val recon
        model.eval()
        val_recon = 0.0
        nb = 0
        with torch.no_grad():
            for x, yy in dl_va:
                x = x.to(DEVICE)
                yy = yy.to(DEVICE)
                xhat, mu, logvar = model(x, yy)
                val_recon += F.l1_loss(xhat, x, reduction="mean").item()
                nb += 1
        val_recon /= max(1, nb)

        print(f"Epoch {epoch:03d} | train_loss={running/len(dl_tr):.4f} | val_recon(L1)={val_recon:.4f} | beta={beta:.3f}")

        if val_recon < best_val:
            best_val = val_recon
            ckpt = {
                "state_dict": model.state_dict(),
                "label_map": label_map,
                "spec_shape": (H, W),
                "latent_dim": LATENT_DIM,
                "label_emb_dim": LABEL_EMB_DIM,
                "sr": SR,
                "n_mels": N_MELS,
                "hop": HOP,
                "loud_seg_sec": LOUD_SEG_SEC,
            }
            torch.save(ckpt, MODEL_OUT)

    print(f"Saved best model to {MODEL_OUT} (best val recon={best_val:.4f})")

    # ---- Generate synthetic spectrograms per class ----
    os.makedirs(OUT_DIR, exist_ok=True)

    # invert label map
    idx_to_label = {v: k for k, v in label_map.items()}

    # load best checkpoint back (just in case last epoch wasn't best)
    ckpt = torch.load(MODEL_OUT, map_location=DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    for cls_idx in range(n_classes):
        cls_name = idx_to_label[cls_idx]
        cls_dir = os.path.join(OUT_DIR, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        # batch-generate for speed
        remaining = SYNTH_PER_CLASS
        gen_id = 0
        while remaining > 0:
            b = min(64, remaining)
            y_tensor = torch.tensor([cls_idx] * b, device=DEVICE, dtype=torch.long)
            xgen = model.generate(y_tensor, n=b)  # [B,1,H,W]
            xgen = xgen.squeeze(1).detach().cpu().numpy()  # [B,H,W]

            for i in range(b):
                npy_path = os.path.join(cls_dir, f"gen_{gen_id:05d}.npy")
                np.save(npy_path, xgen[i])
                gen_id += 1

            remaining -= b

        print(f"Generated {SYNTH_PER_CLASS} specs for class '{cls_name}' -> {cls_dir}")

    print("Done.")


if __name__ == "__main__":
    main()