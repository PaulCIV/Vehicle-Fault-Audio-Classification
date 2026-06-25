import os
import csv
import numpy as np
import librosa
import matplotlib.pyplot as plt

# ================= CONFIG (match your CNN) =================
DATA_DIR = "data"
OUT_DIR = "spec_pngs"
SR = 16000

WINDOW_SEC = 5
STRIDE_SEC = 1
MIN_SEC = 3
LOUD_SEG_SEC = 1.5

ENERGY_FILTER = True
RMS_THRESHOLD = 0.01

N_MELS = 64
HOP = 256
# ===========================================================

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


def mel_spectrogram(y):
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_mels=N_MELS, hop_length=HOP
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


def save_png(mel_db, out_path):
    plt.figure(figsize=(3, 3))
    plt.imshow(mel_db, origin="lower", aspect="auto", cmap="magma")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(out_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    if not os.path.isdir(DATA_DIR):
        raise RuntimeError(f"Missing {DATA_DIR}/ folder")

    os.makedirs(OUT_DIR, exist_ok=True)
    index_path = os.path.join(OUT_DIR, "index.csv")

    rows = []
    total_saved = 0

    for label in sorted(os.listdir(DATA_DIR)):
        class_path = os.path.join(DATA_DIR, label)
        if not os.path.isdir(class_path):
            continue

        out_class = os.path.join(OUT_DIR, label)
        os.makedirs(out_class, exist_ok=True)

        for fname in sorted(os.listdir(class_path)):
            if not fname.lower().endswith(".wav"):
                continue

            fpath = os.path.join(class_path, fname)
            y, _ = librosa.load(fpath, sr=SR, mono=True)
            if len(y) < MIN_SAMPLES:
                continue

            max_start = max(1, len(y) - MIN_SAMPLES + 1)
            clip_idx = 0

            for start in range(0, max_start, STRIDE_SAMPLES):
                clip = y[start:start + WINDOW_SAMPLES]
                if len(clip) < MIN_SAMPLES:
                    continue

                clip = pad_with_silence(clip, WINDOW_SAMPLES)

                if ENERGY_FILTER and rms_energy(clip) < RMS_THRESHOLD:
                    continue

                clip = loudest_segment(clip, SR, LOUD_SEG_SEC)
                mel_db = mel_spectrogram(clip)

                out_name = f"{os.path.splitext(fname)[0]}__s{start}__i{clip_idx}.png"
                out_path = os.path.join(out_class, out_name)

                save_png(mel_db, out_path)

                rows.append({
                    "png_path": out_path,
                    "label": label,
                    "source_wav": fpath,
                    "start_sample": start
                })

                total_saved += 1
                clip_idx += 1

            print(f"[{label}] {fname}: saved {clip_idx} spectrograms")

    with open(index_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["png_path", "label", "source_wav", "start_sample"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nDone.")
    print("Saved PNGs:", total_saved)
    print("Index CSV:", index_path)
    print("Output folder:", OUT_DIR)


if __name__ == "__main__":
    main()