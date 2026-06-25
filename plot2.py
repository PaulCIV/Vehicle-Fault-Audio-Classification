import os
import re
import ast
import json
import subprocess
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt


# ================= CONFIG =================
PYTHON = "python"        # use "python3" if needed
CNN_SCRIPT = "cnn.py"
TRANS_SCRIPT = "shazam.py"
DATA_DIR = "data"

SYNTH_DIR = "synthetic_specs"
OUT_DIR = os.path.join("figures", "cvae_experiment")  # saves plots here

# Regex patterns to extract accuracy from stdout
CNN_ACC_REGEX = r"CNN GroupCV mean=([0-9.]+)"
TRANS_ACC_REGEX = r"Transformer GroupCV mean=([0-9.]+)"

# Parse dataset info
TOTAL_CLIPS_REGEX = r"Total clips:\s*(\d+)"
CLASS_COUNTS_REGEX = r"Class counts:\s*(Counter\(\{.*\}\))"
SYNTH_LOADED_REGEX = r"\[OK\]\s*Loaded synthetic specs:\s*(\d+)\s*from\s*(.+)"
# =========================================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def count_synth_specs(synth_dir: str):
    """
    Counts how many .npy synthetic spectrograms exist per class folder:
      synth_dir/<class>/*.npy
    Returns: (total, {class: count})
    """
    per_class = {}
    total = 0

    if not os.path.isdir(synth_dir):
        return 0, per_class

    for cls in sorted(os.listdir(synth_dir)):
        cls_path = os.path.join(synth_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        n = sum(1 for f in os.listdir(cls_path) if f.endswith(".npy"))
        per_class[cls] = n
        total += n

    return total, per_class


def parse_counter(counter_str: str):
    """
    Parses a string like: Counter({0: 120, 2: 70, ...})
    Returns dict {class_idx:int -> count:int}
    """
    # turn "Counter({...})" into "{...}"
    inner = counter_str.strip()
    if inner.startswith("Counter(") and inner.endswith(")"):
        inner = inner[len("Counter("):-1]
    # inner now like "{0: 120, 2: 70}"
    d = ast.literal_eval(inner)
    return {int(k): int(v) for k, v in d.items()}


def run_model(cmd, acc_regex, label):
    """
    Runs a model script, captures stdout, extracts:
      - accuracy (required)
      - total clips (optional)
      - class counts (optional)
      - synth loaded count (optional)
    """
    print(f"\nRunning: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True
    )
    out = proc.stdout
    print(out)

    # accuracy
    m = re.search(acc_regex, out)
    if not m:
        raise RuntimeError(
            f"Could not extract accuracy for {label}. Expected regex '{acc_regex}'."
        )
    acc = float(m.group(1))

    # total clips
    total_clips = None
    m2 = re.search(TOTAL_CLIPS_REGEX, out)
    if m2:
        total_clips = int(m2.group(1))

    # class counts
    class_counts = None
    m3 = re.search(CLASS_COUNTS_REGEX, out)
    if m3:
        class_counts = parse_counter(m3.group(1))

    # synth loaded info (your scripts print this only when use_synth=1 and found specs)
    synth_loaded = 0
    synth_path_printed = None
    m4 = re.search(SYNTH_LOADED_REGEX, out)
    if m4:
        synth_loaded = int(m4.group(1))
        synth_path_printed = m4.group(2).strip()

    return {
        "acc": acc,
        "total_clips": total_clips,
        "class_counts": class_counts,
        "synth_loaded": synth_loaded,
        "stdout": out,
        "synth_path_printed": synth_path_printed
    }


def plot_accuracy(out_path, cnn_before, cnn_after, trans_before, trans_after):
    models = ["CNN", "Transformer"]
    before = [cnn_before, trans_before]
    after = [cnn_after, trans_after]

    x = np.arange(len(models))
    w = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - w/2, before, w, label="Before (Real)")
    plt.bar(x + w/2, after,  w, label="After (Real + cVAE)")

    plt.xticks(x, models)
    plt.ylabel("Accuracy")
    plt.title("Effect of cVAE Synthetic Spectrogram Augmentation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_data_counts(out_path, real_total, synth_total_available):
    labels = ["Real clips", "Synthetic specs available"]
    vals = [real_total, synth_total_available]

    plt.figure(figsize=(8, 5))
    plt.bar(np.arange(len(labels)), vals)
    plt.xticks(np.arange(len(labels)), labels, rotation=10, ha="right")
    plt.ylabel("Count")
    plt.title("Dataset Size: Real vs Synthetic (Available)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_class_distributions(out_path, real_class_counts_by_name, synth_class_counts_by_name):
    """
    Bar plot per class: real vs synthetic available.
    Both dicts keyed by class folder name.
    """
    classes = sorted(set(real_class_counts_by_name.keys()) | set(synth_class_counts_by_name.keys()))
    real_vals = [real_class_counts_by_name.get(c, 0) for c in classes]
    synth_vals = [synth_class_counts_by_name.get(c, 0) for c in classes]

    x = np.arange(len(classes))
    w = 0.35

    plt.figure(figsize=(max(10, len(classes) * 1.1), 5))
    plt.bar(x - w/2, real_vals, w, label="Real clips (windows)")
    plt.bar(x + w/2, synth_vals, w, label="Synthetic specs available")

    plt.xticks(x, classes, rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("Per-class Distribution: Real vs Synthetic (Available)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ensure_dir(OUT_DIR)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(OUT_DIR, run_id)
    os.makedirs(run_folder, exist_ok=True)
    ensure_dir(run_folder)

    # count synthetic files on disk (this is what matters for report)
    synth_total_available, synth_per_class = count_synth_specs(SYNTH_DIR)

    print("\n================ BASELINE (REAL DATA ONLY) ================")
    cnn_before = run_model(
        [PYTHON, CNN_SCRIPT, "--use_synth", "0", "--synth_dir", SYNTH_DIR],
        CNN_ACC_REGEX,
        "CNN (before)"
    )
    trans_before = run_model(
        [PYTHON, TRANS_SCRIPT, "--use_synth", "0", "--synth_dir", SYNTH_DIR],
        TRANS_ACC_REGEX,
        "Transformer (before)"
    )

    print("\n================ AFTER cVAE (REAL + SYNTHETIC) ================")
    cnn_after = run_model(
        [PYTHON, CNN_SCRIPT, "--use_synth", "1", "--synth_dir", SYNTH_DIR],
        CNN_ACC_REGEX,
        "CNN (after)"
    )
    trans_after = run_model(
        [PYTHON, TRANS_SCRIPT, "--use_synth", "1", "--synth_dir", SYNTH_DIR],
        TRANS_ACC_REGEX,
        "Transformer (after)"
    )

    # real total clips (from baseline parse)
    real_total = cnn_before["total_clips"] if cnn_before["total_clips"] is not None else None

    # Save raw stdout for reproducibility (report-friendly)
    with open(os.path.join(run_folder, "stdout_cnn_before.txt"), "w") as f:
        f.write(cnn_before["stdout"])
    with open(os.path.join(run_folder, "stdout_cnn_after.txt"), "w") as f:
        f.write(cnn_after["stdout"])
    with open(os.path.join(run_folder, "stdout_transformer_before.txt"), "w") as f:
        f.write(trans_before["stdout"])
    with open(os.path.join(run_folder, "stdout_transformer_after.txt"), "w") as f:
        f.write(trans_after["stdout"])

    # Plot 1: accuracy before vs after
    plot_accuracy(
        os.path.join(run_folder, "accuracy_before_after.png"),
        cnn_before["acc"], cnn_after["acc"],
        trans_before["acc"], trans_after["acc"]
    )

    # Plot 2: dataset counts real vs synthetic available
    if real_total is not None:
        plot_data_counts(
            os.path.join(run_folder, "data_counts_real_vs_synth_available.png"),
            real_total,
            synth_total_available
        )

    # Plot 3: per-class distributions (real vs synthetic available)
    # We can only plot real per-class by folder name if we compute it from disk.
    # Doing it from stdout class counts is by index, so we use disk counts for real (fast, no librosa).
    real_per_class_wavs = {}
    if os.path.isdir(DATA_DIR):
        for cls in sorted(os.listdir(DATA_DIR)):
            cls_path = os.path.join(DATA_DIR, cls)
            if not os.path.isdir(cls_path):
                continue
            n_wavs = sum(1 for f in os.listdir(cls_path) if f.lower().endswith(".wav"))
            real_per_class_wavs[cls] = n_wavs

    plot_class_distributions(
        os.path.join(run_folder, "per_class_real_wavs_vs_synth_specs.png"),
        real_per_class_wavs,
        synth_per_class
    )

    # Save JSON summary
    summary = {
        "run_id": run_id,
        "paths": {
            "data_dir": DATA_DIR,
            "synth_dir": SYNTH_DIR,
            "out_dir": run_folder
        },
        "real": {
            "total_windowed_clips_from_cnn_builder": real_total,
            "wavs_per_class_from_disk": real_per_class_wavs
        },
        "synthetic": {
            "total_specs_available_on_disk": synth_total_available,
            "specs_per_class_available_on_disk": synth_per_class,
            "cnn_reported_loaded_specs": cnn_after["synth_loaded"],
            "transformer_reported_loaded_specs": trans_after["synth_loaded"]
        },
        "results": {
            "cnn_before_acc": cnn_before["acc"],
            "cnn_after_acc": cnn_after["acc"],
            "transformer_before_acc": trans_before["acc"],
            "transformer_after_acc": trans_after["acc"],
            "cnn_delta": cnn_after["acc"] - cnn_before["acc"],
            "transformer_delta": trans_after["acc"] - trans_before["acc"]
        }
    }

    with open(os.path.join(run_folder, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ SUMMARY ================")
    print("Saved folder:", run_folder)
    print("Saved plots:")
    print(" - accuracy_before_after.png")
    if real_total is not None:
        print(" - data_counts_real_vs_synth_available.png")
    print(" - per_class_real_wavs_vs_synth_specs.png")
    print("Saved JSON: summary.json")

    print("\nAccuracy:")
    print(f"CNN:         {cnn_before['acc']:.4f}  →  {cnn_after['acc']:.4f}  (Δ {summary['results']['cnn_delta']:+.4f})")
    print(f"Transformer: {trans_before['acc']:.4f}  →  {trans_after['acc']:.4f}  (Δ {summary['results']['transformer_delta']:+.4f})")


if __name__ == "__main__":
    main()