import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# =========================
# CONFIG (matches your setup)
# =========================
FIG_DIR = "figures"
PYTHON = "python3"

METHOD_SCRIPTS = [
    ("kNN", "knn.py"),
    ("LDA", "lda.py"),
    ("SVM(RBF)", "svm.py"),
    ("CNN", "cnn.py"),
    ("Transformer", "shazam.py"),  # your transformer script name
]

N_TRIALS = 5   # <-- CHANGE THIS
# =========================


@dataclass
class MethodResult:
    name: str
    # flattened list: all fold accuracies from all trials
    scores: List[float]
    mean: float
    std: float
    # optional: keep per-trial fold lists for debugging
    trials: List[List[float]]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def compute_mean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.array(vals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name).strip("_")


def parse_fold_scores(stdout: str) -> List[float]:
    """
    First tries to parse:
      FOLD_SCORES_JSON=[0.1,0.2,0.3,0.4,0.5]

    Fallback:
      Fold 1 acc: 0.5207
    """
    m = re.search(r"FOLD_SCORES_JSON\s*=\s*(\[[^\]]*\])", stdout)
    if m:
        raw = m.group(1)
        vals = re.findall(r"[-+]?\d*\.\d+|\d+", raw)
        return [float(v) for v in vals]

    folds = []
    for line in stdout.splitlines():
        m2 = re.search(r"Fold\s+\d+\s+acc:\s*([0-9]*\.?[0-9]+)", line, flags=re.IGNORECASE)
        if m2:
            folds.append(float(m2.group(1)))
    return folds


def run_script(script_path: str) -> str:
    """
    Runs python script and returns its stdout (also prints stdout to screen).
    """
    if not os.path.exists(script_path):
        raise FileNotFoundError(script_path)

    proc = subprocess.run(
        [PYTHON, script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    print("\n" + "=" * 60)
    print(f"OUTPUT: {script_path}")
    print("=" * 60)
    print(proc.stdout)
    return proc.stdout


def plot_solo(method: MethodResult):
    """
    Bar = mean
    Errorbar = std
    Dots = every fold score across all trials (N_TRIALS * N_SPLITS points)
    """
    ensure_dir(FIG_DIR)

    x = [0]
    plt.figure(figsize=(6, 4))
    plt.bar(x, [method.mean], yerr=[method.std], capsize=6)
    plt.xticks(x, [method.name], rotation=15, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(f"{method.name} — Repeated GroupCV (mean ± std), trials={len(method.trials)}")

    if method.scores:
        # jitter dots horizontally so you can see them
        jitter = np.linspace(-0.12, 0.12, num=len(method.scores))
        plt.scatter(np.array(x) + jitter, method.scores)

    out = os.path.join(FIG_DIR, f"solo_{sanitize(method.name)}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)


def plot_compare_bar(methods: List[MethodResult]):
    ensure_dir(FIG_DIR)

    names = [m.name for m in methods]
    means = [m.mean for m in methods]
    stds = [m.std for m in methods]
    x = np.arange(len(methods))

    plt.figure(figsize=(max(8, len(methods) * 1.2), 5))
    plt.bar(x, means, yerr=stds, capsize=6)
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(f"Repeated GroupCV Accuracy Comparison (mean ± std), trials={N_TRIALS}")
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "compare_means.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)


def plot_boxplot(methods: List[MethodResult]):
    ensure_dir(FIG_DIR)

    data = [m.scores for m in methods if m.scores]
    labels = [m.name for m in methods if m.scores]

    plt.figure(figsize=(max(8, len(labels) * 1.2), 5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title(f"Repeated GroupCV Accuracy Distribution (Boxplot), trials={N_TRIALS}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "compare_boxplot.png")
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out)


def main():
    ensure_dir(FIG_DIR)

    results: List[MethodResult] = []

    for display_name, script in METHOD_SCRIPTS:
        if not os.path.exists(script):
            print(f"Skipping {script} (not found)")
            continue

        try:
            all_scores: List[float] = []
            trials: List[List[float]] = []

            for t in range(N_TRIALS):
                print(f"\n--- {display_name}: TRIAL {t+1}/{N_TRIALS} ---")
                stdout = run_script(script)
                folds = parse_fold_scores(stdout)

                if len(folds) < 2:
                    print(f"WARNING: Could not parse fold accuracies from {script} on trial {t+1}.")
                    print("Fix: make the method print:  FOLD_SCORES_JSON=[...]")
                    folds = []

                if folds:
                    trials.append(folds)
                    all_scores.extend(folds)

            if len(all_scores) < 2:
                print(f"WARNING: {display_name} produced no usable fold scores across trials.")
                continue

            mean, std = compute_mean_std(all_scores)
            results.append(MethodResult(display_name, all_scores, mean, std, trials))

        except Exception as e:
            print(f"ERROR running {script}: {e}")

    if not results:
        print("\nNo methods produced fold scores to plot.")
        print("Fix: in each method script, print the fold list like:")
        print("  print('FOLD_SCORES_JSON=', fold_scores)")
        return

    # Sort best → worst
    results.sort(key=lambda r: r.mean, reverse=True)

    # Solo plots
    for r in results:
        plot_solo(r)

    # Comparative plots
    plot_compare_bar(results)
    plot_boxplot(results)

    # Summary text
    summary_path = os.path.join(FIG_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Repeated GroupCV trials={N_TRIALS}\n\n")
        for r in results:
            f.write(f"{r.name}: mean={r.mean:.4f}, std={r.std:.4f}, n_points={len(r.scores)}\n")
            f.write(f"  per-trial folds: {r.trials}\n")
    print("Saved:", summary_path)

    print("\nDone. Check ./figures")


if __name__ == "__main__":
    main()