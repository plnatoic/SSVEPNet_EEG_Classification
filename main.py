"""SSVEP EEG classification with CCA reference features.

Data: 10 subjects x 180 trials x 8 channels x 1024 samples (256 Hz), 12 classes.
Method:
  1. notch 50 Hz + band-pass 0.5-50 Hz
  2. CCA between each trial and sine/cosine references at 12 target
     frequencies (3 harmonics), giving a 12-dimensional correlation vector
  3. standard CCA classifier = argmax of that vector
  4. optional trained classifiers on the same 12 features

The original notebook kept a trial ID as a feature and split rows randomly;
all 8 channels of a trial could appear in both train and test. This script
drops IDs and evaluates by trial and by held-out subject.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, iirnotch, lfilter
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = [9.25, 11.25, 13.25, 9.75, 11.75, 13.75,
           10.25, 12.25, 14.25, 10.75, 12.75, 14.75]
FS = 256
TIME_STEPS = 1024
N_CHANNELS = 8
N_TRIALS = 180
N_SUBJECTS = 10
HARMONICS = 3


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype="band")
    return lfilter(b, a, data)


def notch_filter(data, notch_freq, fs, quality_factor=30):
    b, a = iirnotch(notch_freq, quality_factor, fs)
    return lfilter(b, a, data)


def preprocess_eeg(data, fs=FS, lowcut=0.5, highcut=50.0, notch_freq=50.0):
    data = notch_filter(data, notch_freq, fs)
    return bandpass_filter(data, lowcut, highcut, fs)


def reference_signals(freq, fs=FS, time_steps=TIME_STEPS, harmonics=HARMONICS):
    t = np.arange(time_steps) / fs
    cols = []
    for h in range(1, harmonics + 1):
        cols.append(np.sin(2 * np.pi * freq * h * t))
        cols.append(np.cos(2 * np.pi * freq * h * t))
    return np.stack(cols, axis=1)


def cca_score(trial, freq, fs=FS):
    """Canonical correlation between one EEG trial and a frequency reference."""
    x = trial.T  # time x channels
    y = reference_signals(freq, fs=fs)
    cca = CCA(n_components=1, scale=False, max_iter=1000)
    x_c, y_c = cca.fit_transform(x, y)
    if np.std(x_c[:, 0]) < 1e-12 or np.std(y_c[:, 0]) < 1e-12:
        return 0.0
    return abs(float(np.corrcoef(x_c[:, 0], y_c[:, 0])[0, 1]))


def load_trials(data_dir: Path):
    signals, labels, subjects, trials = [], [], [], []
    for subject in range(1, N_SUBJECTS + 1):
        data = loadmat(data_dir / f"DataSub_{subject}.mat")["Data"]
        lab = loadmat(data_dir / f"LabSub_{subject}.mat")["Label"].ravel()
        for trial in range(N_TRIALS):
            signals.append(preprocess_eeg(data[:, :, trial]))
            labels.append(int(lab[trial]) - 1)
            subjects.append(subject)
            trials.append((subject - 1) * N_TRIALS + trial)
    return np.stack(signals), np.array(labels), np.array(subjects), np.array(trials)


def build_features(signals):
    return np.array([[cca_score(trial, f) for f in TARGETS] for trial in signals])


def evaluate_classifier(X, y, groups, splitter, clf):
    accs, f1s = [], []
    splits = splitter.split(X, y, groups) if isinstance(splitter, LeaveOneGroupOut) else splitter.split(X, y)
    for train_idx, test_idx in splits:
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, average="macro"))
    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "per_fold_accuracy": [float(v) for v in accs],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="SSVEPNet/data/Dial")
    parser.add_argument("--output", default="results.json")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    signals, labels, subjects, trials = load_trials(data_dir)
    print(f"trials: {len(signals)} | subjects: {len(np.unique(subjects))} | classes: {len(np.unique(labels))}")

    print("Computing CCA features (12 frequencies x 3 harmonics)...")
    X = build_features(signals)

    # Standard CCA classifier: pick the frequency with the highest correlation.
    cca_pred = X.argmax(axis=1)
    cca_acc = accuracy_score(labels, cca_pred)
    subject_acc = [
        accuracy_score(labels[subjects == s], cca_pred[subjects == s])
        for s in np.unique(subjects)
    ]
    print(f"\nStandard CCA: accuracy={cca_acc:.4f}, per-subject mean={np.mean(subject_acc):.4f} ± {np.std(subject_acc):.4f}")

    results = {
        "dataset": {
            "subjects": int(len(np.unique(subjects))),
            "trials": int(len(labels)),
            "classes": int(len(np.unique(labels))),
            "channels": N_CHANNELS,
            "sampling_rate_hz": FS,
            "time_steps": TIME_STEPS,
        },
        "features": {"targets_hz": TARGETS, "harmonics": HARMONICS},
        "standard_cca": {
            "accuracy": float(cca_acc),
            "per_subject_accuracy": [float(v) for v in subject_acc],
        },
    }

    classifiers = [
        ("logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
        ("knn_k3", make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))),
    ]
    for name, clf in classifiers:
        results[name] = {
            "5fold_by_trial": evaluate_classifier(
                X, labels, trials,
                StratifiedKFold(n_splits=5, shuffle=True, random_state=42), clf),
            "leave_one_subject_out": evaluate_classifier(
                X, labels, subjects, LeaveOneGroupOut(), clf),
        }
        print(f"\n{name}")
        print(f"  5-fold by trial: {results[name]['5fold_by_trial']['accuracy_mean']:.4f} ± {results[name]['5fold_by_trial']['accuracy_std']:.4f}")
        print(f"  leave-one-subject-out: {results[name]['leave_one_subject_out']['accuracy_mean']:.4f} ± {results[name]['leave_one_subject_out']['accuracy_std']:.4f}")

    Path(args.output).write_text(json.dumps(results, indent=2))
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
