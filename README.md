# SSVEP EEG Classification with CCA

Classify 12 steady-state visual evoked potential (SSVEP) frequencies from
8-channel EEG using canonical correlation analysis (CCA) reference features.

This is a classical CCA baseline on the SSVEPNet Dial dataset, not the
SSVEPNet convolutional model.

## What this is

SSVEP is a brain response to a flickering stimulus. Given a 4-second EEG trial,
the task is to decide which of 12 flicker frequencies the subject was looking
at. The notebook compares each trial with sine/cosine references at all 12
frequencies and uses the resulting correlations as features.

The evaluation is deliberately leakage-free:

- trial IDs are not used as features;
- all 8 channels of a trial stay in the same fold;
- leave-one-subject-out testing shows performance on a person the model has
  never seen.

**Status:** research/portfolio project. Not a real-time BCI or clinical tool.

## Results

| Model | 5-fold by trial | Leave-one-subject-out |
|---|---|---|
| Standard CCA (argmax) | 93.00% | 93.00% |
| CCA + Logistic Regression | 95.11 ± 0.89% | 94.28 ± 10.53% |
| CCA + KNN (k=3) | 93.56 ± 1.47% | 93.28 ± 10.80% |

Standard CCA is training-free, so its overall accuracy is the same under both
partitions. Per-subject accuracy is 93.00 ± 11.86%: subjects 1–2 are much
harder than the rest, which explains the wide leave-one-subject-out spread.

The previous version of this project reported 100% accuracy. That number was
caused by trial leakage and is not cited here. See [Why the old 100% was
invalid](#why-the-old-100-was-invalid).

## Data

- 10 subjects × 180 trials × 8 channels × 1024 samples.
- Sampling rate: 256 Hz (4 seconds per trial).
- 12 classes: 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25,
  13.75, 14.25, 14.75 Hz.
- Channels: Pz, PO3, PO4, O1, O2, Oz, POz, Fz.
- Source: [YuDongPan/SSVEPNet](https://github.com/YuDongPan/SSVEPNet),
  files `data/Dial/DataSub_*.mat` and `data/Dial/LabSub_*.mat`.

The data repository does not specify a license. Check its terms before reuse.

## Method

1. Notch filter at 50 Hz, then band-pass filter at 0.5–50 Hz.
2. For every trial and every target frequency, build sine/cosine references
   with 3 harmonics.
3. Compute the first canonical correlation between the 8 EEG channels and each
   reference. This gives a 12-dimensional correlation vector per trial.
4. Standard CCA classifier: predict the frequency with the highest correlation.
5. Optional trained classifiers on the same 12 features: logistic regression
   and KNN (k=3), with standardization fitted inside each fold.

## Quick start

Requires Python 3.9+.

```bash
git clone https://github.com/plnatoic/SSVEPNet_EEG_Classification.git
cd SSVEPNet_EEG_Classification

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Data repository expected at ./SSVEPNet/data/Dial
git clone https://github.com/YuDongPan/SSVEPNet.git

jupyter notebook SSVEP_CCA.ipynb
```

Run all cells. The notebook prints the results and writes `results.json`.

If Jupyter is not installed, open `SSVEP_CCA.ipynb` in Kaggle or Colab instead.

## Project structure

```text
.
├── SSVEP_CCA.ipynb    # data loading, CCA features, evaluation
├── requirements.txt
├── results.json       # output from the notebook
└── README.md
```

## Why the old 100% was invalid

The old notebook kept the `Trial` column as a feature and split rows randomly.
Each trial contributes 8 rows (one per channel), so the same trial could appear
in both train and test. A nearest-neighbour model can then match rows by trial
identity instead of learning the EEG signal.

After removing the trial ID and splitting by trial, the old CCA + KNN pipeline
drops to about 12.9%. For 12 classes, random guessing is 8.3%, so the old 100%
was not a real result.

## When not to use this

- Do not use it for clinical diagnosis or a consumer BCI product.
- Do not expect it to work on a new dataset without re-tuning the filters and
  checking the stimulus frequencies.
- Do not use it for real-time decoding: short windows and latency were not
  tested.

## Limitations

- One dataset, 12 fixed frequencies, 10 subjects.
- 4-second trials only; shorter windows are not evaluated.
- Leave-one-subject-out variance is high (about 10% standard deviation).
- CCA assumes the stimulus frequencies are known in advance.
- No model checkpoint or real-time inference code is included.

## Acknowledgements

- Data and SSVEPNet reference code:
  [YuDongPan/SSVEPNet](https://github.com/YuDongPan/SSVEPNet).
- CCA implementation: [scikit-learn](https://scikit-learn.org/).

## License

No license is specified. Add one before redistributing or reusing the code.
