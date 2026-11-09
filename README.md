# SSVEP EEG Classification with CCA

12-class SSVEP classification from EEG using canonical correlation analysis
(CCA) reference features, with group-aware evaluation by trial and by subject.

## Overview

This project classifies 12 SSVEP stimulus frequencies from 8-channel EEG
trials. Each trial is compared against sine/cosine references at every target
frequency; the resulting CCA correlations are used both as a training-free
classifier (argmax) and as features for logistic regression and KNN.

The original notebook reported 100% accuracy. That number is invalid: it kept
the trial ID as a feature and split rows randomly, so the same trial could
appear in both train and test. The corrected pipeline drops trial IDs and
evaluates with trial-level and subject-level splits.

## Data

- 10 subjects x 180 trials x 8 channels x 1024 samples.
- Sampling rate: 256 Hz (4 s per trial).
- 12 classes: 9.25, 9.75, 10.25, 10.75, 11.25, 11.75,
  12.25, 12.75, 13.25, 13.75, 14.25, 14.75 Hz.
- Channels: Pz, PO3, PO4, O1, O2, Oz, POz, Fz.
- Data source: [YuDongPan/SSVEPNet](https://github.com/YuDongPan/SSVEPNet),
  files `data/Dial/DataSub_*.mat` and `data/Dial/LabSub_*.mat`.

Clone the data repository next to this project:

```bash
git clone https://github.com/YuDongPan/SSVEPNet.git
```

## Method

1. Notch filter at 50 Hz and band-pass filter at 0.5–50 Hz.
2. For each trial and each target frequency, build sine/cosine references
   with 3 harmonics and compute the first canonical correlation between the
   8 EEG channels and the reference signals.
3. Standard CCA classifier: choose the frequency with the highest correlation.
4. Optional trained classifiers on the 12 correlation features:
   logistic regression and KNN (k=3), with standardization inside each fold.

## Results

| Model | 5-fold by trial | Leave-one-subject-out |
|---|---|---|
| Standard CCA (argmax) | 93.00% | 93.00% |
| CCA + Logistic Regression | 95.11 ± 0.89% | 94.28 ± 10.53% |
| CCA + KNN (k=3) | 93.56 ± 1.47% | 93.28 ± 10.80% |

The standard CCA classifier is training-free, so its overall accuracy is the
same under both partitions. Per-subject mean accuracy is 93.00 ± 11.86%, which
shows the subject-to-subject variation behind the leave-one-subject-out scores.

## Important note about the original notebook

`Miniproject_NeuronScience.ipynb` is kept for reference. It contains the
original evaluation that produced 100% accuracy. Do not cite that number: the
`Trial` column was used as a feature, and random row splits allowed the same
trial (8 channel rows) to appear in both train and test. `main.py` is the
corrected, leakage-free evaluation.

## Quick start

```bash
pip install -r requirements.txt
git clone https://github.com/YuDongPan/SSVEPNet.git
python main.py --data-dir SSVEPNet/data/Dial
```

The script prints the results and writes `results.json`.

Run with a different data location:

```bash
python main.py --data-dir /path/to/Dial --output results.json
```

## Project structure

```text
.
├── main.py                          # corrected pipeline and evaluation
├── requirements.txt
├── Miniproject_NeuronScience.ipynb  # original notebook, contains leakage
└── README.md
```

## Limitations

- Only 12 fixed stimulus frequencies from one dataset.
- 4 s trials; shorter windows and real-time use are not tested.
- Leave-one-subject-out variance is high (about 10% standard deviation).
- CCA assumes the stimulus frequencies are known in advance.
- Not a clinical or consumer BCI product.

## Acknowledgements

- Data and SSVEPNet reference code: [YuDongPan/SSVEPNet](https://github.com/YuDongPan/SSVEPNet).
- CCA implementation: scikit-learn.

## License

No license is specified. Add one before redistributing or reusing the code.
