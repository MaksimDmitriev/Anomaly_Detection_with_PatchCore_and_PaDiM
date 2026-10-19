# BeeX interview lab — PatchCore vs PaDiM

A small **synthetic** anomaly-detection lab designed for learning, not for real structural inspection. Requires Python 3.11 or 3.12. Uses Anomalib 2.6.2 and PyTorch on CPU (suitable as a compatibility-first starting point on Apple Silicon Macs).

## 1. Set up

```bash
cd beex_anomaly_lab
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

First installation downloads pretrained model weights and Python dependencies and needs internet access.

## 2. Generate synthetic concrete-like images

```bash
python generate_dataset.py
open sample_preview.png  # macOS; optional
```

This creates:

```text
data/
  train/good/       60 normal images, fit model here
  test/good/        16 normal images, split between validation and test
  test/defect/      16 defect images, split between validation and test
  demo/good/         2 extra examples (not used during evaluation)
  demo/defect/       2 extra examples (not used during evaluation)
```

Only the 60 `train/good` images are used to fit the normal-feature models. The anomalib `Folder` datamodule splits the labelled test directories into distinct validation and final test subsets (`val_split_mode='from_test'`).

## 3. Fit and evaluate both models

```bash
python compare_models.py
```

The comparison is run on CPU for compatibility; it may be slow on some systems. Results are written to:

- `results/comparison.csv` — confusion-matrix counts, accuracy, precision, recall, F1 and average precision on the held-out evaluation subset.
- `results/patchcore_predictions.csv` and `results/padim_predictions.csv` — per-image labels and model scores.
- `results/patchcore_heatmaps.png` and `results/padim_heatmaps.png` — example test images beside their anomaly heatmaps.

Expect the initial pretrained ResNet-18 weights to download on first run. Exact scores are **not** known in advance; inspect the actual output. This lab's synthetic texture is a teaching aid, not a meaningful benchmark of subsea inspection quality.

## 4. Exercises

1. Compare which model had higher defect **recall** and which produced fewer false alarms.
2. Look at any images listed as `FN` or `FP` using their paths in the per-image predictions.
3. Change `coreset_sampling_ratio` in PatchCore from `0.05` to `0.10`. Did the result change?
4. Change `n_features` in PaDiM from `30` to `60`. Did the result change?
5. Make defects subtler in `generate_dataset.py`. Does recall fall? Why?
6. Explain why **accuracy** is insufficient when defect prevalence is low.

## 5. Technical distinctions

- **PatchCore** compares patches to a representative memory bank of *normal* feature vectors.
- **PaDiM** models the distribution of normal patch features at each spatial location using multivariate Gaussian statistics.
- Neither requires gradient-based fine-tuning of its ImageNet-pretrained CNN backbone for this lab.
- A detected defect at image level is not the same as precise defect localisation. Anomalib can also produce anomaly maps, which are a useful next exercise.
- Avoid cross-contamination between train, validation and test samples. In real field datasets, keep **structures, sites or inspection sessions** separated between splits so near-identical frames do not leak between them.

## References

- [Anomalib Documentation](https://anomalib.readthedocs.io/en/lib-v2.6.2/)
- [PatchCore reference](https://anomalib.readthedocs.io/en/lib-v2.6.2/markdown/guides/reference/models/image/patchcore.html)
- [PaDiM reference](https://anomalib.readthedocs.io/en/lib-v2.6.2/markdown/guides/reference/models/image/padim.html)
- [Folder datamodule](https://anomalib.readthedocs.io/en/lib-v2.6.2/markdown/guides/reference/data/datamodules/image/folder.html)

**Note:** This training script follows Anomalib's documented 2.6.2 API but was not end-to-end executed with anomalib in this environment. If your installation or platform reports an issue, preserve the error traceback for troubleshooting.
