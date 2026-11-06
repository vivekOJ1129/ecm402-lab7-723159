# ECM-402 Lab 7 — Unsupervised Learning (ICA, PCA, Clustering)

Author: Vivekanand Ojha — replace `rollnumber` in the repo/folder name with your roll number before submitting.

## Run

```bash
pip install -r requirements.txt
python src/main.py          # ~10 s on a CPU; regenerates every figure/table in results/
```

`main.py` writes figures to `results/figures/`, CSV tables to `results/tables/`, and all raw numbers to `results/results.json`.
No datasets are stored in the repo; they are regenerated from fixed seeds.

## Layout

| File | Content |
|---|---|
| `src/bss.py` | Part A: sources, mixing, whitening, FastICA vs PCA, Gaussian (unidentifiable) case |
| `src/datasets.py` | Part B: modulation signal generator + 12 AMC features (lab-sheet code) |
| `src/dimred.py` | Part B: PCA scree, reconstruction error, 2-D projection, loadings |
| `src/clustering.py` | Parts C, D: k-means, k selection, PCA-before-clustering, SNR sweep, GMM/agglomerative, moons + spectral |
| `src/main.py` | Runs everything |
| `report.pdf` | Answers, tables, figures |

## Random seeds

| Item | Seed |
|---|---|
| Sources (`np.random.seed`) | 0 |
| Gaussian sources (`default_rng`) | 1 |
| Random-rotation baseline (`default_rng`) | 123 |
| FastICA `random_state` | 0 (seeds 0–9 used for a stability check) |
| Modulation dataset | `seed=0` (seeds 0–9 for supplementary multi-dataset checks) |
| KMeans / GMM / Spectral `random_state` | 0 (`n_init=10` for KMeans) |
| `make_moons` | `random_state=0` |

## Label policy

Ground-truth labels are used **only** for evaluation (ARI, NMI, contingency table) or for colouring plots.
They are never passed to any unsupervised `fit()` and never used to choose `k` or the number of components.
The one exception is the supervised logistic-regression curve in C12(c), which is a supervised classifier by definition and is labelled as a stand-in for the Lab 3 curve.

## Dataset

The synthetic modulation dataset (not `load_digits`) is used in Parts B–D, at 20 dB unless the SNR sweep says otherwise.
