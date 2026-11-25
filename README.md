# ERAPR

**E**ntropy-**R**egularized **A**daptive **P**rojection and **R**otation for One-Step Spectral Clustering

> [!NOTE]
> This is the official code for the paper **"Entropy-Regularized Adaptive Projection and Rotation for One-Step Spectral Clustering"**, currently under review. The authors and full citation (BibTeX) will be added once the paper is accepted.

---

## Overview

ERAPR is a graph-based clustering method that jointly learns:

- an **adaptive projection matrix `P`** that embeds high-dimensional data into a low-dimensional spectral space;
- a **spectral rotation matrix `Q`** that aligns the continuous spectral embedding with the discrete cluster indicator;
- an **adaptive similarity matrix `A`** with an `L0`-style `k`-nearest-neighbor sparsity constraint and an entropy regularization term, which avoids trivial solutions and adaptively captures local structure.

The variables are optimized alternately via **ADMM** and **Generalized Power Iteration (GPI)**. The method further supports a **sliding-window (chunk-wise) streaming setting**, making it suitable for incremental / data-stream clustering scenarios.

**Main contributions**

1. Unified optimization of adaptive projection, spectral rotation, and adaptive graph learning in a single framework.
2. An entropy-regularized sparse similarity graph that adaptively determines neighborhood structure.
3. A label-driven convergence criterion that is more stable than the conventional objective-driven stopping rule.

---

## Repository Structure

This repository ships **source code only** (the paper is under review). Benchmark datasets (`.mat`), experiment result figures (`.png` / `.pdf`), result tables (`.xlsx`), and the baseline method `ERCAN` are intentionally **not** included — see [Reproducing the Results](#reproducing-the-results).

The code is organized into two top-level folders — **`ERAPR/`** (core algorithm implementation) and **`Implement/`** (evaluation & demo scripts) — plus experiment sub-folders:

```
ERAPR code and data/
├── ERAPR/                          # Core algorithm implementation
│   ├── ERAPR.m                     # Main algorithm (P / Q / F / A joint optimization)
│   ├── init_A.m                    # Adaptive similarity matrix initialization
│   └── L2_distance_1.m             # Squared Euclidean distance matrix
├── Implement/                      # Evaluation & demo scripts
│   ├── ClusteringMeasure.m         # ACC / NMI / Purity / F1 (embeds hungarian.m)
│   ├── all_seeds_demo.m            # Multi-seed (10-run) evaluation
│   └── single_seed_demo.m          # Quick single-seed demo
├── ablation/                       # Ablation study scripts
│   ├── ablationP.m
│   ├── ablationPdemo.m
│   └── ablationQ.m
├── convergence analysis/           # Convergence analysis scripts
│   ├── ERAPR_con.m                 # Main algorithm with convergence history
│   └── convergence.m               # Convergence curves (objective + accuracy)
├── parameter analysis/demo/        # Parameter sensitivity scripts
│   ├── Parameter_one.m             # Single-variable sensitivity (lambda / eta / k)
│   ├── Parameter_two.m             # Two-variable synergy (lambda vs eta, lambda vs k)
│   ├── Parameter_w.m               # Window-consistency analysis
│   ├── chunk_size_experiment.m     # Chunk-size sensitivity
│   └── run_multi_seed_eval.m       # Multi-seed evaluation helper
├── data for experiments/           # Dataset helper (countlabels.m); .mat NOT included
│   └── countlabels.m
├── README.md
└── .gitignore
```

---

## Requirements

- **MATLAB R2020a or later** (uses `exportgraphics`, `smoothdata`)

---

## Quick Start

1. Put the repository folder on the MATLAB path:

   ```matlab
   addpath(genpath('ERAPR code and data'));
   ```

2. Run the single-seed demo (edit `DataName` in `Implement/single_seed_demo.m` to pick a dataset):

   ```matlab
   run('Implement/single_seed_demo.m');
   ```

3. Or call the algorithm directly:

   ```matlab
   load('data for experiments/nci9_data.mat');   % X (n x d), Y (n x 1) labels
   X = double(X);
   X = X ./ (max(X(:)) + 1e-9);
   [result, label, time, iter] = ERAPR(X, Y, k, mu, eta, lambda);
   % result = [ACC, NMI, Purity, F1, RunningTime]
   ```

### Parameters

| Parameter | Meaning | Typical range |
|---|---|---|
| `k`      | Number of nearest neighbors (`L0` constraint on each row of `A`) | 2 – 20 |
| `lambda` | Graph regularization strength | `1e-3` – `1e7` |
| `mu`     | Entropy regularization weight (auto-initialized if omitted) | dataset-dependent |
| `eta`    | Scale factor for adaptive initialization of `mu` | `1e-3` – `1e3` |
| `iter`   | Maximum iterations (default 40) | — |

Per-dataset best hyper-parameters used in the paper can be found inside the `*_demo.m` and `convergence.m` scripts.

---

## Reproducing the Results

The following artifacts are **not** included in this repository (the paper is under review). To reproduce the experiments, obtain them from their original public sources and place them in the expected locations:

| Artifact | Expected location | How to obtain |
|---|---|---|
| Benchmark datasets | `data for experiments/*.mat` | Download from the original public sources (see the table below). Each `.mat` is expected to contain `X` (`n × d` data) and `Y` (`n × 1` labels). |
| Baseline method ERCAN | `original_ERCAN/` | Not redistributed here; please cite and obtain from the original ERCAN paper. |
| Result figures / tables | generated by the scripts | Re-run `convergence.m`, `Parameter_*.m`, `all_seeds_demo.m`, etc. |

### Datasets

| Dataset | Type |
|---|---|
| AR / ORL / Yale / warpAR10P | Face images |
| Isolet | Spoken letter recognition |
| nci9 / Yeast | Gene expression / biology |
| abalone / glass | UCI tabular |

All datasets are publicly available benchmarks. Please refer to their original sources for the respective licenses.

---

## Citation

If you find this work useful, please cite:

**Entropy-Regularized Adaptive Projection and Rotation for One-Step Spectral Clustering** (under review).

<!-- TODO: add full BibTeX entry once accepted -->

---

## Acknowledgments

- The baseline method **ERCAN** is used for comparison in the paper but is **not redistributed** in this repository; please cite the original ERCAN paper when reproducing the comparisons.
- `ClusteringMeasure.m` embeds the Hungarian-algorithm implementation `hungarian.m` by Niclas Borlin (Department of Computing Science, Umeå University, Sweden). If you use it in a publication, please include an acknowledgement as requested by its author.

## License

<!-- TODO: choose a license, e.g. MIT -->
