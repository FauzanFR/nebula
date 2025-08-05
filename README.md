# 🌌 Nebula Tuner

**Nebula** is an experimental hyperparameter optimization engine powered by evolutionary strategies.  
It is designed to balance exploration and exploitation efficiently, especially in compute-intensive tuning scenarios.
Especially in compute-intensive tuning scenarios such as high-dimensional search spaces, expensive evaluations, or noisy objectives.

---

## ⚙️ Key Features

- 🔀 **Hybrid Genetic Evolution**  
  Combines BLX-α and SBX crossover operators with Gaussian + Cauchy mutation for both continuous and categorical parameters.

- 🎲 **Sobol Sequence Initialization**  
  Generates the initial population with uniform space-filling distribution for better coverage.

- 🧠 **Numba-Accelerated Surrogate Model**  
  A minimalist MLP predictor compiled via `@njit` to estimate objective scores and reduce unnecessary evaluations.

- ♻️ **Built-in Memoization**  
  Previously evaluated parameter configurations are cached

- ⚡ **Batch Parallel Evaluation (`joblib.Parallel`)**  
  Evaluates population members concurrently using available CPU cores with automatic conflict detection (e.g. Avoids resource conflict with model-level parallelism (e.g., `n_jobs=-1`)).

- 💥 **Adaptive Exploratory Injection**
  Injects novel candidates upon stagnation to escape local optima, inspired by probabilistic diversification strategies.

- 🛑 **Stagnation-Aware Early Stopping**  
  Automatically terminates evolution when no meaningful improvements are observed over time.

- 💾 **Automatic Logging to CSV**  
  The best individual of each generation is saved incrementally for progress tracking and recovery.

---
## How to install
```bash
git clone https://github.com/FauzanFR/nebula
cd nebula
pip install numpy joblib pandas tqdm scikit-learn numba
```
---

## 📊 Lightweight Benchmark

Nebula has been benchmarked against several common optimization libraries:

- Grid Search
- Random Search
- Optuna
- Hyperopt
- Nevergrad

Across datasets such as:

- 🍷 Wine Quality
- 🧔 Adult Income
- 🛳️ Titanic

| Dataset    | Algo       | Best Score | Mean Score | Trials | Std Dev | Avg Time | Max Time | Min Time |
|------------|------------|------------|------------|--------|---------|----------|----------|----------|
| Titanic    | grid       | 0.812495   | 0.720337   | 1500   | 0.073001| 0.174914 | 0.429789 | 0.154182 |
| Titanic    | hyperopt   | 0.821812   | 0.797999   | 1500   | 0.017645| 0.913846 | 2.038795 | 0.185442 |
| Titanic    | nebula     | 0.824067   | 0.764169   | 1500   | 0.060004| 1.629065 | 4.625285 | 0.167891 |
| Titanic    | nevergrad  | 0.820052   | 0.801765   | 1500   | 0.014335| 1.003537 | 2.209618 | 0.177585 |
| Titanic    | optuna     | 0.821812   | 0.806295   | 1500   | 0.014040| 1.034304 | 2.049532 | 0.169083 |
| Titanic    | random     | 0.821970   | 0.778892   | 1500   | 0.022473| 1.030620 | 3.703116 | 0.167729 |
| adult      | grid       | 0.790707   | 0.706903   | 1500   | 0.095360| 0.340328 | 1.051960 | 0.157842 |
| adult      | hyperopt   | 0.802335   | 0.787364   | 1500   | 0.022935| 6.395920 |25.255293 | 0.269537 |
| adult      | nebula     | 0.802089   | 0.744450   | 1500   | 0.094048|16.312931 |67.573750 | 0.309047 |
| adult      | nevergrad  | 0.801967   | 0.796010   | 1500   | 0.016693| 8.562082 |15.669124 | 0.289341 |
| adult      | optuna     | 0.803263   | 0.795591   | 1500   | 0.013381| 8.063505 |18.622150 | 0.362606 |
| adult      | random     | 0.801733   | 0.773376   | 1500   | 0.040074| 5.104347 |16.851778 | 0.259344 |
| wine       | grid       | 0.935324   | 0.852366   | 1500   | 0.048967| 0.108413 | 0.257906 | 0.100452 |
| wine       | hyperopt   | 0.973280   | 0.937339   | 1500   | 0.014614| 0.663781 | 1.909422 | 0.115429 |
| wine       | nebula     | 0.962587   | 0.907383   | 1500   | 0.048849| 1.479980 | 4.501302 | 0.135811 |
| wine       | nevergrad  | 0.957324   | 0.931776   | 1500   | 0.022075| 0.382590 | 1.575513 | 0.107153 |
| wine       | optuna     | 0.973280   | 0.944945   | 1500   | 0.013494| 0.765695 | 3.847182 | 0.112388 |
| wine       | random     | 0.957324   | 0.925481   | 1500   | 0.017225| 0.702832 | 2.717413 | 0.102117 |


⚠️ Note:
Per-trial time may appear higher in Nebula,
but unlike conventional tuners, it executes trials in parallel (e.g., joblib, batch vectorization).

As such, total runtime may be significantly shorter for large populations,
especially on multicore machines.

This is due to its use of parallel evaluation and population-based operations, which amortize better over larger generations.

---
## 📄 License

This project is licensed under the Apache 2.0 License — see the [LICENSE](./LICENSE) file for details.
