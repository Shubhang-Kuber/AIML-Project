# DP Synthetic Tabular Data via Evolutionary Search (Private-GSD reimplementation, UCI Adult)

**What was built.** A partial reimplementation of the Private-GSD idea (ICML 2023) on UCI Adult. The pipeline has four steps:
1. `prepare_adult.py` buckets 8 columns into small integer domains `[5,4,3,7,2,5,3,2]`, learning the rules from train only.
2. `dp_ga_synth.py` measures all 28 pairwise marginals **once** with Laplace noise (scale 28/ε, add/remove-one-record neighbours), clips negatives and normalises.
3. A genetic algorithm then evolves a 2000-row synthetic dataset to minimise the summed L1 error against those noisy marginals. It uses pop=20, elites=4, row-wise crossover, 20-cell mutation and 5000 generations. The real data is never touched after measurement, so the GA is pure post-processing.
4. `evaluate.py` reports fidelity (mean TV distance to the true train marginals) and utility (logistic regression trained on synthetic data, tested on the real Adult test set).

## Results

| eps | fidelity_tv (↓) | acc | f1 |
|---|---|---|---|
| real_train | 0.0000 | 0.8264 | 0.6012 |
| 0.5 | 0.0323 | 0.8216 | 0.5779 |
| 2.0 | 0.0327 | 0.8198 | 0.5708 |
| 8.0 | 0.0293 | 0.8205 | 0.5707 |

The full GA (3 × 5000 generations) took 278 s. Plots: `results/fitness_curve.png`, `results/utility_vs_eps.png`, `results/fidelity_vs_eps.png`.

## Findings
- At every ε, a model trained on DP synthetic data comes within 0.5–0.7 accuracy points of the real-data baseline (0.820–0.822 vs 0.826).
- F1 drops more than accuracy: 0.571–0.578 vs 0.601 on real data.
- The mean pairwise-marginal TV distance to the true data is 0.029–0.033 at all three ε.
- ε=8 gives the best fidelity (0.0293), but the gap between ε values is small. Utility does not rise monotonically with ε.
- The GA's error was still falling at generation 5000. Search error, not DP noise, is probably the main limit at this compute budget, which would explain why the ε values look so similar.

## Limitations / next steps
- Add 3-way marginal workloads, which capture more of the joint structure.
- Run a membership-inference attack to check privacy empirically.
- Analyse fairness by sex and race (per-group accuracy and marginal error).
- Compare against MST and DP-CTGAN baselines.
- Run longer or larger GAs, and use multiple seeds to get error bars. Every result here comes from a single seed (0).

## Decisions made
- Race codes are the 5 train categories sorted alphabetically. "Married" means any `Married-*` status.
- The number of generations was chosen from a `--quick` timing run (5.1 s per 100 gens × 3 ε), giving 5000 gens to fit the ~5 min budget.
- Python: the system Python is not on PATH and pip hit an SSL certificate error, so the scripts were run with an existing venv that has the required packages.

## Reproduce (run order)
```powershell
python prepare_adult.py
python dp_ga_synth.py --quick      # smoke test, ~5 s
python dp_ga_synth.py --gens 5000  # ~4.5 min
python evaluate.py
```
