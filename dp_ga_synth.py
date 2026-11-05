"""DP synthetic data via a genetic algorithm over noisy 2-way marginals (Private-GSD-style)."""
import argparse, itertools, json, os, time
import numpy as np
import pandas as pd

from prepare_adult import KEEP, DOMAIN

# The 28 column pairs are computed once and reused everywhere.
PAIRS = list(itertools.combinations(range(len(DOMAIN)), 2))


def marginals(X):
    """All 28 pairwise marginals as fractions. bincount on the flat index a*dj+b is much
    faster than np.add.at, and fitness evaluation is the GA's hot loop."""
    n = len(X)
    return [np.bincount(X[:, i] * DOMAIN[j] + X[:, j], minlength=DOMAIN[i] * DOMAIN[j]) / n
            for i, j in PAIRS]


def measure(X, eps, rng):
    """The ONLY access to real data. Under add/remove-one-record neighbours, one person changes
    one cell of each of the 28 count tables by 1, so the L1 sensitivity is 28 -> Laplace(28/eps).
    Everything after this is post-processing, so it costs no extra privacy."""
    scale = len(PAIRS) / eps
    noisy = []
    for i, j in PAIRS:
        c = np.bincount(X[:, i] * DOMAIN[j] + X[:, j], minlength=DOMAIN[i] * DOMAIN[j]).astype(float)
        c = np.clip(c + rng.laplace(0, scale, c.shape), 0, None)  # clipping is post-processing
        noisy.append(c / max(c.sum(), 1e-9))
    return noisy


def fitness(X, target):
    # Sum of L1 errors over the 28 marginals; lower is better.
    return float(sum(np.abs(a - b).sum() for a, b in zip(marginals(X), target)))


def ga(target, m, pop, elites, gens, k, rng, log_every=50):
    dom = np.array(DOMAIN)
    # Start uniformly at random over the domain: this uses no data, so it is privacy-free.
    P = [(rng.integers(0, dom, size=(m, len(dom)))) for _ in range(pop)]
    F = [fitness(x, target) for x in P]
    curve = []
    for g in range(gens + 1):
        order = np.argsort(F)
        E, EF = [P[i] for i in order[:elites]], [F[i] for i in order[:elites]]
        curve.append(EF[0])
        if g % log_every == 0:
            print(f"  gen {g:4d}  best L1 = {EF[0]:.4f}")
        if g == gens:
            break
        children, cf = [], []
        for _ in range(pop - elites):
            a, b = rng.choice(elites, 2, replace=False)
            # Row-wise crossover: each synthetic row comes from one of the two parents.
            mask = rng.random(m) < 0.5
            c = np.where(mask[:, None], E[a], E[b])
            # Mutation: resample k random cells inside their column's domain.
            r, col = rng.integers(0, m, k), rng.integers(0, len(dom), k)
            c[r, col] = rng.integers(0, dom[col])
            children.append(c)
            cf.append(fitness(c, target))
        # Elites keep their cached fitness, so only children are scored.
        P, F = E + children, EF + cf
    return P[int(np.argmin(F))], curve


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    gens = 100 if a.quick else a.gens
    os.makedirs("results", exist_ok=True)
    X = pd.read_csv("data/adult_train_enc.csv")[KEEP].to_numpy()
    curves, t0 = {}, time.time()
    for eps in [0.5, 2.0, 8.0]:
        rng = np.random.default_rng(a.seed)
        print(f"eps={eps}")
        target = measure(X, eps, rng)
        # From here on only `target` (noisy marginals) is used; X is never touched by the GA.
        best, curve = ga(target, m=2000, pop=20, elites=4, gens=gens, k=20, rng=rng)
        curves[str(eps)] = curve
        if not a.quick:
            pd.DataFrame(best, columns=KEEP).to_csv(f"results/synthetic_adult_eps{eps}.csv", index=False)
    print(f"total GA time: {time.time() - t0:.1f}s (gens={gens})")
    if not a.quick:
        json.dump(curves, open("results/fitness_curves.json", "w"))
