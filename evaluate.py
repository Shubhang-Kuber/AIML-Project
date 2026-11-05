"""Evaluate fidelity (TV distance on 2-way marginals) and ML utility (train-synthetic, test-real)."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import OneHotEncoder

from prepare_adult import KEEP, DOMAIN
from dp_ga_synth import marginals

EPS = [0.5, 2.0, 8.0]
train = pd.read_csv("data/adult_train_enc.csv")[KEEP]
test = pd.read_csv("data/adult_test_enc.csv")[KEEP]
true_m = marginals(train.to_numpy())
feats = KEEP[:-1]
# Fixed categories mean every model sees the same one-hot columns, even when a synthetic set lacks a value.
enc = OneHotEncoder(categories=[list(range(d)) for d in DOMAIN[:-1]], handle_unknown="ignore")
enc.fit(test[feats])
Xte, yte = enc.transform(test[feats]), test["income"]


def evaluate(df, label):
    tv = float(np.mean([0.5 * np.abs(a - b).sum() for a, b in zip(marginals(df.to_numpy()), true_m)]))
    y = df["income"]
    if y.nunique() < 2:
        acc, f1 = float("nan"), float("nan")
    else:
        clf = LogisticRegression(max_iter=1000).fit(enc.transform(df[feats]), y)
        p = clf.predict(Xte)
        acc, f1 = accuracy_score(yte, p), f1_score(yte, p)
    return {"eps": label, "fidelity_tv": round(tv, 4), "acc": round(acc, 4), "f1": round(f1, 4)}


rows = [evaluate(train, "real_train")]
for e in EPS:
    rows.append(evaluate(pd.read_csv(f"results/synthetic_adult_eps{e}.csv")[KEEP], e))
res = pd.DataFrame(rows)
res.to_csv("results/results.csv", index=False)
print(res.to_string(index=False))

syn, base = res.iloc[1:], res.iloc[0]
curves = json.load(open("results/fitness_curves.json"))
plt.figure(figsize=(6, 4))
for e, c in curves.items():
    plt.plot(c, label=f"eps={e}")
plt.xlabel("Generation"); plt.ylabel("Best L1 error vs noisy marginals (sum of 28)")
plt.title("GA convergence"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("results/fitness_curve.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4))
plt.plot(EPS, syn["acc"], "o-", label="Synthetic (GA)")
plt.axhline(base["acc"], color="k", ls="--", label="Real train baseline")
plt.xscale("log"); plt.xticks(EPS, [str(e) for e in EPS])
plt.xlabel("Privacy budget epsilon"); plt.ylabel("Test accuracy (real Adult test)")
plt.title("Utility vs epsilon"); plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("results/utility_vs_eps.png", dpi=150); plt.close()

plt.figure(figsize=(6, 4))
plt.plot(EPS, syn["fidelity_tv"], "o-")
plt.xscale("log"); plt.xticks(EPS, [str(e) for e in EPS])
plt.xlabel("Privacy budget epsilon"); plt.ylabel("Mean TV distance (28 2-way marginals)")
plt.title("Fidelity vs epsilon (lower = better)"); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig("results/fidelity_vs_eps.png", dpi=150); plt.close()
