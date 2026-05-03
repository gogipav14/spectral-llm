"""Analyze the n=5 min-support MILP results."""
import csv
from collections import Counter

import numpy as np
from scipy import stats

CSV = "/mnt/c/Users/gogip/Spectral-LLM/experiments/n5_min_support_mu.csv"
rows = []
with open(CSV) as f:
    for r in csv.DictReader(f):
        rows.append({
            "seed": int(r["seed"]),
            "min_support": int(r["min_support"]),
            "I": float(r["I"]),
            "mu": float(r["mu"]),
            "log2_mu": float(r["log2_mu"]),
            "H_inf": float(r["H_inf"]),
            "max_inf": float(r["max_inf"]),
        })
print(f"loaded {len(rows)} rows")

ms = np.array([r["min_support"] for r in rows])
print(f"\n=== n=5 min_support distribution (sample of {len(rows)}) ===")
counter = Counter(ms.tolist())
for s in sorted(counter):
    print(f"  s={s:2d}: {counter[s]:5d}  ({100*counter[s]/len(ms):.2f}%)")
print(f"  mean = {ms.mean():.3f}, median = {np.median(ms):.1f}, "
      f"min = {ms.min()}, max = {ms.max()}")
all_odd = all(s % 2 == 1 for s in counter)
print(f"  all-odd: {all_odd}")

valid = [r for r in rows if r["min_support"] > 0]
ms = np.array([r["min_support"] for r in valid])
I = np.array([r["I"] for r in valid])
mu = np.array([r["mu"] for r in valid])
log2_mu = np.array([r["log2_mu"] for r in valid])
H_inf = np.array([r["H_inf"] for r in valid])
max_inf = np.array([r["max_inf"] for r in valid])

print(f"\n=== marginal correlations (n={len(valid)}) ===")
for name, x in [("I", I), ("mu", mu), ("log2_mu", log2_mu),
                ("H_inf", H_inf), ("max_inf", max_inf)]:
    pear = stats.pearsonr(ms, x)
    spear = stats.spearmanr(ms, x)
    print(f"  {name:10s}  Pearson r={pear.statistic:+.4f} p<{pear.pvalue:.2e}    "
          f"Spearman ρ={spear.statistic:+.4f} p<{spear.pvalue:.2e}")

# Conditional on I (rounded to 0.1 since I has more spread at n=5)
I_round = np.round(I * 10) / 10
buckets = {}
for i, ir in enumerate(I_round):
    buckets.setdefault(ir, []).append(i)
print(f"\n=== conditional rho_Spearman(mu, support) and rho(H_inf, support) at fixed I ===")
print(f"  buckets with >= 100 functions:")
print(f"  {'I':>6}  {'n':>5}  {'rho_mu':>10}  {'p_mu':>11}  {'rho_Hinf':>10}  {'p_Hinf':>11}")
for ir in sorted(buckets):
    idxs = buckets[ir]
    if len(idxs) < 100:
        continue
    sub_ms = ms[idxs]
    sub_mu = mu[idxs]
    sub_h = H_inf[idxs]
    if len(set(sub_ms.tolist())) < 2:
        continue
    rho_mu = stats.spearmanr(sub_ms, sub_mu)
    rho_h = stats.spearmanr(sub_ms, sub_h)
    print(f"  {ir:>6.2f}  {len(idxs):>5d}  {rho_mu.statistic:+.4f}  "
          f"{rho_mu.pvalue:.2e}  {rho_h.statistic:+.4f}  {rho_h.pvalue:.2e}")

# Quintile breakdown
print(f"\n=== mean min_support by mu-quintile ===")
quintiles = np.quantile(mu, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
for k in range(5):
    lo, hi = quintiles[k], quintiles[k+1]
    mask = (mu >= lo) & (mu < hi if k < 4 else mu <= hi)
    print(f"  μ ∈ [{lo:.4f}, {hi:.4f}]  n={mask.sum():5d}  "
          f"mean(min_support)={ms[mask].mean():.3f}  std={ms[mask].std():.3f}")
