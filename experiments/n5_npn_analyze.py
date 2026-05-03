"""Compare n=5 NPN-uniform vs n=5 function-uniform samples."""
import csv
from collections import Counter
import numpy as np
from scipy import stats


def load(path, key):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                key: int(r[key]),
                "min_support": int(r["min_support"]),
                "I": float(r["I"]),
                "mu": float(r["mu"]),
                "log2_mu": float(r["log2_mu"]),
                "H_inf": float(r["H_inf"]),
                "max_inf": float(r["max_inf"]),
            })
    return rows


def summarize(rows, label):
    ms = np.array([r["min_support"] for r in rows])
    print(f"\n=== {label} (n={len(rows)}) ===")
    counter = Counter(ms.tolist())
    for s in sorted(counter):
        print(f"  s={s:2d}: {counter[s]:5d}  ({100*counter[s]/len(ms):.2f}%)")
    print(f"  mean = {ms.mean():.3f}  median = {np.median(ms):.1f}  "
          f"min={ms.min()}  max={ms.max()}  all_odd={all(s % 2 == 1 for s in counter)}")

    valid = [r for r in rows if r["min_support"] > 0]
    ms = np.array([r["min_support"] for r in valid])
    I = np.array([r["I"] for r in valid])
    mu = np.array([r["mu"] for r in valid])
    H_inf = np.array([r["H_inf"] for r in valid])
    max_inf = np.array([r["max_inf"] for r in valid])

    print(f"  marginal Spearman:")
    for name, x in [("I", I), ("mu", mu), ("H_inf", H_inf), ("max_inf", max_inf)]:
        spear = stats.spearmanr(ms, x)
        print(f"    {name:10s}  ρ={spear.statistic:+.4f}  p<{spear.pvalue:.2e}")

    # Conditional by I (rounded to 0.1)
    I_round = np.round(I * 10) / 10
    buckets: dict[float, list[int]] = {}
    for i, ir in enumerate(I_round):
        buckets.setdefault(float(ir), []).append(i)
    print(f"  conditional Spearman(mu, support) at fixed I (bins ≥ 100 fns):")
    print(f"  {'I':>5}  {'n':>5}  {'rho_mu':>9}  {'p_mu':>10}  {'rho_Hinf':>9}  {'p_Hinf':>10}")
    for ir in sorted(buckets):
        idx = buckets[ir]
        if len(idx) < 100:
            continue
        sub_ms = ms[idx]; sub_mu = mu[idx]; sub_h = H_inf[idx]
        if len(set(sub_ms.tolist())) < 2:
            continue
        rho_mu = stats.spearmanr(sub_ms, sub_mu)
        rho_h = stats.spearmanr(sub_ms, sub_h)
        print(f"  {ir:>5.2f}  {len(idx):>5d}  {rho_mu.statistic:+.4f}  "
              f"{rho_mu.pvalue:.2e}  {rho_h.statistic:+.4f}  {rho_h.pvalue:.2e}")
    return ms, I, mu, H_inf


print("Comparing n=5 uniform-over-functions vs uniform-over-NPN-classes")
print("=" * 65)
uniform = load("/mnt/c/Users/gogip/Spectral-LLM/experiments/n5_min_support_mu.csv", "seed")
summarize(uniform, "n=5 UNIFORM-over-functions (10k)")
npn = load("/mnt/c/Users/gogip/Spectral-LLM/experiments/n5_npn_min_support.csv", "tt_int")
summarize(npn, "n=5 NPN-canonical (10k of 616,126)")
