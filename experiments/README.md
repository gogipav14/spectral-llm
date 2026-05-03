# Experiments

Reproducible experiment scripts and outputs for the Boolean-theory paper
(`paper/bbt_paper.tex`). All scripts use `scipy.optimize.milp` (HiGHS backend)
or `numba`-jitted Python; no GPUs required.

## Files

| File | What it does | Runtime | Output |
|---|---|---|---|
| `n4_min_support_mu.py` | MILP minimum-support certificate for every Boolean function on 4 variables (all $2^{16}=65{,}536$). Computes $I(f)$, $\mu(f)$, $H(\mathrm{Inf})$, $\max_\ell \mathrm{Inf}_\ell$ alongside. | ~10 min, 7 workers | `n4_min_support_mu.csv` (65,536 rows) |
| `n4_analyze.py` | Marginal + conditional-by-$I$ Spearman/Pearson correlations of min-support against the diagnostics. | seconds | stdout |
| `npn_enumerate.py` | Numba-jit orbit-marking enumeration of all NPN-canonical Boolean functions on $5$ variables. Produces exactly $616{,}126$ representatives, matching OEIS A000370. | ~65 sec | `npn5_canonical.npy` (uint32, 4.7 MB) |
| `n5_min_support_mu.py` | MILP minimum-support on a uniform sample of $10{,}000$ Boolean functions on $5$ variables (env var `N5_SAMPLE` to override). | ~11 min, 7 workers | `n5_min_support_mu.csv` |
| `n5_npn_min_support.py` | Same MILP on a uniform sample of NPN-canonical reps from `npn5_canonical.npy`. Use `N5_NPN_SAMPLE=10000` for the paper's 10k-sample analysis. | ~11 min, 7 workers | `n5_npn_min_support.csv` |
| `n5_npn_analyze.py` | Side-by-side comparison of function-uniform vs.\ NPN-uniform sampling at $n=5$. Confirms that the negative conditional $\rho(\mu, \|\mathrm{supp}\|)$ at $n=5$ is not an orbit-size artifact. | seconds | stdout |
| `n5_analyze.py` | Standalone analysis on the function-uniform sweep. | seconds | stdout |

## Reproducing the paper's tables

From the repo root:

```bash
# n=4 minimum-support sweep + correlation analysis
python -m experiments.n4_min_support_mu
python experiments/n4_analyze.py

# n=5: enumerate NPN reps, run MILP on a 10k uniform NPN sample
python -m experiments.npn_enumerate
N5_NPN_SAMPLE=10000 python -m experiments.n5_npn_min_support
python experiments/n5_npn_analyze.py

# (Optional) function-uniform 10k sample at n=5 for the comparison row
N5_SAMPLE=10000 python -m experiments.n5_min_support_mu
python experiments/n5_analyze.py
```

The CSV / NPY outputs are gitignored (see [`../.gitignore`](../.gitignore));
regenerate via the scripts above, or pull the published artifacts from the
arXiv source tarball if available.

## Headline numbers from these experiments (Section 6 of the theory paper)

- **n=4 minimum-support distribution**: 32 fns (0.05%) at support 1, 1,120
  (1.7%) at 3, 18,176 (27.7%) at 5, 44,800 (68.4%) at 7, 1,408 (2.1%) at 9.
  Mean **6.42**, max **9**, all-odd by a parity argument.
  (Companion paper `draft_nai.tex`'s heuristic-search numbers were mean 10.8 / max 16 — different objective, both valid.)
- **n=4 marginal Spearman** $\rho(\mu, |\mathrm{supp}|) = +0.017$ (weak; support distribution too tight).
- **n=4 conditional Spearman at $I=1.75$** (the largest stratum, 13,568 fns,
  $20\%$ of the universe): $\rho = \mathbf{+0.571}$, $p < 10^{-300}$. Cleanest
  empirical confirmation of the strict-Schur-convexity prediction in our data.
- **n=5 (function-uniform 10k sample)**: same all-odd structure, mean min-support
  10.57, max 17. Marginal $\rho(\mu, |\mathrm{supp}|) = -0.023$. Conditional
  at $I=2.40$ (largest bin): $\rho = \mathbf{-0.326}$ — opposite sign from $n=4$.
- **n=5 (NPN-canonical 10k of 616,126)**: replicates the function-uniform numbers
  closely (mean 10.54, max 17, $\rho(\mu, |\mathrm{supp}|)$ marginal $-0.021$,
  conditional at $I=2.40$: $\rho = \mathbf{-0.378}$). The $n=5$ negative is
  **not** an orbit-size sampling artifact.

So $\mu$ is a valid Schur-convex concentration invariant, but **not** a
universal monotone predictor of minimum support across $n$. Influence
entropy continues to track support qualitatively at $n=5$ where $\mu$
does not — see the theory paper §6.4.
