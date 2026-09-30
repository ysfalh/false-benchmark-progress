# Score generalization: comparison with no attack

The supplementary [comparison data](../site/data/score_comparison.csv) record the first tested submission count at which the attack's average score generalization gap exceeds the genuine-model baseline with statistical support. The estimate averages the paired difference between attack and baseline. It uses the top-five endpoint policy and all 250 released trials per benchmark. The main figure instead shows the gap curves with pointwise intervals; these adjusted comparisons remain a separate check.

For trial i, let A_i(k) be score generalization gap after k new submissions and B_i the genuine-model-only score generalization gap, in percentage points. The estimate is

    d(k) = mean_i [A_i(k) − B_i].

Each bootstrap resample draws 250 trial indices with replacement and uses the same indices for A and B. Use 100,000 resamples and seed 2026081900, with common resamples for every budget. The released score intervals are unchanged; the additional paired-difference intervals are basic (reverse-percentile) bootstrap intervals.

The one-sided lower bound used for the significance comparison is

    L(k) = 2 d(k) − Q_(1 − 0.05/55)(d*(k)).

The upper bootstrap quantile uses `higher`, conservatively retaining an observed tail value. The Bonferroni family contains all 11 queried budgets on all five benchmarks, 55 comparisons total. A budget qualifies when L(k) > 1e-10 pp, where the tolerance excludes numerical roundoff. Report the smallest tested qualifying budget; this is not a claim about untested intermediate budgets or all larger budgets. Budget 1 is a distinct static control, retained in the detailed results with a pointwise interval but excluded from this family. No other endpoint policy is searched for a better crossing.

The comparison data also give pointwise 95% two-sided basic intervals for the difference: [2d − Q_0.975(d*), 2d − Q_0.025(d*)]. These intervals alone do not account for looking across budgets; the adjusted lower bound does. The existing baseline + 2 pp crossing remains a separate effect-size reference. Statistical support does not imply a large effect.

Bootstrap coverage is approximate. The multiplicity adjustment does not turn it into an exact finite-sample test. Inference concerns random splits of these fixed historical snapshots, not fresh benchmark data. Both histories contain the same genuine models; the attack adds candidates and a final router, as in the paper's score gap metric.

The [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) describes paired resampling and basic intervals. [NIST's Bonferroni discussion](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm) gives the simultaneous-coverage principle. The implementation uses NumPy directly and needs no new dependency.

## Regeneration

```bash
python -m scripts.score_comparison --output runs/score_comparison.csv
python site/build.py
```

The first command refuses to overwrite an existing output. The site build recomputes this analysis from `results/release_v2/trials.parquet` and writes `site/data/score_comparison.csv` plus the method and input hash in `site/data/score_comparison_method.json`. The full score intervals remain in `results/release_v2/budget_curves.csv`. The trial-level outcomes are unchanged.
