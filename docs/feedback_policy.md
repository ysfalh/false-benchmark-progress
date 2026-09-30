# Practical feedback-policy comparison

This 250-trial LiveBench comparison restricts endpoint selection and every submission to the same feedback policy. At 16 submissions, false-winner rates are 98.4% with 18 task scores, 99.2% with six category scores, and 71.6% with one aggregate. At 128 they are 99.2%, 100%, and 100%. Aggregate feedback delays this attack; category feedback is not uniformly better. No policy is established as universally safe.

The comparison uses paired splits, prompt-hash routing seeds, tie seeds and budgets through 128. Endpoint selection and later submissions receive only the policy’s allowed scores. The common rule selects the maximum weighted absolute-distance pair among the top five genuine models ranked by the allowed overall. This differs from the original Pareto-frontier rule, which can leave only one model with scalar feedback.

Native task ticks are averaged exactly within natural categories, or into an equal-category aggregate, then rounded once to 0.001 pp. Reused ranks use the allowed released overall; held-out ranks and score regret use the common unrounded evaluator score. Regret is best genuine held-out score minus router score, averaged over all trials. The final submission counts toward the budget.

All three policies retain about 94.3% of genuine-model pair orderings for held-out gaps of at least 1 pp (trial-average fraction, public ties count as failures). This measures broad overall discrimination, not task diagnostics or small improvements. Intervals are descriptive pointwise percentiles from 2,000 common trial bootstrap resamples; overlapping splits are not independent datasets.

The original controlled study varies 1, 2, 3, 6, 9 and 18 artificial balanced task groups while holding full-task-selected endpoints fixed. This comparison instead uses natural categories and restricts endpoint selection too. The frozen tables retain their original static controls. Neither study is altered by the new selected-model analysis.

One snapshot, one endpoint heuristic and one fixed attack limit generalization. Endpoint and decoding information change together, so this is not a pure dimensionality effect. [No-attack references](no_attack.md) separately describe genuine-model selection before new submissions.

## Comparison using model-selection loss

The website now also applies its [model-selection loss](selection_loss.md) to these same saved trials. Each policy selects its own genuine public winner and then selects again after adding the final router. Both choices use only that policy's released scores; a public tie keeps the genuine model. The held-out score difference is averaged over all 250 trials.

| Feedback | Mean selection loss at 128 submissions (pp) |
| --- | ---: |
| 18 task scores | 4.48 |
| 6 category scores | 4.84 |
| 1 overall score | 4.03 |

This is a later analysis of the saved records. It leaves the original protocol, regret, static-gain and false-winner fields unchanged. A false winner can improve on the no-attack choice, so its rate is not a substitute for this score difference. The selected-model summaries use the main website's bootstrap seed (2026081900), while the original policy intervals retain seed 2026092201. [Derived trials and intervals](selection_loss.md#secondary-comparisons).

The documented maintainer command uses `scripts.maintainer_report` and `benchmark_progress.maintainer_view` to report this no-attack comparison. The bundled feedback records retain every public endpoint choice, decoder rule, and evaluation aggregate needed for verification.

[Protocol](../results/feedback_policy_v1/protocol.json) · [Package integrity lock](../results/feedback_policy_v1/protocol_lock.json) · [All metrics and intervals](../results/feedback_policy_v1/summary.csv) · [Paired differences](../results/feedback_policy_v1/paired_differences.csv). The bundled lock covers the packaged reproduction code and protocol; it is not an external preregistration. A new `freeze` command records input and code hashes before a fresh local run. Raw feedback, rules and evaluator aggregates are retained.

```bash
python -m scripts.feedback_policy_study verify
python -m scripts.feedback_policy_study freeze --output runs/policy-repeat
OPENBLAS_NUM_THREADS=1 python -m scripts.feedback_policy_study run --output runs/policy-repeat --workers 4
```

Existing output directories are refused. The run checks frozen input/code hashes before execution and saves all policy decisions in each trial before held-out scoring.
