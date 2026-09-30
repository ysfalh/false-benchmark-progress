# Model-selection loss

Select the genuine public winner before attack. Select again among the same genuine models and the final router afterward. Report

**held-out score of the no-attack choice − held-out score of the post-attack choice.**

Positive means selection worsened; negative means it improved. Average the signed difference over eligible paired trials (250 per snapshot, or 212 for Terminal-Bench-Science), including unchanged choices and unsuccessful attacks. Intermediate random routers are excluded from this comparison pool, matching the original final-router diagnostic. Selection uses public scores only; ties at ten-decimal loss precision favor genuine models, then lexical order.

The secondary rank drop is the post-attack choice’s held-out rank minus the no-attack choice’s held-out rank. Rank both against the **same** pool of genuine models plus the final router. Competition ties share rank. Pool sizes differ across benchmarks, so places are within-benchmark context.

HELM Lite retains mean-win-rate selection and ranking; its performance score is the equal-category mean. Win rates are recomputed on the comparison pool. Other scores retain their original evaluator weights and normalization.

The best-score generalization gap is the maximum absolute difference between separate running-best reused and held-out scores over the submission history, averaged across eligible trials. It measures score accuracy. A false winner is the final router ranking first reused but below first held out; that event does not, on its own, measure added selection harm.

## Rebuild and verify

```bash
python site/build.py
python -m pytest tests/test_selection_loss.py tests/test_presentation.py
```

[`scripts/selection_loss.py`](../scripts/selection_loss.py) reconstructs choices, scores and ranks from the canonical compact records. Scalar calculations independently check selection and ranking; archived false-winner events and final-router deficits must match. The site build combines these with the released-outcome bank in shared [trial records](../site/data/selection_loss_trials.csv), [summaries](../site/data/selection_loss.csv) and [definitions with input hashes](../site/data/selection_loss_method.json). Budget zero describes all original choices; budget one is static routing. Ineligible attack outcomes remain in the trial table and are omitted within each common 250-index bootstrap resample.

Intervals use 2,000 common paired bootstrap resamples, seed 2026081900. They are pointwise descriptive intervals across overlapping partitions of fixed snapshots, not fresh-data guarantees.

## Worked example and additional datasets

[`scripts/helm_walkthrough.py`](../scripts/helm_walkthrough.py) reconstructs HELM Lite trial 165 at 32 submissions from item scores and verifies the archived outcomes. The documented, outcome-dependent rule selects the median paired loss among 69 trials where the no-attack choice stays first held out, the router becomes a false winner, and the original score gap grows. The example illustrates the mechanism; it is not a representative estimate. [Exact record](../site/data/walkthrough.json).

The [Terminal-Bench counterexample](../results/terminal_counterexample/) retains its frozen inputs, code hashes, trial records and all budgets separately from the five-benchmark release. Run `python -m scripts.terminal_counterexample` to replay all 250 trials and compare them with the published records. Terminal-Bench is an exploratory counterexample: selection improves on average despite many false winners. It was not a prespecified negative control. Its 66 task identities and 43-task held-out splits limit interpretation.

The [practical feedback-policy study](feedback_policy.md) restricts both endpoint choice and subsequent feedback. Its feedback records, rules, and numerical outcomes are preserved. Package checksums cover the source and records used by the verification command.

## Secondary comparisons

The site build also runs [`scripts/secondary_selection.py`](../scripts/secondary_selection.py) over saved records. It uses the same public-selection rule, held-out score difference and common-pool ranks as the main analysis, with 2,000 paired bootstrap resamples and seed 2026081900. These are post-hoc summaries, not new attack experiments or part of the original pre-run protocol.

- **Static routing:** compare the original public winner with the choice after one static submission, then with the choice after 128 attack submissions. Static routing can improve selection; it is neither no attack nor one random submission. The comparison keeps endpoints paired.
- **Endpoint pool:** on HELM Capabilities at 2,048 submissions, the top-five rule gives 1.02 pp mean selection loss; the full-frontier rule gives 7.51 pp. The top-five rule nevertheless has the larger best-score gap. This now compares the selected models, rather than substituting the final routers' regret for selection loss.
- **Practical feedback policies:** select the genuine winner and the post-attack winner using each policy's own allowed scores, retaining genuine models on public ties. At 128 submissions, mean loss is 4.48 pp for task scores, 4.84 pp for categories and 4.03 pp for the aggregate. These differ from the main LiveBench result because the policy experiment uses a different endpoint rule and public-ranking interface.
- **Non-leading endpoints:** the original diagnostic excludes all public leaders before selecting its pair. Its 145 false winners in 250 trials give the reported 58%. That event includes public ties and does not mean 58% of choices become worse than the no-attack choice. The archive retains router ranks and regret, but not the public scores needed to resolve selection ties under the new rule; we therefore retain the original diagnostic without relabeling it as selection loss.

[Trial records](../site/data/secondary_selection_trials.csv) · [Summaries and intervals](../site/data/secondary_selection.csv) · [Definitions and source hashes](../site/data/secondary_selection_method.json).

The controlled feedback-dimension study also reports selection loss from its saved scores. Its false-winner thresholds, and the predictor’s score-gap thresholds, remain separate diagnostics. The original release and policy bundles keep their existing field names and values; the website links them as supporting diagnostics.
