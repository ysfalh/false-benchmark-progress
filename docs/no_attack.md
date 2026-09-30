# No-attack references

Choose the genuine model with the highest public ranking statistic, breaking ties lexically at ten-decimal precision. Freeze that choice before evaluating held-out data. It consumes no new submissions.

| Benchmark | Reused score (%) | Held-out score (%) |
| --- | ---: | ---: |
| LiveBench | 60.70 | 60.41 |
| HELM Capabilities | 82.01 | 81.46 |
| HELM Lite | 74.20 | 74.12 |
| Open LLM v2 | 48.13 | 47.71 |
| SWE-bench Verified | 80.44 | 77.71 |

Each row averages the same public-selected model on both sets over 250 trials. Scores follow benchmark-specific scales. HELM Lite selects by mean win rate but reports category-average performance.

These references distinguish ordinary selection error from the [additional loss after attack](selection_loss.md). A selected model can already trail another held out, especially among near-tied systems; exact winner reversals do not measure the size of that error.

[Trial records](../results/no_attack_v1/trials.csv) · [Summaries and intervals](../results/no_attack_v1/summary.csv) · [Definitions and source hashes](../results/no_attack_v1/method.json) · [Verification](../results/no_attack_v1/verification.json).

```bash
python -m scripts.no_attack --output runs/no-attack
```

The command refuses an existing output directory. The separately recorded genuine-only score-gap baseline uses the full running-best history; it is not the selected model’s reused-minus-held-out score.
