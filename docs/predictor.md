# Predicting submission thresholds

The predictor estimates the submission count needed to increase the score generalization gap by 2 pp above its genuine-model baseline. Both gaps average the trial-level maximum absolute running-best gap. It does not predict model-selection loss.

The training data use 250 paired LiveBench trials, reused-set sizes of 30, 66, 126, and 204, and 1, 2, 3, 6, 9, or 18 released scores. Smaller reused splits are nested, with equal category totals and identical prompts kept together; the held-out split retains 408 examples. Subsampling uses seed 202609190000 plus the trial index. Endpoint pairs are selected using the full reused split and held fixed across sizes and dimensions. Candidate routers, the three score-blind task groupings, and 0.001 pp precision are shared.

Score-gap curves average the three grouping-specific means. A threshold is the first tested submission count reaching the genuine-model baseline +2 pp. Tested counts are 1, 2, 4, …, 2,048, including the final router. The baseline is recomputed for each reused-set size.

The fit is `k = 1 + c*n/d`, with `c = 0.6927637936637241`: the geometric mean of `d*(k−1)/n` over 23 training combinations that cross after the static router. One combination crosses at the static router and is excluded. The separate 90% false-winner predictor uses `k = 1 + c/d`, with `c = 58.779370191787386`, fitted on 18 combinations; six unreached thresholds are excluded.

The coefficients are applied to other benchmarks without fitting to their outcomes:

| Dataset | Observed score-gap threshold | Predicted submissions |
|---|---:|---:|
| HELM Capabilities | 128 | 148.4 |
| HELM Lite | 256 | 189.5 |
| Open LLM | 64 | 712.5 |
| SWE-bench Verified | Not reached through 2,048 | 12.5 |

Open LLM supplies seven released coordinates, including its separately rounded aggregate; the decoder uses six task scores. SWE-bench Verified uses ten repository-group coordinates. The observed thresholds come from the same top-five endpoint policy as the main website. An unreached threshold is right-censored, so no finite observed crossing or error factor is assigned. Its plot marker appears at the tested limit and is labeled “>2,048”.

The large external errors mean the fit is not a general vulnerability predictor. Thresholds describe the tested budget grid; they do not locate a continuous crossing or establish that every larger budget also crosses.

## Reproduction

[`results/predictor_v2`](../results/predictor_v2/) contains the training curves, thresholds, fitted coefficients, external predictions, and figure points. The training curves are summarized from the same fixed routing outcomes; individual trials are not included in this summary bundle. Checksums record the saved inputs and outputs.

```sh
python -m scripts.predictor_report --output runs/predictor
python -m pytest tests/test_predictor.py
python site/build.py
```

The report command refits the saved training curves and regenerates external predictions. Tests check threshold crossings, coefficient arithmetic, censoring, and agreement with the main benchmark curves.
