# Adapting another benchmark

The public interface in `benchmark_progress/adapter.py` has two methods:

```python
genuine_public_profiles() -> PublicProfiles
public_feedback(submission) -> numpy.ndarray
```

`PublicProfiles` contains unique model names in lexical order, public coordinate names, and a model-by-coordinate score matrix on a 0–1 scale (higher is better). Supply only released values. `scale=100000` represents percentages published to three decimal places; `scale=None` preserves public floating-point JSON. Optional `categories` specifies equal-category aggregation; optional `overall` supplies a separately public overall column, or a public ranking statistic such as HELM Lite mean win rate. No private overall statistic belongs here.

A submission is a function `(prompt_hash, coordinate) -> model_name`. Both arguments must be public metadata. Hash normalized prompt text consistently; duplicate observable prompts must receive the same hash. Never encode row position, labels, or S/T membership. The evaluator runs the chosen endpoint locally, aggregates the outcomes, and releases only the allowed task profile.

```python
from benchmark_progress.adapter import run_attack
result = run_attack(benchmark.public_access(), budget=32)
metrics = benchmark.evaluate(result)  # Private scoring, only after completion.
```

`public_access()` can return a `PublicAccess` object containing just the two callbacks. The attack submits 31 random routers and one final router. Existing model profiles are free. Budget 1 gives the static task router. Seeds, endpoint selection, decoder arithmetic, and tie rules are the same functions used by the historical experiments. A frontier with fewer than two models raises an error; there is no added fallback.

The public interface contains no held-out scorer. Keep S and T private inside the evaluator and open T only after `run_attack` returns. This is an auditable code boundary, not a security sandbox for untrusted Python: maintainers run the code in their own environment.

The synthetic example is the smallest complete implementation:

```bash
python -m examples.custom_benchmark.run
```

It bundles 80 generated binary outcomes for three models and two tasks. Public feedback rounds percentages to three decimals. The completed attack is scored on S and T, with ranks comparing genuine models plus the final router. The output includes score generalization gap, the genuine baseline, both ranks, false selection, and held-out regret. These illustrate the API; they are not evidence about an additional real benchmark.

For private evaluation, retain genuine/candidate/final category profiles, call `compact_trace`, then `outcomes` from `benchmark_progress.evaluation`. Its default aggregation is an equal-category mean; it also implements the reported HELM Lite rank convention. An adapter with different ranking or weighting conventions must supply an evaluator appropriate to that benchmark. The public adapter does not prescribe or infer private evaluation conventions.


## Generate a maintainer report

```bash
python -m scripts.maintainer_report --policy category --budget 8 --output runs/maintainer-example
```

Open `runs/maintainer-example/report.html`. Keep its `style.css` and `fonts/` alongside it; the report uses the research website’s existing style. `report.md` is a text copy, `report.json` contains evaluator-approved aggregate metrics, and `public_transcript.json` records only allowed genuine profiles, submitted feedback and final weights. Existing output directories are refused. [Generated example](../examples/maintainer_report/report.md) · [HTML example](../site/maintainer-example.html).

Choose `--policy task`, `category`, or `aggregate`, all at 0.001 pp precision. Category scores average native published task ticks within categories. The aggregate averages exact category means before one rounding step. Categories receive equal weight even with unequal task counts. If `PublicProfiles.categories` is absent, each coordinate is a category. This path expects native feedback at 0.001 pp and equal-category scoring; benchmarks with other conventions must use an explicit matching evaluator instead of silently inheriting these assumptions.

Endpoint selection and every submission receive only the restricted surface. The report passes `endpoint_selector=select_policy_endpoints`: choose the maximum weighted absolute-distance pair among the five highest-scoring genuine models, with lexical ties. This stays defined for scalar feedback. The default historical selector remains unchanged.

The **no-attack reference** is the genuine model with the highest allowed public overall, using lexical ties at 1e-10 precision. It consumes zero new submissions. The attack uses `budget−1` random candidates and one final submission. Before held-out evaluation, also choose between the genuine winner and router using allowed public scores; ties retain the genuine winner. The report shows both models and the resulting model-selection loss.

### Connect your evaluator

```bash
python -m scripts.maintainer_report \
  --adapter my_benchmark.report:make_evaluator \
  --policy aggregate --budget 32 --output runs/my-benchmark
```

The zero-argument factory returns an object with:

- `public_access()`: the existing two-callback `PublicAccess`, publishing only reused native task feedback.
- `evaluate(result)`: private scoring of a completed router, returning `reused_score_pp`, `heldout_score_pp`, `heldout_regret_pp`, and `rank_t`. The command computes reused rank from allowed feedback. The final router is remapped to original task metadata before evaluation.
- `evaluate_genuine(model_name)`: the same four aggregate fields for the already selected genuine model. Rank among genuine models only; do not reselect a model using held-out outcomes.
- `describe()`: `name`, `scope`, `coverage` (key/value map), `notes` (list), and `reproduction` (key/value map). Include input version/checksum, split rule, seeds, reused/held-out counts, model/task counts, missing-score exclusions and duplicate-prompt checks. The report does not infer dataset completeness.

Scores are percentages. Regret is the best genuine held-out score minus the evaluated model’s score. Ranks compare genuine models plus the final router for attack evaluation, and genuine models only for the no-attack reference. Ties share rank. The evaluator retains private item outcomes and returns only aggregates.

[`SyntheticBenchmark`](../examples/custom_benchmark/run.py) implements the complete contract. This is a single-run diagnostic, not an estimated false-winner rate. [Five-dataset no-attack references](no_attack.md) and the separate [paired policy study](feedback_policy.md) supply population summaries. The callback boundary is for trusted local code, not a sandbox for arbitrary malicious Python.
