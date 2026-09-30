# Historical benchmark interfaces

The experiment models public aggregate feedback. It does not call a live leaderboard, assume automatic acceptance of arbitrary submissions, or provide all historically downloadable information to the attacker.

| Benchmark | Snapshot | Genuine models | Public feedback | Overall used for endpoint ordering |
| --- | --- | ---: | --- | --- |
| LiveBench | October 22, 2024 source snapshot | 33 | 18 task scores, three decimal percentage places | Equal task means within six categories, then equal category means |
| HELM Capabilities | v1.15.0 | 68 | Five scenario scores, full-precision JSON | Equal scenario mean |
| HELM Lite | v1.13.0 | 14 | Ten scenario scores, full-precision JSON | Mean win rate |
| Open LLM v2 | Pinned archived runs | 8 | Six normalized scores and Average, two decimal percentage places | Separately published Average |
| SWE-bench Verified | September 21, 2026 frozen snapshot | 169 systems | Ten repository aggregates reconstructed from public item resolutions | Instance-weighted resolution rate |
| Terminal-Bench-Science | v0.1 | 17 model–agent configurations | Five domain means reconstructed from public task outcomes, full precision | Task-weighted mean of three runs per task |

## LiveBench

The historical [table component](https://github.com/LiveBench/livebench.github.io/blob/1fadceff99bda3f8044aefda9a55e15f8c148c56/src/Table/CSVTable_2024_08_31.js) fetches a public task CSV with three decimal places on the percentage scale. The [reporting script](https://github.com/LiveBench/LiveBench/blob/3953c6cb2f478285ad28e7372383276ccfea692c/livebench/show_livebench_result.py) creates that task table by multiplying item scores by 100, averaging by task, then rounding. The website's category and overall displays derive from those public task values. Every listed model has a score profile; this interface does not release only Pareto or scalar records.

October 22 is the source snapshot date; the latest selectable release in the pinned reporting code is August 31, 2024. The balanced analysis frame is retained. Public item judgments also existed, but they are deliberately excluded from the modeled aggregate-feedback interface. Evaluation ranks use unrounded equal-category scores with the paper's numerical comparison convention, not the website's formatted sorting column.

## HELM

The [Capabilities core table](https://storage.googleapis.com/crfm-helm-public/capabilities/benchmark_output/releases/v1.15.0/groups/core_scenarios.json) exposes five scenario coordinates. The [Lite core table](https://storage.googleapis.com/crfm-helm-public/lite/benchmark_output/releases/v1.13.0/groups/core_scenarios.json) exposes ten: NarrativeQA, open-book NaturalQuestions, closed-book NaturalQuestions, OpenbookQA, MMLU, MATH, GSM8K, LegalBench, MedQA, and WMT 2014.

The Lite coordinates cover 28 constituent tasks used internally in splitting and aggregation. The recorded mapping checks all 140 scenario/model combinations; see `helm_lite_coordinates.json`. The attacker receives ten scenario values, not 28 task values.

The historical [JSON loader](https://github.com/stanford-crfm/helm/blob/12ab30bc79e44881597e2486e38c96b5b5931ef6/helm-frontend/src/services/getGroupTablesByName.ts) reads full-precision public numbers. The browser rounds for display; the experiment uses the public JSON. The [mean-win-rate implementation](https://github.com/stanford-crfm/helm/blob/12ab30bc79e44881597e2486e38c96b5b5931ef6/src/helm/benchmark/presentation/summarize.py#L213-L260) counts the fraction of other models beaten on each scenario, gives ties half credit, and averages across scenarios. Final evaluation recomputes these win rates over the genuine pool plus the final router. Score gap and score regret still use equal-category scores.

The source revisions used for frontend evidence are the final official revisions on the release dates, not verified deployment hashes. The archived release JSON fixes the scenario coordinates directly.

## Open LLM v2

The archived [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) reports IFEval, BBH, MATH Lvl 5, GPQA, MuSR, MMLU-Pro, and their separately rounded Average. The six task scores and Average are rounded to two decimal percentage places. The attacker uses the six task coordinates; endpoint ordering uses Average.

Follow the [official normalization](https://huggingface.co/docs/leaderboards/en/open_llm_leaderboard/normalization): subtract chance performance, divide by remaining range, and clip below zero. BBH and MuSR normalize each constituent task before averaging. GPQA pools its subset occurrences before normalization. IFEval equally averages prompt-level and instruction-level strict accuracy; instruction counts weight the latter. MATH uses the corrected math-verify evaluator pinned in the source lock. Overall evaluation averages the six unrounded normalized scores.

The eight-model common frame contains 21,599 occurrences (20,791 distinct prompt hashes). Seven MMLU-Pro items with inconsistent prompts across model records are excluded. Duplicate prompt hashes stay together across S/T. Source scores and normalization records are `data/openllm_v2_source_scores.json` and `data/openllm_v2_normalization.json`; all source revisions and hashes are pinned in `data/extension_sources/openllm_v2_source_lock.json`.

## SWE-bench Verified

The frozen [Verified leaderboard](https://www.swebench.com/) and official [experiments repository](https://github.com/SWE-bench/experiments) supply per-instance resolutions for 169 systems over all 500 instances. This is an offline replay of saved agent outcomes. It models ten repository-group feedback coordinates reconstructed from public item outcomes, rather than claiming that the leaderboard publishes ten scalar columns.

Pool repositories with fewer than ten instances using repository identity and counts alone: Flask, Seaborn, and Requests form one group of eleven instances. Keep the other nine repositories separate. Both overall ranking and score-gap evaluation use the instance-weighted resolution rate. Exact count-derived feedback retains ties and needs no additional independent overall coordinate.

The pool comprises 133 complete resolved-ID lists and 36 complete embedded per-instance records whose aggregates match the frozen official scores. Missing records and aggregate mismatches are excluded. `data/swe_verified_sources.json` records inclusion decisions; `data/extension_sources/swe_verified_source_lock.json` pins the dataset, leaderboard, and result records. The one-third reused split contains 166 instances and the held-out split 334.

Open LLM v1 remains only in the archived original release and its source evidence. The website uses v2.

## Terminal-Bench-Science

The released bank contains 70 task identities with three runs each. Repeated runs are averaged within a task and kept together when splitting. Feedback is emulated from success counts, using exact domain means and a task-weighted overall. The standard top-five Pareto endpoint rule is undefined in 38 of the 250 partitions; reported attack comparisons use the remaining 212, with paired baselines. Source records and hashes are in `results/outcome_banks/terminal_science/`.
