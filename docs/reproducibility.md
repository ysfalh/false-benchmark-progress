# Data and reproduction

The five configurations specify 250 paired trials and submission budgets 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, and 2048. Budget 1 is the static task router; all others use random candidates and task-linear decoding. Top-five and full-frontier endpoint selection are reported separately.

## Inputs

`data/` contains compressed score matrices and prompt metadata. These compact inputs are included so reproduction needs no network download. Prompt text, model answers, and unused fields are omitted. Large original model-detail files remain at their pinned source URLs; no model evaluation is repeated.

The original unused calibration partition is excluded before constructing these files. Its values never enter endpoint selection or routing. The remaining rows, normalized prompt hashes, original score values, model pools, and split seeds are unchanged. Task-stratified S/T allocation keeps duplicate observable prompts together.

| Benchmark | S items | T items |
| --- | ---: | ---: |
| LiveBench | 204 | 408 |
| HELM Capabilities | 1,064 | 2,126 |
| HELM Lite | 2,721 | 5,438 |
| Open LLM v2 | 7,189 | 14,410 |
| SWE-bench Verified | 166 | 334 |

`data/manifest.json` records the original source dimensions; benchmark reconstruction records and source locks are under `data/` and `data/extension_sources/`. Active input hashes are in `results/release_v2/checksums.json`. `configs/` pins historical releases and models. Source interface and MC2 evidence is under `docs/`. The random hash domain bytes are retained verbatim so renaming the software does not change any candidate routing decisions.

## Metrics

The main selection result is **model-selection loss**: the held-out score of the genuine public winner before attack minus the held-out score of the model selected after adding the final router. Selection uses only public ranking values, with ties favoring genuine models and then lexical order. The signed mean includes all 250 trials, including unchanged choices. Positive means the new ranking leads to worse selection. This analysis is derived from the existing records; it does not replace their original fields. See [selection definitions and secondary comparisons](selection_loss.md).

Score gap is the maximum absolute gap between separate running-best aggregate losses on S and T. Both histories begin with genuine models in Unicode lexical order and include candidate submissions and the final router. The two endpoint entries repeat genuine models and do not change the running best. The genuine-model baseline is the same statistic restricted to the initial genuine-model history.

Aggregate loss follows each benchmark’s scoring rule. Open LLM v2 averages six normalized scores; SWE-bench Verified uses the instance-weighted resolution rate. The other benchmarks equally average task losses within each category, then average categories. The score gap is averaged across trials. This is not the final router's own S/T score gap.

Ranks compare only genuine models and the final router. Competition ties share rank: compare losses after rounding to ten decimal places, nearest with ties to even. HELM Lite ranks use mean win rate on its ten public scenarios; SWE-bench Verified ranks by instance-weighted resolution rate, and the other benchmarks rank by equal-category score. The selection-loss figure's rank drop compares the before/after choices in the same genuine-plus-router pool.

The archived `first_s_not_t` event means the final router shares first place on S but is below first on T. It does not establish that following the ranking worsens selection. Archived `heldout_regret_pp` is the best genuine T score minus the final router's T score, in percentage points, averaged over all trials in the release tables. The supplemental severity figure instead conditions its mean regret on false-winner trials. Neither is the new model-selection loss. Lite also retains regret in mean win rate. These fields and the one-submission static control keep their original meanings.

Intervals use 2,000 common bootstrap resamples of the 250 trials, seed 2026081900. Means and medians are recomputed in each resample; interval endpoints use the 2.5th and 97.5th percentiles. There is no per-trial tuning or best-of-policy selection. Threshold crossings describe the tested budget grid, not a continuous optimum.

## Release verification

```bash
python -m scripts.verify results/release_v2 --allow-code-changes
```

This verifies input and release hashes; independently checks every public Pareto pool and maximum-MAD pair; recomputes all 30,000 reported trial outcomes from compact evaluation records; and regenerates the complete summaries and intervals. This default check does not reconstruct feedback from private item scores. The record for each trial retains the running-best loss histories and the genuine/final category profiles needed for these metrics. It omits redundant candidate task tables. The separate non-leading-endpoint diagnostic retains all its original method outcomes.

The adapter and reporting code have been extended since the release. `--allow-code-changes` permits changed Python files under `benchmark_progress/` and `scripts/`, names them in the output, and evaluates the saved records with the current implementation. It never waives input, configuration or result hashes. Without this flag, source identity is checked too and the current extensions cause a checksum failure. The manifest records the summary aggregation. Summary tables and their checksums are regenerated from the saved trial outcomes; input and outcome records retain their original hashes.

To reconstruct genuine-model public task profiles and overall columns from bundled item scores, add `--inputs`. This uses separate aggregation code and the documented input reader and split procedure. It requires the item inputs; the default metric checks use only release records after checking input hashes. Candidate profiles are not retained in the compact release, so checking their computation requires a complete rerun.

`--smoke` checks both policies for trial 0 on each benchmark (120 outcomes), plus file integrity and table consistency; it skips bootstrap regeneration. CI uses this limited mode and the synthetic example. The full command checks all 30,000 outcomes and regenerates the 2,000-resample summaries using the documented statistical implementation.

Checksums establish integrity, not correctness by themselves. The verifier's endpoint and rank calculations are separate from the attack implementation. Reproduction from item scores supplies the additional end-to-end check:

```bash
python -m scripts.reproduce --benchmark livebench
python -m scripts.verify results/release_v2 --allow-code-changes --run runs/livebench
```

The comparison requires identical public profiles, endpoint choices, ranks, and events. Floating-point metrics must agree within 1e-10 percentage points, allowing only summation-order roundoff. `results/release_v2/verification.json` records the performed checks. New runs contain their configuration, input/code hashes, all rule records, public profiles, private evaluation records, and trial outcomes. They never overwrite the release.

## Reporting

The website also gives a [paired comparison of score gap with no attack](score_comparison.md), derived from the same released trials. Its first significant query budget uses one-sided basic-bootstrap lower bounds adjusted across all 55 primary benchmark–budget comparisons. This additional analysis leaves the canonical score intervals and effect-size thresholds unchanged.

```bash
python -m scripts.report --output runs/tables
python site/build.py
```

The first command regenerates the release tables, including false-winner rates and router regret. The second builds the website and the newer selection-loss summaries, including the secondary controls and practical-policy comparisons. `site/data/provenance.json` connects displayed values to their sources. Equation source is written in LaTeX in the HTML template and compiled into SVG with the paper's Computer Modern math. The bundled Palatino-family text fonts carry their upstream license.

These are within-snapshot held-out evaluations. No fresh-data generalization is claimed.

## Controlled feedback-dimension supplement

The `results/feedback_dimension_v1/` bundle contains the LiveBench results for different numbers of released score coordinates. Run `python -m scripts.verify_feedback` for canonical-record and summary checks, or `python -m scripts.reproduce_feedback` for the complete 250-trial rerun. The [study note](feedback_dimension.md) specifies task mappings, precision, aliases, and verification boundaries.

## Predictor comparison

The `results/predictor_v2/` summary bundle contains the LiveBench fits and their predictions on HELM Capabilities, HELM Lite, Open LLM v2, and SWE-bench Verified. The [methods note](predictor.md) describes the fits and saved inputs. `python -m pytest tests/test_predictor.py` checks their arithmetic against the saved LiveBench curves and canonical benchmark thresholds. The website build reads these saved values directly, including right-censored thresholds.

## Released-outcome banks

`results/outcome_banks/` contains compact task outcomes, prompt hashes, public router records, evaluation arrays, and checksums. Each bank has a configuration in `configs/` and uses the shared splitter, endpoint selector, gate generator, and decoder. Repeated runs stay within their task. Ineligible endpoint selections are retained without a fallback or redraw.

```bash
python -m scripts.reproduce --benchmark terminal_science
python -m scripts.verify_outcomes --benchmark terminal_science --rerun
```

The verifier reconstructs routing from saved integer weights, checks selection against the same no-attack choice, regenerates paired-bootstrap summaries, and optionally compares every replayed decision and outcome with the archive. It reads only local data and uses no model inference.
