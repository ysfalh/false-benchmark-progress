# Data provenance and licenses

The repository distributes compact numerical evaluation records, task identities, and normalized prompt hashes. It omits prompt text, reference answers, and model responses. The inputs are sufficient to reproduce the reported routing experiments without downloading the much larger original detail records.

| Input | Historical source | Retained information |
| --- | --- | --- |
| LiveBench | [2024-08-31 public table](https://github.com/LiveBench/livebench.github.io/blob/1fadceff99bda3f8044aefda9a55e15f8c148c56/public/table_2024_08_31.csv) and model evaluation records | 33 models; 612 non-calibration items |
| HELM Capabilities | [HELM public results](https://crfm.stanford.edu/helm/capabilities/latest/) at the release pinned in `configs/helm_capabilities.json` | 68 models; 3,190 items; five public scenarios |
| HELM Lite | [HELM public results](https://crfm.stanford.edu/helm/lite/latest/) at the release pinned in `configs/helm_lite.json` | 14 models; 8,159 items; ten public scenarios |
| Open LLM v2 | [Archived model-detail repositories](https://huggingface.co/open-llm-leaderboard), pinned in `data/extension_sources/openllm_v2_source_lock.json` | Eight models; 21,599 occurrences; six normalized scores |
| SWE-bench Verified | [Official experiment records](https://github.com/SWE-bench/experiments), pinned in `data/extension_sources/swe_verified_source_lock.json` | 169 systems; 500 instances; ten repository groups |
| Terminal-Bench-Science 0.1 | [Versioned public outcomes](https://www.terminal-bench-science.ai/api/leaderboard?package=terminal-bench-science%2Fterminal-bench-science&name=v0-1-eval), with source hash in `results/outcome_banks/terminal_science/provenance.json` | 17 configurations; 70 tasks; three runs per task; five domains |

The HELM landing pages above aid discovery; the configuration releases and source records, not a mutable latest page, define the experiment. `data/manifest.json` records source hashes and retained dimensions. `results/release_v2/checksums.json` checks active inputs and canonical results. Open LLM v1 and its MC2 evidence remain archived. Source locks and reconstruction records record exact per-item reconstruction and exclusions. `docs/interface_sources.json` pins historical interface evidence; `docs/openllm_mc2_sources.json` pins the model-specific MC2 files and revisions. See [benchmark interfaces](benchmark_interfaces.md) for aggregation, coordinate mapping, and precision.

The website’s introductory Terminal-Bench-Science excerpt uses `results/outcome_banks/terminal_science/inputs.json`, checked against the input hash in the adjacent `provenance.json`. Its domain scores are reconstructed from the saved task outcomes: three runs are averaged within each task, then tasks are weighted equally within each domain and in the aggregate. The HELM excerpt comes from the unchanged public JSON file in `data/interfaces/`, checked against `docs/interface_sources.json`. The site build records each displayed value in the site’s source ledger. Both excerpts show percentages rounded to one decimal.

## Terms

The MIT license covers this project's code and original documentation. It does not relicense upstream benchmark content, model outputs, or fonts.

- The pinned [LiveBench evaluator code](https://github.com/LiveBench/LiveBench/blob/3953c6cb2f478285ad28e7372383276ccfea692c/LICENSE) and [HELM code](https://github.com/stanford-crfm/helm/blob/12ab30bc79e44881597e2486e38c96b5b5931ef6/LICENSE) carry Apache-2.0 licenses. Those code licenses do not by themselves establish terms for every constituent dataset.
- The inspected historical [LiveBench math card](https://huggingface.co/datasets/livebench/math/blob/17808a1857c4504ca390b380c40ab6dc604fd8c4/README.md) and [Open LLM detail card](https://huggingface.co/datasets/open-llm-leaderboard-old/details_meta-llama__Meta-Llama-3-8B-Instruct/blob/0bb67ddda727931a9e7428fdab9f3654a737b1a8/README.md) do not state a separate dataset license. No blanket license claim is made for the bundled numerical records. A redistribution license for every source has not been established here.
- Original questions and answers must be obtained under their respective source terms. They are not included in this artifact.
- TeX Gyre Pagella fonts retain the [GUST Font License](../site/fonts/GUST-FONT-LICENSE.txt); their provenance is in [the font notes](../site/fonts/README.md).
- The custom-benchmark example generates synthetic values and uses no upstream dataset.

Source attribution and omission of question text do not establish a redistribution grant. Before a public release is presented as fully licensed, the maintainers should resolve the terms for the compact numerical dataset derivatives. The code can be used under MIT independently of those inputs.

## Rebuilding the new inputs

Run `python -m scripts.import_openllm_v2 --cache /path/to/cache --output /path/to/new-staging` with Python 3.10+, a locally authenticated Hugging Face account, huggingface_hub, math-verify 0.5.2, sympy 1.14.0, and latex2sympy2-extended 1.0.6. The importer verifies pinned raw and compact hashes and applies the archived MATH correction. Raw prompts and responses stay in the external cache.

Run `python -m scripts.import_swe_verified --cache /path/to/cache --output /path/to/new-staging` to rebuild Verified. Bundled compressed result records contain instance IDs and resolutions; its prompt-bearing dataset is fetched into the external cache. Both importers refuse to overwrite an existing staging directory. Ordinary experiment reproduction uses the bundled compact inputs and requires no downloads.

Repeated runs in the outcome bank are averaged within each task before splitting. The public domain feedback is emulated from released outcomes, with equal task weights. The full original agent inputs and environment bytes were not independently certified. The bank retains the 38 partitions where fewer than two models remain on the public Pareto frontier; conditional estimates use the other 212.
