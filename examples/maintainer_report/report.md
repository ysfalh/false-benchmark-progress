# Synthetic two-task example — feedback report

One reproducible synthetic trial demonstrating the adapter and report. It is not evidence about a real benchmark or an estimate of a false-winner rate.

## 1. Feedback and coverage

Category feedback: 2 score(s) per submission at 0.001 pp precision. Endpoint selection and all submissions see this same surface. Category means and the aggregate preserve equal-category weighting. No separate overall score is supplied.

| Coverage | Value |
| --- | --- |
| Genuine models | 3 |
| Reused items | 40 |
| Held-out items | 40 |
| Tasks / categories | 2 |
| Per task | 20 reused + 20 held out |
| Missing scores | 0 |
| Duplicate prompts crossing split | 0 |

## 2. No attack and repeated submissions

Without attack, the genuine public winner is model_b. Select this model from public feedback, then evaluate the same model held out. It adds no submissions.

The attack uses endpoint A, model_a, and endpoint B, model_b. Choose the most different pair among the top five genuine models by the allowed public score, using equal-category-weighted absolute differences and lexical ties.

| Outcome | No attack (0 submissions) | Final router (8 submissions) |
| --- | --- | --- |
| Reused score (%) | 57.500 | 72.500 |
| Held-out score (%) | 45.000 | 50.000 |
| Reused rank | 1 | 1 |
| Held-out rank | 3 | 3 |
| Public leader fails held out | Yes | Yes |
| Score below best genuine (pp) | 15.000 | 10.000 |

Following the public ranking selects Final router. Model-selection loss is -5.000 pp: the no-attack choice’s held-out score minus this choice’s held-out score. Positive means worse selection; negative means an improvement. Public ties retain the genuine model.

The columns evaluate the genuine public winner and final router separately. A false winner ranks first on the allowed reused score and below a genuine model held out. The last row compares each model with the best genuine held-out score; it can be negative for a router. No-attack ranks use genuine models; router ranks add the final submission.

The evaluator averages the two tasks equally. Random binary outcomes come from fixed synthetic abilities and seed 7.

One split and one attack cannot establish a safe feedback policy. Task and category feedback coincide here because each category has one task.

## 3. Reproduce this run

python -m scripts.maintainer_report --adapter examples.custom_benchmark.run:SyntheticBenchmark --policy category --budget 8 --routing-seed maintainer-v1 --tie-seed maintainer-ties-v1 --output runs/maintainer-rerun

| Setting | Value |
| --- | --- |
| adapter | examples.custom_benchmark.run:SyntheticBenchmark |
| policy | category |
| routing_seed | maintainer-v1 |
| tie_seed | maintainer-ties-v1 |
| submissions | 8 attack; 0 no-attack submissions; genuine profiles free |
| python | 3.9.7 |
| numpy | 1.26.4 |
| adapter_sha256 | 213bd55a5f35c4823f556a831ac17f516c1c8184cc2cce39002f6d3d369e3749 |
| data | Generated locally; no downloads or model calls |
| data_seed | 7 |
| split | First 20 of 40 examples per task reused; remainder held out |

The evaluator keeps private item outcomes and held-out scoring. This report contains only approved aggregate results. The two-callback adapter is a boundary for trusted local code, not a sandbox for untrusted Python.

