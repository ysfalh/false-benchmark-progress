# Attack and information boundary

A submission is a binary router over two genuine endpoint models. The submitter can compute a normalized prompt hash and its public task identity. The endpoint selector receives only genuine model names, public task profiles on the reused set S, and any separately exposed public overall score. The decoder receives only the two endpoint profiles and the candidate profiles.

Neither function receives item scores, labels, calibration data, S/T membership, or any held-out feedback. The runner records all final routing rules for both endpoint policies and all budgets before evaluating T. Private score matrices belong to the evaluator, not to the attack API.

## Endpoint selection

Compute the Pareto frontier of the public genuine-model profiles. Order that frontier by its public overall score, with lexical model-name ties. The primary rule retains the first min(5, frontier size) models, then chooses the pair with maximum unweighted mean absolute task-profile difference (MAD). The comparison rule maximizes the same MAD over the full frontier. Pair ties use canonical lexical order. If fewer than two frontier models exist, stop: no fallback is defined.

LiveBench's overall is the equal-category average derived from its rounded task profile. HELM Capabilities uses the mean public scenario score. HELM Lite uses mean win rate. Open LLM uses its separately rounded public Average column. The lower equal-category-score endpoint is A; orienting the pair this way keeps the random routing sequence reproducible.

## Task-weighted combination

Write a_j and b_j for public endpoint losses, r_tj for candidate t's public loss, and g_t(x) in {-1,+1} for its routing sign (+1 selects A). The decoder sets

    alpha_tj = (a_j + b_j)/2 - r_tj
    f_j(x) = sign(sum_t alpha_tj g_t(x)).

It selects A if f_j(x) is +1 and B if it is -1. A zero weighted sum uses the recorded prompt-hash tie rule. With no candidate submissions, the static control selects the publicly stronger endpoint for each task, using the same tie rule.

The item loss identity is

    loss_t(x) = (loss_A(x) + loss_B(x))/2
                + g_t(x) (loss_A(x) - loss_B(x))/2.

A task average therefore supplies a linear measurement of the endpoint loss differences. Correlating that measurement with the random routing signs estimates which endpoint is better for a prompt. Repeated prompt hashes share a sign; the measurement then concerns their combined contribution. Public rounding perturbs the measurements, and the attack retains that rounding. It does not recover an unrounded score.

The implementation uses public scores rather than losses, with weights 2*p_tj - p_Aj - p_Bj. These equal 2*alpha_tj, so the final signs are identical. Decimal feedback is represented as exact rational values at the released precision. HELM JSON numbers are represented by their exact binary floating-point values. Scaling each coordinate to integers preserves exact zero decisions without a fitted tolerance.

## Submission accounting

At budget k, submit k-1 random candidates and one final router. Already-public genuine models and endpoint profiles are free. Candidate prefixes and trial seeds are shared across budgets and policies. The static control costs one submission. The release also retains the historical count k+2, which includes the two endpoint records.

The secondary LiveBench diagnostic excludes every public S leader before full-frontier pair selection. It is not an alternative primary policy. Its saved ranks, router regret and original method outcomes, including the cumulative-router comparison, are retained in `non_top_trials.parquet`. It tests whether non-leading endpoints can produce a false winner; its 58% rate is not a measurement of the added model-selection loss. [Interpretation and archive limits](selection_loss.md#secondary-comparisons).
