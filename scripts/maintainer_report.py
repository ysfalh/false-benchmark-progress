"""Run a local evaluator factory through restricted feedback and render its report."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import platform
import shlex
import numpy as np
from benchmark_progress.adapter import run_attack
from benchmark_progress.feedback_policy import FeedbackPolicy, MODES, overall_scores, select_policy_endpoints
from benchmark_progress.maintainer_view import write_report
from scripts.no_attack import choose_public_winner


def main(factory, mode, budget, output, routing_seed='maintainer-v1', tie_seed='maintainer-ties-v1'):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    module,attribute=factory.split(':',1)
    evaluator=getattr(importlib.import_module(module),attribute)()
    public=evaluator.public_access()
    native=public.genuine_public_profiles()
    policy=FeedbackPolicy(mode,tuple(native.coordinates),tuple(native.categories or native.coordinates))
    restricted=policy.restrict(public)
    profiles=restricted.genuine_public_profiles()
    leader=choose_public_winner(profiles.models,overall_scores(profiles.scores,profiles.categories))
    leader_name=profiles.models[leader]
    attack=run_attack(restricted,budget,routing_seed=routing_seed,tie_seed=tie_seed,endpoint_selector=select_policy_endpoints)
    final_public=float(overall_scores(attack.public_feedback[-1],profiles.categories))
    leader_public=float(overall_scores(profiles.scores,profiles.categories)[leader])
    select_router=round(1-final_public,10)<round(1-leader_public,10)
    # The genuine model and final router are fixed before held-out evaluation.
    def evaluate(result):
        row=dict(evaluator.evaluate(policy.for_evaluator(result)))
        genuine=overall_scores(profiles.scores,profiles.categories)
        final=overall_scores(result.public_feedback[-1],profiles.categories)
        row['rank_s']=1+int((np.round(1-genuine,10)<round(1-float(final),10)).sum())
        row['false_winner']=row['rank_s']==1 and row['rank_t']>1
        for key in ['reused_score_pp','heldout_score_pp','heldout_regret_pp','rank_t']:
            if key not in row or not np.isfinite(row[key]): raise ValueError('Evaluator missing finite '+key)
        return row
    a=evaluate(attack)
    s=dict(evaluator.evaluate_genuine(leader_name))
    s.update(rank_s=1,false_winner=s['rank_t']>1)
    for key in ['reused_score_pp','heldout_score_pp','heldout_regret_pp','rank_t']:
        if key not in s or not np.isfinite(s[key]):raise ValueError('Evaluator missing finite no-attack '+key)
    description=evaluator.describe()
    command='python -m scripts.maintainer_report --adapter '+shlex.quote(factory)+' --policy '+mode+' --budget '+str(budget)+' --routing-seed '+shlex.quote(routing_seed)+' --tie-seed '+shlex.quote(tie_seed)+' --output runs/maintainer-rerun'
    report=dict(name=description['name'],scope=description['scope'],
        feedback=f'{mode.title()} feedback: {len(policy.coordinates)} score(s) per submission at 0.001 pp precision. Endpoint selection and all submissions see this same surface. Category means and the aggregate preserve equal-category weighting. No separate overall score is supplied.',
        coverage=description['coverage'], endpoints=[attack.choice['base_a'],attack.choice['base_b']],
        endpoint_rule='Choose the most different pair among the top five genuine models by the allowed public score, using equal-category-weighted absolute differences and lexical ties.',
        budget=budget,no_attack=s,no_attack_model=leader_name,attack=a,
        selection=dict(model='Final router' if select_router else leader_name,
            heldout_score_pp=a['heldout_score_pp'] if select_router else s['heldout_score_pp'],
            model_selection_loss_pp=s['heldout_score_pp']-a['heldout_score_pp'] if select_router else 0.),
        utility=description.get('utility','No population distinguishability estimate from a single trial; use the paired study for that measure.'),
        notes=description['notes'],reproduction=dict(command=command,adapter=factory,policy=mode,
            routing_seed=routing_seed,tie_seed=tie_seed,submissions=f'{budget} attack; 0 no-attack submissions; genuine profiles free',
            python=platform.python_version(),numpy=np.__version__,
            adapter_sha256=hashlib.sha256(Path(importlib.import_module(module).__file__).read_bytes()).hexdigest(),
            **description['reproduction']))
    write_report(report,output)
    transcript=dict(coordinates=profiles.coordinates,public_profiles=profiles.scores.tolist(),
        models=profiles.models,choice=attack.choice,candidate_and_final_feedback=attack.public_feedback.tolist(),
        rule=dict(weights=attack.final.rule.weights,fallback=attack.final.rule.fallback),
        routing_seed=routing_seed,tie_seed=tie_seed)
    (output/'public_transcript.json').write_text(json.dumps(transcript,indent=2)+'\n')
    print(f'Report: {output / "report.html"}',flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adapter',default='examples.custom_benchmark.run:SyntheticBenchmark',help='Local module:zero_argument_evaluator_factory')
    p.add_argument('--policy',choices=MODES,default='category')
    p.add_argument('--budget',type=int,default=8)
    p.add_argument('--routing-seed',default='maintainer-v1')
    p.add_argument('--tie-seed',default='maintainer-ties-v1')
    p.add_argument('--output',type=Path,default=Path('runs/maintainer-example'))
    a=p.parse_args();main(a.adapter,a.policy,a.budget,a.output,a.routing_seed,a.tie_seed)
