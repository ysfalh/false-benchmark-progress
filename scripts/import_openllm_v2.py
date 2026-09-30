"""Rebuild Open LLM v2 inputs from pinned public sources (HF login required).

Python >=3.10, huggingface_hub, math-verify==0.5.2, sympy==1.14.0.
Raw benchmark prompts/responses stay in the supplied cache, outside the repo.
"""
import argparse,gzip,hashlib,importlib.metadata,json,re,shutil,urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from benchmark_progress.benchmarks import ROOT

def normalized(s):return ' '.join(s.split())
def digest(s):return hashlib.sha256(normalized(s).encode()).hexdigest()
def content(task,doc):
 if 'bbh_' in task:return doc['input']
 if 'math_' in task:return doc['problem']
 if task.endswith('ifeval'):return doc['prompt']
 if 'gpqa_' in task:return doc['Question']+'\n'+'\n'.join(sorted(doc[f'choice{i}'] for i in range(1,5)))
 if 'mmlu_pro' in task:return doc['question']+'\n'+'\n'.join(doc['options'])
 if 'musr_' in task:return doc['narrative']+'\n'+doc['question']+'\n'+str(doc['choices'])
 raise ValueError(task)
def compact(task,row):
 doc=row['doc'];metric='prompt_level_strict_acc' if task.endswith('ifeval') else 'exact_match' if 'math_' in task else 'acc' if task.endswith('mmlu_pro') else 'acc_norm'
 out={'doc_id':row['doc_id'],'content_hash':digest(content(task,doc)), 'score':float(row[metric])}
 if task.endswith('ifeval'):
  inst=row['inst_level_strict_acc'];out.update(instruction_count=len(inst),instruction_score=sum(inst)/len(inst))
 return out

def h(s):return hashlib.sha256(' '.join(s.split()).encode()).hexdigest()
def save(name,q,s,meta,**extras):
 q=pd.DataFrame(q).sort_values('question_id').reset_index(drop=True);s=pd.DataFrame(s).sort_values(['question_id','model']).reset_index(drop=True)
 assert not q.question_id.duplicated().any();assert not s[['question_id','model']].duplicated().any()
 models=sorted(s.model.unique());assert len(s)==len(q)*len(models)
 config=dict(benchmark=name,snapshot=meta,categories=sorted(q.category.unique()),models=models,trials=250,budgets=[1,2,4,8,16,32,64,128,256,512,1024,2048],split_seed_base=202609210000,routing_seed=name+'-bank0-20260921',tie_seed=name+'-ties0-20260921',holdout_fraction=2/3,**extras)
 q.to_parquet(R/'data'/f'{name}_questions.parquet',index=False);s.to_parquet(R/'data'/f'{name}_scores.parquet',index=False);(R/'configs'/f'{name}.json').write_text(json.dumps(config,indent=2)+'\n')
 print(name,len(q),'items',len(models),'models',q.groupby('category').size().to_dict(),flush=True)
 return config

def openllm():
 qmap={};s=[];audits=[];baselines={};reference=None;conflicts=set()
 for model in json.loads((P/'ollm_pool.json').read_text()):
  name=model['model'];slug=name.replace('/','__');result=json.loads((P/(slug+'.result.json')).read_text());seen=set()
  files=sorted((P/'openllm_samples').glob(slug+'--*.json'));assert len(files)==39
  for file in files:
   archived=json.loads(file.read_text());task=archived['task'];d=json.loads((P/'rescored_math'/file.name).read_text()) if 'math_' in task else archived
   if 'math_' in task:assert d['rescoring']['matches']
   category=('BBH' if 'bbh_' in task else 'MATH Lvl 5' if 'math_' in task else 'GPQA' if 'gpqa_' in task else 'MuSR' if 'musr_' in task else 'IFEval' if task.endswith('ifeval') else 'MMLU-Pro')
   metric='prompt_level_strict_acc,none' if category=='IFEval' else 'exact_match,none' if category=='MATH Lvl 5' else 'acc,none' if category=='MMLU-Pro' else 'acc_norm,none'
   exp=result['results'][task][metric];actual=np.mean([r['score'] for r in d['rows']]);assert abs(actual-exp)<1e-12,(name,task,actual,exp)
   if category=='IFEval':
    expected_inst=result['results'][task]['inst_level_strict_acc,none'];actual_inst=sum(r['instruction_count']*r['instruction_score'] for r in d['rows'])/sum(r['instruction_count'] for r in d['rows']);assert abs(expected_inst-actual_inst)<1e-12
   audits.append(dict(model=name,task=task,items=len(d['rows']),archived_score=float(np.mean([r['score'] for r in archived['rows']])),reconstructed_score=float(actual),published_score=exp,rescored=category=='MATH Lvl 5'))
   if category=='BBH':baselines[task]=1/len(result['configs'][task]['doc_to_choice'])
   elif category=='MuSR':baselines[task]=1/({'leaderboard_musr_murder_mysteries':2,'leaderboard_musr_object_placements':5,'leaderboard_musr_team_allocation':3}[task])
   for row in d['rows']:
    iid=task+'/'+str(row['doc_id']);seen.add(iid)
    qr=dict(question_id=iid,content_hash=row['content_hash'],category=category,task='GPQA' if category=='GPQA' else task,source_task=task,instruction_count=row.get('instruction_count',0))
    if iid in qmap:
     if qmap[iid]!=qr:conflicts.add(iid)
    else:qmap[iid]=qr
    s.append(dict(question_id=iid,model=name,score=float(row['score']),instruction_score=float(row.get('instruction_score',0))))
  if reference is None:reference=seen
  else:assert reference==seen,('Incomplete item universe',name)
 save('openllm_v2',[v for k,v in qmap.items() if k not in conflicts],[r for r in s if r['question_id'] not in conflicts],dict(project='Open LLM Leaderboard',release='v2 archived eight-model pool',release_url='https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard',math_correction_commit='79492751739f92ba3c6503410d32510c55880291',math_verify_version='0.5.2'),repeats=1,chance_baselines=baselines,excluded_mismatched_prompts=sorted(conflicts))
 (R/'data'/'openllm_v2_source_scores.json').write_text(json.dumps(audits,indent=2)+'\n')


def prepare_cache(cache, lock):
    from huggingface_hub import get_token
    token=get_token()
    def fetch(rec):
        path=cache/'raw'/rec['sha256'];path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            if rec.get('bundled'):
                path.write_bytes(gzip.decompress((ROOT/rec['bundled']).read_bytes()))
            else:
                headers={'Authorization':'Bearer '+token} if token and (rec.get('authenticated') or '/open-llm-leaderboard/' in rec['url']) else {}
                with urllib.request.urlopen(urllib.request.Request(rec['url'],headers=headers),timeout=180) as response,path.open('wb') as out:
                    shutil.copyfileobj(response,out)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=rec['sha256']:
            raise ValueError(f'Source checksum mismatch: {rec["url"]}')
        return path
    for rec in lock['files']:
        src=fetch(rec);out=cache/rec['destination'];out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,out)
    for name,key in [('ollm_pool.json','models')]:
        (cache/name).write_text(json.dumps(lock[key]))
    assert importlib.metadata.version('math-verify')=='0.5.2'
    assert importlib.metadata.version('sympy')=='1.14.0'
    from math_verify import LatexExtractionConfig,parse,verify
    for i,rec in enumerate(lock['samples'],1):
        path=fetch(rec);rows=[];repaired=[]
        with path.open() as f:
            for line in f:
                if not line.strip():continue
                raw=json.loads(line);row=compact(rec['task'],raw);rows.append(row)
                if 'math_' in rec['task']:
                    new=dict(row);new['archived_score']=row['score']
                    new['score']=int(verify(parse(raw['doc']['solution'],extraction_config=[LatexExtractionConfig()]),parse(raw['filtered_resps'][0])))
                    repaired.append(new)
        actual=repaired or rows
        digest=hashlib.sha256(json.dumps(actual,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if len(actual)!=rec['rows'] or digest!=rec['compact_rows_sha256']:
            raise ValueError(f'Extracted item mismatch: {rec["model"]}/{rec["task"]}')
        data={k:rec[k] for k in ['model','task','url','sha256','bytes']};data['rows']=rows
        out=cache/'openllm_samples'/rec['file'];out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(data))
        if repaired:
            data['rows']=repaired;data['rescoring']={'matches':True}
            out=cache/'rescored_math'/rec['file'];out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(data))
        if i%39==0:print(f'Imported {i}/{len(lock["samples"])} source files',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path,help='New staging directory; never overwrites repo inputs')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    R=args.output;P=args.cache
    R.mkdir(parents=True);(R/'data').mkdir();(R/'configs').mkdir();P.mkdir(parents=True,exist_ok=True)
    prepare_cache(P,json.loads((ROOT/'data'/'extension_sources'/'openllm_v2_source_lock.json').read_text()))
    openllm()
