"""Reader-facing report in the research website's existing style."""
import html
import json
from pathlib import Path
import shutil
from .benchmarks import ROOT


def write_report(report, output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(ROOT/'site/style.css',output/'style.css')
    (output/'fonts').mkdir()
    for font in (ROOT/'site/fonts').glob('*.woff'):shutil.copyfile(font,output/'fonts'/font.name)
    for name in ['GUST-FONT-LICENSE.txt','README.md']:
        shutil.copyfile(ROOT/'site/fonts'/name,output/'fonts'/name)
    e=lambda value:html.escape(str(value))
    title=report['name']+' — feedback report'
    blocks=[];md=['# '+title,'']
    def p(text):blocks.append('<p>'+e(text)+'</p>');md.extend([text,''])
    def heading(text):blocks.append('<h2>'+e(text)+'</h2>');md.extend(['## '+text,''])
    def table(headers,rows):
        kind='report-comparison' if len(headers)>2 else 'report-metadata'
        if len(headers)>2:blocks.append('<p class="table-hint">Scroll to see both outcome columns.</p>')
        blocks.append('<div class="table-scroll" tabindex="0"><table class="'+kind+'"><thead><tr>'+''.join('<th scope="col">'+e(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+e(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>')
        md.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x).replace('|','\\|') for x in row)+' |' for row in rows]+[''])
    p(report['scope']);heading('1. Feedback and coverage');p(report['feedback']);table(['Coverage','Value'],list(report['coverage'].items()))
    heading('2. No attack and repeated submissions')
    p('Without attack, the genuine public winner is '+report['no_attack_model']+'. Select this model from public feedback, then evaluate the same model held out. It adds no submissions.')
    p('The attack uses endpoint A, '+report['endpoints'][0]+', and endpoint B, '+report['endpoints'][1]+'. '+report['endpoint_rule'])
    def display(x):return ('Yes' if x else 'No') if isinstance(x,bool) else f'{x:.3f}' if isinstance(x,float) else str(x)
    fields=[('Reused score (%)','reused_score_pp'),('Held-out score (%)','heldout_score_pp'),('Reused rank','rank_s'),('Held-out rank','rank_t'),('Public leader fails held out','false_winner'),('Score below best genuine (pp)','heldout_regret_pp')]
    table(['Outcome','No attack (0 submissions)',f'Final router ({report["budget"]} submissions)'],[[label,display(report['no_attack'][key]),display(report['attack'][key])] for label,key in fields])
    p(f"Following the public ranking selects {report['selection']['model']}. Model-selection loss is {report['selection']['model_selection_loss_pp']:+.3f} pp: the no-attack choice’s held-out score minus this choice’s held-out score. Positive means worse selection; negative means an improvement. Public ties retain the genuine model.")
    p('The columns evaluate the genuine public winner and final router separately. A false winner ranks first on the allowed reused score and below a genuine model held out. The last row compares each model with the best genuine held-out score; it can be negative for a router. No-attack ranks use genuine models; router ranks add the final submission.')
    for note in report['notes']:p(note)
    heading('3. Reproduce this run');p(report['reproduction']['command']);table(['Setting','Value'],[(k,v) for k,v in report['reproduction'].items() if k!='command'])
    p('The evaluator keeps private item outcomes and held-out scoring. This report contains only approved aggregate results. The two-callback adapter is a boundary for trusted local code, not a sandbox for untrusted Python.')
    extra='p,td{overflow-wrap:anywhere}h2{margin-top:38px}table{font-size:.84rem}.report-comparison{min-width:520px}.report-metadata{table-layout:fixed;width:100%}.report-metadata th:first-child{width:40%}.report-metadata td{min-width:0}pre{white-space:pre-wrap}'
    document='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+e(title)+'</title><link rel="stylesheet" href="style.css"><link rel="icon" href="data:,"><style>'+extra+'</style></head><body><main class="page"><header class="article-header"><h1>'+e(title)+'</h1></header>'+''.join(blocks)+'</main></body></html>'
    (output/'report.html').write_text(document);(output/'report.md').write_text('\n'.join(md)+'\n')
    (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
