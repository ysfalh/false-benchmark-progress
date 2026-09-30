"""Compile small mathematical expressions with the paper's LaTeX typography."""

import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

SITE = Path(__file__).resolve().parent


def render(expression, display=False):
    formula = (r'\displaystyle ' if display else '') + expression
    key = hashlib.sha256(formula.encode()).hexdigest()[:16]
    folder = SITE/'math'
    folder.mkdir(exist_ok=True)
    svg, dimensions = folder/f'{key}.svg', folder/f'{key}.json'
    if not svg.exists() or not dimensions.exists():
        env = dict(os.environ, PATH=os.environ.get('PATH','')+':/Library/TeX/texbin')
        for command in ['latex','dvisvgm']:
            if not shutil.which(command, path=env['PATH']):
                raise RuntimeError('Install a TeX distribution to render new equations. Existing equations need no TeX installation.')
        source = r'''\documentclass[11pt]{article}
\usepackage{amsmath,amssymb,mathtools}
\usepackage[active,tightpage]{preview}
\pagestyle{empty}
\begin{document}
\setbox0=\hbox{$FORMULA$}
\typeout{FORMULAHEIGHT:\the\ht0}
\typeout{FORMULADEPTH:\the\dp0}
\begin{preview}\box0\end{preview}
\end{document}
'''.replace('FORMULA',formula,1)
        with tempfile.TemporaryDirectory() as work:
            p=Path(work);(p/'math.tex').write_text(source)
            result=subprocess.run(['latex','-interaction=nonstopmode','-halt-on-error','math.tex'],cwd=p,env=env,capture_output=True,text=True,check=True)
            subprocess.run(['dvisvgm','--no-fonts','--exact','--bbox=min','-o','math.svg','math.dvi'],cwd=p,env=env,capture_output=True,text=True,check=True)
            text=(p/'math.svg').read_text()
            width=float(re.search(r"width='([\d.]+)pt'",text).group(1))
            height=float(re.search(r"height='([\d.]+)pt'",text).group(1))
            depth=float(re.search(r'FORMULADEPTH:([\d.]+)pt',result.stdout).group(1))
            # DVI SVG points use 72/inch, TeX points 72.27/inch.
            dimensions.write_text(json.dumps({'width_em':width/10.95*72.27/72,'height_em':height/10.95*72.27/72,'depth_em':depth/10.95})+'\n')
            svg.write_text(text)
    size=json.loads(dimensions.read_text())
    style=f'width:{size["width_em"]:.5f}em;height:{size["height_em"]:.5f}em;vertical-align:-{size["depth_em"]:.5f}em'
    return f'<img class="latex {"display" if display else "inline"}" src="math/{key}.svg" style="{style}" alt="{html.escape(expression,quote=True)}">'
