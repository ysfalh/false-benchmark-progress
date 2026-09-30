/* Responsive inline SVG. Values and interval endpoints come from frozen tables. */
(() => {
  'use strict';
  const specifications = JSON.parse(document.getElementById('native-chart-data').textContent);
  const NS = 'http://www.w3.org/2000/svg';
  const number = n => n.toLocaleString('en-US', {maximumFractionDigits: 2, minimumFractionDigits: 2});
  const short = n => n.toLocaleString('en-US', {maximumFractionDigits: 2});
  function svgEl(tag, attrs = {}, text) {
    const node = document.createElementNS(NS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function marker(shape, x, y, size, color, className = 'plot-dot', filled = false) {
    const attrs = {stroke: color, class: className, fill: filled ? color : 'var(--paper)'};
    if (shape === 'square') return svgEl('rect', {...attrs, x: x-size, y: y-size, width: size*2, height: size*2});
    if (shape === 'diamond') return svgEl('path', {...attrs, d: `M${x},${y-size*1.25}L${x+size*1.1},${y}L${x},${y+size*1.25}L${x-size*1.1},${y}Z`});
    if (shape === 'triangle' || shape === 'triangle-down') {
      const direction = shape === 'triangle' ? 1 : -1;
      return svgEl('path', {...attrs, d: `M${x},${y-direction*size*1.2}L${x+size*1.1},${y+direction*size*.85}L${x-size*1.1},${y+direction*size*.85}Z`});
    }
    if (shape === 'cross') return svgEl('path', {...attrs, fill: 'none', d: `M${x-size},${y-size}L${x+size},${y+size}M${x-size},${y+size}L${x+size},${y-size}`});
    return svgEl('circle', {...attrs, cx: x, cy: y, r: size});
  }
  function extent(panel) {
    const values = panel.series.flatMap(s => s.points.flatMap(p => [p.y, p.low ?? p.y, p.high ?? p.y]));
    values.push(...panel.references.map(r => r.value));
    const minimum = Math.min(...values);
    let low = minimum, high = Math.max(...values), span = high - low;
    if (span === 0) span = Math.max(Math.abs(low) * .1, 1);
    low -= span * .09; high += span * .09;
    const rough = (high - low) / 4, power = 10 ** Math.floor(Math.log10(rough));
    const step = [1, 2, 2.5, 5, 10].map(x => x * power).find(x => x >= rough) || 10 * power;
    const edgeStep = step / 5;
    low = +(Math.floor(low / edgeStep) * edgeStep).toPrecision(10);
    high = +(Math.ceil(high / edgeStep) * edgeStep).toPrecision(10);
    if (minimum >= 0) low = Math.max(0, low);
    if (panel.bounds) { low = Math.max(panel.bounds[0], low); high = Math.min(panel.bounds[1], high); }
    const ticks = [];
    for (let v = Math.ceil((low - 1e-9) / step) * step; v <= high + 1e-9; v += step) {
      ticks.push(Math.abs(v) < 1e-8 ? 0 : +v.toFixed(8));
    }
    return {low, high, ticks};
  }
  function legend(container, items) {
    const row = el('div', `chart-legend${items.length === 6 ? ' six-keys' : ''}`);
    row.setAttribute('role', 'list');
    row.setAttribute('aria-label', 'Chart legend');
    items.forEach(s => {
      const item = el('span', 'chart-legend-item');
      item.setAttribute('role', 'listitem');
      const swatch = svgEl('svg', {class: `chart-key${s.pointOnly ? ' point-key' : ''}`, viewBox: `0 0 ${s.pointOnly ? 16 : 36} 16`, 'aria-hidden': 'true', focusable: 'false'});
      if (!s.pointOnly) {
        const line = svgEl('line', {x1: 1, x2: 35, y1: 8, y2: 8, stroke: s.color, 'stroke-width': s.reference ? 1.3 : 2});
        const dash = s.dash || (s.reference ? '4 4' : null);
        if (dash) line.setAttribute('stroke-dasharray', dash);
        swatch.append(line);
      }
      if (!s.reference) swatch.append(marker(s.marker, s.pointOnly ? 8 : 18, 8, 3.5, s.color, 'legend-dot', s.pointOnly));
      item.append(swatch, document.createTextNode(s.name)); row.append(item);
    });
    container.append(row);
  }
  function lineChart(container, panel, caption) {
    const range = extent(panel);
    const summary = panel.series.map(s => {
      const first = s.points[0], last = s.points[s.points.length - 1];
      return `${s.name}: ${number(first.y)} at ${short(first.x)}, ${number(last.y)} at ${short(last.x)}.`;
    }).join(' ');
    const svg = svgEl('svg', {
      class: 'native-plot', role: 'img', 'aria-describedby': caption,
      'aria-label': `${panel.title}. ${panel.metric}, ${panel.unit}. ${panel.xLabel} on a logarithmic axis. Y-axis ${short(range.low)} to ${short(range.high)}. ${summary}`
    });
    container.append(svg);
    container.dataset.yMin = range.low; container.dataset.yMax = range.high;
    const xs = [...new Set(panel.series.flatMap(s => s.points.map(p => p.x)))].sort((a, b) => a - b);
    function render() {
      const width = container.clientWidth;
      if (width < 100) return;
      const height = svg.clientHeight || 240, left = 40, right = width - 22, top = 16, bottom = height - 28;
      const lx = Math.log2(xs[0]), hx = Math.log2(xs[xs.length - 1]);
      const x = v => left + (Math.log2(v) - lx) / (hx - lx) * (right - left);
      const y = v => bottom - (v - range.low) / (range.high - range.low) * (bottom - top);
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.replaceChildren();
      range.ticks.forEach(v => svg.append(
        svgEl('line', {x1: left, x2: right, y1: y(v), y2: y(v), class: 'plot-grid'}),
        svgEl('text', {x: left - 10, y: y(v) + 4, 'text-anchor': 'end'}, short(v))
      ));
      let ticks;
      if (panel.xLabel === 'Released score coordinates') {
        ticks = [1, 4, 16, 64, 256, 1024, 4096].filter(v => v >= xs[0] && v <= xs[xs.length - 1]);
      } else {
        const stride = xs.length <= 7 ? 1 : 2;
        ticks = xs.filter((_, i) => i % stride === 0);
        if (ticks[ticks.length - 1] !== xs[xs.length - 1]) ticks.push(xs[xs.length - 1]);
      }
      ticks.forEach(v => svg.append(svgEl('text', {
        x: x(v), y: bottom + 22, 'text-anchor': 'middle'
      }, short(v))));
      svg.append(svgEl('line', {x1: left, x2: right, y1: bottom, y2: bottom, class: 'plot-axis'}));
      panel.references.forEach(r => svg.append(svgEl('line', {
        x1: left, x2: right, y1: y(r.value), y2: y(r.value), stroke: r.color,
        'stroke-width': 1.3, 'stroke-dasharray': r.dash || '4 4'
      })));
      // Draw intervals behind the trends, at their exact observed budgets.
      panel.series.forEach(s => s.points.forEach(p => {
        if (!Number.isFinite(p.low) || !Number.isFinite(p.high)) return;
        const cx = x(p.x), low = y(p.low), high = y(p.high), cap = 3;
        svg.append(svgEl('path', {
          d: `M${cx},${low}V${high} M${cx - cap},${low}H${cx + cap} M${cx - cap},${high}H${cx + cap}`,
          stroke: s.color, class: 'plot-interval'
        }));
      }));
      panel.series.forEach(s => {
        const path = svgEl('path', {
          d: s.points.map((p, i) => `${i ? 'L' : 'M'}${x(p.x)},${y(p.y)}`).join(' '),
          stroke: s.color, class: 'plot-line'
        });
        if (s.dash) path.setAttribute('stroke-dasharray', s.dash);
        svg.append(path);
        s.points.forEach(p => svg.append(marker(s.marker, x(p.x), y(p.y), s.marker && s.marker !== 'circle' ? 2.8 : 2.6, s.color)));
      });
    }
    new ResizeObserver(render).observe(container); render();
  }
  function dumbbell(container, panel, caption) {
    const summary = panel.rows.map(r => `${r.label}: reused ${number(r.reused)}%, held out ${number(r.heldout)}%.`).join(' ');
    const svg = svgEl('svg', {class: 'native-plot', role: 'img', 'aria-describedby': caption, 'aria-label': summary});
    container.append(svg);
    function render() {
      const width = container.clientWidth;
      if (width < 100) return;
      const all = panel.rows.flatMap(r => [r.reused, r.heldout]);
      const low = Math.floor(Math.min(...all) - .5), high = Math.ceil(Math.max(...all) + .5);
      const left = width < 500 ? 132 : 177, right = width - 22;
      const x = v => left + (v - low) / (high - low) * (right - left), y = i => 28 + i * 52;
      svg.setAttribute('viewBox', `0 0 ${width} 240`); svg.replaceChildren();
      for (let v = low; v <= high; v += 2) svg.append(
        svgEl('line', {x1: x(v), x2: x(v), y1: 8, y2: 210, class: 'plot-grid'}),
        svgEl('text', {x: x(v), y: 234, 'text-anchor': 'middle'}, v)
      );
      panel.rows.forEach((r, i) => {
        svg.append(svgEl('text', {x: 0, y: y(i) + 4, class: 'model-label'}, r.label));
        svg.append(svgEl('line', {x1: x(r.reused), x2: x(r.heldout), y1: y(i), y2: y(i), stroke: '#626972', 'stroke-width': 1.5}));
        [['reused', '#9abaff', -10], ['heldout', '#d8b47a', 18]].forEach(([key, color, offset]) => {
          svg.append(svgEl('circle', {cx: x(r[key]), cy: y(i), r: 4, fill: color}));
          svg.append(svgEl('text', {x: x(r[key]), y: y(i) + offset, 'text-anchor': 'middle', class: 'score-value', style: `fill:${color}`}, number(r[key])));
        });
      });
    }
    new ResizeObserver(render).observe(container); render();
  }
  specifications.forEach(group => {
    const host = document.querySelector(`[data-chart-group="${group.id}"]`);
    if (!host) return;
    const first = group.panels[0];
    // Share the key once per figure; diagnostic thresholds are defined in captions.
    const items = first.type === 'dumbbell' ? [{name: 'Reused', color: '#9abaff', pointOnly: true}, {name: 'Held out', color: '#d8b47a', pointOnly: true}]
      : [...first.series, ...(group.id.startsWith('feedback') || group.id === 'false-winners' ? [] : first.references.map(r => ({...r, reference: true})))];
    if (items.length > 1) legend(host, items);
    const grid = el('div', `chart-grid${group.panels.length === 1 ? ' single' : group.panels.length === 6 ? ' six' : ''}`); host.append(grid);
    const sharedX = group.panels.every(p => p.xLabel === first.xLabel);
    group.panels.forEach(panel => {
      const container = el('div', 'chart-panel');
      let title = panel.title;
      if (group.id.startsWith('feedback')) title += ` (${panel.unit})`;
      if (group.panels.length > 1) container.append(el('h4', '', title));
      grid.append(container);
      if (panel.type === 'dumbbell') dumbbell(container, panel, group.caption);
      else lineChart(container, panel, group.caption);
      if (!sharedX) container.append(el('div', 'chart-x-label', `${panel.xLabel} (log scale)`));
      if (panel.footer) container.append(el('div', 'chart-footnote', panel.footer));
    });
    if (sharedX) host.append(el('div', 'chart-x-label', first.type === 'dumbbell' ? 'Mean category score (%)' : `${first.xLabel} (log scale)`));
  });
})();
