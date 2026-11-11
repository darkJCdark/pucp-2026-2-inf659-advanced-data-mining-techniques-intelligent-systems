// ============================================================
// OUTPUT COMPOSITION VIEW — composition.js
// Two focused plots:
//   1. Value vectors in 2D, sized by α weight
//   2. Chain vector sum → out_i
// α weights shown inline on the word pills
// ============================================================

let compQueryIdx = 0;
let currentView  = 'main'; // 'main' | 'composition'

const DARK_HOVER = { bgcolor: '#1e293b', bordercolor: '#475569', font: { size: 11, color: '#e2e8f0' } };

// ── View switching ────────────────────────────────────────────
function setView(v) {
    currentView = v;
    const isComp = v === 'composition';

    document.getElementById('view-main').classList.toggle('hidden', isComp);
    document.getElementById('view-composition').classList.toggle('hidden', !isComp);

    const btnMain = document.getElementById('view-btn-main');
    const btnComp = document.getElementById('view-btn-comp');
    if (btnMain) {
        btnMain.classList.toggle('bg-slate-200',   !isComp);
        btnMain.classList.toggle('text-slate-900', !isComp);
        btnMain.classList.toggle('text-slate-400',  isComp);
    }
    if (btnComp) {
        btnComp.classList.toggle('bg-rose-600',    isComp);
        btnComp.classList.toggle('text-white',     isComp);
        btnComp.classList.toggle('text-slate-400', !isComp);
    }

    if (isComp) {
        renderCompositionView();
        setTimeout(() => {
            ['comp-values','comp-chain'].forEach(id => Plotly.Plots.resize(id));
        }, 50);
        setTimeout(() => {
            ['comp-values','comp-chain'].forEach(id => Plotly.Plots.resize(id));
        }, 300);
    } else {
        renderAll();
        setTimeout(() => {
            ['plot-embeddings','plot-qkspace','plot-heatmap','plot-output']
                .forEach(id => Plotly.Plots.resize(id));
        }, 120);
    }
}

// ── Top-level render ──────────────────────────────────────────
function renderCompositionView() {
    if (currentView !== 'composition') return;
    const n = words.length;
    compQueryIdx = Math.min(compQueryIdx, n - 1);

    const result = computeAttention();
    if (!result) return;

    renderCompWordBtns(result);
    renderCompValues(result);
    renderCompChain(result);
}

// ── Word selector pills — with inline α values ────────────────
function renderCompWordBtns(result) {
    const el = document.getElementById('comp-word-pills');
    if (!el) return;
    el.innerHTML = '';

    const hint = document.createElement('span');
    hint.className = 'text-[10px] text-slate-500 self-center flex-shrink-0 mr-1 select-none';
    hint.textContent = 'Query word:';
    el.appendChild(hint);

    const alphas = result.attentionWeights[compQueryIdx];

    words.forEach((w, i) => {
        const c  = WORD_COLORS[i % WORD_COLORS.length];
        const on = i === compQueryIdx;
        const btn = document.createElement('button');
        btn.className = 'flex items-center gap-1 px-2.5 py-1 rounded-full text-[11px] font-bold transition-all border';
        btn.style.cssText = `border-color:${on ? c : '#334155'};background:${on ? c + '28' : 'transparent'};color:${c};`;
        btn.innerHTML = `<span>${w.word}</span>`;
        btn.onclick = () => { compQueryIdx = i; renderCompositionView(); };
        el.appendChild(btn);
    });

    // α legend: show how much each word contributes
    const divider = document.createElement('div');
    divider.className = 'h-5 w-px bg-slate-700 mx-2 self-center flex-shrink-0';
    el.appendChild(divider);

    const alphaLabel = document.createElement('span');
    alphaLabel.className = 'text-[10px] text-slate-500 self-center flex-shrink-0 mr-1 select-none';
    alphaLabel.textContent = 'α:';
    el.appendChild(alphaLabel);

    // Sort by α descending for the legend
    const order = [...Array(words.length).keys()].sort((a, b) => alphas[b] - alphas[a]);
    order.forEach(j => {
        const c   = WORD_COLORS[j % WORD_COLORS.length];
        const a   = alphas[j];
        const pill = document.createElement('div');
        pill.className = 'flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] border select-none';
        pill.style.cssText = `border-color:${c}44;background:${c}18;color:${c};`;
        // Mini bar + value
        pill.innerHTML = `
            <span style="font-weight:bold">${words[j].word}</span>
            <div style="width:${Math.round(a*40)}px;height:3px;background:${c};border-radius:2px;opacity:0.7;min-width:2px"></div>
            <span style="color:#94a3b8">${a.toFixed(3)}</span>
        `;
        el.appendChild(pill);
    });
}

// ── Plot 1: value vectors sized by α ─────────────────────────
function renderCompValues(result) {
    const { values, outputs, attentionWeights } = result;
    const n      = words.length;
    const qIdx   = compQueryIdx;
    const alphas = attentionWeights[qIdx];
    const out    = outputs[qIdx];
    const qWord  = words[qIdx].word;

    const traces = [];

    // Faint spokes from origin to each v_j
    for (let j = 0; j < n; j++) {
        const a = alphas[j];
        if (a < 0.003) continue;
        traces.push({
            x: [0, values[j][0]], y: [0, values[j][1]],
            mode: 'lines', type: 'scatter',
            line: { color: WORD_COLORS[j % WORD_COLORS.length], width: 1, dash: 'dot' },
            opacity: 0.2, showlegend: false, hoverinfo: 'skip',
        });
    }

    // v_j dots — sized by α
    for (let j = 0; j < n; j++) {
        const c    = WORD_COLORS[j % WORD_COLORS.length];
        const a    = alphas[j];
        const size = 12 + a * 32;
        const isQ  = j === qIdx;
        traces.push({
            x: [values[j][0]], y: [values[j][1]],
            mode: 'markers+text', type: 'scatter',
            text: [`v_${words[j].word}`],
            textposition: 'top center',
            textfont: { size: 10, color: c },
            marker: {
                size,
                color: c,
                opacity: 0.85,
                symbol: isQ ? 'diamond' : 'circle',
                line: { color: isQ ? '#fff' : '#0f172a', width: isQ ? 2 : 1 },
            },
            showlegend: false,
            hovertemplate:
                `<b>v_${words[j].word}</b><br>` +
                `(${values[j][0].toFixed(3)}, ${values[j][1].toFixed(3)})<br>` +
                `α = <b>${a.toFixed(3)}</b><extra></extra>`,
        });
    }

    // Line from origin to output
    traces.push({
        x: [0, out[0]], y: [0, out[1]],
        mode: 'lines', type: 'scatter',
        line: { color: '#ef4444', width: 1.5, dash: 'dot' },
        opacity: 0.4, showlegend: false, hoverinfo: 'skip',
    });

    // Output star
    traces.push({
        x: [out[0]], y: [out[1]],
        mode: 'markers+text', type: 'scatter',
        text: [`out_${qWord}`],
        textposition: 'top center',
        textfont: { size: 11, color: '#ef4444' },
        marker: { size: 18, color: '#ef4444', symbol: 'star',
                  line: { color: '#fff', width: 1.5 } },
        showlegend: false,
        hovertemplate: `<b>out_${qWord}</b><br>(${out[0].toFixed(3)}, ${out[1].toFixed(3)})<extra></extra>`,
    });

    // Tight axis range — pad 25% around actual data
    const allPts = [...values, out, [0, 0]];
    const xs = allPts.map(p => p[0]), ys = allPts.map(p => p[1]);
    const xmn = Math.min(...xs), xmx = Math.max(...xs);
    const ymn = Math.min(...ys), ymx = Math.max(...ys);
    const xpad = Math.max((xmx - xmn) * 0.3, 0.4);
    const ypad = Math.max((ymx - ymn) * 0.3, 0.4);

    Plotly.react('comp-values', traces, {
        paper_bgcolor: '#0f172a', plot_bgcolor: '#0f172a',
        font:  { color: '#94a3b8', size: 10 },
        xaxis: { range: [xmn - xpad, xmx + xpad],
                 zeroline: true, zerolinecolor: '#334155', zerolinewidth: 1,
                 gridcolor: '#1e293b', tickfont: { size: 8 } },
        yaxis: { range: [ymn - ypad, ymx + ypad],
                 zeroline: true, zerolinecolor: '#334155', zerolinewidth: 1,
                 gridcolor: '#1e293b', tickfont: { size: 8 },
                 scaleanchor: 'x', scaleratio: 1 },
        margin: { l: 38, r: 16, t: 10, b: 36 },
        showlegend: false,
        hoverlabel: DARK_HOVER,
    }, { responsive: true, displayModeBar: false });
}

// ── Plot 2: chain vector sum ──────────────────────────────────
function renderCompChain(result) {
    const { values, outputs, attentionWeights } = result;
    const n      = words.length;
    const qIdx   = compQueryIdx;
    const alphas = attentionWeights[qIdx];
    const out    = outputs[qIdx];
    const qWord  = words[qIdx].word;

    // Sort largest α first
    const order = [...Array(n).keys()].sort((a, b) => alphas[b] - alphas[a]);

    // Build cumulative chain
    const chain = [[0, 0]];
    for (const j of order) {
        const prev = chain[chain.length - 1];
        chain.push([
            prev[0] + alphas[j] * values[j][0],
            prev[1] + alphas[j] * values[j][1],
        ]);
    }

    const traces = [];

    // Faint drop lines from each chain tip to zero axes
    for (let s = 1; s < chain.length; s++) {
        const [cx, cy] = chain[s];
        traces.push({
            x: [cx, cx], y: [0, cy],
            mode: 'lines', type: 'scatter',
            line: { color: '#1e3a5f', width: 1, dash: 'dot' },
            showlegend: false, hoverinfo: 'skip',
        });
    }

    // Chain arrow segments
    for (let s = 0; s < order.length; s++) {
        const j   = order[s];
        const c   = WORD_COLORS[j % WORD_COLORS.length];
        const a   = alphas[j];
        if (a < 0.002) continue;

        const [x0, y0] = chain[s];
        const [x1, y1] = chain[s + 1];
        const dx = x1 - x0, dy = y1 - y0;
        const len = Math.sqrt(dx*dx + dy*dy);

        // Arrow shaft
        traces.push({
            x: [x0, x1], y: [y0, y1],
            mode: 'lines', type: 'scatter',
            line: { color: c, width: Math.max(2, a * 9) },
            showlegend: false, hoverinfo: 'skip',
        });

        // Arrowhead dot at tip
        traces.push({
            x: [x1], y: [y1],
            mode: 'markers', type: 'scatter',
            marker: { size: 7, color: c, symbol: 'circle' },
            showlegend: false,
            hovertemplate:
                `<b>+${a.toFixed(3)} · v_${words[j].word}</b><br>` +
                `step: (${dx.toFixed(3)}, ${dy.toFixed(3)})<br>` +
                `total so far: (${x1.toFixed(3)}, ${y1.toFixed(3)})<extra></extra>`,
        });

        // Label only if segment long enough
        if (len > 0.05) {
            // Perpendicular offset so label doesn't sit on the line
            const nx = -dy / len * 0.06, ny = dx / len * 0.06;
            traces.push({
                x: [(x0 + x1) / 2 + nx], y: [(y0 + y1) / 2 + ny],
                mode: 'text', type: 'scatter',
                text: [`${a.toFixed(2)}·v_${words[j].word}`],
                textfont: { size: 9, color: c },
                showlegend: false, hoverinfo: 'skip',
            });
        }
    }

    // Origin marker
    traces.push({
        x: [0], y: [0],
        mode: 'markers+text', type: 'scatter',
        text: ['  0,0'],
        textposition: 'middle right',
        textfont: { size: 9, color: '#475569' },
        marker: { size: 7, color: '#475569', symbol: 'cross-thin',
                  line: { color: '#475569', width: 2 } },
        showlegend: false, hoverinfo: 'skip',
    });

    // Output star
    traces.push({
        x: [out[0]], y: [out[1]],
        mode: 'markers+text', type: 'scatter',
        text: [`  out_${qWord}  ★`],
        textposition: 'middle right',
        textfont: { size: 12, color: '#ef4444' },
        marker: { size: 20, color: '#ef4444', symbol: 'star',
                  line: { color: '#fff', width: 1.5 } },
        showlegend: false,
        hovertemplate: `<b>out_${qWord}</b><br>(${out[0].toFixed(3)}, ${out[1].toFixed(3)})<extra></extra>`,
    });

    // Tight axis — pad 30% around chain + origin
    const allPts = [...chain, [0,0]];
    const xs = allPts.map(p => p[0]), ys = allPts.map(p => p[1]);
    const xmn = Math.min(...xs), xmx = Math.max(...xs);
    const ymn = Math.min(...ys), ymx = Math.max(...ys);
    const xpad = Math.max((xmx - xmn) * 0.32, 0.15);
    const ypad = Math.max((ymx - ymn) * 0.32, 0.15);

    Plotly.react('comp-chain', traces, {
        paper_bgcolor: '#0f172a', plot_bgcolor: '#0f172a',
        font:  { color: '#94a3b8', size: 10 },
        xaxis: { range: [xmn - xpad, xmx + xpad],
                 zeroline: true, zerolinecolor: '#334155', zerolinewidth: 1,
                 gridcolor: '#1e293b', tickfont: { size: 8 } },
        yaxis: { range: [ymn - ypad, ymx + ypad],
                 zeroline: true, zerolinecolor: '#334155', zerolinewidth: 1,
                 gridcolor: '#1e293b', tickfont: { size: 8 },
                 scaleanchor: 'x', scaleratio: 1 },
        margin: { l: 38, r: 90, t: 10, b: 36 },
        showlegend: false,
        hoverlabel: DARK_HOVER,
    }, { responsive: true, displayModeBar: false });
}
