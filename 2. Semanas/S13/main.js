// ============================================================
// TRANSFORMER SELF / CROSS-ATTENTION VISUALIZER — main.js
// ============================================================

// ════════════════════════════════════════════════════════════
// 1. MATH UTILITIES
// ════════════════════════════════════════════════════════════

function matVec(M, v) {
    return [
        M[0][0] * v[0] + M[0][1] * v[1],
        M[1][0] * v[0] + M[1][1] * v[1],
    ];
}

function dot(a, b) { return a[0] * b[0] + a[1] * b[1]; }
function norm(v)   { return Math.sqrt(v[0] * v[0] + v[1] * v[1]); }

function cosineSim(a, b) {
    const na = norm(a), nb = norm(b);
    return (na < 1e-10 || nb < 1e-10) ? 0 : dot(a, b) / (na * nb);
}

function softmax(arr, temp) {
    const t  = Math.max(temp, 0.01);
    const s  = arr.map(x => x / t);
    const mx = Math.max(...s);
    const ex = s.map(x => Math.exp(x - mx));
    const sm = ex.reduce((a, b) => a + b, 0);
    return ex.map(x => x / sm);
}

// ════════════════════════════════════════════════════════════
// 2. PAIR-FOCUS COLOR HELPERS
// ════════════════════════════════════════════════════════════

const DIM = 0.07;  // opacity for non-focused elements

// Convert a #rrggbb color to rgba(r,g,b,a)
function fadeColor(hex, alpha) {
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
}

// Apply focus: indices in brightSet stay original color, others get dimmed rgba
function applyFocus(baseColors, brightSet) {
    if (focusPairQ < 0) return baseColors;
    return baseColors.map((c, i) => brightSet.has(i) ? c : fadeColor(c, DIM));
}

function hasPairFocus() { return focusPairQ >= 0 && focusPairK >= 0; }

// Line opacity for an arrow connecting word i
function arrowOpacity(i, focusedIdx, defaultOp) {
    if (!hasPairFocus()) return defaultOp;
    return i === focusedIdx ? 0.85 : 0.0;  // hide non-focused arrows entirely
}

// ════════════════════════════════════════════════════════════
// 3. CORE ATTENTION COMPUTATION
// ════════════════════════════════════════════════════════════

function computeAttention() {
    if (attentionMode === 'cross') return computeCrossAttention();

    const n = words.length;
    if (n === 0) return null;

    const embeddings = words.map(w => [w.x, w.y]);
    const queries    = embeddings.map(e => matVec(Q, e));
    const keys       = embeddings.map(e => matVec(K, e));
    const values     = embeddings.map(e => matVec(V, e));

    const rawScores = [], cosineMatrix = [];
    for (let i = 0; i < n; i++) {
        rawScores.push([]);
        cosineMatrix.push([]);
        for (let j = 0; j < n; j++) {
            rawScores[i].push(dot(queries[i], keys[j]) / Math.sqrt(2));
            cosineMatrix[i].push(cosineSim(queries[i], keys[j]));
        }
    }

    const attentionWeights = rawScores.map(row => softmax(row, temperature));
    const outputs = embeddings.map((_, i) => {
        const out = [0, 0];
        for (let j = 0; j < n; j++) {
            out[0] += attentionWeights[i][j] * values[j][0];
            out[1] += attentionWeights[i][j] * values[j][1];
        }
        return out;
    });

    return {
        mode: 'self',
        embeddings, queries, keys, values,
        rawScores, cosineMatrix, attentionWeights, outputs,
        rowLabels: words.map(w => w.word),
        colLabels: words.map(w => w.word),
    };
}

function computeCrossAttention() {
    const nA = words.length, nB = wordsB.length;
    if (!nA || !nB) return null;

    const embA    = words.map(w => [w.x, w.y]);
    const embB    = wordsB.map(w => [w.x, w.y]);
    const queries = embA.map(e => matVec(Q, e));
    const keys    = embB.map(e => matVec(K, e));
    const values  = embB.map(e => matVec(V, e));

    const rawScores = [], cosineMatrix = [];
    for (let i = 0; i < nA; i++) {
        rawScores.push([]);
        cosineMatrix.push([]);
        for (let j = 0; j < nB; j++) {
            rawScores[i].push(dot(queries[i], keys[j]) / Math.sqrt(2));
            cosineMatrix[i].push(cosineSim(queries[i], keys[j]));
        }
    }

    const attentionWeights = rawScores.map(row => softmax(row, temperature));
    const outputs = embA.map((_, i) => {
        const out = [0, 0];
        for (let j = 0; j < nB; j++) {
            out[0] += attentionWeights[i][j] * values[j][0];
            out[1] += attentionWeights[i][j] * values[j][1];
        }
        return out;
    });

    return {
        mode: 'cross',
        embA, embB, queries, keys, values,
        rawScores, cosineMatrix, attentionWeights, outputs,
        rowLabels: words.map(w => w.word),
        colLabels: wordsB.map(w => w.word),
    };
}

// ════════════════════════════════════════════════════════════
// 4. RENDERING
// ════════════════════════════════════════════════════════════

function renderAll() {
    const result = computeAttention();
    if (!result) return;
    if (attentionMode === 'cross') {
        renderEmbeddingSpaceCross(result);
        renderQKSpaceCross(result);
    } else {
        renderEmbeddingSpace(result);
        renderQKSpace(result);
    }
    renderHeatmap(result);
    attentionMode === 'cross' ? renderOutputSpaceCross(result) : renderOutputSpace(result);
    updateAttentionBars(result);
    if (typeof renderCompositionView === 'function') renderCompositionView();
}

function baseLayout(title, axisRange) {
    const rng = axisRange || [-2.6, 2.6];
    return {
        paper_bgcolor: '#0f172a', plot_bgcolor: '#0f172a',
        font:  { color: '#94a3b8', size: 10 },
        title: { text: title, font: { size: 11, color: '#475569' }, x: 0.5 },
        xaxis: {
            gridcolor: '#1e293b', zeroline: true, zerolinecolor: '#334155',
            title: { text: 'dim 1', font: { size: 10 } }, range: rng,
        },
        yaxis: {
            gridcolor: '#1e293b', zeroline: true, zerolinecolor: '#334155',
            title: { text: 'dim 2', font: { size: 10 } }, range: rng,
            scaleanchor: 'x', scaleratio: 1,
        },
        legend: {
            bgcolor: 'rgba(15,23,42,0.0)', borderwidth: 0,
            font: { size: 9, color: '#94a3b8' },
            orientation: 'h',
            x: 0.5, xanchor: 'center',
            y: -0.06, yanchor: 'top',
        },
        margin: { l: 45, r: 12, t: 30, b: 62 },
        hovermode: 'closest',
    };
}

// ─────────────────────────────────────────────────────────────
// PLOT 1a — Self-Attention: Raw Word Embeddings only
// Purpose: show where words live in 2D space before any transformation
// ─────────────────────────────────────────────────────────────
function renderEmbeddingSpace(result) {
    const { embeddings } = result;
    const n          = words.length;
    const labels     = words.map(w => w.word);
    const baseColors = words.map((_, i) => WORD_COLORS[i % WORD_COLORS.length]);
    const colors     = applyFocus(baseColors, new Set([focusPairQ, focusPairK]));
    const sizes      = words.map((_, i) =>
        (i === focusPairQ || i === focusPairK || i === selectedIdx) ? 15 : 11);

    const traces = [{
        x: embeddings.map(e => e[0]),
        y: embeddings.map(e => e[1]),
        mode: 'markers+text', type: 'scatter',
        name: 'e_i',
        text: labels, textposition: 'top center',
        textfont: { size: 11, color: colors },
        marker: { size: sizes, color: colors, line: { color: '#0f172a', width: 2 } },
        hovertemplate: labels.map((l, i) =>
            `<b>${l}</b><br>e = (${embeddings[i][0].toFixed(3)}, ${embeddings[i][1].toFixed(3)})<extra></extra>`),
        showlegend: false,
    }];

    // Pair focus: draw a faint connector between the two focused words
    if (hasPairFocus() && focusPairQ < n && focusPairK < n) {
        traces.push({
            x: [embeddings[focusPairQ][0], embeddings[focusPairK][0]],
            y: [embeddings[focusPairQ][1], embeddings[focusPairK][1]],
            mode: 'lines', type: 'scatter',
            line: { color: '#c084fc', width: 1.5, dash: 'dot' },
            showlegend: false, hoverinfo: 'none', opacity: 0.5,
        });
    }

    const mx = Math.max(2.2, ...embeddings.flatMap(p => [Math.abs(p[0]), Math.abs(p[1])])) * 1.35;
    Plotly.react('plot-embeddings', traces,
        { ...baseLayout('① Word Embeddings  e_i', [-mx, mx]) },
        { responsive: true, displayModeBar: false });
}

// ─────────────────────────────────────────────────────────────
// PLOT 1b — Cross-Attention: Raw Embeddings of Seq A and Seq B
// Purpose: show where both sequences live before any projection
// ─────────────────────────────────────────────────────────────
function renderEmbeddingSpaceCross(result) {
    const { embA, embB } = result;
    const labA     = words.map(w => w.word);
    const labB     = wordsB.map(w => w.word);
    const baseColA = words.map((_, i) => WORD_COLORS[i % WORD_COLORS.length]);
    const baseColB = wordsB.map((_, i) => WORD_COLORS_B[i % WORD_COLORS_B.length]);
    const colA     = applyFocus(baseColA, new Set([focusPairQ]));
    const colB     = applyFocus(baseColB, new Set([focusPairK]));
    const traces   = [];

    // Seq B — diamonds (context / key+value source)
    traces.push({
        x: embB.map(e => e[0]), y: embB.map(e => e[1]),
        mode: 'markers+text', type: 'scatter', name: 'Seq B  b_j',
        text: labB, textposition: 'bottom center',
        textfont: { size: 11, color: colB },
        marker: {
            size: wordsB.map((_, i) => (selectedSeq === 'B' && i === selectedIdx) || i === focusPairK ? 14 : 11),
            color: colB, symbol: 'diamond', line: { color: '#f87171', width: 1 },
        },
        hovertemplate: labB.map((l, i) =>
            `<b>${l} [B]</b><br>b = (${embB[i][0].toFixed(3)}, ${embB[i][1].toFixed(3)})<extra></extra>`),
    });

    // Seq A — circles (query source)
    traces.push({
        x: embA.map(e => e[0]), y: embA.map(e => e[1]),
        mode: 'markers+text', type: 'scatter', name: 'Seq A  a_i',
        text: labA, textposition: 'top center',
        textfont: { size: 11, color: colA },
        marker: {
            size: words.map((_, i) => (selectedSeq === 'A' && i === selectedIdx) || i === focusPairQ ? 14 : 11),
            color: colA, line: { color: '#0f172a', width: 2 },
        },
        hovertemplate: labA.map((l, i) =>
            `<b>${l} [A]</b><br>a = (${embA[i][0].toFixed(3)}, ${embA[i][1].toFixed(3)})<extra></extra>`),
    });

    // Pair focus: connector between the two focused words in raw space
    if (hasPairFocus() && focusPairQ < words.length && focusPairK < wordsB.length) {
        traces.push({
            x: [embA[focusPairQ][0], embB[focusPairK][0]],
            y: [embA[focusPairQ][1], embB[focusPairK][1]],
            mode: 'lines', type: 'scatter',
            line: { color: '#c084fc', width: 1.5, dash: 'dot' },
            showlegend: false, hoverinfo: 'none', opacity: 0.5,
        });
    }

    const all = [...embA, ...embB].flatMap(p => [Math.abs(p[0]), Math.abs(p[1])]);
    const mx  = Math.max(2.2, ...all) * 1.35;
    Plotly.react('plot-embeddings', traces,
        { ...baseLayout('① Word Embeddings  [● Seq A   ◆ Seq B]', [-mx, mx]) },
        { responsive: true, displayModeBar: false });
}

// ─────────────────────────────────────────────────────────────
// PLOT 2a — Self-Attention: Q & K Projections (the transformation)
// Purpose: show how Q and K matrices move each word from embedding space
//          into query-space and key-space. Arrows make the projection visible.
// ─────────────────────────────────────────────────────────────
function renderQKSpace(result) {
    const { embeddings, queries, keys } = result;
    const n          = words.length;
    const labels     = words.map(w => w.word);
    const baseColors = words.map((_, i) => WORD_COLORS[i % WORD_COLORS.length]);
    const traces     = [];

    // Which words to show at full brightness
    const showAll = !hasPairFocus();
    const brightQ = new Set(showAll ? [...Array(n).keys()] : [focusPairQ]);
    const brightK = new Set(showAll ? [...Array(n).keys()] : [focusPairK]);

    // ── Arrows: e_i → q_i  (dashed amber) ──────────────────
    for (let i = 0; i < n; i++) {
        if (!brightQ.has(i)) continue;
        traces.push({
            x: [embeddings[i][0], queries[i][0]],
            y: [embeddings[i][1], queries[i][1]],
            mode: 'lines', type: 'scatter',
            line: { color: baseColors[i], width: 1.5, dash: 'dash' },
            showlegend: false, hoverinfo: 'none',
            opacity: hasPairFocus() ? 0.85 : 0.45,
        });
    }

    // ── Arrows: e_i → k_i  (dotted green) ──────────────────
    for (let i = 0; i < n; i++) {
        if (!brightK.has(i)) continue;
        traces.push({
            x: [embeddings[i][0], keys[i][0]],
            y: [embeddings[i][1], keys[i][1]],
            mode: 'lines', type: 'scatter',
            line: { color: baseColors[i], width: 1.5, dash: 'dot' },
            showlegend: false, hoverinfo: 'none',
            opacity: hasPairFocus() ? 0.85 : 0.38,
        });
    }

    // ── When pair focused: purple line q_pQ → k_pK ──────────
    if (hasPairFocus()) {
        traces.push({
            x: [queries[focusPairQ][0], keys[focusPairK][0]],
            y: [queries[focusPairQ][1], keys[focusPairK][1]],
            mode: 'lines', type: 'scatter',
            line: { color: '#c084fc', width: 2.5 },
            showlegend: false, hoverinfo: 'none', opacity: 0.85,
        });
        const mid = [(queries[focusPairQ][0] + keys[focusPairK][0]) / 2,
                     (queries[focusPairQ][1] + keys[focusPairK][1]) / 2];
        const alpha  = result.attentionWeights[focusPairQ][focusPairK];
        traces.push({
            x: [mid[0]], y: [mid[1]],
            mode: 'markers+text', type: 'scatter',
            text: [`α = ${alpha.toFixed(3)}`], textposition: 'top center',
            textfont: { size: 10, color: '#c084fc' },
            marker: { size: 0, color: 'transparent' },
            showlegend: false, hoverinfo: 'none',
        });
    }

    // ── Original embeddings (faded, always shown) ───────────
    const embColors = applyFocus(baseColors, new Set([focusPairQ, focusPairK]));
    traces.push({
        x: embeddings.map(e => e[0]), y: embeddings.map(e => e[1]),
        mode: 'markers', type: 'scatter', name: 'embedding  e_i',
        marker: {
            size: words.map((_, i) => (brightQ.has(i) || brightK.has(i)) ? 8 : 5),
            color: embColors, opacity: 0.45,
            line: { color: '#334155', width: 1 },
        },
        hovertemplate: labels.map((l, i) =>
            `<b>${l}</b> (original)<br>${embeddings[i][0].toFixed(3)}, ${embeddings[i][1].toFixed(3)}<extra></extra>`),
    });

    // ── Query endpoints  ●  (amber) ─────────────────────────
    const qColors = applyFocus(baseColors, brightQ);
    traces.push({
        x: queries.map(q => q[0]), y: queries.map(q => q[1]),
        mode: 'markers+text', type: 'scatter', name: 'q_i = Q·e_i',
        text: labels.map(l => 'q_' + l),
        textposition: 'top right',
        textfont: { size: 9, color: applyFocus(baseColors.map(() => '#f59e0b'), brightQ) },
        marker: {
            size: words.map((_, i) => i === focusPairQ ? 13 : 9),
            color: qColors, symbol: 'circle',
            line: { color: applyFocus(baseColors.map(() => '#f59e0b'), brightQ), width: 2 },
        },
        hovertemplate: labels.map((l, i) =>
            `<b>q_${l}</b> = Q·e_${l}<br>${queries[i][0].toFixed(3)}, ${queries[i][1].toFixed(3)}<extra></extra>`),
    });

    // ── Key endpoints  ■  (green) ───────────────────────────
    const kColors = applyFocus(baseColors, brightK);
    traces.push({
        x: keys.map(k => k[0]), y: keys.map(k => k[1]),
        mode: 'markers+text', type: 'scatter', name: 'k_i = K·e_i',
        text: labels.map(l => 'k_' + l),
        textposition: 'bottom left',
        textfont: { size: 9, color: applyFocus(baseColors.map(() => '#10b981'), brightK) },
        marker: {
            size: words.map((_, i) => i === focusPairK ? 13 : 9),
            color: kColors, symbol: 'square',
            line: { color: applyFocus(baseColors.map(() => '#10b981'), brightK), width: 2 },
        },
        hovertemplate: labels.map((l, i) =>
            `<b>k_${l}</b> = K·e_${l}<br>${keys[i][0].toFixed(3)}, ${keys[i][1].toFixed(3)}<extra></extra>`),
    });

    const all = [...embeddings, ...queries, ...keys].flatMap(p => [Math.abs(p[0]), Math.abs(p[1])]);
    const mx  = Math.max(2.2, ...all) * 1.28;

    Plotly.react('plot-qkspace', traces, {
        ...baseLayout('② Q & K Projections  (e_i → q_i, k_i)', [-mx, mx]),
    }, { responsive: true, displayModeBar: false });
}

// ─────────────────────────────────────────────────────────────
// PLOT 2b — Cross-Attention: Q & K Projections
// Queries come from Seq A (Q·a_i), Keys come from Seq B (K·b_j)
// ─────────────────────────────────────────────────────────────
function renderQKSpaceCross(result) {
    const { embA, embB, queries, keys } = result;
    const nA       = words.length, nB = wordsB.length;
    const labA     = words.map(w => w.word);
    const labB     = wordsB.map(w => w.word);
    const baseColA = words.map((_, i) => WORD_COLORS[i % WORD_COLORS.length]);
    const baseColB = wordsB.map((_, i) => WORD_COLORS_B[i % WORD_COLORS_B.length]);
    const showAll  = !hasPairFocus();
    const brightQ  = new Set(showAll ? [...Array(nA).keys()] : [focusPairQ]);
    const brightK  = new Set(showAll ? [...Array(nB).keys()] : [focusPairK]);
    const traces   = [];

    // ── Arrows: a_i → q_i  (dashed amber, from A) ──────────
    for (let i = 0; i < nA; i++) {
        if (!brightQ.has(i)) continue;
        traces.push({
            x: [embA[i][0], queries[i][0]], y: [embA[i][1], queries[i][1]],
            mode: 'lines', type: 'scatter',
            line: { color: baseColA[i], width: 1.5, dash: 'dash' },
            showlegend: false, hoverinfo: 'none',
            opacity: hasPairFocus() ? 0.85 : 0.45,
        });
    }

    // ── Arrows: b_j → k_j  (dotted green, from B) ──────────
    for (let j = 0; j < nB; j++) {
        if (!brightK.has(j)) continue;
        traces.push({
            x: [embB[j][0], keys[j][0]], y: [embB[j][1], keys[j][1]],
            mode: 'lines', type: 'scatter',
            line: { color: baseColB[j], width: 1.5, dash: 'dot' },
            showlegend: false, hoverinfo: 'none',
            opacity: hasPairFocus() ? 0.85 : 0.38,
        });
    }

    // ── Pair-focused: purple line q_pQ → k_pK ───────────────
    if (hasPairFocus()) {
        traces.push({
            x: [queries[focusPairQ][0], keys[focusPairK][0]],
            y: [queries[focusPairQ][1], keys[focusPairK][1]],
            mode: 'lines', type: 'scatter',
            line: { color: '#c084fc', width: 2.5 },
            showlegend: false, hoverinfo: 'none', opacity: 0.85,
        });
        const mid   = [(queries[focusPairQ][0] + keys[focusPairK][0]) / 2,
                       (queries[focusPairQ][1] + keys[focusPairK][1]) / 2];
        const alpha = result.attentionWeights[focusPairQ][focusPairK];
        traces.push({
            x: [mid[0]], y: [mid[1]],
            mode: 'markers+text', type: 'scatter',
            text: [`α = ${alpha.toFixed(3)}`], textposition: 'top center',
            textfont: { size: 10, color: '#c084fc' },
            marker: { size: 0, color: 'transparent' },
            showlegend: false, hoverinfo: 'none',
        });
    }

    // ── Original A embeddings (faded) ───────────────────────
    const embColA = applyFocus(baseColA, brightQ);
    traces.push({
        x: embA.map(e => e[0]), y: embA.map(e => e[1]),
        mode: 'markers', type: 'scatter', name: 'a_i',
        marker: { size: 6, color: embColA, opacity: 0.35, line: { color: '#334155', width: 1 } },
        hovertemplate: labA.map((l, i) =>
            `<b>${l} [A]</b> original<br>${embA[i][0].toFixed(3)}, ${embA[i][1].toFixed(3)}<extra></extra>`),
    });

    // ── Original B embeddings (faded) ───────────────────────
    const embColB = applyFocus(baseColB, brightK);
    traces.push({
        x: embB.map(e => e[0]), y: embB.map(e => e[1]),
        mode: 'markers', type: 'scatter', name: 'b_j',
        marker: { size: 6, color: embColB, opacity: 0.35, symbol: 'diamond', line: { color: '#334155', width: 1 } },
        hovertemplate: labB.map((l, i) =>
            `<b>${l} [B]</b> original<br>${embB[i][0].toFixed(3)}, ${embB[i][1].toFixed(3)}<extra></extra>`),
    });

    // ── Query endpoints (A → amber circles) ─────────────────
    const qColors = applyFocus(baseColA, brightQ);
    traces.push({
        x: queries.map(q => q[0]), y: queries.map(q => q[1]),
        mode: 'markers+text', type: 'scatter', name: 'q_i = Q·a_i',
        text: labA.map(l => 'q_' + l), textposition: 'top right',
        textfont: { size: 9, color: applyFocus(baseColA.map(() => '#f59e0b'), brightQ) },
        marker: {
            size: words.map((_, i) => i === focusPairQ ? 13 : 9),
            color: qColors, symbol: 'circle',
            line: { color: applyFocus(baseColA.map(() => '#f59e0b'), brightQ), width: 2 },
        },
        hovertemplate: labA.map((l, i) =>
            `<b>q_${l}</b> = Q·a_${l}<br>${queries[i][0].toFixed(3)}, ${queries[i][1].toFixed(3)}<extra></extra>`),
    });

    // ── Key endpoints (B → green squares) ───────────────────
    const kColors = applyFocus(baseColB, brightK);
    traces.push({
        x: keys.map(k => k[0]), y: keys.map(k => k[1]),
        mode: 'markers+text', type: 'scatter', name: 'k_j = K·b_j',
        text: labB.map(l => 'k_' + l), textposition: 'bottom left',
        textfont: { size: 9, color: applyFocus(baseColB.map(() => '#10b981'), brightK) },
        marker: {
            size: wordsB.map((_, i) => i === focusPairK ? 13 : 9),
            color: kColors, symbol: 'square',
            line: { color: applyFocus(baseColB.map(() => '#10b981'), brightK), width: 2 },
        },
        hovertemplate: labB.map((l, i) =>
            `<b>k_${l}</b> = K·b_${l}<br>${keys[i][0].toFixed(3)}, ${keys[i][1].toFixed(3)}<extra></extra>`),
    });

    const all = [...embA, ...embB, ...queries, ...keys].flatMap(p => [Math.abs(p[0]), Math.abs(p[1])]);
    const mx  = Math.max(2.2, ...all) * 1.28;

    Plotly.react('plot-qkspace', traces, {
        ...baseLayout('② Q & K Projections  (a_i → q_i,  b_j → k_j)', [-mx, mx]),
    }, { responsive: true, displayModeBar: false });
}

// ─────────────────────────────────────────────────────────────
// PLOT 3 — Attention Heatmap
// ─────────────────────────────────────────────────────────────
function renderHeatmap(result) {
    const { rawScores, cosineMatrix, attentionWeights, rowLabels, colLabels } = result;

    let matrix, titleText, colorscale, zmin, zmax, cbTitle;

    if (heatmapMode === 'rawscore') {
        matrix = rawScores;
        titleText = attentionMode === 'cross' ? '③ Cross Raw Scores  ( q_A·k_B / √2 )' : '③ Raw Scores  ( q·k / √2 )';
        colorscale = [[0,'#1a0533'],[0.5,'#581c87'],[1,'#ec4899']];
        zmin = null; zmax = null; cbTitle = 'score';
    } else if (heatmapMode === 'cosine') {
        matrix = cosineMatrix;
        titleText = attentionMode === 'cross' ? '④ Cross Cosine Similarity' : '④ Cosine Similarity';
        colorscale = [[0,'#0c1a4a'],[0.5,'#1d4ed8'],[1,'#7c3aed']];
        zmin = -1; zmax = 1; cbTitle = 'cos';
    } else {
        matrix = attentionWeights;
        titleText = attentionMode === 'cross' ? '⑤ Cross-Attention Weights  α_ij' : '⑤ Attention Weights  α_ij';
        colorscale = [[0,'#0f172a'],[0.4,'#1e3a5f'],[1,'#06b6d4']];
        zmin = 0; zmax = 1; cbTitle = 'α';
    }

    const nR = rowLabels.length, nC = colLabels.length;
    const showText = nR <= 6 && nC <= 6;

    const traces = [{
        z: matrix,
        x: colLabels.map(l => 'k_' + l),
        y: rowLabels.map(l => 'q_' + l),
        type: 'heatmap', colorscale, zmin, zmax,
        colorbar: {
            title: { text: cbTitle, font: { size: 10, color: '#94a3b8' } },
            thickness: 12, len: 0.8,
            tickfont: { size: 9, color: '#94a3b8' }, tickformat: '.2f',
        },
        hovertemplate: '<b>%{y}</b> → <b>%{x}</b><br>value: %{z:.3f}<extra></extra>',
        ...(showText ? {
            text: matrix.map(row => row.map(v => v.toFixed(2))),
            texttemplate: '%{text}', textfont: { size: 9, color: '#e2e8f0' },
        } : {}),
    }];

    const shapes = [];

    // Single-word selection highlights (row/col)
    if (!hasPairFocus()) {
        if (attentionMode === 'self' && selectedIdx >= 0 && selectedIdx < nR) {
            shapes.push({
                type: 'rect', xref: 'paper', yref: 'y',
                x0: 0, x1: 1, y0: selectedIdx - 0.5, y1: selectedIdx + 0.5,
                fillcolor: 'rgba(245,158,11,0.08)', line: { color: '#f59e0b', width: 1.5 },
            });
            shapes.push({
                type: 'rect', xref: 'x', yref: 'paper',
                x0: selectedIdx - 0.5, x1: selectedIdx + 0.5, y0: 0, y1: 1,
                fillcolor: 'rgba(16,185,129,0.08)', line: { color: '#10b981', width: 1.5 },
            });
        }
        if (attentionMode === 'cross') {
            if (selectedSeq === 'A' && selectedIdx >= 0 && selectedIdx < nR) {
                shapes.push({
                    type: 'rect', xref: 'paper', yref: 'y',
                    x0: 0, x1: 1, y0: selectedIdx - 0.5, y1: selectedIdx + 0.5,
                    fillcolor: 'rgba(245,158,11,0.08)', line: { color: '#f59e0b', width: 1.5 },
                });
            }
            if (selectedSeq === 'B' && selectedIdx >= 0 && selectedIdx < nC) {
                shapes.push({
                    type: 'rect', xref: 'x', yref: 'paper',
                    x0: selectedIdx - 0.5, x1: selectedIdx + 0.5, y0: 0, y1: 1,
                    fillcolor: 'rgba(248,113,113,0.08)', line: { color: '#f87171', width: 1.5 },
                });
            }
        }
    }

    // Pair-focus: bright box around the focused cell
    if (hasPairFocus() && focusPairQ < nR && focusPairK < nC) {
        shapes.push({
            type: 'rect', xref: 'x', yref: 'y',
            x0: focusPairK - 0.5, x1: focusPairK + 0.5,
            y0: focusPairQ - 0.5, y1: focusPairQ + 0.5,
            fillcolor: 'rgba(192,132,252,0.12)',
            line: { color: '#c084fc', width: 2.5 },
        });
        // Faint row and col guides
        shapes.push({
            type: 'rect', xref: 'paper', yref: 'y',
            x0: 0, x1: 1, y0: focusPairQ - 0.5, y1: focusPairQ + 0.5,
            fillcolor: 'rgba(245,158,11,0.04)', line: { color: '#f59e0b', width: 1, dash: 'dot' },
        });
        shapes.push({
            type: 'rect', xref: 'x', yref: 'paper',
            x0: focusPairK - 0.5, x1: focusPairK + 0.5, y0: 0, y1: 1,
            fillcolor: 'rgba(16,185,129,0.04)', line: { color: '#10b981', width: 1, dash: 'dot' },
        });
    }

    const xTitle = attentionMode === 'cross' ? 'Seq B word  (key / context)' : 'Key word  (click cell to focus pair)';
    const yTitle = attentionMode === 'cross' ? 'Seq A word  (query)' : 'Query word';

    Plotly.react('plot-heatmap', traces, {
        paper_bgcolor: '#0f172a', plot_bgcolor: '#0f172a',
        font:   { color: '#94a3b8', size: 10 },
        title:  { text: titleText, font: { size: 11, color: '#475569' }, x: 0.5 },
        xaxis:  { tickfont: { size: 9 }, tickangle: -30, gridcolor: 'transparent',
                  title: { text: xTitle, font: { size: 10 } }, autorange: true },
        yaxis:  { tickfont: { size: 9 }, gridcolor: 'transparent',
                  title: { text: yTitle, font: { size: 10 } }, autorange: 'reversed' },
        margin: { l: 70, r: 60, t: 34, b: 68 },
        hoverlabel: { bgcolor: '#1e293b', bordercolor: '#475569', font: { size: 11, color: '#e2e8f0' } },
        shapes,
    }, { responsive: true, displayModeBar: false });
}

// ─────────────────────────────────────────────────────────────
// PLOT 4a — Self-Attention: Output Space
// ─────────────────────────────────────────────────────────────
function renderOutputSpace(result) {
    const { embeddings, values, outputs, attentionWeights } = result;
    const n      = words.length;
    const labels = words.map(w => w.word);
    const baseColors = words.map((_, i) => WORD_COLORS[i % WORD_COLORS.length]);
    // When pair focused: pQ is the query whose output we track; pK is the contributing value
    const embColors = applyFocus(baseColors, new Set([focusPairQ, focusPairK]));
    const valColors = applyFocus(baseColors, new Set([focusPairK]));
    const outColors = applyFocus(baseColors, new Set([focusPairQ]));
    const traces = [];

    // Arrows: input → output (dim non-pQ when focused)
    for (let i = 0; i < n; i++) {
        const op = hasPairFocus() ? (i === focusPairQ ? 0.9 : 0.0) : (i === selectedIdx ? 0.9 : 0.4);
        if (op === 0) continue;
        traces.push({
            x: [embeddings[i][0], outputs[i][0]],
            y: [embeddings[i][1], outputs[i][1]],
            mode: 'lines', type: 'scatter',
            line: { color: baseColors[i], width: 2 },
            showlegend: false, hoverinfo: 'none', opacity: op,
        });
    }

    // When pair focused: draw a contribution arrow from V[pK] to out[pQ], scaled by alpha
    if (hasPairFocus() && focusPairQ < n && focusPairK < n) {
        const vk   = values[focusPairK];
        const outQ = outputs[focusPairQ];
        traces.push({
            x: [vk[0], outQ[0]], y: [vk[1], outQ[1]],
            mode: 'lines', type: 'scatter',
            line: { color: '#c084fc', width: 2, dash: 'dash' },
            showlegend: false, hoverinfo: 'none', opacity: 0.7,
        });
    }

    // Input embeddings
    traces.push({
        x: embeddings.map(e => e[0]), y: embeddings.map(e => e[1]),
        mode: 'markers+text', type: 'scatter', name: 'e_i (input)',
        text: labels, textposition: 'top center',
        textfont: { size: 9, color: embColors },
        marker: { size: 8, color: embColors, opacity: hasPairFocus() ? 1 : 0.35, line: { color: '#334155', width: 1 } },
        hovertemplate: labels.map((l, i) =>
            `<b>${l} input</b><br>${embeddings[i][0].toFixed(3)}, ${embeddings[i][1].toFixed(3)}<extra></extra>`),
    });

    // Value projections
    traces.push({
        x: values.map(v => v[0]), y: values.map(v => v[1]),
        mode: 'markers', type: 'scatter', name: 'v_i = V·e_i',
        marker: {
            size: words.map((_, i) => i === focusPairK ? 13 : 9),
            color: valColors, symbol: 'diamond-open', line: { color: '#14b8a6', width: 2 },
        },
        hovertemplate: labels.map((l, i) =>
            `<b>v_${l}</b><br>${values[i][0].toFixed(3)}, ${values[i][1].toFixed(3)}<extra></extra>`),
    });

    // Output stars
    traces.push({
        x: outputs.map(o => o[0]), y: outputs.map(o => o[1]),
        mode: 'markers+text', type: 'scatter', name: 'out_i  ★',
        text: labels, textposition: 'bottom center',
        textfont: { size: 10, color: outColors },
        marker: {
            size: words.map((_, i) => i === focusPairQ ? 20 : (i === selectedIdx ? 18 : 13)),
            color: outColors, symbol: 'star', line: { color: '#0f172a', width: 1 },
        },
        hovertemplate: labels.map((l, i) =>
            `<b>out_${l}  ★</b><br>${outputs[i][0].toFixed(3)}, ${outputs[i][1].toFixed(3)}<extra></extra>`),
    });

    const all = [...embeddings, ...values, ...outputs].flatMap(p => [Math.abs(p[0]), Math.abs(p[1])]);
    const mx  = Math.max(2.2, ...all) * 1.25;
    Plotly.react('plot-output', traces, {
        ...baseLayout('⑥ Value Projection & Attended Output', [-mx, mx]),
    }, { responsive: true, displayModeBar: false });
}

// ─────────────────────────────────────────────────────────────
// PLOT 4b — Cross-Attention: Output Space
// ─────────────────────────────────────────────────────────────
function renderOutputSpaceCross(result) {
    const { embA, embB, values, outputs, attentionWeights } = result;
    const nA   = words.length;
    const labA = words.map(w => w.word);
    const labB = wordsB.map(w => w.word);
    const baseColA = words.map((_, i) => WORD_COLORS[i % WORD_COLORS.length]);
    const baseColB = wordsB.map((_, i) => WORD_COLORS_B[i % WORD_COLORS_B.length]);
    const colA = applyFocus(baseColA, new Set([focusPairQ]));
    const colB = applyFocus(baseColB, new Set([focusPairK]));
    const traces = [];

    // Arrows A input → A output
    for (let i = 0; i < nA; i++) {
        const op = hasPairFocus()
            ? (i === focusPairQ ? 0.9 : 0.0)
            : (selectedSeq === 'A' && i === selectedIdx ? 0.9 : 0.4);
        if (op === 0) continue;
        traces.push({
            x: [embA[i][0], outputs[i][0]], y: [embA[i][1], outputs[i][1]],
            mode: 'lines', type: 'scatter',
            line: { color: baseColA[i], width: 2 },
            showlegend: false, hoverinfo: 'none', opacity: op,
        });
    }

    if (hasPairFocus() && focusPairQ < nA && focusPairK < wordsB.length) {
        const vk   = values[focusPairK];
        const outQ = outputs[focusPairQ];
        traces.push({
            x: [vk[0], outQ[0]], y: [vk[1], outQ[1]],
            mode: 'lines', type: 'scatter',
            line: { color: '#c084fc', width: 2, dash: 'dash' },
            showlegend: false, hoverinfo: 'none', opacity: 0.7,
        });
    }

    // B context embeddings (faded)
    traces.push({
        x: embB.map(e => e[0]), y: embB.map(e => e[1]),
        mode: 'markers+text', type: 'scatter', name: 'b_j (input)',
        text: labB, textposition: 'bottom center',
        textfont: { size: 9, color: colB },
        marker: { size: 9, color: colB, opacity: hasPairFocus() ? 1 : 0.3, symbol: 'diamond', line: { color: '#374151', width: 1 } },
        hovertemplate: labB.map((l, i) =>
            `<b>${l} [B]</b><br>${embB[i][0].toFixed(3)}, ${embB[i][1].toFixed(3)}<extra></extra>`),
    });

    // B value projections
    traces.push({
        x: values.map(v => v[0]), y: values.map(v => v[1]),
        mode: 'markers+text', type: 'scatter', name: 'v_j = V·b_j',
        text: labB.map(l => 'v_' + l), textposition: 'top right',
        textfont: { size: 9, color: '#14b8a6' },
        marker: {
            size: wordsB.map((_, i) => i === focusPairK ? 13 : 9),
            color: colB, symbol: 'diamond-open', line: { color: '#14b8a6', width: 2 },
        },
        hovertemplate: labB.map((l, i) =>
            `<b>v_${l}</b><br>${values[i][0].toFixed(3)}, ${values[i][1].toFixed(3)}<extra></extra>`),
    });

    // A input embeddings
    traces.push({
        x: embA.map(e => e[0]), y: embA.map(e => e[1]),
        mode: 'markers+text', type: 'scatter', name: 'a_i (input)',
        text: labA, textposition: 'top center',
        textfont: { size: 9, color: colA },
        marker: { size: 8, color: colA, opacity: hasPairFocus() ? 1 : 0.35, line: { color: '#334155', width: 1 } },
        hovertemplate: labA.map((l, i) =>
            `<b>${l} [A]</b><br>${embA[i][0].toFixed(3)}, ${embA[i][1].toFixed(3)}<extra></extra>`),
    });

    // A output stars
    traces.push({
        x: outputs.map(o => o[0]), y: outputs.map(o => o[1]),
        mode: 'markers+text', type: 'scatter', name: 'out_i  ★',
        text: labA, textposition: 'bottom center',
        textfont: { size: 10, color: colA },
        marker: {
            size: words.map((_, i) => i === focusPairQ ? 20 : 13),
            color: colA, symbol: 'star', line: { color: '#0f172a', width: 1 },
        },
        hovertemplate: labA.map((l, i) =>
            `<b>out_${l}  ★</b><br>${outputs[i][0].toFixed(3)}, ${outputs[i][1].toFixed(3)}<extra></extra>`),
    });

    const all = [...embA, ...embB, ...values, ...outputs].flatMap(p => [Math.abs(p[0]), Math.abs(p[1])]);
    const mx  = Math.max(2.2, ...all) * 1.25;
    Plotly.react('plot-output', traces, {
        ...baseLayout('⑥ A attends to B  →  Output A  ★', [-mx, mx]),
    }, { responsive: true, displayModeBar: false });
}

// ════════════════════════════════════════════════════════════
// 5. UI UPDATE FUNCTIONS
// ════════════════════════════════════════════════════════════

function updateMatrixUI() {
    const entries = [['Q', Q, 'q'], ['K', K, 'k'], ['V', V, 'v']];
    entries.forEach(([, M, pfx]) => {
        [[0,0],[0,1],[1,0],[1,1]].forEach(([r,c]) => {
            const id = `${pfx}${r+1}${c+1}`;
            const slider  = document.getElementById(`slider-${id}`);
            const display = document.getElementById(`val-${id}`);
            if (slider)  slider.value        = M[r][c];
            if (display) display.textContent = M[r][c].toFixed(2);
        });
    });
    const ts = document.getElementById('temp-slider');
    const tv = document.getElementById('temp-val');
    if (ts) ts.value       = temperature;
    if (tv) tv.textContent = temperature.toFixed(2);
}

function updateWordsList() {
    const container = document.getElementById('words-list');
    if (!container) return;

    if (attentionMode === 'cross') {
        const makeList = (arr, colorSet, seqLabel, selectFn, removeFn, curSeq) =>
            !arr.length
                ? `<div class="text-[10px] text-slate-600 text-center py-2">No words</div>`
                : arr.map((w, i) => {
                    const color = colorSet[i % colorSet.length];
                    const sel   = curSeq === seqLabel && i === selectedIdx;
                    const fq    = seqLabel === 'A' && i === focusPairQ;
                    const fk    = seqLabel === 'B' && i === focusPairK;
                    const ring  = (fq || fk) ? 'ring-1 ring-purple-500/50' : '';
                    return `
                    <div class="flex items-center gap-1.5 px-2 py-0.5 rounded cursor-pointer ${ring}
                                ${sel ? 'bg-slate-800' : 'hover:bg-slate-800/50'}"
                         onclick="${selectFn}(${i})">
                        <div class="w-2 h-2 rounded-full flex-shrink-0" style="background:${color}"></div>
                        <span class="text-[10px] font-mono font-bold text-slate-200 flex-1">${w.word}</span>
                        <span class="text-[10px] font-mono text-slate-500">(${w.x.toFixed(1)},${w.y.toFixed(1)})</span>
                        <button onclick="event.stopPropagation();${removeFn}(${i})"
                                class="text-red-500 hover:text-red-300 text-[10px] px-1">✕</button>
                    </div>`;
                }).join('');

        container.innerHTML = `
        <div class="space-y-1 pb-1">
            <div class="px-2 py-0.5 text-[9px] font-bold text-cyan-500 uppercase tracking-wider bg-cyan-950/30">
                Seq A — Queries (${words.length})
            </div>
            ${makeList(words, WORD_COLORS, 'A', 'selectWord', 'removeWord', selectedSeq)}
        </div>
        <div class="space-y-1 pt-1 border-t border-slate-800">
            <div class="px-2 py-0.5 text-[9px] font-bold text-red-400 uppercase tracking-wider bg-red-950/30">
                Seq B — Context (${wordsB.length})
            </div>
            ${makeList(wordsB, WORD_COLORS_B, 'B', 'selectWordB', 'removeWordB', selectedSeq)}
        </div>`;
        return;
    }

    if (!words.length) {
        container.innerHTML = '<div class="text-[10px] text-slate-500 text-center py-3">No words. Add some below.</div>';
        return;
    }
    container.innerHTML = words.map((w, i) => {
        const color = WORD_COLORS[i % WORD_COLORS.length];
        const sel   = i === selectedIdx;
        const fp    = (i === focusPairQ || i === focusPairK);
        return `
        <div class="flex items-center gap-1.5 px-2 py-1 rounded cursor-pointer
                    ${sel ? 'bg-slate-800' : 'hover:bg-slate-800/50'}
                    ${fp  ? 'ring-1 ring-inset ring-purple-500/40' : ''}"
             onclick="selectWord(${i})">
            <div class="w-2 h-2 rounded-full flex-shrink-0" style="background:${color}"></div>
            <span class="text-[10px] font-mono font-bold text-slate-200 flex-1">${w.word}</span>
            ${fp ? `<span class="text-[9px] text-purple-400">${i === focusPairQ ? 'Q' : 'K'}</span>` : ''}
            <span class="text-[10px] font-mono text-slate-500">(${w.x.toFixed(2)},${w.y.toFixed(2)})</span>
            <button onclick="event.stopPropagation();removeWord(${i})"
                    class="text-red-500 hover:text-red-300 text-[10px] px-1">✕</button>
        </div>`;
    }).join('');
}

function updateEditPanel() {
    const panel = document.getElementById('edit-panel');
    if (!panel) return;
    if (selectedIdx < 0) { panel.classList.add('hidden'); return; }
    panel.classList.remove('hidden');
    const w   = (attentionMode === 'cross' && selectedSeq === 'B') ? wordsB[selectedIdx] : words[selectedIdx];
    const tag = (attentionMode === 'cross' && selectedSeq === 'B') ? ' [B]' : '';
    if (!w) { panel.classList.add('hidden'); return; }
    document.getElementById('edit-word-name').textContent = w.word + tag;
    document.getElementById('edit-x').value = w.x.toFixed(2);
    document.getElementById('edit-y').value = w.y.toFixed(2);
}

function updateAttentionBars(result) {
    const container = document.getElementById('attn-bars');
    const header    = document.getElementById('attn-bars-header');
    if (!container) return;

    const displayIdx = hasPairFocus() ? focusPairQ : selectedIdx;
    if (displayIdx < 0 || !result) {
        if (header) header.textContent = 'Click a heatmap cell or select a word';
        container.innerHTML = '<div class="text-[10px] text-slate-600 text-center py-2">—</div>';
        return;
    }

    if (attentionMode === 'cross') {
        if (hasPairFocus() || selectedSeq === 'A') {
            const idx   = hasPairFocus() ? focusPairQ : selectedIdx;
            const row   = result.attentionWeights[idx];
            const qWord = words[idx].word;
            if (header) header.textContent = `"${qWord}"${hasPairFocus() ? ' [focused]' : ''} attends to B:`;
            container.innerHTML = wordsB.map((w, j) => {
                const alpha    = row[j];
                const color    = WORD_COLORS_B[j % WORD_COLORS_B.length];
                const barW     = Math.round(alpha * 100);
                const isFocusK = hasPairFocus() && j === focusPairK;
                return `
                <div class="space-y-0.5 ${isFocusK ? 'bg-purple-950/30 rounded px-1 py-0.5 -mx-1' : ''}">
                    <div class="flex justify-between items-center">
                        <span class="text-[10px] font-mono font-bold" style="color:${color};opacity:${isFocusK || !hasPairFocus() ? 1 : 0.3}">${w.word}[B]</span>
                        <span class="text-[10px] font-mono ${isFocusK ? 'text-purple-300 font-bold' : 'text-slate-600'}" style="opacity:${isFocusK || !hasPairFocus() ? 1 : 0.4}">${(alpha*100).toFixed(1)}%</span>
                    </div>
                    <div class="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div class="h-full rounded-full" style="width:${barW}%;background:${color};opacity:${isFocusK || !hasPairFocus() ? 1 : 0.2}"></div>
                    </div>
                </div>`;
            }).join('');
        } else {
            const col   = result.attentionWeights.map(row => row[selectedIdx]);
            const kWord = wordsB[selectedIdx].word;
            if (header) header.textContent = `A words attending to "${kWord}"[B]:`;
            container.innerHTML = words.map((w, i) => {
                const alpha = col[i], color = WORD_COLORS[i % WORD_COLORS.length];
                const barW  = Math.round(alpha * 100);
                return `
                <div class="space-y-0.5">
                    <div class="flex justify-between items-center">
                        <span class="text-[10px] font-mono font-bold" style="color:${color}">${w.word}[A]</span>
                        <span class="text-[10px] font-mono text-slate-400">${(alpha*100).toFixed(1)}%</span>
                    </div>
                    <div class="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                        <div class="h-full rounded-full" style="width:${barW}%;background:${color}"></div>
                    </div>
                </div>`;
            }).join('');
        }
        return;
    }

    // Self mode
    const row   = result.attentionWeights[displayIdx];
    const qWord = words[displayIdx].word;
    if (header) header.textContent = hasPairFocus()
        ? `"${qWord}" → "${words[focusPairK].word}"  [focused]`
        : `"${qWord}" attends to:`;

    container.innerHTML = words.map((w, j) => {
        const alpha    = row[j];
        const color    = WORD_COLORS[j % WORD_COLORS.length];
        const barW     = Math.round(alpha * 100);
        const isFocusK = hasPairFocus() && j === focusPairK;
        return `
        <div class="space-y-0.5 ${isFocusK ? 'bg-purple-950/30 rounded px-1 py-0.5 -mx-1' : ''}">
            <div class="flex justify-between items-center">
                <span class="text-[10px] font-mono font-bold" style="color:${color};opacity:${isFocusK || !hasPairFocus() ? 1 : 0.3}">${w.word}</span>
                <span class="text-[10px] font-mono ${isFocusK ? 'text-purple-300 font-bold' : 'text-slate-500'}" style="opacity:${isFocusK || !hasPairFocus() ? 1 : 0.35}">${(alpha*100).toFixed(1)}%</span>
            </div>
            <div class="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                <div class="h-full rounded-full transition-all duration-200"
                     style="width:${barW}%;background:${color};opacity:${isFocusK || !hasPairFocus() ? 1 : 0.2}"></div>
            </div>
        </div>`;
    }).join('');
}

function updateStepGuide() {
    const step  = STEPS[currentStep];
    const numEl = document.getElementById('step-number');
    const ttlEl = document.getElementById('step-title');
    const frmEl = document.getElementById('step-formula');
    const dscEl = document.getElementById('step-description');
    const tipEl = document.getElementById('step-tip');
    const ctrEl = document.getElementById('step-counter');

    if (numEl)  { numEl.textContent = step.number; numEl.style.background = step.color; }
    if (ttlEl)  ttlEl.textContent  = step.title;
    if (frmEl)  frmEl.textContent  = step.formula;
    if (dscEl)  dscEl.textContent  = step.description;
    if (tipEl)  tipEl.textContent  = '💡 ' + step.tip;
    if (ctrEl)  ctrEl.textContent  = `${currentStep + 1} / ${STEPS.length}`;

    const listEl = document.getElementById('steps-list');
    if (listEl) {
        listEl.innerHTML = STEPS.map((s, i) => `
        <button onclick="goToStep(${i})"
                class="w-full flex items-center gap-2 px-2 py-1.5 rounded text-left transition-colors
                       ${i === currentStep ? 'bg-slate-800' : 'hover:bg-slate-800/50'}">
            <div class="w-5 h-5 rounded-full flex-shrink-0 flex items-center justify-center text-[9px] font-bold border"
                 style="background:${i === currentStep ? s.color : 'transparent'};
                        border-color:${i <= currentStep ? s.color : '#334155'};
                        color:${i === currentStep ? '#fff' : i < currentStep ? s.color : '#64748b'}">
                ${s.number}
            </div>
            <span class="text-[10px] ${i === currentStep ? 'text-slate-200 font-bold' : 'text-slate-400'}">
                ${s.title}
            </span>
        </button>`).join('');
    }

    heatmapMode = step.heatmap;
    syncHeatmapButtons();
}

function syncHeatmapButtons() {
    ['rawscore', 'cosine', 'attention'].forEach(mode => {
        const btn = document.getElementById(`heatmap-${mode}`);
        if (!btn) return;
        const active = heatmapMode === mode;
        btn.className = `px-2 py-0.5 rounded text-[9px] font-bold transition-colors ${
            active ? 'bg-purple-600 text-white' : 'bg-slate-800 text-slate-400 hover:bg-slate-700'
        }`;
    });
}

function updateModeUI() {
    const selfBtn   = document.getElementById('mode-self');
    const crossBtn  = document.getElementById('mode-cross');
    const crossSec  = document.getElementById('cross-section');
    const crossInfo = document.getElementById('cross-info-card');
    const titleEl   = document.getElementById('header-title');

    if (selfBtn)  selfBtn.className  = `px-3 py-1 rounded-md text-[11px] font-bold transition-colors ${
        attentionMode === 'self'  ? 'bg-slate-200 text-slate-900' : 'text-slate-400 hover:text-white'
    }`;
    if (crossBtn) crossBtn.className = `px-3 py-1 rounded-md text-[11px] font-bold transition-colors ${
        attentionMode === 'cross' ? 'bg-orange-400 text-slate-900' : 'text-slate-400 hover:text-white'
    }`;
    if (crossSec)  crossSec.classList.toggle('hidden',  attentionMode !== 'cross');
    if (crossInfo) crossInfo.classList.toggle('hidden', attentionMode !== 'cross');
    if (titleEl)   titleEl.textContent = attentionMode === 'cross'
        ? 'Transformer Cross-Attention — Interactive Lab'
        : 'Transformer Self-Attention — Interactive Lab';
}

// ════════════════════════════════════════════════════════════
// 6. INTERACTION HANDLERS
// ════════════════════════════════════════════════════════════

function updateFocusClearBtn() {
    const btn = document.getElementById('focus-clear-btn');
    if (!btn) return;
    if (hasPairFocus()) {
        const qW = words[focusPairQ] ? words[focusPairQ].word : '?';
        const kArr = attentionMode === 'cross' ? wordsB : words;
        const kW = kArr[focusPairK] ? kArr[focusPairK].word : '?';
        btn.textContent = `✕  q_${qW} → k_${kW}`;
        btn.classList.remove('hidden');
    } else {
        btn.classList.add('hidden');
    }
}

function focusPair(qIdx, kIdx) {
    if (focusPairQ === qIdx && focusPairK === kIdx) {
        focusPairQ  = -1;
        focusPairK  = -1;
        selectedIdx = -1;
    } else {
        focusPairQ = qIdx;
        focusPairK = kIdx;
        selectedIdx = qIdx;
        if (attentionMode === 'cross') selectedSeq = 'A';
    }
    updateFocusClearBtn();
    updateWordsList();
    updateEditPanel();
    renderAll();
}

function clearPairFocus() {
    focusPairQ  = -1;
    focusPairK  = -1;
    selectedIdx = -1;
    updateFocusClearBtn();
    updateWordsList();
    updateEditPanel();
    renderAll();
}

function selectWord(idx) {
    // Clear pair focus when clicking a word directly
    focusPairQ = -1; focusPairK = -1;
    if (attentionMode === 'cross') selectedSeq = 'A';
    selectedIdx = (selectedIdx === idx && (attentionMode !== 'cross' || selectedSeq === 'A')) ? -1 : idx;
    updateWordsList();
    updateEditPanel();
    renderAll();
}

function selectWordB(idx) {
    focusPairQ = -1; focusPairK = -1; // clear pair focus
    const alreadySelected = selectedSeq === 'B' && selectedIdx === idx;
    selectedSeq = 'B';
    selectedIdx = alreadySelected ? -1 : idx;
    updateWordsList();
    updateEditPanel();
    renderAll();
}

function updateSelectedWordCoord(axis, val) {
    if (selectedIdx < 0) return;
    const v = parseFloat(val);
    if (isNaN(v)) return;
    const arr = (attentionMode === 'cross' && selectedSeq === 'B') ? wordsB : words;
    arr[selectedIdx][axis] = v;
    updateWordsList();
    renderAll();
}

function addWord() {
    const wordEl = document.getElementById('new-word');
    const xEl    = document.getElementById('new-x');
    const yEl    = document.getElementById('new-y');
    const word   = wordEl.value.trim();
    const x      = parseFloat(xEl.value);
    const y      = parseFloat(yEl.value);
    if (!word) { wordEl.focus(); return; }
    if (words.length >= 8) { showNotif('Maximum 8 words for clarity.'); return; }
    words.push({ word, x: isNaN(x) ? 0 : x, y: isNaN(y) ? 0 : y });
    wordEl.value = xEl.value = yEl.value = '';
    wordEl.focus();
    updateWordsList();
    renderAll();
}

function addWordB() {
    const wordEl = document.getElementById('new-word-b');
    const xEl    = document.getElementById('new-x-b');
    const yEl    = document.getElementById('new-y-b');
    const word   = wordEl.value.trim();
    const x      = parseFloat(xEl.value);
    const y      = parseFloat(yEl.value);
    if (!word) { wordEl.focus(); return; }
    if (wordsB.length >= 8) { showNotif('Maximum 8 words in Seq B.'); return; }
    wordsB.push({ word, x: isNaN(x) ? 0 : x, y: isNaN(y) ? 0 : y });
    wordEl.value = xEl.value = yEl.value = '';
    wordEl.focus();
    updateWordsList();
    renderAll();
}

function removeWord(idx) {
    words.splice(idx, 1);
    if (selectedSeq === 'A' && selectedIdx >= words.length) selectedIdx = -1;
    if (focusPairQ >= words.length) { focusPairQ = -1; focusPairK = -1; }
    updateWordsList(); updateEditPanel(); renderAll();
}

function removeWordB(idx) {
    wordsB.splice(idx, 1);
    if (selectedSeq === 'B' && selectedIdx >= wordsB.length) selectedIdx = -1;
    if (focusPairK >= wordsB.length) { focusPairQ = -1; focusPairK = -1; }
    updateWordsList(); updateEditPanel(); renderAll();
}

function loadPreset(name) {
    words = WORD_PRESETS[name].words.map(w => ({ ...w }));
    selectedIdx = -1; focusPairQ = -1; focusPairK = -1;
    updateWordsList(); updateEditPanel(); renderAll();
}

function loadCrossPreset(name) {
    const p = CROSS_PRESETS[name];
    words   = p.wordsA.map(w => ({ ...w }));
    wordsB  = p.wordsB.map(w => ({ ...w }));
    selectedIdx = -1; selectedSeq = 'A'; focusPairQ = -1; focusPairK = -1;
    updateWordsList(); updateEditPanel(); renderAll();
}

function setAttentionMode(mode) {
    attentionMode = mode;
    selectedIdx = -1; selectedSeq = 'A'; focusPairQ = -1; focusPairK = -1;
    updateFocusClearBtn();
    updateModeUI(); updateWordsList(); updateEditPanel(); renderAll();
}

function loadMatrixPreset(name) {
    const p = MATRIX_PRESETS[name];
    Q = p.Q.map(r => [...r]);
    K = p.K.map(r => [...r]);
    V = p.V.map(r => [...r]);
    updateMatrixUI(); renderAll();
}

function resetMatrices() {
    Q = [[1,0],[0,1]]; K = [[1,0],[0,1]]; V = [[1,0],[0,1]];
    temperature = 1.0;
    updateMatrixUI(); renderAll();
}

function setMatrixValue(mat, row, col, val) {
    const v = parseFloat(val);
    if (isNaN(v)) return;
    const M = mat === 'Q' ? Q : mat === 'K' ? K : V;
    M[row][col] = v;
    const dispEl = document.getElementById(`val-${mat.toLowerCase()}${row+1}${col+1}`);
    if (dispEl) dispEl.textContent = v.toFixed(2);
    renderAll();
}

function setTemperature(val) {
    temperature = Math.max(0.01, parseFloat(val));
    const tv = document.getElementById('temp-val');
    if (tv) tv.textContent = temperature.toFixed(2);
    renderAll();
}

function setHeatmapMode(mode) {
    heatmapMode = mode;
    syncHeatmapButtons();
    const result = computeAttention();
    if (result) renderHeatmap(result);
}

function goToStep(idx) { currentStep = idx; updateStepGuide(); renderAll(); }
function nextStep()    { if (currentStep < STEPS.length - 1) { currentStep++; updateStepGuide(); renderAll(); } }
function prevStep()    { if (currentStep > 0) { currentStep--; updateStepGuide(); renderAll(); } }

function togglePanel(panelId) {
    const key   = panelId.replace('panel-', '');
    const panel = document.getElementById(panelId);
    panelVis[key] = !panelVis[key];
    if (panelVis[key]) panel.classList.remove('hidden');
    else               panel.classList.add('hidden');

    const colorMap = { words: 'bg-cyan-600', matrices: 'bg-amber-600', steps: 'bg-violet-600' };
    const btn = document.getElementById(`toggle-${key}`);
    if (btn) {
        if (panelVis[key]) { btn.classList.remove('bg-slate-700'); btn.classList.add(colorMap[key]); }
        else               { btn.classList.remove(colorMap[key]); btn.classList.add('bg-slate-700'); }
    }
    setTimeout(() => {
        ['plot-embeddings', 'plot-qkspace', 'plot-heatmap', 'plot-output']
            .forEach(id => Plotly.Plots.resize(id));
    }, 160);
}

function showNotif(msg) {
    const el = document.getElementById('notif');
    if (!el) return;
    el.textContent = msg;
    el.classList.remove('opacity-0');
    el.classList.add('opacity-100');
    setTimeout(() => { el.classList.remove('opacity-100'); el.classList.add('opacity-0'); }, 2200);
}

// ════════════════════════════════════════════════════════════
// 7. INITIALIZATION
// ════════════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', () => {
    updateWordsList();
    updateMatrixUI();
    updateStepGuide();
    updateModeUI();
    updateFocusClearBtn();
    renderAll();

    setTimeout(() => {
        // Embedding plot: click word
        const embPlot = document.getElementById('plot-embeddings');
        if (embPlot) {
            embPlot.on('plotly_click', data => {
                const pt = data.points[0];
                if (!pt) return;
                if (attentionMode === 'cross') {
                    if (pt.data.name && pt.data.name.includes('Seq A')) selectWord(pt.pointIndex);
                    if (pt.data.name && pt.data.name.includes('Seq B')) selectWordB(pt.pointIndex);
                } else {
                    if (pt.data.name === 'Embeddings') selectWord(pt.pointIndex);
                }
            });
        }

        // Heatmap: click cell → focus pair; double-click → clear focus
        const hmPlot = document.getElementById('plot-heatmap');
        if (hmPlot) {
            hmPlot.on('plotly_click', data => {
                const pt = data.points[0];
                if (!pt) return;
                // y-axis = query (row index), x-axis = key (col index)
                const qWord = String(pt.y).replace('q_', '');
                const kWord = String(pt.x).replace('k_', '');
                const qIdx  = words.findIndex(w => w.word === qWord);
                const kIdx  = attentionMode === 'cross'
                    ? wordsB.findIndex(w => w.word === kWord)
                    : words.findIndex(w => w.word === kWord);
                if (qIdx >= 0 && kIdx >= 0) focusPair(qIdx, kIdx);
            });
            hmPlot.on('plotly_doubleclick', () => { clearPairFocus(); return false; });
        }

        // Output plot: click output star → select word
        const outPlot = document.getElementById('plot-output');
        if (outPlot) {
            outPlot.on('plotly_click', data => {
                const pt = data.points[0];
                if (!pt || !pt.data.name) return;
                if (pt.data.name.startsWith('out_i')) selectWord(pt.pointIndex);
                if (pt.data.name === 'b_j (input)' || pt.data.name === 'v_j = V·b_j') selectWordB(pt.pointIndex);
            });
        }

        // Q/K space: click to clear pair focus
        const qkPlot = document.getElementById('plot-qkspace');
        if (qkPlot) {
            qkPlot.on('plotly_click', () => { if (hasPairFocus()) clearPairFocus(); });
        }
    }, 600);
});
