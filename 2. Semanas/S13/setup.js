// ============================================================
// TRANSFORMER SELF-ATTENTION VISUALIZER — setup.js
// Global config, presets, and initial state
// ============================================================

// Word colors — each word gets a unique color across all 4 plots
const WORD_COLORS = [
    '#06b6d4', // cyan
    '#f59e0b', // amber
    '#10b981', // emerald
    '#ec4899', // pink
    '#8b5cf6', // violet
    '#f97316', // orange
    '#14b8a6', // teal
    '#ef4444', // red
];

// ── Word Presets ─────────────────────────────────────────────
const WORD_PRESETS = {
    animals: {
        label: 'Animals',
        words: [
            { word: 'cat',  x:  0.80, y:  0.70 },
            { word: 'dog',  x:  0.70, y:  0.50 },
            { word: 'fish', x: -0.50, y:  0.80 },
            { word: 'bird', x: -0.40, y:  0.90 },
            { word: 'car',  x:  0.20, y: -0.80 },
        ],
    },
    sentence: {
        label: 'Simple Sentence',
        words: [
            { word: 'The',  x:  0.10, y:  0.90 },
            { word: 'cat',  x:  0.80, y:  0.70 },
            { word: 'sat',  x: -0.60, y: -0.40 },
            { word: 'on',   x:  0.00, y: -0.90 },
            { word: 'mat',  x:  0.60, y: -0.60 },
        ],
    },
    geometric: {
        label: 'Geometric (4 pts)',
        words: [
            { word: 'A', x:  1.00, y:  0.00 },
            { word: 'B', x:  0.00, y:  1.00 },
            { word: 'C', x: -1.00, y:  0.00 },
            { word: 'D', x:  0.00, y: -1.00 },
        ],
    },
    clusters: {
        label: 'Two Clusters',
        words: [
            { word: 'W1', x:  0.90, y:  0.80 },
            { word: 'W2', x:  0.80, y:  0.90 },
            { word: 'W3', x: -0.80, y: -0.70 },
            { word: 'W4', x: -0.90, y: -0.80 },
        ],
    },
};

// ── Matrix Presets ────────────────────────────────────────────
const MATRIX_PRESETS = {
    identity: {
        label: 'Identity (no change)',
        Q: [[1, 0], [0, 1]],
        K: [[1, 0], [0, 1]],
        V: [[1, 0], [0, 1]],
    },
    rotation45: {
        label: 'Rotate Q by 45°',
        Q: [[ 0.707, -0.707], [ 0.707,  0.707]],
        K: [[ 1,      0    ], [ 0,      1    ]],
        V: [[ 1,      0    ], [ 0,      1    ]],
    },
    xProject: {
        label: 'Project onto X',
        Q: [[1, 0], [0, 0]],
        K: [[1, 0], [0, 0]],
        V: [[1, 0], [0, 1]],
    },
    divergeQK: {
        label: 'Diverge Q vs K',
        Q: [[ 1.0,  0.5], [-0.5,  1.0]],
        K: [[-0.5,  1.0], [ 1.0,  0.5]],
        V: [[ 1.0,  0.0], [ 0.0,  1.0]],
    },
    scaleV: {
        label: 'Scale Values ×2',
        Q: [[1, 0], [0, 1]],
        K: [[1, 0], [0, 1]],
        V: [[2, 0], [0, 2]],
    },
    rotateV: {
        label: 'Rotate Values 90°',
        Q: [[1, 0], [0, 1]],
        K: [[1, 0], [0, 1]],
        V: [[0, -1], [1, 0]],
    },
};

// ── Step-by-Step Guide Data ───────────────────────────────────
const STEPS = [
    {
        number: 1,
        title: 'Word Embeddings',
        formula: 'e_i  ∈  ℝ²',
        description: 'Each word is a 2D vector in embedding space. Similar words are close together. These are the raw inputs to the attention mechanism.',
        color: '#06b6d4',
        highlight: 'embeddings',
        heatmap: 'attention',
        tip: 'Drag words around in the first plot to see how position affects attention.',
    },
    {
        number: 2,
        title: 'Query Projection',
        formula: 'q_i  =  Q · e_i',
        description: 'The Query matrix Q projects each embedding into a "query vector" — what each word is searching for. Q rotates and scales the embedding space.',
        color: '#f59e0b',
        highlight: 'queries',
        heatmap: 'attention',
        tip: 'Change Q matrix values and watch how the query arrows move in the first plot.',
    },
    {
        number: 3,
        title: 'Key Projection',
        formula: 'k_i  =  K · e_i',
        description: 'The Key matrix K projects embeddings into "key vectors" — what each word broadcasts to others. Keys are matched against queries to compute attention.',
        color: '#10b981',
        highlight: 'keys',
        heatmap: 'attention',
        tip: 'When Q = K = Identity, queries and keys are the same as the embeddings.',
    },
    {
        number: 4,
        title: 'Raw Attention Scores',
        formula: 'score(i,j)  =  q_i · k_j  /  √2',
        description: 'The dot product of query i with key j measures compatibility. Dividing by √d_k prevents scores from growing too large and making softmax saturate.',
        color: '#ec4899',
        highlight: 'rawscores',
        heatmap: 'rawscore',
        tip: 'Look at the heatmap: bright = high raw score, dark = low. Rows sum before softmax.',
    },
    {
        number: 5,
        title: 'Cosine Similarity',
        formula: 'cos(i,j)  =  (q_i · k_j) / (|q_i| · |k_j|)',
        description: 'Normalizing by vector magnitudes gives cosine similarity — purely directional. A value of 1 means they point the same way; -1 means opposite.',
        color: '#8b5cf6',
        highlight: 'cosine',
        heatmap: 'cosine',
        tip: 'Cosine similarity removes the effect of vector length — only direction matters here.',
    },
    {
        number: 6,
        title: 'Attention Weights',
        formula: 'α_ij  =  softmax( score_ij / τ )',
        description: 'Softmax converts scores into a probability distribution (each row sums to 1). Temperature τ controls sharpness: low τ → sharp focus on top match; high τ → spread evenly.',
        color: '#f97316',
        highlight: 'attention',
        heatmap: 'attention',
        tip: 'Try dragging the Temperature slider to see how τ affects the attention distribution.',
    },
    {
        number: 7,
        title: 'Value Projection',
        formula: 'v_i  =  V · e_i',
        description: 'The Value matrix V transforms each embedding into what it "contributes" when attended to. V is completely independent from Q and K — it only affects the output, not who attends to whom.',
        color: '#14b8a6',
        highlight: 'values',
        heatmap: 'attention',
        tip: 'Change V and watch the output points (stars) move — Q/K/attention stay the same.',
    },
    {
        number: 8,
        title: 'Attended Output',
        formula: 'out_i  =  Σ_j  α_ij · v_j',
        description: 'The output for word i is the weighted average of all value vectors, using the attention weights. Words with high α_ij pull the output toward their value vector.',
        color: '#ef4444',
        highlight: 'output',
        heatmap: 'attention',
        tip: 'The star (★) in the output plot shows where each word "ends up" after attending.',
    },
];

// ── Cross-Attention Color Set (Sequence B — warm palette) ─────
const WORD_COLORS_B = [
    '#f87171', // red-400
    '#fb923c', // orange-400
    '#facc15', // yellow-400
    '#a3e635', // lime-400
    '#4ade80', // green-400
    '#2dd4bf', // teal-300
    '#60a5fa', // blue-400
    '#c084fc', // purple-400
];

// ── Cross-Attention Word Presets ──────────────────────────────
const CROSS_PRESETS = {
    translation: {
        label: 'EN → FR Translation',
        wordsA: [
            { word: 'the',  x:  0.20, y:  0.90 },
            { word: 'cat',  x:  0.80, y:  0.70 },
            { word: 'sat',  x: -0.60, y: -0.40 },
        ],
        wordsB: [
            { word: 'le',    x:  0.15, y:  0.85 },
            { word: 'chat',  x:  0.75, y:  0.65 },
            { word: 'assis', x: -0.55, y: -0.45 },
        ],
    },
    qa: {
        label: 'Question → Answer',
        wordsA: [
            { word: 'What', x:  0.50, y:  0.80 },
            { word: 'is',   x:  0.10, y:  0.60 },
            { word: 'it',   x: -0.30, y:  0.70 },
        ],
        wordsB: [
            { word: 'sky',   x:  0.40, y: -0.70 },
            { word: 'is',    x:  0.10, y: -0.50 },
            { word: 'blue',  x: -0.60, y: -0.80 },
            { word: 'today', x: -0.80, y: -0.40 },
        ],
    },
    clusters: {
        label: 'Two Clusters A→B',
        wordsA: [
            { word: 'A1', x:  0.90, y:  0.80 },
            { word: 'A2', x:  0.80, y:  0.95 },
            { word: 'A3', x:  0.95, y:  0.60 },
        ],
        wordsB: [
            { word: 'B1', x: -0.90, y: -0.80 },
            { word: 'B2', x: -0.80, y: -0.95 },
            { word: 'B3', x: -0.95, y: -0.60 },
            { word: 'B4', x: -0.70, y: -0.70 },
        ],
    },
    mirror: {
        label: 'Mirror A ↔ B',
        wordsA: [
            { word: 'X',  x:  1.00, y:  0.00 },
            { word: 'Y',  x:  0.00, y:  1.00 },
            { word: 'Z',  x: -1.00, y:  0.00 },
        ],
        wordsB: [
            { word: "X'", x:  1.00, y:  0.00 },
            { word: "Y'", x:  0.00, y:  1.00 },
            { word: "Z'", x: -1.00, y:  0.00 },
            { word: "W'", x:  0.00, y: -1.00 },
        ],
    },
};

// ── Global Mutable State ──────────────────────────────────────
let words       = WORD_PRESETS.animals.words.map(w => ({ ...w }));
let selectedIdx = -1;  // selected word index (-1 = none)

// Query, Key, Value matrices (2×2)
let Q = [[1, 0], [0, 1]];
let K = [[1, 0], [0, 1]];
let V = [[1, 0], [0, 1]];

let temperature    = 1.0;
let heatmapMode    = 'attention';   // 'rawscore' | 'cosine' | 'attention'
let currentStep    = 0;

// ── Pair-focus state (set when a heatmap cell is clicked) ─────
let focusPairQ     = -1;   // focused query index  (-1 = none)
let focusPairK     = -1;   // focused key index    (-1 = none)

// ── Cross-Attention State ─────────────────────────────────────
let attentionMode = 'self';   // 'self' | 'cross'
let wordsB        = CROSS_PRESETS.translation.wordsB.map(w => ({ ...w }));
let selectedSeq   = 'A';      // 'A' | 'B'  (which sequence holds selectedIdx)

// Panel visibility
let panelVis = { words: true, matrices: true, steps: true };
