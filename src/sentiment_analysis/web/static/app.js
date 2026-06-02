/* ============================================================
   SENTIMENT ANALYSIS DASHBOARD — Frontend Logic
   ============================================================ */

const API_BASE = '/api';
let comparisonChart = null;
let batchChart = null;
let analysisHistory = JSON.parse(localStorage.getItem('sentimentHistory') || '[]');

// ——— Utility ———

const SENTIMENT_CONFIG = {
    VERY_POSITIVE: { emoji: '🤩', color: '#10b981', class: 'very-positive' },
    POSITIVE:      { emoji: '😊', color: '#22c55e', class: 'positive' },
    NEUTRAL:       { emoji: '😐', color: '#eab308', class: 'neutral' },
    NEGATIVE:      { emoji: '😞', color: '#f97316', class: 'negative' },
    VERY_NEGATIVE: { emoji: '😠', color: '#ef4444', class: 'very-negative' },
};

function getSentimentConfig(label) {
    return SENTIMENT_CONFIG[label] || SENTIMENT_CONFIG.NEUTRAL;
}

function setStatus(state, text) {
    const dot = document.querySelector('.status-dot');
    const txt = document.querySelector('.status-text');
    dot.className = 'status-dot' + (state === 'loading' ? ' loading' : state === 'error' ? ' error' : '');
    txt.textContent = text || (state === 'loading' ? 'Analyzing...' : state === 'error' ? 'Error' : 'Ready');
}

function truncate(str, len = 80) {
    return str.length > len ? str.slice(0, len) + '…' : str;
}

// ——— Character Counter ———

document.getElementById('text-input').addEventListener('input', function () {
    document.getElementById('char-count').textContent = this.value.length;
});

// ——— Analyze Text ———

async function analyzeText() {
    const text = document.getElementById('text-input').value.trim();
    if (!text) return;

    const model = document.getElementById('model-select').value;
    const btn = document.getElementById('analyze-btn');

    btn.disabled = true;
    setStatus('loading');

    try {
        const res = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, model }),
        });

        if (!res.ok) {
            let errMessage = 'Analysis failed';
            const errorText = await res.text();
            try {
                const err = JSON.parse(errorText);
                errMessage = err.detail || errMessage;
            } catch (parseErr) {
                // If it's not JSON (e.g. Render 502/504 HTML page)
                errMessage = errorText;
            }
            throw new Error(errMessage);
        }

        const result = await res.json();
        displayResult(result);
        addToHistory(result);
        setStatus('ready', 'Done');
    } catch (err) {
        setStatus('error', err.message);
        console.error(err);
    } finally {
        btn.disabled = false;
    }
}

function displayResult(result) {
    const section = document.getElementById('results-section');
    section.style.display = '';
    section.classList.add('fade-in-up');

    const cfg = getSentimentConfig(result.sentiment);

    // Gauge arc: polarity -1 to 1 mapped to 0-100%
    const pct = ((result.polarity + 1) / 2) * 100;
    const circumference = 2 * Math.PI * 85; // ~534
    const offset = circumference - (pct / 100) * circumference;
    document.getElementById('gauge-arc').style.strokeDashoffset = offset;

    document.getElementById('gauge-emoji').textContent = cfg.emoji;
    document.getElementById('gauge-label').textContent = result.sentiment.replace('_', ' ');
    document.getElementById('gauge-label').style.color = cfg.color;
    document.getElementById('gauge-polarity').textContent = result.polarity.toFixed(2);

    document.getElementById('model-badge').textContent = result.model_used.toUpperCase();

    // Sarcasm Badge
    const sarcasmBadge = document.getElementById('sarcasm-badge');
    if (sarcasmBadge) {
        sarcasmBadge.style.display = result.is_sarcastic ? 'inline-block' : 'none';
    }

    // Emotions Container
    const emotionsContainer = document.getElementById('emotions-container');
    if (emotionsContainer) {
        emotionsContainer.innerHTML = '';
        if (result.metadata && result.metadata.emotions && Object.keys(result.metadata.emotions).length > 0) {
            emotionsContainer.style.display = 'flex';
            for (const [emotion, score] of Object.entries(result.metadata.emotions)) {
                const badge = document.createElement('span');
                badge.className = 'badge';
                badge.style.background = 'rgba(139, 92, 246, 0.2)';
                badge.style.border = '1px solid #8b5cf6';
                badge.style.color = '#c4b5fd';
                badge.textContent = `${emotion.toUpperCase()} ${(score * 100).toFixed(0)}%`;
                emotionsContainer.appendChild(badge);
            }
        } else {
            emotionsContainer.style.display = 'none';
        }
    }

    // Metrics
    document.getElementById('metric-polarity').textContent = result.polarity.toFixed(4);
    document.getElementById('metric-subjectivity').textContent = result.subjectivity.toFixed(4);
    document.getElementById('metric-confidence').textContent = (result.confidence * 100).toFixed(0) + '%';

    // Bars — polarity: map -1..1 to 0..100
    document.getElementById('polarity-bar').style.width = pct + '%';
    document.getElementById('subjectivity-bar').style.width = (result.subjectivity * 100) + '%';
    document.getElementById('confidence-bar').style.width = (result.confidence * 100) + '%';

    // Hide comparison if showing single result
    document.getElementById('comparison-section').style.display = 'none';

    // Scroll to result
    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ——— Compare Models ———

async function compareModels() {
    const text = document.getElementById('text-input').value.trim();
    if (!text) return;

    const btn = document.getElementById('compare-btn');
    btn.disabled = true;
    setStatus('loading', 'Comparing models...');

    try {
        const res = await fetch(`${API_BASE}/analyze/compare`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Comparison failed');
        }

        const data = await res.json();
        displayComparison(data);
        setStatus('ready', 'Done');
    } catch (err) {
        setStatus('error', err.message);
        console.error(err);
    } finally {
        btn.disabled = false;
    }
}

function displayComparison(data) {
    const section = document.getElementById('comparison-section');
    section.style.display = '';
    section.classList.add('fade-in-up');

    // Hide single result
    document.getElementById('results-section').style.display = 'none';

    // Consensus badge
    const badge = document.getElementById('consensus-badge');
    if (data.consensus) {
        const cfg = getSentimentConfig(data.consensus);
        badge.textContent = 'Consensus: ' + data.consensus.replace('_', ' ');
        badge.style.color = cfg.color;
        badge.style.borderColor = cfg.color + '33';
        badge.style.background = cfg.color + '1a';
    } else {
        badge.textContent = 'No consensus';
    }

    // Build comparison cards
    const grid = document.getElementById('comparison-grid');
    grid.innerHTML = '';
    const labels = [];
    const polarities = [];
    const confidences = [];
    const bgColors = [];

    for (const [name, result] of Object.entries(data.results)) {
        const cfg = getSentimentConfig(result.sentiment);
        labels.push(name.toUpperCase());
        polarities.push(result.polarity);
        confidences.push(result.confidence);
        bgColors.push(cfg.color + '80');

        const card = document.createElement('div');
        card.className = 'comparison-card scale-in';
        card.innerHTML = `
            <div class="model-name">${name}</div>
            <div class="emoji">${cfg.emoji}</div>
            <div class="sentiment-label" style="color:${cfg.color}">${result.sentiment.replace('_', ' ')}</div>
            <div class="polarity-value" style="color:${cfg.color}">${result.polarity.toFixed(4)}</div>
        `;
        grid.appendChild(card);
    }

    // Chart
    renderComparisonChart(labels, polarities, confidences, bgColors);

    section.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function renderComparisonChart(labels, polarities, confidences, bgColors) {
    const ctx = document.getElementById('comparison-chart').getContext('2d');

    if (comparisonChart) comparisonChart.destroy();

    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels,
            datasets: [
                {
                    label: 'Polarity',
                    data: polarities,
                    backgroundColor: bgColors,
                    borderColor: bgColors.map(c => c.replace('80', 'ff')),
                    borderWidth: 1,
                    borderRadius: 6,
                    barPercentage: 0.6,
                },
                {
                    label: 'Confidence',
                    data: confidences,
                    backgroundColor: 'rgba(129, 140, 248, 0.25)',
                    borderColor: 'rgba(129, 140, 248, 0.6)',
                    borderWidth: 1,
                    borderRadius: 6,
                    barPercentage: 0.6,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: { color: '#9ca3af', font: { family: "'Inter', sans-serif", size: 11 } },
                },
            },
            scales: {
                x: {
                    ticks: { color: '#6b7280', font: { family: "'Inter', sans-serif" } },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                },
                y: {
                    min: -1,
                    max: 1,
                    ticks: { color: '#6b7280', font: { family: "'Inter', sans-serif" } },
                    grid: { color: 'rgba(255,255,255,0.03)' },
                },
            },
        },
    });
}

// ——— CSV Upload ———

const dropZone = document.getElementById('drop-zone');
const csvInput = document.getElementById('csv-input');

dropZone.addEventListener('click', () => csvInput.click());
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    if (e.dataTransfer.files.length) handleCSV(e.dataTransfer.files[0]);
});
csvInput.addEventListener('change', (e) => {
    if (e.target.files.length) handleCSV(e.target.files[0]);
});

async function handleCSV(file) {
    if (!file.name.endsWith('.csv')) {
        setStatus('error', 'Please upload a .csv file');
        return;
    }

    const model = document.getElementById('model-select').value;
    const textColumn = document.getElementById('csv-column').value || 'text';
    const formData = new FormData();
    formData.append('file', file);

    setStatus('loading', 'Analyzing CSV...');

    try {
        const res = await fetch(`${API_BASE}/analyze/csv?text_column=${encodeURIComponent(textColumn)}&model=${model}`, {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'CSV analysis failed');
        }

        const data = await res.json();
        displayBatchResults(data);
        setStatus('ready', `Analyzed ${data.total} rows`);
    } catch (err) {
        setStatus('error', err.message);
        console.error(err);
    }
}

let currentBatchResults = null;

function displayBatchResults(data) {
    currentBatchResults = data;
    document.getElementById('batch-results').style.display = '';

    // Stats
    const stats = {};
    data.results.forEach(r => {
        const s = r.sentiment || 'UNKNOWN';
        stats[s] = (stats[s] || 0) + 1;
    });

    const statsEl = document.getElementById('batch-stats');
    statsEl.innerHTML = Object.entries(stats).map(([label, count]) => {
        const cfg = getSentimentConfig(label);
        return `<div class="batch-stat">
            <span class="dot" style="background:${cfg.color}"></span>
            ${label.replace('_', ' ')}: <strong>${count}</strong>
        </div>`;
    }).join('');

    // Table
    const tbody = document.getElementById('batch-table-body');
    tbody.innerHTML = data.results.map((r, i) => {
        const cfg = getSentimentConfig(r.sentiment);
        const text = r.text || r[Object.keys(r).find(k => typeof r[k] === 'string' && k !== 'sentiment' && k !== 'model_used')] || '';
        return `<tr>
            <td>${i + 1}</td>
            <td class="text-cell" title="${text}">${truncate(text, 60)}</td>
            <td><span class="sentiment-pill ${cfg.class}">${cfg.emoji} ${r.sentiment.replace('_', ' ')}</span></td>
            <td>${(r.polarity || 0).toFixed(4)}</td>
            <td>${((r.confidence || 0) * 100).toFixed(0)}%</td>
        </tr>`;
    }).join('');

    // Donut chart
    renderBatchChart(stats);
}

function renderBatchChart(stats) {
    const ctx = document.getElementById('batch-chart').getContext('2d');
    if (batchChart) batchChart.destroy();

    const labels = Object.keys(stats);
    const values = Object.values(stats);
    const colors = labels.map(l => getSentimentConfig(l).color);

    batchChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels.map(l => l.replace('_', ' ')),
            datasets: [{
                data: values,
                backgroundColor: colors.map(c => c + '99'),
                borderColor: colors,
                borderWidth: 2,
                hoverOffset: 8,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'right',
                    labels: { color: '#9ca3af', font: { family: "'Inter', sans-serif", size: 11 }, padding: 16 },
                },
            },
        },
    });
}

function exportResults() {
    if (!currentBatchResults) return;

    const rows = currentBatchResults.results;
    if (!rows.length) return;

    const headers = Object.keys(rows[0]);
    const csv = [
        headers.join(','),
        ...rows.map(r => headers.map(h => {
            const val = r[h];
            return typeof val === 'string' ? `"${val.replace(/"/g, '""')}"` : val;
        }).join(',')),
    ].join('\n');

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'sentiment_results.csv';
    a.click();
    URL.revokeObjectURL(url);
}

// ——— History ———

function addToHistory(result) {
    analysisHistory.unshift({
        text: result.text,
        sentiment: result.sentiment,
        polarity: result.polarity,
        model: result.model_used,
        time: new Date().toLocaleTimeString(),
    });

    // Keep last 50
    if (analysisHistory.length > 50) analysisHistory = analysisHistory.slice(0, 50);
    localStorage.setItem('sentimentHistory', JSON.stringify(analysisHistory));
    renderHistory();
}

function renderHistory() {
    const list = document.getElementById('history-list');

    if (!analysisHistory.length) {
        list.innerHTML = '<p class="empty-state">No analysis history yet. Start analyzing to see results here.</p>';
        return;
    }

    list.innerHTML = analysisHistory.map((item, i) => {
        const cfg = getSentimentConfig(item.sentiment);
        return `<div class="history-item" onclick="loadHistoryItem(${i})">
            <span class="history-emoji">${cfg.emoji}</span>
            <div class="history-content">
                <div class="history-text">${truncate(item.text, 60)}</div>
                <div class="history-meta">${item.model} • ${item.time}</div>
            </div>
            <span class="history-polarity" style="color:${cfg.color}">${item.polarity.toFixed(2)}</span>
        </div>`;
    }).join('');
}

function loadHistoryItem(index) {
    const item = analysisHistory[index];
    if (!item) return;
    document.getElementById('text-input').value = item.text;
    document.getElementById('char-count').textContent = item.text.length;
    analyzeText();
}

function clearHistory() {
    analysisHistory = [];
    localStorage.removeItem('sentimentHistory');
    renderHistory();
}

// ——— Clear ———

function clearAll() {
    document.getElementById('text-input').value = '';
    document.getElementById('char-count').textContent = '0';
    document.getElementById('results-section').style.display = 'none';
    document.getElementById('comparison-section').style.display = 'none';
    setStatus('ready');
}

// ——— Keyboard shortcut ———

document.getElementById('text-input').addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        analyzeText();
    }
});

// ——— Init ———

renderHistory();
