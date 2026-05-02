// static/js/ui.js

export function updateDateTime() {
    const now = new Date();
    document.getElementById('currentDate').textContent = now.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
    document.getElementById('currentTime').textContent = now.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

export function populateMetricsTable(metrics) {
    const tbody = document.querySelector('.performance-table tbody');
    if (!tbody) return;
    tbody.innerHTML = '';
    
    metrics.forEach(m => {
        const tr = document.createElement('tr');
        if(m.is_best) {
            tr.className = 'best-model-row';
            tr.setAttribute('data-bs-toggle', 'tooltip');
            tr.setAttribute('title', 'Dynamically selected based on lowest RMSE and highest R².');
            tr.innerHTML = `
                <td class="fw-bold text-success"><i class="fa-solid fa-crown me-1"></i> ${m.name}</td>
                <td class="text-center font-monospace">${m.mae}</td>
                <td class="text-center font-monospace">${m.rmse}</td>
                <td class="text-end pe-3 fw-bold text-success font-monospace">${m.r2} (Best)</td>
            `;
        } else {
            tr.innerHTML = `
                <td>${m.name}</td>
                <td class="text-center font-monospace">${m.mae}</td>
                <td class="text-center font-monospace">${m.rmse}</td>
                <td class="text-end pe-3 font-monospace">${m.r2}</td>
            `;
        }
        tbody.appendChild(tr);
    });
}

export function showSimulationLoading() {
    const displayElem = document.getElementById('simResultDisplay');
    if (displayElem) {
        displayElem.innerHTML = `<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Predicting...`;
    }
}

export function updateSimulationUI(data) {
    const displayElem = document.getElementById('simResultDisplay');
    if (!displayElem) return;

    if (!data.success) {
        displayElem.innerHTML = `<span class="text-danger">Error: ${data.error}</span>`;
        return;
    }

    const simAqi = data.aqi;
    const catText = data.category;
    const isBest = data.is_best;
    
    let colorClass = "";
    if(simAqi <= 50) colorClass = "text-success";
    else if(simAqi <= 100) colorClass = "text-warning";
    else if(simAqi <= 200) { colorClass = "text-orange"; displayElem.style.color = '#fd7e14'; }
    else if(simAqi <= 300) colorClass = "text-danger";
    else colorClass = "text-danger text-decoration-underline";

    const bestBadge = isBest 
        ? `<span class="text-success fw-bold">(Best Model)</span>` 
        : ``;
    
    const modelNameColor = isBest ? "text-success" : "text-secondary";

    displayElem.innerHTML = `
        <div class="mb-2">Predicted AQI: <span class="counter-val">${simAqi}</span> <span class="fs-6 fw-normal ${colorClass}">(${catText})</span></div>
        <div class="fs-6 ${modelNameColor}">Model Used: ${data.model_used} ${bestBadge}</div>
    `;
    
    // Trigger CSS transition
    displayElem.classList.remove('fade-in');
    void displayElem.offsetWidth; // trigger reflow
    displayElem.classList.add('fade-in');

    // Re-init tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
}

export function updateComparisonDashboard(comparisonData) {
    const tbody = document.getElementById('comparisonTableBody');
    const insightsPanel = document.getElementById('aiInsightsPanel');
    const rankingPanel = document.getElementById('aiRankingPanel');

    if (!tbody || !comparisonData.success) return;
    
    tbody.innerHTML = '';
    const bestModelName = comparisonData.best_model;
    const modelEntries = Object.entries(comparisonData.models);
    
    // Sort entries by rank for consistent display
    const sortedModels = modelEntries.sort((a, b) => a[1].rank - b[1].rank);

    for (const [modelName, m] of sortedModels) {
        const tr = document.createElement('tr');
        const isBest = modelName === bestModelName;
        const rankSuffix = m.rank === 1 ? 'st' : m.rank === 2 ? 'nd' : m.rank === 3 ? 'rd' : 'th';
        
        let modelDisplayName = modelName;
        let badgeHTML = '';

        if (modelName === "Hybrid Model") {
            modelDisplayName = `Hybrid Model (LR + GBR) <i class="fa-solid fa-circle-info ms-1 text-muted" data-bs-toggle="tooltip" title="Combines linear and non-linear models for better balance"></i>`;
        }

        if (isBest) {
            badgeHTML = `<span class="badge bg-success ms-2">🏆 Best Model</span>`;
        } else if (modelName === "Hybrid Model") {
            badgeHTML = `<span class="badge bg-warning text-dark ms-2 opacity-75">${m.rank}${rankSuffix} Best Model</span>`;
        } else {
            badgeHTML = `<span class="badge bg-secondary opacity-50 ms-2">${m.rank}${rankSuffix}</span>`;
        }
        
        const rowClass = isBest ? 'bg-success bg-opacity-10' : '';
        const nameColor = isBest ? 'text-success' : '';
        
        tr.className = rowClass;
        tr.innerHTML = `
            <td class="fw-bold ${nameColor}">${modelDisplayName} ${badgeHTML}</td>
            <td class="text-center fw-bold font-monospace">${m.aqi} <span class="fs-8 text-muted">(${m.category})</span></td>
            <td class="text-center ${isBest ? 'text-success fw-bold' : ''} font-monospace">${m.r2}</td>
            <td class="text-center font-monospace">${m.rmse}</td>
            <td class="text-center font-monospace">${m.mae}</td>
        `;
        tbody.appendChild(tr);
    }
    
    // Re-init tooltips
    const tooltipTriggerList = document.querySelectorAll('#comparisonTableBody [data-bs-toggle="tooltip"]');
    [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));

    // Populate AI Insights
    if (insightsPanel) {
        const bestModelData = comparisonData.models[bestModelName];
        let explanationText = "Linear models perform better on simple linear datasets.";
        if (["Random Forest", "Gradient Boosting", "AdaBoost"].includes(bestModelName)) {
            explanationText = "Tree-based models perform better due to non-linear AQI patterns and complex interactions between pollutants.";
        } else if (bestModelName === "Hybrid Model") {
            explanationText = "The Hybrid model provides the best balance by capturing both linear trends and non-linear outliers.";
        }

        insightsPanel.innerHTML = `
            <div class="mb-2"><span class="text-secondary">Best Model:</span> <strong class="text-success">${bestModelName}</strong></div>
            <div class="mb-2"><span class="text-secondary">Why:</span>
                <ul class="mb-1 ps-3 mt-1 text-muted">
                    <li>Highest R² (<span class="font-monospace text-light">${bestModelData.r2}</span>)</li>
                    <li>Lowest RMSE (<span class="font-monospace text-light">${bestModelData.rmse}</span>)</li>
                </ul>
            </div>
            <div class="fst-italic border-top border-secondary pt-2 mt-2 text-info opacity-75">
                <i class="fa-solid fa-quote-left me-1"></i> ${explanationText}
            </div>
        `;
    }

    // Populate Ranking
    if (rankingPanel) {
        rankingPanel.innerHTML = '';
        sortedModels.forEach(([modelName, m]) => {
            const li = document.createElement('li');
            li.className = "mb-1 pb-1 border-bottom border-secondary border-opacity-25";
            if (m.rank === 1) {
                li.innerHTML = `<strong class="text-success">${modelName} <i class="fa-solid fa-trophy ms-2 fs-8 text-warning"></i></strong>`;
            } else {
                li.innerHTML = `<span class="${modelName === 'Hybrid Model' ? 'text-light' : 'text-secondary'}">${modelName}</span>`;
            }
            rankingPanel.appendChild(li);
        });
    }
}
