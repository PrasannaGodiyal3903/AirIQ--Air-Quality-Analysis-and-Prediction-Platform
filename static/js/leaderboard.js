// static/js/leaderboard.js

import { fetchLeaderboard, fetchCityComparison } from './api.js';

let comparisonChart = null;

document.addEventListener("DOMContentLoaded", async () => {
    const cleanestList = document.getElementById('cleanestList');
    const pollutedList = document.getElementById('pollutedList');
    const city1Select = document.getElementById('compareCity1');
    const city2Select = document.getElementById('compareCity2');

    // 1. Fetch and render rankings
    const leaderboardData = await fetchLeaderboard();
    renderRankingList(cleanestList, leaderboardData.cleanest, 'success');
    renderRankingList(pollutedList, leaderboardData.polluted, 'danger');

    // 2. Populate comparison dropdowns
    await populateComparisonCities(city1Select, city2Select);

    // 3. Setup comparison change listeners
    [city1Select, city2Select].forEach(select => {
        select.addEventListener('change', updateComparisonBattle);
    });

    // Initial comparison
    if (city1Select.options.length > 1 && city2Select.options.length > 2) {
        city1Select.selectedIndex = 1;
        city2Select.selectedIndex = 2;
        updateComparisonBattle();
    }
});

function renderRankingList(container, data, type) {
    if (!container) return;
    container.innerHTML = '';
    
    data.forEach((item, index) => {
        const div = document.createElement('div');
        div.className = `d-flex align-items-center justify-content-between p-2 mb-2 rounded-3 bg-${type} bg-opacity-10 border border-${type} border-opacity-10 hover-scale`;
        div.innerHTML = `
            <div class="d-flex align-items-center">
                <span class="fw-bold text-${type} me-2 small">#${index + 1}</span>
                <span class="text-white small">${item.city}</span>
            </div>
            <span class="badge bg-${type} bg-opacity-25 text-${type} font-monospace fs-8">${Math.round(item.AQI)}</span>
        `;
        container.appendChild(div);
    });
}

async function populateComparisonCities(...selects) {
    try {
        const response = await fetch('/api/aqi-heatmap');
        const json = await response.json();
        const cities = [...new Set(json.data.map(item => item.city))].sort();

        selects.forEach(select => {
            if (!select) return;
            select.innerHTML = '<option value="" disabled>Select City</option>';
            cities.forEach(city => {
                const opt = document.createElement('option');
                opt.value = city;
                opt.textContent = city;
                select.appendChild(opt);
            });
        });
    } catch (err) {
        console.error("Error populating comparison cities:", err);
    }
}

async function updateComparisonBattle() {
    const city1 = document.getElementById('compareCity1').value;
    const city2 = document.getElementById('compareCity2').value;

    if (!city1 || !city2) return;

    const data = await fetchCityComparison(city1, city2);
    if (data.error) return;

    renderComparisonChart(data);
}

function renderComparisonChart(data) {
    const ctx = document.getElementById('comparisonBattleChart').getContext('2d');
    
    const labels = ['PM2.5', 'PM10', 'NO2', 'CO', 'OZONE', 'AQI'];
    const city1Data = labels.map(l => data.city1.data[l]);
    const city2Data = labels.map(l => data.city2.data[l]);

    if (comparisonChart) {
        comparisonChart.destroy();
    }

    comparisonChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: data.city1.name,
                    data: city1Data,
                    backgroundColor: 'rgba(78, 115, 223, 0.7)',
                    borderColor: '#4e73df',
                    borderWidth: 1
                },
                {
                    label: data.city2.name,
                    data: city2Data,
                    backgroundColor: 'rgba(244, 67, 54, 0.7)',
                    borderColor: '#f44336',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#858796' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#858796' }
                }
            },
            plugins: {
                legend: {
                    labels: { color: '#e4e6eb', boxWidth: 12 }
                }
            }
        }
    });
}
