// static/js/comparison.js

import { populateCities } from './ui.js';

let battleChart = null;

document.addEventListener('DOMContentLoaded', async () => {
    const city1Select = document.getElementById('compareCity1');
    const city2Select = document.getElementById('compareCity2');

    if (!city1Select || !city2Select) return;

    // 1. Populate city dropdowns
    await populateCities(city1Select, city2Select);

    // 2. Event Listeners for Battle
    city1Select.addEventListener('change', updateBattle);
    city2Select.addEventListener('change', updateBattle);
});

async function updateBattle() {
    const city1 = document.getElementById('compareCity1').value;
    const city2 = document.getElementById('compareCity2').value;

    if (!city1 || !city2) return;

    try {
        const response = await fetch('/api/compare-cities', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ city1, city2 })
        });
        const data = await response.json();

        if (data.error) {
            console.error(data.error);
            return;
        }

        renderBattleChart(data);
    } catch (err) {
        console.error("Battle failed:", err);
    }
}

function renderBattleChart(data) {
    const ctx = document.getElementById('comparisonBattleChart').getContext('2d');
    if (!ctx) return;

    if (battleChart) battleChart.destroy();

    const pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'OZONE', 'AQI'];
    
    battleChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: pollutants,
            datasets: [
                {
                    label: data.city1.name,
                    data: pollutants.map(p => data.city1.data[p]),
                    backgroundColor: 'rgba(79, 172, 254, 0.8)',
                    borderColor: '#4facfe',
                    borderWidth: 1,
                    borderRadius: 5
                },
                {
                    label: data.city2.name,
                    data: pollutants.map(p => data.city2.data[p]),
                    backgroundColor: 'rgba(255, 107, 107, 0.8)',
                    borderColor: '#ff6b6b',
                    borderWidth: 1,
                    borderRadius: 5
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#fff' } }
            },
            scales: {
                y: { 
                    beginAtZero: true,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#8b949e' }
                },
                x: { 
                    grid: { display: false },
                    ticks: { color: '#8b949e' }
                }
            }
        }
    });
}
