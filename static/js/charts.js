// static/js/charts.js

// Set global Chart.js defaults
Chart.defaults.color = '#8b949e';
Chart.defaults.font.family = "'Poppins', sans-serif";

let scatterChartInstance = null;
let barChartInstance = null;
let comparisonChartInstance = null;

export function renderScatterChart(scatterData) {
    const ctx = document.getElementById('scatterChart')?.getContext('2d');
    if (!ctx) return;
    
    if (scatterChartInstance) scatterChartInstance.destroy();

    scatterChartInstance = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Real Predictions (Hybrid)',
                    data: scatterData,
                    backgroundColor: '#BA68C8',
                    borderColor: 'rgba(186, 104, 200, 0.8)',
                    pointRadius: 5,
                    pointHoverRadius: 7
                },
                {
                    label: 'Perfect Prediction',
                    data: [{x: 0, y: 0}, {x: 450, y: 450}],
                    type: 'line',
                    borderColor: '#4CAF50',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#c9d1d9' } },
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            if(ctx.datasetIndex === 1) return 'Perfect Prediction';
                            return `Actual: ${ctx.raw.x}, Predicted: ${ctx.raw.y.toFixed(0)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear', position: 'bottom',
                    title: { display: true, text: 'Actual AQI (Ground Truth)', color: '#8b949e' },
                    grid: { color: 'rgba(255,255,255,0.05)' }, max: 450
                },
                y: {
                    title: { display: true, text: 'Predicted AQI', color: '#8b949e' },
                    grid: { color: 'rgba(255,255,255,0.05)' }, max: 450
                }
            }
        }
    });
}

export function renderPollutantChart() {
    const ctx = document.getElementById('barChart')?.getContext('2d');
    if (!ctx) return;

    if (barChartInstance) barChartInstance.destroy();

    barChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['CO', 'NO2', 'OZONE', 'PM10'],
            datasets: [{
                label: 'Relative Contribution',
                data: [1.5, 45, 35, 180],
                backgroundColor: ['rgba(255, 193, 7, 0.8)', 'rgba(76, 175, 80, 0.8)', 'rgba(33, 150, 243, 0.8)', 'rgba(244, 67, 54, 0.8)'],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { grid: { display: false } }
            }
        }
    });
}

export function renderComparisonChart(predictions) {
    const ctx = document.getElementById('comparisonChart')?.getContext('2d');
    if (!ctx) return;

    if (comparisonChartInstance) comparisonChartInstance.destroy();

    const labels = predictions.map(p => p.model_name);
    const data = predictions.map(p => p.aqi);
    
    // Distinct color for Hybrid Model
    const backgroundColors = labels.map(label => 
        label === 'Hybrid Model' ? 'rgba(255, 193, 7, 0.9)' : 'rgba(79, 172, 254, 0.5)'
    );
    const borderColors = labels.map(label => 
        label === 'Hybrid Model' ? 'rgba(255, 193, 7, 1)' : 'rgba(79, 172, 254, 1)'
    );

    comparisonChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Predicted AQI',
                data: data,
                backgroundColor: backgroundColors,
                borderColor: borderColors,
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(ctx) {
                            return `AQI: ${ctx.raw} (${predictions[ctx.dataIndex].category})`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#8b949e' }
                },
                x: {
                    grid: { display: false },
                    ticks: { color: '#8b949e', font: { size: 10 } }
                }
            }
        }
    });
}
