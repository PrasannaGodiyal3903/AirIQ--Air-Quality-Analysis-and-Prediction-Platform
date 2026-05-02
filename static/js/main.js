// static/js/main.js

import { fetchMetrics, simulateAQI, compareModels, fetchMapData } from './api.js';
import { renderScatterChart, renderPollutantChart, renderComparisonChart } from './charts.js';
import { initMap, renderMapMarkers } from './map.js';
import { updateDateTime, populateMetricsTable, updateSimulationUI, showSimulationLoading, updateComparisonDashboard } from './ui.js';

let debounceTimer;

document.addEventListener("DOMContentLoaded", async () => {
    // 1. Initialize Date/Time in Navbar
    setInterval(updateDateTime, 1000);
    updateDateTime();

    // 2. Initialize Charts & Maps
    renderPollutantChart();
    initMap();

    // 3. Fetch Initial Data (Metrics & Map)
    loadDashboardData();

    // 4. Setup Simulation Event Listeners
    setupSimulationPanel();
});

async function loadDashboardData() {
    try {
        const [metricsData, mapData] = await Promise.all([
            fetchMetrics(),
            fetchMapData()
        ]);

        if (!metricsData.error) {
            populateMetricsTable(metricsData.metrics);
            renderScatterChart(metricsData.scatter);
        }

        if (!mapData.error) {
            renderMapMarkers(mapData.data);
        }
    } catch (err) {
        console.error("Error loading dashboard data:", err);
    }
}

function setupSimulationPanel() {
    const sliders = ['simCO', 'simNO2', 'simPM10', 'simOzone'];
    
    // Setup slider value displays
    sliders.forEach(id => {
        const slider = document.getElementById(id);
        if(!slider) return;
        
        slider.addEventListener('input', (e) => {
            const valSpan = id.replace('sim', '').toLowerCase() + 'Val';
            document.getElementById(valSpan).textContent = e.target.value;
            
            // Trigger Debounced API call
            triggerSimulation();
        });
    });

    // Initial simulation run
    triggerSimulation();
}

function triggerSimulation() {
    clearTimeout(debounceTimer);
    showSimulationLoading();

    debounceTimer = setTimeout(async () => {
        const payload = {
            PM10: parseInt(document.getElementById('simPM10').value),
            NO2: parseInt(document.getElementById('simNO2').value),
            OZONE: parseInt(document.getElementById('simOzone').value),
            CO: parseFloat(document.getElementById('simCO').value)
        };

        const selectedModel = "Hybrid Model"; // Locked to production model

        // Parallel requests for single prediction AND model comparison
        const [predictionData, comparisonData] = await Promise.all([
            simulateAQI(payload, selectedModel),
            compareModels(payload)
        ]);

        updateSimulationUI(predictionData);
        
        if (!comparisonData.error) {
            updateComparisonDashboard(comparisonData);
        }

    }, 300); // 300ms debounce
}
