// static/js/commute.js

import { fetchCommuteScore } from './api.js';

document.addEventListener("DOMContentLoaded", async () => {
    const commuteForm = document.getElementById('commuteForm');
    const startSelect = document.getElementById('startCity');
    const endSelect = document.getElementById('endCity');
    const placeholder = document.getElementById('commuteResultPlaceholder');
    const resultCard = document.getElementById('commuteResultCard');

    // 1. Populate City Dropdowns
    await populateCities(startSelect, endSelect);

    // 2. Handle Form Submission
    commuteForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const start = startSelect.value;
        const end = endSelect.value;
        const mode = document.querySelector('input[name="travelMode"]:checked').value;

        // Show loading state
        const btn = document.getElementById('calcCommuteBtn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Calculating...';
        btn.disabled = true;

        const data = await fetchCommuteScore({ start, end, mode });

        btn.innerHTML = originalText;
        btn.disabled = false;

        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        updateCommuteUI(data);
    });
});

async function populateCities(...selects) {
    try {
        // Fetch cities from heatmap data endpoint
        const response = await fetch('/api/aqi-heatmap');
        const json = await response.json();
        const cities = [...new Set(json.data.map(item => item.city))].sort();

        selects.forEach(select => {
            if (!select) return;
            select.innerHTML = '<option value="" disabled selected>Select City</option>';
            cities.forEach(city => {
                const opt = document.createElement('option');
                opt.value = city;
                opt.textContent = city;
                select.appendChild(opt);
            });
        });
    } catch (err) {
        console.error("Error populating cities:", err);
    }
}

function updateCommuteUI(data) {
    const placeholder = document.getElementById('commuteResultPlaceholder');
    const resultCard = document.getElementById('commuteResultCard');
    
    placeholder.classList.add('d-none');
    resultCard.classList.remove('d-none');

    document.getElementById('resultAqi').textContent = data.exposure_aqi;
    document.getElementById('resultPm').textContent = `${data.exposure_pm} µg/m³`;
    document.getElementById('resultRoute').textContent = `${data.start} to ${data.end}`;
    document.getElementById('resultAdvice').textContent = data.advice;
    
    const statusElem = document.getElementById('resultStatus');
    statusElem.textContent = data.status.toUpperCase();
    statusElem.style.backgroundColor = data.color;
    statusElem.style.color = (data.status === 'Moderate') ? '#000' : '#fff';

    // Update styling based on status
    resultCard.style.borderColor = `${data.color}44`;
    resultCard.style.backgroundColor = `${data.color}11`;
    
    const iconBg = document.getElementById('statusIconBg');
    iconBg.style.color = data.color;

    // Transition effect
    resultCard.classList.remove('fade-in');
    void resultCard.offsetWidth;
    resultCard.classList.add('fade-in');
}
