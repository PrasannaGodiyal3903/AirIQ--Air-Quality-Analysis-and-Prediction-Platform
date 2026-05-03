// static/js/api.js

export async function fetchMetrics() {
    try {
        const response = await fetch('/api/metrics');
        if (!response.ok) throw new Error('Failed to fetch metrics');
        return await response.json();
    } catch (error) {
        console.error(error);
        return { error: error.message };
    }
}

export async function simulateAQI(payload, modelName) {
    try {
        const payloadWithModel = { ...payload, model: modelName };
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payloadWithModel)
        });
        if (!response.ok) throw new Error('Failed to predict AQI');
        return await response.json();
    } catch (error) {
        console.error(error);
        return { error: error.message };
    }
}

export async function compareModels(payload) {
    try {
        const response = await fetch('/api/compare-models', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) throw new Error('Failed to compare models');
        return await response.json();
    } catch (error) {
        console.error(error);
        return { error: error.message };
    }
}

export async function fetchMapData() {
    try {
        const response = await fetch('/api/aqi-heatmap');
        if (!response.ok) throw new Error('Failed to fetch heatmap data');
        const json = await response.json();
        // Return the data array directly for map rendering
        return json.data || [];
    } catch (error) {
        console.error(error);
        return [];
    }
}

export async function fetchCommuteScore(payload) {
    try {
        const response = await fetch('/api/commute-score', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!response.ok) throw new Error('Failed to calculate commute score');
        return await response.json();
    } catch (error) {
        console.error(error);
        return { error: error.message };
    }
}

export async function fetchLeaderboard() {
    try {
        const response = await fetch('/api/leaderboard');
        if (!response.ok) throw new Error('Failed to fetch leaderboard');
        return await response.json();
    } catch (error) {
        console.error(error);
        return { cleanest: [], polluted: [] };
    }
}

export async function fetchCityComparison(city1, city2) {
    try {
        const response = await fetch('/api/compare-cities', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ city1, city2 })
        });
        if (!response.ok) throw new Error('Failed to fetch city comparison');
        return await response.json();
    } catch (error) {
        console.error(error);
        return { error: error.message };
    }
}
