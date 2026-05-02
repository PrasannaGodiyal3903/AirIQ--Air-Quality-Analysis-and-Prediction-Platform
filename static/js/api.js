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
