// static/js/map.js

let mapInstance = null;
let markersLayer = null;

// AQI color scale matching Indian CPCB standards
function getAqiColor(aqi) {
    if (aqi <= 50)  return '#00e400';  // Good
    if (aqi <= 100) return '#92d050';  // Satisfactory
    if (aqi <= 200) return '#ffff00';  // Moderate
    if (aqi <= 300) return '#ff7e00';  // Poor
    if (aqi <= 400) return '#ff0000';  // Very Poor
    return '#7e0023';                  // Severe
}

function makeIcon(aqi, isActual) {
    const color = getAqiColor(aqi);
    const shape = isActual
        ? `border-radius:50%;`           // Circle = Actual
        : `border-radius:3px;`;          // Square = Predicted
    return L.divIcon({
        className: '',
        html: `<div style="background:${color};width:13px;height:13px;${shape}border:2px solid rgba(255,255,255,0.8);box-shadow:0 0 6px ${color};"></div>`,
        iconSize: [13, 13],
        iconAnchor: [6, 6]
    });
}

export function initMap() {
    const mapContainer = document.getElementById('aqiMap');
    if (!mapContainer) return;

    mapInstance = L.map('aqiMap').setView([20.5937, 78.9629], 5);

    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 20
    }).addTo(mapInstance);

    markersLayer = L.layerGroup().addTo(mapInstance);
}

export function renderMapMarkers(mapData) {
    if (!mapInstance || !mapData || !mapData.length) return;

    // Clear previous markers
    markersLayer.clearLayers();

    mapData.forEach(loc => {
        const actualPos  = [loc.lat, loc.lon - 0.08];
        const predPos    = [loc.lat, loc.lon + 0.08];

        // Actual Marker (circle)
        L.marker(actualPos, { icon: makeIcon(loc.actual_aqi, true) })
            .addTo(markersLayer)
            .bindPopup(`
                <div style="font-family:sans-serif;min-width:160px;">
                    <strong style="font-size:13px;">📍 ${loc.city}</strong>
                    <hr style="margin:4px 0;">
                    <div style="color:#333;">
                        <b>● Actual AQI:</b> ${loc.actual_aqi}<br>
                        <b>Category:</b> ${loc.actual_category}
                    </div>
                </div>
            `);

        // Predicted Marker (square)
        L.marker(predPos, { icon: makeIcon(loc.predicted_aqi, false) })
            .addTo(markersLayer)
            .bindPopup(`
                <div style="font-family:sans-serif;min-width:160px;">
                    <strong style="font-size:13px;">🤖 ${loc.city}</strong>
                    <hr style="margin:4px 0;">
                    <div style="color:#333;">
                        <b>■ Predicted AQI:</b> ${loc.predicted_aqi}<br>
                        <b>Category:</b> ${loc.predicted_category}
                    </div>
                </div>
            `);

        // Dashed line connecting actual vs predicted for same city
        L.polyline([actualPos, predPos], {
            color: getAqiColor(loc.actual_aqi),
            weight: 1.5,
            dashArray: '5,4',
            opacity: 0.5
        }).addTo(markersLayer);
    });
}
