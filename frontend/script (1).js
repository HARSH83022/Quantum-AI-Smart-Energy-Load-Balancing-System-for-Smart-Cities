// Frontend JavaScript for Smart Grid AI

// API Base URL
const API_BASE = 'http://localhost:8000';

// DOM Elements
const apiBadge = document.getElementById('api-badge');
const clockElement = document.getElementById('clock');
const dateInput = document.getElementById('date');
const predictBtn = document.getElementById('predictBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');
const alertBar = document.getElementById('alert-bar');

// Card Elements
const loadCard = document.getElementById('load');
const timeCard = document.getElementById('time');
const tempCard = document.getElementById('temp');
const humidityCard = document.getElementById('humidity');
const rainCard = document.getElementById('rain');
const dayCard = document.getElementById('day');
const holidayCard = document.getElementById('holiday');
const weekendCard = document.getElementById('weekend');
const transformerCard = document.getElementById('transformer');
const riskCard = document.getElementById('risk');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    checkAPIStatus();
    updateClock();
    setInterval(updateClock, 1000);
});

// Check API Status
async function checkAPIStatus() {
    try {
        const response = await fetch(`${API_BASE}/health`);
        if (response.ok) {
            apiBadge.className = 'badge badge-connected';
            apiBadge.innerHTML = '<span class="dot"></span> Connected';
        } else {
            throw new Error();
        }
    } catch (error) {
        apiBadge.className = 'badge badge-error';
        apiBadge.innerHTML = '<span class="dot"></span> Disconnected';
    }
}

// Update Live Clock
function updateClock() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', {
        hour12: false,
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
    });
    clockElement.textContent = timeString;
}

// Predict Function
async function predict() {
    const date = dateInput.value;
    if (!date) {
        showAlert('Please select a date', 'error');
        return;
    }

    // Show loading
    predictBtn.disabled = true;
    btnText.textContent = 'Predicting...';
    btnLoader.classList.remove('hidden');

    try {
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ date: date })
        });

        const result = await response.json();

        if (result.error) {
            showAlert(result.error, 'error');
        } else {
            updateCards(result);
            showAlert('Prediction successful!', 'success');
        }
    } catch (error) {
        showAlert('Failed to connect to API', 'error');
    } finally {
        // Hide loading
        predictBtn.disabled = false;
        btnText.textContent = 'Run Prediction';
        btnLoader.classList.add('hidden');
    }
}

// Update Cards with Prediction Data
function updateCards(data) {
    loadCard.textContent = data.predicted_peak_load_MW || '--';
    timeCard.textContent = data.predicted_peak_hour || '--';
    tempCard.textContent = data.temperature || '--';
    humidityCard.textContent = data.humidity || '--';
    rainCard.textContent = data.rain || '--';
    dayCard.textContent = data.day_name || '--';
    holidayCard.textContent = data.is_holiday ? 'Yes' : 'No';
    weekendCard.textContent = data.is_weekend ? 'Yes' : 'No';
    transformerCard.textContent = data.transformer_status || '--';
    riskCard.textContent = data.outage_risk || '--';

    // Update chart with prediction
    updateChart(data);
}

// Update Chart
function updateChart(data) {
    const ctx = document.getElementById('chart').getContext('2d');

    // Destroy existing chart if any
    if (window.myChart) {
        window.myChart.destroy();
    }

    // Create new chart
    window.myChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['00:00', '06:00', '12:00', '18:00', '24:00'],
            datasets: [{
                label: 'Predicted Load (MW)',
                data: [data.predicted_peak_load_MW * 0.5, data.predicted_peak_load_MW * 0.7, data.predicted_peak_load_MW, data.predicted_peak_load_MW * 0.8, data.predicted_peak_load_MW * 0.6],
                borderColor: '#00ff88',
                backgroundColor: 'rgba(0, 255, 136, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#ffffff'
                    }
                },
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: '#ffffff'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#ffffff'
                    }
                }
            }
        }
    });
}

// Show Alert
function showAlert(message, type) {
    alertBar.textContent = message;
    alertBar.className = `alert-bar ${type}`;
    alertBar.classList.remove('hidden');

    setTimeout(() => {
        alertBar.classList.add('hidden');
    }, 5000);
}
