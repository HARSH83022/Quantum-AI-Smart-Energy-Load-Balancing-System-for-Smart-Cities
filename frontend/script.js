// Quantum AI Smart Grid — Frontend Controller
const API_BASE = 'http://localhost:8000';
const apiBadge = document.getElementById('api-badge');
const clockElement = document.getElementById('clock');
const dateInput = document.getElementById('date');
const predictBtn = document.getElementById('predictBtn');
const btnText = document.getElementById('btnText');
const btnLoader = document.getElementById('btnLoader');
const alertBar = document.getElementById('alert-bar');
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
const qBalance = document.getElementById('q-balance');
const qQubits = document.getElementById('q-qubits');
const qEnergy = document.getElementById('q-energy');
const qTime = document.getElementById('q-time');
const quantumMethod = document.getElementById('quantum-method');
const transformerGrid = document.getElementById('transformer-grid');

document.addEventListener('DOMContentLoaded', () => {
    checkAPIStatus(); updateClock(); setInterval(updateClock, 1000);
});

async function checkAPIStatus() {
    try {
        const r = await fetch(`${API_BASE}/health`);
        if (r.ok) { apiBadge.className='badge badge-connected'; apiBadge.innerHTML='<span class="dot"></span> Connected'; }
        else throw new Error();
    } catch(e) { apiBadge.className='badge badge-error'; apiBadge.innerHTML='<span class="dot"></span> Disconnected'; }
}

function updateClock() {
    clockElement.textContent = new Date().toLocaleTimeString('en-US',{hour12:false,hour:'2-digit',minute:'2-digit',second:'2-digit'});
}

async function predict() {
    const date = dateInput.value;
    if (!date) { showAlert('Please select a date','error'); return; }
    predictBtn.disabled = true; btnText.textContent = '⚛️ Running Quantum Prediction...'; btnLoader.classList.remove('hidden');
    try {
        const response = await fetch(`${API_BASE}/predict`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({date}) });
        const result = await response.json();
        if (result.error) { showAlert(result.error,'error'); }
        else { updateCards(result); updateQuantumPanel(result.quantum_optimization); updateZones(result.quantum_optimization); updateChart(result); showAlert('✅ Quantum prediction & optimization complete!','success'); }
    } catch(e) { showAlert('Failed to connect to API','error'); }
    finally { predictBtn.disabled=false; btnText.textContent='⚛️ Run Quantum Prediction'; btnLoader.classList.add('hidden'); }
}

function updateCards(d) {
    loadCard.textContent = d.predicted_peak_load_MW || '--';
    timeCard.textContent = d.predicted_peak_hour || '--';
    tempCard.textContent = d.temperature || '--';
    humidityCard.textContent = d.humidity || '--';
    rainCard.textContent = d.rain || '--';
    dayCard.textContent = d.day_name || '--';
    holidayCard.textContent = d.is_holiday ? 'Yes' : 'No';
    weekendCard.textContent = d.is_weekend ? 'Yes' : 'No';
    transformerCard.textContent = d.transformer_status || '--';
    riskCard.textContent = d.outage_risk || '--';
}

function getUtilColor(p) {
    if(p>95) return 'linear-gradient(90deg,#ff1744,#ff5252)';
    if(p>85) return 'linear-gradient(90deg,#ff6d00,#ffab00)';
    if(p>70) return 'linear-gradient(90deg,#ff9100,#ffc107)';
    if(p>40) return 'linear-gradient(90deg,#00e676,#00e5ff)';
    return 'linear-gradient(90deg,#ffd54f,#ffee58)';
}
function getUtilClass(p) {
    if(p>95) return 'status-critical'; if(p>85) return 'status-overload';
    if(p>70) return 'status-high'; if(p>40) return 'status-optimal'; return 'status-low';
}
function getBalClass(s) { if(s>=0.85) return 'balance-good'; if(s>=0.65) return 'balance-moderate'; return 'balance-poor'; }

function updateQuantumPanel(q) {
    if (!q || q.status==='error') { qBalance.textContent='--'; qQubits.textContent='--'; qEnergy.textContent='--'; qTime.textContent='--'; return; }
    const bp = (q.balance_score*100).toFixed(1);
    qBalance.textContent = bp+'%'; qBalance.className = 'quantum-stat-value '+getBalClass(q.balance_score);
    qQubits.textContent = q.n_qubits||'--';
    qEnergy.textContent = q.quantum_energy!=null ? q.quantum_energy.toFixed(2) : '--';
    qTime.textContent = q.execution_time_sec ? q.execution_time_sec+'s' : '--';
    quantumMethod.textContent = q.optimization_method || 'QAOA';
    if (q.transformers && q.transformers.length>0) {
        transformerGrid.innerHTML = '';
        q.transformers.forEach((t,i) => {
            const c = document.createElement('div'); c.className='transformer-card glass'; c.style.animationDelay=i*0.1+'s';
            c.innerHTML = `<div class="t-card-header"><span class="t-card-name">${t.name}</span><span class="t-card-status ${getUtilClass(t.utilization_pct)}">${t.status}</span></div>
            <div class="t-card-load"><span class="t-load-value">${t.assigned_load_mw}</span><span class="t-load-unit"> MW</span><span class="t-load-cap"> / ${t.capacity_mw} MW</span></div>
            <div class="t-bar-container"><div class="t-bar-fill" style="width:${Math.min(t.utilization_pct,100)}%;background:${getUtilColor(t.utilization_pct)}"></div></div>
            <div class="t-card-footer"><span class="t-util-pct">${t.utilization_pct}%</span><span class="t-blocks">${t.assigned_blocks} blocks</span></div>`;
            transformerGrid.appendChild(c);
        });
        const s = document.createElement('div'); s.className='transformer-card glass summary-card';
        s.innerHTML = `<div class="t-card-header"><span class="t-card-name">⚛️ Summary</span><span class="t-card-status status-info">${q.status==='success'?'Quantum ✓':'Fallback'}</span></div>
        <div class="summary-stats"><div class="summary-item"><span class="summary-label">Total</span><span class="summary-value">${q.total_assigned_mw} MW</span></div>
        <div class="summary-item"><span class="summary-label">Target</span><span class="summary-value">${q.predicted_load_mw} MW</span></div>
        <div class="summary-item"><span class="summary-label">Balance</span><span class="summary-value ${getBalClass(q.balance_score)}">${bp}%</span></div></div>`;
        transformerGrid.appendChild(s);
    }
}

function updateZones(q) {
    const z = document.getElementById('zones');
    if (!q||!q.transformers) { z.innerHTML='<p class="placeholder-text">No quantum data</p>'; return; }
    const mx = Math.max(...q.transformers.map(t=>t.capacity_mw));
    z.innerHTML = q.transformers.map(t => `<div class="zone-item"><div class="zone-header"><span class="zone-name">${t.name}</span><span class="zone-mw">${t.assigned_load_mw} MW</span></div><div class="zone-bar-bg"><div class="zone-bar-fill" style="width:${(t.assigned_load_mw/mx)*100}%;background:${getUtilColor(t.utilization_pct)}"></div></div></div>`).join('');
}

function updateChart(data) {
    const ctx = document.getElementById('chart').getContext('2d');
    if (window.myChart) window.myChart.destroy();
    const pk = data.predicted_peak_load_MW, ph = data.predicted_peak_hour;
    const hrs = Array.from({length:24},(_,i)=>`${String(i).padStart(2,'0')}:00`);
    const curve = hrs.map((_,i)=>{ const d=Math.abs(i-ph); return Math.round(pk*(0.35+0.65*Math.exp(-0.08*d*d))); });
    window.myChart = new Chart(ctx, {
        type:'line', data:{ labels:hrs, datasets:[
            {label:'Predicted Load (MW)',data:curve,borderColor:'#00e5ff',backgroundColor:'rgba(0,229,255,0.08)',tension:0.4,fill:true,borderWidth:2.5,pointRadius:0},
            {label:'Grid Capacity (MW)',data:hrs.map(()=>7000),borderColor:'rgba(255,82,82,0.6)',borderDash:[8,4],borderWidth:1.5,pointRadius:0,fill:false}
        ]}, options:{responsive:true,maintainAspectRatio:false,scales:{
            y:{beginAtZero:true,grid:{color:'rgba(255,255,255,0.06)'},ticks:{color:'#7a9cc0'}},
            x:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#7a9cc0',maxTicksLimit:12}}
        },plugins:{legend:{labels:{color:'#e0eaff'}},tooltip:{backgroundColor:'rgba(6,13,31,0.9)',borderColor:'rgba(0,229,255,0.3)',borderWidth:1}}}
    });
}

function showAlert(msg, type) {
    alertBar.textContent = msg; alertBar.className = `alert-bar ${type}`; alertBar.classList.remove('hidden');
    setTimeout(()=>alertBar.classList.add('hidden'), 5000);
}
