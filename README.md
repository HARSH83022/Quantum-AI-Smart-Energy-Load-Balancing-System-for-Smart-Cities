# ⚡ Quantum-AI Smart Energy Load Balancing System  
### 🧠 Deep Learning × ⚛️ Quantum Optimization × 📊 Risk Intelligence

<p align="center">
  <b>Forecast Volatility • Optimize Dispatch • Minimize Risk</b><br>
  Production-Ready Hybrid Quantum-Classical Smart Grid Platform
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue"/>
  <img src="https://img.shields.io/badge/FastAPI-Production-green"/>
  <img src="https://img.shields.io/badge/PyTorch-LSTM-red"/>
  <img src="https://img.shields.io/badge/Qiskit-QAOA-purple"/>
  <img src="https://img.shields.io/badge/PostgreSQL-15+-blue"/>
  <img src="https://img.shields.io/badge/License-MIT-lightgrey"/>
</p>

---

# 🌍 Overview

Modern smart grids face:

- ⚡ Demand volatility  
- 🌬 Renewable intermittency  
- 📉 Risk exposure  
- 💸 Economic inefficiencies  
- 🛑 Combinatorial optimization challenges  

This system delivers a **hybrid AI + Quantum optimization pipeline** to forecast demand, generate uncertainty scenarios, and compute risk-aware optimal dispatch decisions.

---

# 🏗️ System Architecture

```mermaid
flowchart LR
A[Historical Grid Data] --> B[Preprocessing]
B --> C[LSTM Forecasting]
C --> D[Scenario Generation]
D --> E[Robust QUBO Formulation]
E --> F[QAOA Optimization]
F --> G[CVaR Risk Analysis]
G --> H[Optimized Load Dispatch]
```

---

# ✨ Core Features

## 🔮 1. LSTM Demand Forecasting
- Time-series normalization
- Sequence generation
- Multi-step forecasting
- PyTorch-based training pipeline

## ⚛️ 2. Quantum Optimization Engine
- Load balancing as QUBO
- QAOA implementation (Qiskit)
- Parameter warm-starting
- Convergence monitoring

## 🎲 3. Scenario Simulation
- Probabilistic demand generation
- Monte Carlo stress testing
- Renewable uncertainty modeling

## 📉 4. Risk-Aware Optimization
- CVaR (Conditional Value at Risk)
- Robust QUBO penalties
- Tail-risk minimization

## 🌐 5. REST API Layer
- FastAPI backend
- JWT authentication
- Modular endpoints
- Production-ready deployment

---

# 📂 Project Structure

```bash
quantum-energy-system/
│
├── src/
│   ├── data_sources/          # CSV & IoT loaders
│   ├── preprocessing/         # Cleaning & scaling
│   ├── forecasting/           # LSTM models
│   ├── frequency_analysis/    # FFT & QFT
│   ├── scenario_generation/   # Probabilistic scenarios
│   ├── monte_carlo/           # Stress testing
│   ├── optimization/
│   │   ├── qubo/
│   │   ├── robust_qubo/
│   │   ├── qaoa/
│   │   └── risk_analysis/
│   ├── api/
│   └── database/
│
├── tests/
├── docs/
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

# 🚀 Installation

## 🔹 Prerequisites

- Python 3.11+
- PostgreSQL 15+
- (Optional) IBM Quantum API Key

---

## 🔹 Local Setup

```bash
git clone <repository-url>
cd quantum-energy-system
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn src.main:app --reload
```

API available at:

```
http://localhost:8000
```

---

## 🐳 Docker Setup

```bash
docker-compose up --build
```

---

# 🌐 API Endpoints

## 🟢 Core

| Method | Endpoint | Description |
|--------|----------|------------|
| GET | `/` | Root |
| GET | `/health` | Health check |
| POST | `/api/forecast` | Generate forecast |
| POST | `/api/optimize` | Run QAOA optimization |
| GET | `/api/results` | Retrieve results |

## 🔬 Research Extensions

| Method | Endpoint | Purpose |
|--------|----------|--------|
| POST | `/api/scenarios/generate` | Generate uncertainty scenarios |
| POST | `/api/risk/analyze` | CVaR analysis |
| GET | `/api/frequency/features` | FFT vs QFT comparison |
| POST | `/api/optimize/robust` | Robust optimization |

---

# ⚙️ Configuration

## Required

```env
DATABASE_URL=postgresql://...
JWT_SECRET=your_secret_key
```

## Optional

```env
IBM_QUANTUM_API_KEY=
QAOA_LAYERS=3
LSTM_EPOCHS=100
N_SCENARIOS=100
RISK_WEIGHT=15.0
CVAR_CONFIDENCE=0.95
LOG_LEVEL=INFO
```

---

# 📊 Research Contributions

### 🔹 Hybrid FFT vs QFT Analysis
Classical vs quantum frequency domain comparison.

### 🔹 Robust Quantum Optimization
Scenario-weighted penalties embedded in QUBO.

### 🔹 Quantum Risk Minimization
CVaR-aware parameterized quantum circuits.

### 🔹 Monte Carlo Quantum Stress Testing
Grid robustness under extreme demand conditions.

### 🔹 QAOA Warm-Start Strategy
Improved convergence using classical heuristics.

---

# 🧪 Testing

```bash
pytest tests/ -v
```

Property-based testing:

```bash
pytest tests/ -v --hypothesis-show-statistics
```

---

# ☁️ Deployment

## 🌍 Render

Build:
```
pip install -r requirements.txt
```

Start:
```
uvicorn src.main:app --host 0.0.0.0 --port $PORT
```

---

## 🐘 Neon PostgreSQL

1. Create database  
2. Copy connection string  
3. Set `DATABASE_URL`

---

# 📈 Use Cases

🏙 Smart Cities  
⚡ Renewable Grid Integration  
🏭 Industrial Energy Optimization  
📊 Energy Market Risk Modeling  
🧠 AI × Quantum Research Platforms  

---

# 👥 Contributors

- Harsh Mishra  
- Ramya Sharma  
- Harshit Verma  

---

# 📜 License

MIT License

---

<p align="center">
  ⚡ Built for the Future of Intelligent Energy Systems ⚛️
</p>
