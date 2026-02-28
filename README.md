# Quantum-AI Smart Energy Load Balancing System

[![CI/CD Pipeline](https://github.com/HARSH83022/Quantum-AI-Smart-Energy-Load-Balancing-System-for-Smart-Cities/actions/workflows/ci-cd.yml/badge.svg?branch=T2-Quant)](https://github.com/HARSH83022/Quantum-AI-Smart-Energy-Load-Balancing-System-for-Smart-Cities/actions/workflows/ci-cd.yml)
[![Fast Tests](https://github.com/HARSH83022/Quantum-AI-Smart-Energy-Load-Balancing-System-for-Smart-Cities/actions/workflows/test-fast.yml/badge.svg)](https://github.com/HARSH83022/Quantum-AI-Smart-Energy-Load-Balancing-System-for-Smart-Cities/actions/workflows/test-fast.yml)
[![Code Quality](https://github.com/HARSH83022/Quantum-AI-Smart-Energy-Load-Balancing-System-for-Smart-Cities/actions/workflows/code-quality.yml/badge.svg)](https://github.com/HARSH83022/Quantum-AI-Smart-Energy-Load-Balancing-System-for-Smart-Cities/actions/workflows/code-quality.yml)
[![Docker Build](https://github.com/HARSH83022/Quantum-AI-Smart-Energy-Load-Balancing-System-for-Smart-Cities/actions/workflows/docker-publish.yml/badge.svg)](https://github.com/HARSH83022/Quantum-AI-Smart-Energy-Load-Balancing-System-for-Smart-Cities/actions/workflows/docker-publish.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready backend system combining LSTM forecasting with quantum-inspired optimization for smart grid load balancing.

## Features

### Core Features
- **Data Loading**: CSV-based historical smart grid data ingestion
- **Preprocessing**: Data cleaning, normalization, and sequence generation
- **LSTM Forecasting**: PyTorch-based demand forecasting
- **QUBO Formulation**: Load balancing as optimization problem
- **QAOA Optimization**: Quantum-inspired optimization using Qiskit
- **REST API**: FastAPI-based endpoints for all operations

### Research Extensions
- **Frequency Analysis**: Classical FFT and Quantum Fourier Transform comparison
- **Scenario Generation**: Probabilistic demand scenarios for robust optimization
- **Monte Carlo Simulation**: Risk assessment across multiple scenarios
- **Robust QUBO**: Risk-aware optimization formulation
- **Enhanced QAOA**: Parameter warm-starting and convergence monitoring
- **Risk Analysis**: CVaR-based risk minimization

## Project Structure

```
quantum-energy-system/
├── src/
│   ├── data_sources/          # CSV and IoT data loaders
│   ├── preprocessing/          # Data preprocessing
│   ├── frequency_analysis/     # FFT and QFT analysis
│   ├── forecasting/            # LSTM models
│   ├── scenario_generation/    # Probabilistic scenarios
│   ├── monte_carlo/            # Stress testing
│   ├── optimization/
│   │   ├── qubo/              # QUBO formulation
│   │   ├── robust_qubo/       # Robust optimization
│   │   ├── qaoa/              # Quantum optimization
│   │   └── risk_analysis/     # Risk metrics
│   ├── api/                    # REST API endpoints
│   └── database/               # Database models
├── tests/                      # Test suite
├── .kiro/specs/               # Specification documents
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Installation

### Prerequisites
- Python 3.11+
- PostgreSQL 15+ (or use Docker Compose)
- IBM Quantum API key (optional, for quantum backend)

### Local Setup

1. Clone the repository
```bash
git clone <repository-url>
cd quantum-energy-system
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run the application
```bash
uvicorn src.main:app --reload
```

### Docker Setup

1. Build and run with Docker Compose
```bash
docker-compose up --build
```

2. Access the API at `http://localhost:8000`

## API Endpoints

### Core Endpoints
- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /api/data/load` - Load historical data
- `POST /api/forecast` - Trigger forecasting
- `POST /api/optimize` - Run optimization
- `GET /api/results` - Get optimization results

### Research Extension Endpoints
- `POST /api/scenarios/generate` - Generate demand scenarios
- `POST /api/risk/analyze` - Perform risk analysis
- `GET /api/frequency/features` - Get frequency analysis results
- `POST /api/optimize/robust` - Run robust optimization

## Configuration

### Environment Variables

**Required:**
- `DATABASE_URL` - PostgreSQL connection string
- `JWT_SECRET` - Secret key for API authentication

**Optional:**
- `IBM_QUANTUM_API_KEY` - IBM Quantum API key
- `GEMINI_API_KEY` - Google Gemini API key
- `LOG_LEVEL` - Logging level (default: INFO)
- `QAOA_LAYERS` - QAOA circuit depth (default: 3)
- `LSTM_EPOCHS` - Training epochs (default: 100)
- `N_SCENARIOS` - Number of scenarios (default: 100)
- `RISK_WEIGHT` - Risk penalty weight (default: 15.0)
- `CVAR_CONFIDENCE` - CVaR confidence level (default: 0.95)

## Testing

Run the test suite:
```bash
pytest tests/ -v
```

Run property-based tests:
```bash
pytest tests/ -v --hypothesis-show-statistics
```

## Development Status

### Completed Tasks
✅ Project structure and dependencies
✅ Database models and connection
✅ Data source interface and CSV loader
✅ Property-based tests for database operations
✅ CSV schema validation tests

### In Progress
🔄 Preprocessing module
🔄 Frequency analysis (FFT/QFT)
🔄 LSTM forecasting
🔄 Scenario generation
🔄 QUBO formulation
🔄 QAOA optimization
🔄 API endpoints

## Research Contributions

This system implements several novel research contributions:

1. **Hybrid Quantum-Classical Frequency Analysis**: Compares classical FFT with QFT for periodicity detection
2. **Robust Quantum Optimization**: Scenario-based risk penalties in QUBO formulation
3. **Quantum Risk Minimization**: CVaR-based optimization using quantum algorithms
4. **Monte Carlo Quantum Stress Testing**: System robustness evaluation
5. **QAOA Parameter Warm-Starting**: Improved convergence through parameter initialization

## Deployment

### Render Deployment

1. Create a new Web Service on Render
2. Connect your repository
3. Configure:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables from `.env.example`
5. Deploy

### Database (Neon PostgreSQL)

1. Create a Neon PostgreSQL database
2. Copy the connection string
3. Set `DATABASE_URL` environment variable

## License

MIT License

## Contributors

- Your Name

## Acknowledgments

- IBM Quantum for quantum computing resources
- Qiskit team for quantum algorithms
- PyTorch team for deep learning framework
