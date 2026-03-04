# Quantum Energy Optimization Backend

This directory contains the backend implementation of the Quantum Energy Optimization system.

## Structure

- `src/` - Main source code
  - `api/` - REST API routes and endpoints
  - `database/` - Database models and connection handling
  - `data_sources/` - Data loading and processing
  - `forecasting/` - LSTM-based energy forecasting
  - `frequency_analysis/` - Classical FFT and Quantum QFT analysis
  - `monte_carlo/` - Monte Carlo simulation
  - `optimization/` - Quantum optimization algorithms (QAOA, QUBO)
  - `preprocessing/` - Data preprocessing utilities
  - `scenario_generation/` - Scenario generation for analysis
  - `utils/` - Utility functions and error handling

- `tests/` - Test suite
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker configuration
- `docker-compose.yml` - Docker Compose setup

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python src/main.py
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

## Docker

Build and run with Docker:
```bash
docker-compose up --build
```