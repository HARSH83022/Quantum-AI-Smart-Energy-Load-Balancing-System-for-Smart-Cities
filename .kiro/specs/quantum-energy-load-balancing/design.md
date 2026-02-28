# Design Document

## Overview

The Quantum-AI Smart Energy Load Balancing System is a modular backend application that combines deep learning forecasting with quantum-inspired optimization. The system follows a pipeline architecture: data ingestion → preprocessing → LSTM forecasting → QUBO formulation → QAOA optimization → result storage → API exposure.

The architecture emphasizes modularity to support future extensibility (e.g., switching from CSV to live IoT streams) and uses industry-standard tools: PyTorch for ML, Qiskit for quantum computing, PostgreSQL for persistence, and FastAPI for REST endpoints.

## Architecture

### High-Level Architecture (Research-Enhanced)

```
┌─────────────────┐
│  Data Sources   │
│  (CSV/IoT API)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Loader    │
│   & Validator   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│  Preprocessing  │─────▶│  PostgreSQL  │
│     Module      │      │   Database   │
└────────┬────────┘      └──────────────┘
         │                       ▲
         ▼                       │
┌─────────────────┐              │
│ Frequency       │──────────────┤
│ Analysis (FFT/  │              │
│ QFT)            │              │
└────────┬────────┘              │
         │                       │
         ▼                       │
┌─────────────────┐              │
│ Enhanced LSTM   │──────────────┤
│ Forecasting     │              │
│ (Time+Freq)     │              │
└────────┬────────┘              │
         │                       │
         ▼                       │
┌─────────────────┐              │
│ Scenario        │──────────────┤
│ Generator       │              │
└────────┬────────┘              │
         │                       │
         ▼                       │
┌─────────────────┐              │
│ Monte Carlo     │──────────────┤
│ Stress Tester   │              │
└────────┬────────┘              │
         │                       │
         ▼                       │
┌─────────────────┐              │
│ Robust QUBO     │──────────────┤
│ Formulation     │              │
└────────┬────────┘              │
         │                       │
         ▼                       │
┌─────────────────┐              │
│ Enhanced QAOA   │──────────────┤
│ Optimizer       │              │
└────────┬────────┘              │
         │                       │
         ▼                       │
┌─────────────────┐              │
│ Quantum Risk    │──────────────┤
│ Analyzer        │              │
└────────┬────────┘              │
         │                       │
         ▼                       │
┌─────────────────┐              │
│   REST API      │◀─────────────┘
│   (FastAPI)     │
└─────────────────┘
```

### Technology Stack

- **Backend Framework**: FastAPI (async support, automatic OpenAPI docs)
- **ML Framework**: PyTorch (LSTM implementation)
- **Quantum Framework**: Qiskit, qiskit-ibm-runtime (QAOA implementation)
- **Database**: PostgreSQL (Neon hosted)
- **ORM**: SQLAlchemy (async support)
- **Data Processing**: Pandas, NumPy
- **Statistical Analysis**: scikit-learn (Research Extension)
- **Visualization**: matplotlib (Research Extension)
- **Testing**: pytest, Hypothesis (property-based testing)
- **Deployment**: Docker, Render
- **Environment Management**: python-dotenv

## Components and Interfaces

### 1. Data Source Interface

**Purpose**: Abstract interface for data ingestion to support multiple sources.

**Interface Definition**:
```python
class DataSource(ABC):
    @abstractmethod
    async def load_data(self) -> pd.DataFrame:
        """Load data and return as DataFrame with standard schema"""
        pass
    
    @abstractmethod
    async def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate that DataFrame has required columns"""
        pass
```

**Implementations**:
- `CSVDataLoader`: Loads from delhi_smart_grid_dataset.csv
- `IoTAPILoader`: (Future) Loads from live IoT endpoints

**Standard Schema**:
- timestamp: datetime
- zone: str (North, South, East, West)
- demand_MW: float
- transformer: str (T1, T2, T3, T4)
- capacity_MW: float
- voltage: float
- current: float
- temperature: float
- humidity: float
- hour: int
- day_of_week: int
- month: int

### 2. Data Loader Module

**Purpose**: Load and validate CSV data, store in database.

**Components**:
- `CSVDataLoader`: Reads CSV file
- `DataValidator`: Validates schema and data quality
- `DatabaseWriter`: Writes to raw_data table

**Key Functions**:
- `load_csv(file_path: str) -> pd.DataFrame`
- `validate_columns(df: pd.DataFrame) -> ValidationResult`
- `store_raw_data(df: pd.DataFrame) -> bool`

### 3. Preprocessing Module

**Purpose**: Clean, normalize, and prepare data for LSTM training.

**Components**:
- `MissingValueHandler`: Imputes or removes missing values
- `Normalizer`: Scales demand values using MinMaxScaler
- `SequenceGenerator`: Creates rolling window sequences
- `DataSplitter`: Splits into train/test sets

**Key Functions**:
- `handle_missing_values(df: pd.DataFrame, strategy: str) -> pd.DataFrame`
- `normalize_demand(df: pd.DataFrame) -> Tuple[pd.DataFrame, Scaler]`
- `create_sequences(df: pd.DataFrame, window_size: int) -> np.ndarray`
- `train_test_split(sequences: np.ndarray, test_size: float) -> Tuple`

**Configuration**:
- Window size: 24 hours (96 data points at 15-min intervals)
- Train/test split: 80/20
- Normalization: MinMaxScaler (0, 1)

### 4. LSTM Forecasting Module

**Purpose**: Train LSTM model and forecast 30-minute ahead demand.

**Architecture**:
```
Input Layer (24 timesteps × features)
    ↓
LSTM Layer 1 (128 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 2 (64 units, return_sequences=True)
    ↓
Dropout (0.2)
    ↓
LSTM Layer 3 (32 units)
    ↓
Dense Layer (4 outputs for 4 zones)
```

**Components**:
- `LSTMModel`: PyTorch nn.Module implementation
- `ModelTrainer`: Training loop with early stopping
- `Forecaster`: Generates predictions
- `MetricsCalculator`: Computes MAE, RMSE

**Key Functions**:
- `train_model(train_data, val_data, epochs: int) -> LSTMModel`
- `forecast(model: LSTMModel, input_sequence: np.ndarray) -> np.ndarray`
- `calculate_mae(y_true, y_pred) -> float`
- `calculate_rmse(y_true, y_pred) -> float`

**Hyperparameters**:
- Learning rate: 0.001
- Batch size: 32
- Epochs: 100 (with early stopping)
- Optimizer: Adam
- Loss function: MSE

### 5. QUBO Formulation Module

**Purpose**: Convert load balancing problem to QUBO matrix.

**Problem Formulation**:

Binary variables: `x_ij ∈ {0, 1}` where i = transformer index, j = zone index

Objective function:
```
minimize: α·(overload_penalty) + β·(imbalance_penalty) + γ·(switching_cost)
```

Where:
- Overload penalty: `Σ_i max(0, load_i - capacity_i)²`
- Imbalance penalty: `Σ_i (load_i - avg_load)²`
- Switching cost: `Σ_ij c_ij · x_ij` (cost of assigning zone j to transformer i)

Constraints (as penalties):
- Capacity: `λ₁ · Σ_i max(0, Σ_j x_ij · demand_j - capacity_i)²`
- Assignment: `λ₂ · Σ_j (Σ_i x_ij - 1)²` (each zone assigned to exactly one transformer)

**Components**:
- `QUBOBuilder`: Constructs QUBO matrix
- `ConstraintEncoder`: Adds penalty terms
- `MatrixOptimizer`: Optimizes matrix representation

**Key Functions**:
- `build_qubo_matrix(demands: np.ndarray, capacities: np.ndarray) -> np.ndarray`
- `add_capacity_constraints(Q: np.ndarray, penalty: float) -> np.ndarray`
- `add_assignment_constraints(Q: np.ndarray, penalty: float) -> np.ndarray`

**Parameters**:
- α (overload weight): 10.0
- β (imbalance weight): 5.0
- γ (switching weight): 1.0
- λ₁ (capacity penalty): 100.0
- λ₂ (assignment penalty): 100.0

### 6. QAOA Optimization Module

**Purpose**: Solve QUBO using quantum-inspired optimization.

**QAOA Configuration**:
- Layers (p): 3
- Optimizer: COBYLA
- Max iterations: 1000
- Backend: Qiskit Aer simulator (default) or IBM simulator

**Components**:
- `QAOACircuitBuilder`: Constructs parameterized quantum circuit
- `QAOAOptimizer`: Runs variational optimization
- `ResultDecoder`: Converts bitstring to solution vector

**Key Functions**:
- `build_qaoa_circuit(Q: np.ndarray, p: int) -> QuantumCircuit`
- `optimize(circuit: QuantumCircuit, backend: Backend) -> OptimizationResult`
- `decode_solution(bitstring: str) -> np.ndarray`

**Circuit Structure**:
```
|0⟩ ─ H ─ Rz(γ₁) ─ CNOT ─ Rx(β₁) ─ ... ─ Rz(γₚ) ─ CNOT ─ Rx(βₚ) ─ Measure
```

### 7. Frequency Analysis Module (Research Extension)

**Purpose**: Extract periodic patterns using classical FFT and Quantum Fourier Transform for hybrid forecasting.

**Components**:
- `ClassicalFFTAnalyzer`: Computes FFT spectrum of demand time-series
- `QFTAnalyzer`: Implements Quantum Fourier Transform using Qiskit
- `FrequencyFeatureExtractor`: Extracts dominant frequencies and periodic patterns
- `FrequencyComparator`: Compares FFT vs QFT outputs

**Key Functions**:
- `compute_fft_spectrum(time_series: np.ndarray) -> np.ndarray`
- `extract_dominant_frequencies(spectrum: np.ndarray, top_k: int) -> List[float]`
- `encode_amplitude_state(signal: np.ndarray) -> QuantumCircuit`
- `apply_qft(circuit: QuantumCircuit) -> QuantumCircuit`
- `extract_qft_spectrum(statevector: np.ndarray) -> np.ndarray`
- `compare_spectra(fft_spectrum: np.ndarray, qft_spectrum: np.ndarray) -> Dict`

**Frequency Features**:
- Daily cycle strength (24-hour periodicity)
- Weekly cycle strength (7-day periodicity)
- Peak frequency magnitudes
- Harmonic components
- Spectral entropy

### 8. Probabilistic Scenario Generator (Research Extension)

**Purpose**: Generate multiple demand scenarios for robust optimization and risk analysis.

**Components**:
- `ScenarioGenerator`: Creates probabilistic demand scenarios
- `GaussianNoiseModel`: Adds uncertainty based on forecast variance
- `BootstrapSampler`: Generates scenarios using bootstrap resampling
- `ScenarioValidator`: Ensures scenarios preserve statistical properties

**Key Functions**:
- `generate_scenarios(forecast_mean: np.ndarray, residual_variance: float, n_scenarios: int) -> np.ndarray`
- `apply_gaussian_noise(base_forecast: np.ndarray, variance: float) -> np.ndarray`
- `bootstrap_sample(historical_residuals: np.ndarray, n_samples: int) -> np.ndarray`
- `validate_scenario_statistics(scenarios: np.ndarray, historical_data: np.ndarray) -> bool`

**Configuration**:
- Number of scenarios: 100 (default)
- Confidence level: 95%
- Noise model: Gaussian with empirical variance
- Bootstrap method: Block bootstrap for time series

### 9. Monte Carlo Stress Tester (Research Extension)

**Purpose**: Evaluate system robustness by running optimization across multiple demand scenarios.

**Components**:
- `MonteCarloSimulator`: Runs optimization across all scenarios
- `RiskMetricsCalculator`: Computes overload probability and stress metrics
- `StressAnalyzer`: Identifies vulnerable transformers and zones

**Key Functions**:
- `run_monte_carlo(scenarios: np.ndarray, qubo_builder: QUBOBuilder, optimizer: QAOAOptimizer) -> List[OptimizationResult]`
- `calculate_overload_frequency(results: List[OptimizationResult], capacities: np.ndarray) -> np.ndarray`
- `calculate_stress_probability(results: List[OptimizationResult]) -> Dict[str, float]`
- `calculate_expected_imbalance(results: List[OptimizationResult]) -> float`
- `compute_confidence_intervals(metrics: np.ndarray, confidence: float) -> Tuple[float, float]`

**Risk Metrics**:
- Overload frequency per transformer
- Transformer stress probability
- Expected load imbalance
- Worst-case scenario cost
- Value at Risk (VaR) at 95%
- Conditional Value at Risk (CVaR)

### 10. Robust QUBO Formulation Module (Research Extension)

**Purpose**: Extend QUBO formulation to account for demand uncertainty and risk.

**Components**:
- `RobustQUBOBuilder`: Constructs scenario-aware QUBO matrices
- `RiskPenaltyCalculator`: Computes risk penalty terms from scenarios
- `OptimizationModeSelector`: Switches between deterministic, scenario-based, and worst-case modes

**Objective Function (Robust)**:
```
minimize: α·(overload_penalty) + β·(imbalance_penalty) + γ·(switching_cost) + δ·(risk_penalty)
```

Where risk_penalty = `Σ_s p_s · max(0, overload_s)²` across scenarios s with probability p_s

**Key Functions**:
- `build_robust_qubo(scenarios: np.ndarray, scenario_probs: np.ndarray, capacities: np.ndarray) -> np.ndarray`
- `calculate_risk_penalty(scenarios: np.ndarray, solution: np.ndarray) -> float`
- `optimize_worst_case(scenarios: np.ndarray) -> np.ndarray`
- `optimize_scenario_based(scenarios: np.ndarray, weights: np.ndarray) -> np.ndarray`

**Optimization Modes**:
- Deterministic: Uses point forecast only
- Scenario-based: Minimizes expected cost across scenarios
- Worst-case: Minimizes maximum cost across scenarios
- CVaR-based: Minimizes conditional value at risk

**Parameters**:
- δ (risk weight): 15.0
- Scenario probability: Uniform (1/N) or weighted by likelihood
- CVaR confidence level: 95%

### 11. Enhanced QAOA Module (Research Extension)

**Purpose**: Improve QAOA performance with parameter warm-starting and convergence monitoring.

**Enhancements**:
- Multi-layer QAOA with configurable depth (p=1 to 10)
- Parameter warm-starting from previous solutions
- Adaptive layer depth based on problem size
- Convergence monitoring and early stopping
- Expectation value variance tracking

**Components**:
- `AdaptiveQAOA`: Dynamically adjusts circuit depth
- `ParameterWarmStarter`: Initializes parameters from previous runs
- `ConvergenceMonitor`: Tracks optimization progress
- `VarianceAnalyzer`: Measures solution stability

**Key Functions**:
- `warm_start_parameters(previous_params: np.ndarray, new_problem_size: int) -> np.ndarray`
- `monitor_convergence(energy_history: List[float], threshold: float) -> bool`
- `calculate_expectation_variance(measurements: Dict[str, int]) -> float`
- `adaptive_layer_selection(problem_size: int, time_budget: float) -> int`

**Performance Metrics**:
- Circuit depth
- Gate count
- Convergence rate (iterations to threshold)
- Energy variance
- Approximation ratio
- Time to solution

### 12. Quantum Risk Analyzer (Research Extension)

**Purpose**: Evaluate solution stability and optimize for risk-adjusted objectives using quantum-enhanced methods.

**Components**:
- `QuantumRiskAnalyzer`: Evaluates solution robustness across scenarios
- `CVaRCalculator`: Computes Conditional Value at Risk
- `RiskOptimizer`: Optimizes risk-adjusted objective functions
- `StabilityEvaluator`: Measures solution sensitivity to demand variations

**Key Functions**:
- `evaluate_solution_stability(solution: np.ndarray, scenarios: np.ndarray) -> Dict`
- `calculate_expected_cost(solution: np.ndarray, scenarios: np.ndarray) -> float`
- `calculate_cost_variance(solution: np.ndarray, scenarios: np.ndarray) -> float`
- `calculate_cvar(costs: np.ndarray, confidence: float) -> float`
- `optimize_cvar_objective(scenarios: np.ndarray, lambda_risk: float) -> np.ndarray`

**Risk-Adjusted Objective**:
```
minimize: E[cost] + λ·Var[cost]
```

Or CVaR-based:
```
minimize: E[cost] + λ·CVaR_α[cost]
```

**Risk Metrics**:
- Expected cost: E[cost]
- Cost variance: Var[cost]
- Cost standard deviation: σ[cost]
- Value at Risk (VaR): 95th percentile cost
- Conditional Value at Risk (CVaR): Expected cost in worst 5% scenarios
- Sharpe ratio: (E[cost] - baseline) / σ[cost]

### 13. REST API Module

**Purpose**: Expose system functionality via HTTP endpoints.

**Endpoints**:

1. `GET /api/data/load`
   - Query params: start_date, end_date, zone (optional)
   - Returns: Historical data JSON

2. `POST /api/forecast`
   - Body: { "start_timestamp": "ISO8601", "horizon_minutes": 30 }
   - Returns: { "predictions": [...], "mae": float, "rmse": float }

3. `POST /api/optimize`
   - Body: { "forecast_id": int, "use_ibm_backend": bool }
   - Returns: { "solution": [...], "objective_value": float, "execution_time": float }

4. `GET /api/results`
   - Query params: optimization_id, limit
   - Returns: Optimization results JSON

5. `POST /api/scenarios/generate` (Research Extension)
   - Body: { "forecast_id": int, "n_scenarios": 100, "noise_model": "gaussian" }
   - Returns: { "scenario_id": int, "n_scenarios": int, "statistics": {...} }

6. `POST /api/risk/analyze` (Research Extension)
   - Body: { "scenario_id": int, "confidence_level": 0.95 }
   - Returns: { "expected_cost": float, "var": float, "cvar": float, "overload_probabilities": [...] }

7. `GET /api/frequency/features` (Research Extension)
   - Query params: data_id, method (fft|qft|both)
   - Returns: { "dominant_frequencies": [...], "cycle_strengths": {...}, "comparison": {...} }

8. `POST /api/optimize/robust` (Research Extension)
   - Body: { "scenario_id": int, "mode": "scenario-based|worst-case|cvar", "risk_weight": 15.0 }
   - Returns: { "solution": [...], "expected_cost": float, "cost_variance": float, "cvar": float }

**Authentication**:
- JWT tokens for API access
- Token validation middleware

**Error Handling**:
- 400: Bad Request (invalid input)
- 401: Unauthorized (missing/invalid token)
- 404: Not Found (resource doesn't exist)
- 500: Internal Server Error (system failure)

### 8. Database Module

**Purpose**: Persist all pipeline data and results.

**Schema**:

```sql
-- Raw data from CSV
CREATE TABLE raw_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    zone VARCHAR(10) NOT NULL,
    demand_mw FLOAT NOT NULL,
    transformer VARCHAR(10) NOT NULL,
    capacity_mw FLOAT NOT NULL,
    voltage FLOAT,
    current FLOAT,
    temperature FLOAT,
    humidity FLOAT,
    hour INT,
    day_of_week INT,
    month INT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Preprocessed sequences
CREATE TABLE preprocessed_data (
    id SERIAL PRIMARY KEY,
    sequence_data JSONB NOT NULL,
    target_data JSONB NOT NULL,
    is_training BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Forecast results
CREATE TABLE forecasts (
    id SERIAL PRIMARY KEY,
    forecast_timestamp TIMESTAMP NOT NULL,
    zone VARCHAR(10) NOT NULL,
    predicted_demand_mw FLOAT NOT NULL,
    actual_demand_mw FLOAT,
    mae FLOAT,
    rmse FLOAT,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- QUBO matrices
CREATE TABLE qubo_matrices (
    id SERIAL PRIMARY KEY,
    forecast_id INT REFERENCES forecasts(id),
    matrix_data JSONB NOT NULL,
    num_variables INT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Optimization results
CREATE TABLE optimization_results (
    id SERIAL PRIMARY KEY,
    qubo_id INT REFERENCES qubo_matrices(id),
    solution_vector JSONB NOT NULL,
    objective_value FLOAT NOT NULL,
    execution_time_seconds FLOAT,
    backend_used VARCHAR(50),
    circuit_depth INT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Frequency features (Research Extension)
CREATE TABLE frequency_features (
    id SERIAL PRIMARY KEY,
    data_id INT REFERENCES raw_data(id),
    method VARCHAR(10) NOT NULL, -- 'fft' or 'qft'
    dominant_frequencies JSONB NOT NULL,
    cycle_strengths JSONB NOT NULL,
    spectral_entropy FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Demand scenarios (Research Extension)
CREATE TABLE scenarios (
    id SERIAL PRIMARY KEY,
    forecast_id INT REFERENCES forecasts(id),
    scenario_matrix JSONB NOT NULL,
    n_scenarios INT NOT NULL,
    noise_model VARCHAR(20) NOT NULL,
    mean_demand FLOAT,
    std_demand FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Risk metrics (Research Extension)
CREATE TABLE risk_metrics (
    id SERIAL PRIMARY KEY,
    scenario_id INT REFERENCES scenarios(id),
    expected_cost FLOAT NOT NULL,
    cost_variance FLOAT NOT NULL,
    var_95 FLOAT NOT NULL,
    cvar_95 FLOAT NOT NULL,
    overload_frequencies JSONB NOT NULL,
    stress_probabilities JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Robust QUBO matrices (Research Extension)
CREATE TABLE robust_qubo_matrices (
    id SERIAL PRIMARY KEY,
    scenario_id INT REFERENCES scenarios(id),
    optimization_mode VARCHAR(20) NOT NULL, -- 'deterministic', 'scenario-based', 'worst-case', 'cvar'
    matrix_data JSONB NOT NULL,
    risk_weight FLOAT NOT NULL,
    num_variables INT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## Data Models

### Python Data Classes

```python
@dataclass
class GridDataPoint:
    timestamp: datetime
    zone: str
    demand_mw: float
    transformer: str
    capacity_mw: float
    voltage: float
    current: float
    temperature: float
    humidity: float

@dataclass
class ForecastResult:
    forecast_timestamp: datetime
    zone: str
    predicted_demand_mw: float
    mae: float
    rmse: float
    model_version: str

@dataclass
class QUBOMatrix:
    matrix: np.ndarray
    num_variables: int
    forecast_id: int

@dataclass
class OptimizationResult:
    solution_vector: np.ndarray
    objective_value: float
    execution_time: float
    backend_used: str
    circuit_depth: int
```


## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: CSV Schema Validation
*For any* CSV file with missing or invalid required columns (timestamp, demand, transformer, capacity), the validation function should reject the file and return an error.
**Validates: Requirements 1.3**

### Property 2: Data Persistence Round-Trip
*For any* valid data object (raw data, preprocessed data, forecast, QUBO matrix, or optimization result), storing it in the database and then retrieving it should yield an equivalent object.
**Validates: Requirements 1.4, 2.5, 3.6, 4.8, 5.4, 10.1, 10.2, 10.3, 10.4, 10.5**

### Property 3: Missing Value Elimination
*For any* dataset containing missing values, after preprocessing with the missing value handler, the resulting dataset should contain zero missing values.
**Validates: Requirements 2.1**

### Property 4: Normalization Range
*For any* demand data, after normalization, all values should fall within the range [0, 1].
**Validates: Requirements 2.2**

### Property 5: Rolling Window Sequence Length
*For any* time series data with sufficient length, all generated rolling window sequences should have exactly 24 hours worth of data points.
**Validates: Requirements 2.3**

### Property 6: Train-Test Split Completeness
*For any* preprocessed dataset, the union of training and testing sets should equal the original dataset size, and the intersection should be empty.
**Validates: Requirements 2.4**

### Property 7: Forecast Horizon Consistency
*For any* input sequence, the forecast output should contain predictions for exactly the specified horizon (30 minutes).
**Validates: Requirements 3.3**

### Property 8: Forecast Metrics Non-Negativity
*For any* forecast with ground truth data, the computed MAE and RMSE metrics should both be non-negative values.
**Validates: Requirements 3.4, 3.5**

### Property 9: QUBO Matrix Symmetry
*For any* load balancing problem instance, the constructed QUBO matrix should be square and symmetric.
**Validates: Requirements 4.7**

### Property 10: QUBO Objective Completeness
*For any* QUBO formulation, the objective function should include all three required terms: overload penalty, imbalance penalty, and switching cost, plus capacity and assignment constraint penalties.
**Validates: Requirements 4.2, 4.3, 4.4, 4.5, 4.6**

### Property 11: QAOA Solution Vector Length
*For any* QUBO problem with n binary variables, the QAOA solution vector should have exactly n elements.
**Validates: Requirements 5.3**

### Property 12: QAOA Performance Metrics Positivity
*For any* QAOA execution, the logged circuit depth and execution time should both be positive values.
**Validates: Requirements 5.5**

### Property 13: API Error Response Format
*For any* invalid API request, the response should have an HTTP status code in the 4xx range and include a JSON body with an error message field.
**Validates: Requirements 6.5**

### Property 14: API Success Response Format
*For any* valid API request, the response should have an HTTP status code in the 2xx range and return valid JSON containing the requested data fields.
**Validates: Requirements 6.6**

### Property 15: Error Logging Completeness
*For any* error occurring during data loading, model training, or optimization, the system should log an entry containing the error message and a timestamp.
**Validates: Requirements 9.1, 9.2, 9.3**

### Property 16: Database Retry Behavior
*For any* database operation that fails, the system should retry exactly 3 times before raising a final error.
**Validates: Requirements 9.4**

### Property 17: API Request Logging
*For any* API request (successful or failed), the system should log an entry containing the timestamp, endpoint path, and HTTP status code.
**Validates: Requirements 9.5**

### Property 18: Referential Integrity Preservation
*For any* pair of related records (e.g., forecast and its associated QUBO matrix), the foreign key relationship should be maintained and queryable.
**Validates: Requirements 10.6**

### Property 19: CSV Column Parsing Completeness
*For any* valid CSV file with all required columns, the parser should successfully extract all column values (timestamp, demand, transformer, capacity, temperature, etc.).
**Validates: Requirements 1.2**

### Property 20: Dominant Frequency Detection (Research Extension)
*For any* time series with known periodic components, FFT analysis should correctly identify the dominant frequencies within a tolerance threshold.
**Validates: Requirements 11.2**

### Property 21: QFT Output State Validity (Research Extension)
*For any* normalized demand signal encoded into quantum amplitude states, the QFT output state should have unit norm and valid probability amplitudes.
**Validates: Requirements 11.4**

### Property 22: Scenario Variance Preservation (Research Extension)
*For any* generated scenario set, the empirical variance of scenarios should match the specified residual variance within a statistical tolerance.
**Validates: Requirements 13.6**

### Property 23: Monte Carlo Stability (Research Extension)
*For any* set of demand scenarios, running Monte Carlo simulation twice with the same scenarios should produce identical risk metrics.
**Validates: Requirements 14.6**

### Property 24: Robust QUBO Risk Term Inclusion (Research Extension)
*For any* robust QUBO formulation, the objective function should include the risk penalty term weighted by scenario probabilities.
**Validates: Requirements 15.2, 15.3**

### Property 25: QAOA Parameter Warm-Start Improvement (Research Extension)
*For any* sequence of related QUBO problems, using parameter warm-starting should reduce the number of iterations to convergence compared to random initialization.
**Validates: Requirements 16.2**

### Property 26: CVaR Monotonicity (Research Extension)
*For any* cost distribution, CVaR at confidence level α should be greater than or equal to VaR at the same confidence level.
**Validates: Requirements 17.4**

## Error Handling

### Error Categories

1. **Data Loading Errors**
   - File not found
   - Invalid CSV format
   - Schema validation failure
   - Encoding issues

2. **Preprocessing Errors**
   - Insufficient data for sequences
   - Normalization failures
   - Invalid data types

3. **Model Errors**
   - Training convergence failure
   - Invalid input dimensions
   - Model loading errors

4. **Optimization Errors**
   - QUBO construction failure
   - QAOA execution timeout
   - Invalid solution decoding

5. **Database Errors**
   - Connection failures
   - Query execution errors
   - Transaction rollback

6. **API Errors**
   - Invalid request format
   - Authentication failures
   - Resource not found

### Error Handling Strategy

**Retry Logic**:
- Database operations: 3 retries with exponential backoff (1s, 2s, 4s)
- API calls to external services: 2 retries with 5s delay
- No retries for validation errors or client errors (4xx)

**Logging**:
- All errors logged with: timestamp, error type, error message, stack trace
- Log levels: ERROR for failures, WARNING for retries, INFO for recovery
- Structured logging using JSON format for easy parsing

**Graceful Degradation**:
- If LSTM training fails, use simple moving average fallback
- If QAOA fails, use classical optimization (scipy.optimize)
- If database write fails, cache results in memory and retry

**User Feedback**:
- API responses include clear error messages
- Error codes follow standard HTTP conventions
- Detailed errors in logs, sanitized errors in API responses

## Testing Strategy

### Unit Testing

**Framework**: pytest

**Coverage Areas**:
- Data validation functions
- Preprocessing transformations
- QUBO matrix construction
- API endpoint handlers
- Database operations

**Key Unit Tests**:
1. Test CSV loader with valid file
2. Test CSV loader with missing columns (error case)
3. Test normalization with known input/output
4. Test sequence generation with small dataset
5. Test QUBO matrix construction with 2x2 problem
6. Test API authentication with valid/invalid tokens
7. Test database connection with mock database

**Mocking Strategy**:
- Mock database connections for isolated testing
- Mock file I/O for data loader tests
- Mock quantum backend for QAOA tests (use statevector simulator)

### Property-Based Testing

**Framework**: Hypothesis (Python)

**Configuration**:
- Minimum 100 iterations per property test
- Use custom strategies for domain-specific data generation
- Seed random generator for reproducibility

**Property Test Implementation Requirements**:
- Each property test MUST be tagged with a comment: `# Feature: quantum-energy-load-balancing, Property X: [property text]`
- Each correctness property MUST be implemented by exactly ONE property-based test
- Tests should use Hypothesis strategies to generate diverse inputs

**Key Property Tests**:

1. **Property 1 Test**: CSV Schema Validation
   - Strategy: Generate CSV files with random missing columns
   - Verify: All invalid schemas are rejected

2. **Property 2 Test**: Data Persistence Round-Trip
   - Strategy: Generate random valid data objects
   - Verify: store → retrieve yields equivalent object

3. **Property 3 Test**: Missing Value Elimination
   - Strategy: Generate datasets with random missing values
   - Verify: Preprocessed data has zero NaN values

4. **Property 4 Test**: Normalization Range
   - Strategy: Generate random demand values
   - Verify: All normalized values in [0, 1]

5. **Property 5 Test**: Rolling Window Sequence Length
   - Strategy: Generate time series of varying lengths
   - Verify: All sequences have exactly 24-hour length

6. **Property 6 Test**: Train-Test Split Completeness
   - Strategy: Generate random datasets
   - Verify: train ∪ test = original, train ∩ test = ∅

7. **Property 7 Test**: Forecast Horizon Consistency
   - Strategy: Generate random input sequences
   - Verify: Output length matches 30-minute horizon

8. **Property 8 Test**: Forecast Metrics Non-Negativity
   - Strategy: Generate random predictions and ground truth
   - Verify: MAE ≥ 0 and RMSE ≥ 0

9. **Property 9 Test**: QUBO Matrix Symmetry
   - Strategy: Generate random problem instances
   - Verify: Q = Q^T and Q is square

10. **Property 10 Test**: QUBO Objective Completeness
    - Strategy: Generate random QUBO matrices
    - Verify: All five terms present in objective

11. **Property 11 Test**: QAOA Solution Vector Length
    - Strategy: Generate QUBO problems of varying sizes
    - Verify: Solution length = number of variables

12. **Property 12 Test**: QAOA Performance Metrics Positivity
    - Strategy: Run QAOA on random problems
    - Verify: circuit_depth > 0 and execution_time > 0

13. **Property 13 Test**: API Error Response Format
    - Strategy: Generate invalid API requests
    - Verify: 4xx status code and error message present

14. **Property 14 Test**: API Success Response Format
    - Strategy: Generate valid API requests
    - Verify: 2xx status code and valid JSON with required fields

15. **Property 15 Test**: Error Logging Completeness
    - Strategy: Trigger random errors in different modules
    - Verify: Log contains error message and timestamp

16. **Property 16 Test**: Database Retry Behavior
    - Strategy: Simulate database failures
    - Verify: Exactly 3 retry attempts occur

17. **Property 17 Test**: API Request Logging
    - Strategy: Make random API requests
    - Verify: Log contains timestamp, endpoint, status code

18. **Property 18 Test**: Referential Integrity Preservation
    - Strategy: Create related records
    - Verify: Foreign key relationships maintained

19. **Property 19 Test**: CSV Column Parsing Completeness
    - Strategy: Generate valid CSV files with all columns
    - Verify: All columns successfully parsed

### Integration Testing

**Scope**: End-to-end pipeline testing

**Test Scenarios**:
1. Full pipeline: CSV → Preprocessing → LSTM → QUBO → QAOA → API
2. Database integration: Write and read from actual test database
3. API integration: Test all endpoints with real backend

**Test Environment**:
- Use Docker Compose for test database
- Use test fixtures for sample data
- Clean database between tests

### Performance Testing

**Metrics**:
- Data loading time (target: < 10s for 100MB file)
- LSTM training time (target: < 5 minutes for 1 year data)
- QAOA optimization time (target: < 30s for 16-variable problem)
- API response time (target: < 200ms for GET requests)

**Tools**:
- pytest-benchmark for Python performance tests
- Locust for API load testing

## Deployment Architecture

### Container Structure

```
quantum-energy-system/
├── Dockerfile
├── docker-compose.yml (for local development)
├── requirements.txt
├── .env.example
└── src/
    ├── main.py (FastAPI app entry point)
    ├── data_sources/
    ├── preprocessing/
    ├── frequency_analysis/ (Research Extension)
    ├── forecasting/
    ├── scenario_generation/ (Research Extension)
    ├── monte_carlo/ (Research Extension)
    ├── optimization/
    │   ├── qubo/
    │   ├── robust_qubo/ (Research Extension)
    │   ├── qaoa/
    │   └── risk_analysis/ (Research Extension)
    ├── api/
    └── database/
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY delhi_smart_grid_dataset.csv ./data/

# Expose API port
EXPOSE 8000

# Start command
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Environment Variables

Required:
- `DATABASE_URL`: PostgreSQL connection string
- `JWT_SECRET`: Secret key for JWT token generation

Optional:
- `IBM_QUANTUM_API_KEY`: IBM Quantum API key for cloud backend
- `GEMINI_API_KEY`: Google Gemini API key for AI-enhanced analysis (Research Extension)
- `LOG_LEVEL`: Logging level (default: INFO)
- `QAOA_LAYERS`: Number of QAOA layers (default: 3)
- `LSTM_EPOCHS`: Maximum training epochs (default: 100)
- `N_SCENARIOS`: Number of demand scenarios for Monte Carlo (default: 100)
- `RISK_WEIGHT`: Risk penalty weight in robust QUBO (default: 15.0)
- `CVAR_CONFIDENCE`: CVaR confidence level (default: 0.95)

### Render Deployment

**Configuration**:
- Service type: Web Service
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn src.main:app --host 0.0.0.0 --port $PORT`
- Environment: Python 3.11
- Plan: Free tier (512MB RAM, shared CPU)

**Database**:
- Use Neon PostgreSQL (free tier: 0.5GB storage)
- Connection pooling enabled
- SSL mode: require

### Monitoring and Observability

**Logging**:
- Structured JSON logs
- Log aggregation via Render logs
- Log retention: 7 days (free tier)

**Metrics**:
- API request count and latency
- Model inference time
- QAOA execution time
- Database query performance

**Health Checks**:
- Endpoint: `GET /health`
- Checks: Database connectivity, model loaded, API responsive
- Interval: 30 seconds

## Security Considerations

1. **API Authentication**: JWT tokens with expiration
2. **Environment Variables**: Never commit secrets to version control
3. **Database**: Use connection pooling, parameterized queries to prevent SQL injection
4. **Input Validation**: Validate all API inputs, sanitize CSV data
5. **Rate Limiting**: Implement rate limiting on API endpoints (100 requests/minute)
6. **CORS**: Configure CORS for allowed origins only

## Research Contributions

This system implements several novel research contributions:

1. **Hybrid Quantum-Classical Frequency Analysis**: Compares classical FFT with Quantum Fourier Transform for time series periodicity detection
2. **Robust Quantum Optimization**: Extends QUBO formulation to account for demand uncertainty using scenario-based risk penalties
3. **Quantum Risk Minimization**: Implements CVaR-based optimization using quantum algorithms for risk-adjusted load balancing
4. **Monte Carlo Quantum Stress Testing**: Evaluates system robustness across probabilistic demand scenarios
5. **Parameter Warm-Starting for QAOA**: Improves convergence by initializing quantum circuit parameters from previous solutions

## Future Extensions

### Phase 2: Live IoT Integration

- Implement `IoTAPILoader` class
- Add WebSocket support for real-time data streaming
- Implement incremental model retraining
- Add data buffering and batching

### Phase 3: Real Quantum Hardware

- Deploy on IBM Quantum hardware (beyond simulator)
- Implement error mitigation techniques
- Add quantum circuit optimization
- Benchmark quantum vs classical performance

### Phase 4: Visualization Dashboard

- Real-time load monitoring dashboard
- Forecast visualization with confidence intervals
- Optimization results visualization
- Historical trend analysis
- Risk heatmaps and scenario comparisons

### Phase 5: Scalability

- Implement distributed training for LSTM
- Add caching layer (Redis)
- Implement message queue for async processing (Celery)
- Add horizontal scaling support
- Implement parallel scenario generation
