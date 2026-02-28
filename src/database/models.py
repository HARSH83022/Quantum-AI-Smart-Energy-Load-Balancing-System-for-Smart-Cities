"""
SQLAlchemy database models for all tables
"""
from sqlalchemy import Column, Integer, String, Float, Boolean, TIMESTAMP, ForeignKey, JSON
from sqlalchemy.sql import func
from .connection import Base


class RawData(Base):
    """Raw data from CSV"""
    __tablename__ = "raw_data"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(TIMESTAMP, nullable=False, index=True)
    zone = Column(String(10), nullable=False)
    demand_mw = Column(Float, nullable=False)
    transformer = Column(String(10), nullable=False)
    capacity_mw = Column(Float, nullable=False)
    voltage = Column(Float)
    current = Column(Float)
    temperature = Column(Float)
    humidity = Column(Float)
    hour = Column(Integer)
    day_of_week = Column(Integer)
    month = Column(Integer)
    created_at = Column(TIMESTAMP, server_default=func.now())


class PreprocessedData(Base):
    """Preprocessed sequences for training"""
    __tablename__ = "preprocessed_data"
    
    id = Column(Integer, primary_key=True, index=True)
    sequence_data = Column(JSON, nullable=False)
    target_data = Column(JSON, nullable=False)
    is_training = Column(Boolean, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())


class Forecast(Base):
    """Forecast results"""
    __tablename__ = "forecasts"
    
    id = Column(Integer, primary_key=True, index=True)
    forecast_timestamp = Column(TIMESTAMP, nullable=False, index=True)
    zone = Column(String(10), nullable=False)
    predicted_demand_mw = Column(Float, nullable=False)
    actual_demand_mw = Column(Float)
    mae = Column(Float)
    rmse = Column(Float)
    model_version = Column(String(50))
    created_at = Column(TIMESTAMP, server_default=func.now())


class QUBOMatrix(Base):
    """QUBO matrices"""
    __tablename__ = "qubo_matrices"
    
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(Integer, ForeignKey("forecasts.id"), nullable=True)
    matrix_data = Column(JSON, nullable=False)
    num_variables = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())


class OptimizationResult(Base):
    """Optimization results"""
    __tablename__ = "optimization_results"
    
    id = Column(Integer, primary_key=True, index=True)
    qubo_id = Column(Integer, ForeignKey("qubo_matrices.id"), nullable=True)
    solution_vector = Column(JSON, nullable=False)
    objective_value = Column(Float, nullable=False)
    execution_time_seconds = Column(Float)
    backend_used = Column(String(50))
    circuit_depth = Column(Integer)
    created_at = Column(TIMESTAMP, server_default=func.now())


# Research Extension Tables

class FrequencyFeature(Base):
    """Frequency features from FFT and QFT analysis"""
    __tablename__ = "frequency_features"
    
    id = Column(Integer, primary_key=True, index=True)
    data_id = Column(Integer, ForeignKey("raw_data.id"), nullable=True)
    method = Column(String(10), nullable=False)  # 'fft' or 'qft'
    dominant_frequencies = Column(JSON, nullable=False)
    cycle_strengths = Column(JSON, nullable=False)
    spectral_entropy = Column(Float)
    created_at = Column(TIMESTAMP, server_default=func.now())


class Scenario(Base):
    """Demand scenarios for robust optimization"""
    __tablename__ = "scenarios"
    
    id = Column(Integer, primary_key=True, index=True)
    forecast_id = Column(Integer, ForeignKey("forecasts.id"), nullable=True)
    scenario_matrix = Column(JSON, nullable=False)
    n_scenarios = Column(Integer, nullable=False)
    noise_model = Column(String(20), nullable=False)
    mean_demand = Column(Float)
    std_demand = Column(Float)
    created_at = Column(TIMESTAMP, server_default=func.now())


class RiskMetric(Base):
    """Risk metrics from Monte Carlo simulation"""
    __tablename__ = "risk_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=True)
    expected_cost = Column(Float, nullable=False)
    cost_variance = Column(Float, nullable=False)
    var_95 = Column(Float, nullable=False)
    cvar_95 = Column(Float, nullable=False)
    overload_frequencies = Column(JSON, nullable=False)
    stress_probabilities = Column(JSON, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())


class RobustQUBOMatrix(Base):
    """Robust QUBO matrices with risk penalties"""
    __tablename__ = "robust_qubo_matrices"
    
    id = Column(Integer, primary_key=True, index=True)
    scenario_id = Column(Integer, ForeignKey("scenarios.id"), nullable=True)
    optimization_mode = Column(String(20), nullable=False)  # 'deterministic', 'scenario-based', 'worst-case', 'cvar'
    matrix_data = Column(JSON, nullable=False)
    risk_weight = Column(Float, nullable=False)
    num_variables = Column(Integer, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
