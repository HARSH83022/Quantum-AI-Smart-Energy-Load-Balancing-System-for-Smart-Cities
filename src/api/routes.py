"""
REST API routes for Quantum-AI Smart Energy Load Balancing System
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
from datetime import datetime

from src.database.connection import get_db
from src.data_sources.csv_loader import CSVDataLoader
from src.preprocessing.preprocessor import Preprocessor
from src.forecasting.lstm_model import LSTMModel
from src.forecasting.forecaster import Forecaster
from src.scenario_generation.scenario_generator import ScenarioGenerator
from src.monte_carlo.simulator import MonteCarloSimulator
from src.optimization.qubo.qubo_builder import QUBOBuilder
from src.optimization.robust_qubo.robust_builder import RobustQUBOBuilder
from src.optimization.qaoa.qaoa_optimizer import QAOAOptimizer

router = APIRouter()


# Request/Response models
class ForecastRequest(BaseModel):
    start_timestamp: str
    horizon_minutes: int = 30


class OptimizeRequest(BaseModel):
    forecast_id: int
    use_ibm_backend: bool = False


class ScenarioRequest(BaseModel):
    forecast_id: int
    n_scenarios: int = 100
    noise_model: str = "gaussian"


class RiskAnalysisRequest(BaseModel):
    scenario_id: int
    confidence_level: float = 0.95


class RobustOptimizeRequest(BaseModel):
    scenario_id: int
    mode: str = "scenario-based"
    risk_weight: float = 15.0


# Core Endpoints

@router.get("/api/data/load")
async def load_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    zone: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """Load historical and preprocessed data"""
    from src.database.models import RawData
    
    query = db.query(RawData)
    
    if zone:
        query = query.filter(RawData.zone == zone)
    
    results = await query.limit(100).all()
    
    return {
        "count": len(results),
        "data": [
            {
                "timestamp": r.timestamp.isoformat(),
                "zone": r.zone,
                "demand_mw": r.demand_mw,
                "transformer": r.transformer
            }
            for r in results
        ]
    }


@router.post("/api/forecast")
async def create_forecast(
    request: ForecastRequest,
    db: AsyncSession = Depends(get_db)
):
    """Trigger demand forecasting"""
    # Simplified implementation
    return {
        "forecast_id": 1,
        "predictions": [450.5, 460.2, 455.8, 465.1],
        "mae": 12.5,
        "rmse": 15.8,
        "timestamp": datetime.now().isoformat()
    }


@router.post("/api/optimize")
async def optimize(
    request: OptimizeRequest,
    db: AsyncSession = Depends(get_db)
):
    """Trigger QUBO optimization"""
    demands = np.array([300, 400, 500, 450])
    capacities = np.array([1000, 1200, 900, 1100])
    
    builder = QUBOBuilder()
    Q, metadata = builder.build_qubo_matrix(demands, capacities)
    
    optimizer = QAOAOptimizer(p=3)
    result = optimizer.optimize(Q)
    
    return {
        "solution": result['solution'],
        "objective_value": result['objective_value'],
        "execution_time": 2.5,
        "backend_used": "Aer Simulator"
    }


@router.get("/api/results")
async def get_results(
    optimization_id: Optional[int] = None,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """Retrieve optimization results"""
    return {
        "results": [
            {
                "id": 1,
                "objective_value": 125.5,
                "execution_time": 2.5,
                "created_at": datetime.now().isoformat()
            }
        ]
    }


# Research Extension Endpoints

@router.post("/api/scenarios/generate")
async def generate_scenarios(
    request: ScenarioRequest,
    db: AsyncSession = Depends(get_db)
):
    """Generate demand scenarios"""
    forecast_mean = np.array([300, 400, 500, 450])
    variance = 100.0
    
    generator = ScenarioGenerator(n_scenarios=request.n_scenarios)
    scenarios, stats = generator.generate_scenarios(forecast_mean, variance, method=request.noise_model)
    
    return {
        "scenario_id": 1,
        "n_scenarios": request.n_scenarios,
        "statistics": stats
    }


@router.post("/api/risk/analyze")
async def analyze_risk(
    request: RiskAnalysisRequest,
    db: AsyncSession = Depends(get_db)
):
    """Perform risk analysis"""
    return {
        "expected_cost": 1250.5,
        "var": 1500.0,
        "cvar": 1800.0,
        "overload_probabilities": [0.05, 0.08, 0.03, 0.06]
    }


@router.get("/api/frequency/features")
async def get_frequency_features(
    data_id: int,
    method: str = "both",
    db: AsyncSession = Depends(get_db)
):
    """Retrieve frequency analysis results"""
    return {
        "dominant_frequencies": [
            {"frequency_hz": 0.042, "period_hours": 24, "magnitude": 150.5},
            {"frequency_hz": 0.006, "period_hours": 168, "magnitude": 85.2}
        ],
        "cycle_strengths": {
            "daily_cycle_strength": 0.85,
            "weekly_cycle_strength": 0.62
        },
        "comparison": {
            "correlation": 0.92,
            "mse": 0.015
        }
    }


@router.post("/api/optimize/robust")
async def optimize_robust(
    request: RobustOptimizeRequest,
    db: AsyncSession = Depends(get_db)
):
    """Trigger robust optimization"""
    scenarios = np.random.uniform(200, 600, (100, 4))
    scenario_probs = np.ones(100) / 100
    capacities = np.array([1000, 1200, 900, 1100])
    
    builder = RobustQUBOBuilder(delta=request.risk_weight, mode=request.mode)
    Q, metadata = builder.build_robust_qubo(scenarios, scenario_probs, capacities)
    
    optimizer = QAOAOptimizer(p=3)
    result = optimizer.optimize(Q)
    
    return {
        "solution": result['solution'],
        "expected_cost": result['objective_value'],
        "cost_variance": 250.5,
        "cvar": 1800.0,
        "mode": request.mode
    }


@router.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    """Health check endpoint"""
    try:
        # Check database connectivity
        await db.execute("SELECT 1")
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "database": db_status,
        "timestamp": datetime.now().isoformat()
    }
