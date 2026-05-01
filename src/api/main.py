from fastapi import FastAPI
from pydantic import BaseModel
from src.forecasting.predict import Predictor
from src.optimization.load_balancer import QuantumLoadBalancer

from fastapi.middleware.cors import CORSMiddleware


# -----------------------------
# Initialize App
# -----------------------------
app = FastAPI(title="Quantum AI Smart Grid API")


# -----------------------------
# Enable CORS (for frontend)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow all (for local dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Load Modules
# -----------------------------
predictor = Predictor()
quantum_balancer = QuantumLoadBalancer(reps=2, shots=2048, block_size_mw=200.0)


# -----------------------------
# Input Schemas
# -----------------------------
class InputData(BaseModel):
    date: str


class OptimizeInput(BaseModel):
    predicted_load_mw: float


# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def home():
    return {
        "message": "⚡ Quantum AI Smart Grid API Running",
        "status": "OK",
        "modules": ["ML Prediction", "Quantum Optimization (QAOA)"]
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(data: InputData):
    """
    Predict peak load & time for the given date,
    then run quantum optimization for load distribution.
    """
    result = predictor.predict_from_date(data.date)

    # If prediction failed, return error immediately
    if "error" in result:
        return result

    # ── Run Quantum Optimization ──
    predicted_load = result.get("predicted_peak_load_MW", 0)

    try:
        quantum_result = quantum_balancer.optimize(predicted_load)
        result["quantum_optimization"] = quantum_result
    except Exception as e:
        result["quantum_optimization"] = {
            "status": "error",
            "message": str(e)
        }

    return result


from datetime import datetime

@app.get("/predict/realtime")
def predict_realtime():
    """
    Real-time prediction using current date and live weather API.
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    result = predictor.predict_from_date(today_str)

    if "error" in result:
        return result

    predicted_load = result.get("predicted_peak_load_MW", 0)

    try:
        quantum_result = quantum_balancer.optimize(predicted_load)
        result["quantum_optimization"] = quantum_result
    except Exception as e:
        result["quantum_optimization"] = {
            "status": "error",
            "message": str(e)
        }

    return result


@app.post("/optimize")
def optimize(data: OptimizeInput):
    """
    Standalone quantum optimization endpoint.
    Accepts a predicted load and returns optimized transformer distribution.
    """
    try:
        result = quantum_balancer.optimize(data.predicted_load_mw)
        return result
    except Exception as e:
        return {"status": "error", "message": str(e)}