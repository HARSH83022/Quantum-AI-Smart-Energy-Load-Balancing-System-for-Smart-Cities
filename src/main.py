"""
Main FastAPI application entry point for Quantum-AI Smart Energy Load Balancing System
"""
import logging
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from dotenv import load_dotenv
import os

# Import routes
from src.api.routes import router
from src.utils.logging_config import setup_logging
from src.utils.error_handlers import log_api_request, APIError

# Load environment variables
load_dotenv()

# Setup structured logging
setup_logging(os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI application"""
    logger.info("Starting Quantum-AI Smart Energy Load Balancing System")
    # Initialize database connection
    from src.database.connection import init_db
    await init_db()
    logger.info("Database initialized")
    yield
    logger.info("Shutting down Quantum-AI Smart Energy Load Balancing System")


# Create FastAPI application
app = FastAPI(
    title="Quantum-AI Smart Energy Load Balancing System",
    description="Production-ready backend system combining LSTM forecasting with quantum-inspired optimization",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all API requests"""
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    log_api_request(
        endpoint=str(request.url.path),
        method=request.method,
        status_code=response.status_code,
        duration=duration
    )
    
    return response


# Error handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.error(f"Validation error: {exc}", extra={
        'endpoint': str(request.url.path),
        'method': request.method,
        'errors': exc.errors()
    })
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Validation Error",
            "message": "Invalid request format",
            "details": exc.errors()
        }
    )


@app.exception_handler(APIError)
async def api_error_handler(request: Request, exc: APIError):
    """Handle API errors"""
    logger.error(f"API error: {exc}", extra={
        'endpoint': str(request.url.path),
        'method': request.method
    })
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "API Error",
            "message": str(exc)
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}", extra={
        'endpoint': str(request.url.path),
        'method': request.method
    }, exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred"
        }
    )


# Include API routes
app.include_router(router)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Quantum-AI Smart Energy Load Balancing System",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "core": [
                "GET /api/data/load",
                "POST /api/forecast",
                "POST /api/optimize",
                "GET /api/results"
            ],
            "research": [
                "POST /api/scenarios/generate",
                "POST /api/risk/analyze",
                "GET /api/frequency/features",
                "POST /api/optimize/robust"
            ],
            "health": [
                "GET /health"
            ]
        }
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
