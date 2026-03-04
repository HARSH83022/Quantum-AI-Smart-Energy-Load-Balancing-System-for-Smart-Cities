"""
Property-based tests for database operations
Feature: quantum-energy-load-balancing
"""
import pytest
import asyncio
from hypothesis import given, strategies as st, settings
from datetime import datetime
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.exc import OperationalError
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.models import Base, RawData, Forecast, QUBOMatrix, OptimizationResult
from src.database.connection import retry_db_operation

# Test database URL (use in-memory SQLite for testing)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Create test engine and session
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture
async def test_db():
    """Create test database"""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


# Feature: quantum-energy-load-balancing, Property 2: Data Persistence Round-Trip
@pytest.mark.asyncio
@given(
    zone=st.sampled_from(["North", "South", "East", "West"]),
    demand_mw=st.floats(min_value=100.0, max_value=1000.0),
    transformer=st.sampled_from(["T1", "T2", "T3", "T4"]),
    capacity_mw=st.floats(min_value=500.0, max_value=1500.0)
)
@settings(max_examples=100, deadline=None)
async def test_raw_data_round_trip(test_db, zone, demand_mw, transformer, capacity_mw):
    """
    Property 2: Data Persistence Round-Trip
    For any valid data object, storing it in the database and then retrieving it 
    should yield an equivalent object.
    Validates: Requirements 1.4, 2.5, 3.6, 4.8, 5.4, 10.1, 10.2, 10.3, 10.4, 10.5
    """
    async with TestSessionLocal() as session:
        # Create raw data object
        raw_data = RawData(
            timestamp=datetime.now(),
            zone=zone,
            demand_mw=demand_mw,
            transformer=transformer,
            capacity_mw=capacity_mw,
            voltage=230.0,
            current=1000.0,
            temperature=25.0,
            humidity=50.0,
            hour=12,
            day_of_week=1,
            month=1
        )
        
        # Store in database
        session.add(raw_data)
        await session.commit()
        await session.refresh(raw_data)
        
        stored_id = raw_data.id
        
        # Retrieve from database
        retrieved = await session.get(RawData, stored_id)
        
        # Verify equivalence
        assert retrieved is not None
        assert retrieved.zone == zone
        assert abs(retrieved.demand_mw - demand_mw) < 0.01
        assert retrieved.transformer == transformer
        assert abs(retrieved.capacity_mw - capacity_mw) < 0.01


# Feature: quantum-energy-load-balancing, Property 16: Database Retry Behavior
@pytest.mark.asyncio
async def test_database_retry_behavior():
    """
    Property 16: Database Retry Behavior
    For any database operation that fails, the system should retry exactly 3 times 
    before raising a final error.
    Validates: Requirements 9.4
    """
    attempt_count = 0
    
    async def failing_operation():
        nonlocal attempt_count
        attempt_count += 1
        raise OperationalError("Test error", None, None)
    
    with pytest.raises(OperationalError):
        await retry_db_operation(failing_operation, max_retries=3, backoff_factor=0.1)
    
    # Verify exactly 3 attempts were made
    assert attempt_count == 3


# Feature: quantum-energy-load-balancing, Property 18: Referential Integrity Preservation
@pytest.mark.asyncio
async def test_referential_integrity(test_db):
    """
    Property 18: Referential Integrity Preservation
    For any pair of related records (e.g., forecast and its associated QUBO matrix), 
    the foreign key relationship should be maintained and queryable.
    Validates: Requirements 10.6
    """
    async with TestSessionLocal() as session:
        # Create forecast
        forecast = Forecast(
            forecast_timestamp=datetime.now(),
            zone="North",
            predicted_demand_mw=500.0,
            mae=10.0,
            rmse=15.0,
            model_version="v1.0"
        )
        session.add(forecast)
        await session.commit()
        await session.refresh(forecast)
        
        # Create QUBO matrix referencing forecast
        qubo = QUBOMatrix(
            forecast_id=forecast.id,
            matrix_data={"matrix": [[1, 0], [0, 1]]},
            num_variables=2
        )
        session.add(qubo)
        await session.commit()
        await session.refresh(qubo)
        
        # Verify foreign key relationship
        assert qubo.forecast_id == forecast.id
        
        # Query related records
        retrieved_qubo = await session.get(QUBOMatrix, qubo.id)
        assert retrieved_qubo.forecast_id == forecast.id
