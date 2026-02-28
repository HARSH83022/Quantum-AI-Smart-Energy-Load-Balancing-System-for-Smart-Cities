"""
Database connection module with retry logic
"""
import os
import logging
import time
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.exc import OperationalError

logger = logging.getLogger(__name__)

# Create base class for models
Base = declarative_base()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./quantum_energy.db")

# For PostgreSQL, use psycopg (not asyncpg) on Windows
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://")
elif DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql+psycopg://")

logger.info(f"Using database: {DATABASE_URL.split('@')[0] if '@' in DATABASE_URL else DATABASE_URL.split('///')[0]}")

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20
)

# Create session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)


async def get_db():
    """Dependency for getting database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created successfully")


async def retry_db_operation(operation, max_retries=3, backoff_factor=1):
    """
    Retry database operation with exponential backoff
    
    Args:
        operation: Async function to execute
        max_retries: Maximum number of retry attempts
        backoff_factor: Base delay for exponential backoff (seconds)
    
    Returns:
        Result of the operation
    
    Raises:
        Exception: If all retries fail
    """
    for attempt in range(max_retries):
        try:
            return await operation()
        except OperationalError as e:
            if attempt == max_retries - 1:
                logger.error(f"Database operation failed after {max_retries} attempts: {e}")
                raise
            
            delay = backoff_factor * (2 ** attempt)
            logger.warning(f"Database operation failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s: {e}")
            time.sleep(delay)
    
    raise Exception("Unexpected error in retry logic")
