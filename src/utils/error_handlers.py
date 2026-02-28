"""
Error handling utilities and custom exceptions
"""
import logging
import traceback
from datetime import datetime
from typing import Optional, Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)


# Custom Exception Classes
class DataLoadingError(Exception):
    """Error during data loading"""
    pass


class PreprocessingError(Exception):
    """Error during preprocessing"""
    pass


class ModelError(Exception):
    """Error during model training or inference"""
    pass


class OptimizationError(Exception):
    """Error during optimization"""
    pass


class DatabaseError(Exception):
    """Error during database operations"""
    pass


class APIError(Exception):
    """Error in API operations"""
    pass


def handle_error(
    error: Exception,
    error_type: str,
    context: Optional[Dict[str, Any]] = None,
    reraise: bool = True
) -> None:
    """
    Handle and log errors with structured logging
    
    Args:
        error: The exception that occurred
        error_type: Type of error (data_loading, model_training, optimization, database, api)
        context: Additional context information
        reraise: Whether to reraise the exception
    """
    timestamp = datetime.utcnow().isoformat()
    
    error_info = {
        'timestamp': timestamp,
        'error_type': error_type,
        'error_message': str(error),
        'error_class': error.__class__.__name__,
        'stack_trace': traceback.format_exc()
    }
    
    if context:
        error_info.update(context)
    
    # Log based on error type
    if error_type == 'data_loading':
        logger.error("Data loading error occurred", extra=error_info)
    elif error_type == 'model_training':
        logger.error("Model training error occurred", extra=error_info)
    elif error_type == 'optimization':
        logger.error("Optimization error occurred", extra=error_info)
    elif error_type == 'database':
        logger.error("Database error occurred", extra=error_info)
    elif error_type == 'api':
        logger.error("API error occurred", extra=error_info)
    else:
        logger.error("Unknown error occurred", extra=error_info)
    
    if reraise:
        raise error


def retry_on_failure(max_retries: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """
    Decorator to retry function on failure with exponential backoff
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries (seconds)
        backoff: Backoff multiplier for delay
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            import asyncio
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}",
                            extra={
                                'function': func.__name__,
                                'attempt': attempt + 1,
                                'max_retries': max_retries,
                                'error': str(e),
                                'retry_delay': current_delay
                            }
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_retries} retry attempts failed for {func.__name__}",
                            extra={
                                'function': func.__name__,
                                'max_retries': max_retries,
                                'final_error': str(e)
                            }
                        )
            
            raise last_exception
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            import time
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_retries} failed for {func.__name__}",
                            extra={
                                'function': func.__name__,
                                'attempt': attempt + 1,
                                'max_retries': max_retries,
                                'error': str(e),
                                'retry_delay': current_delay
                            }
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"All {max_retries} retry attempts failed for {func.__name__}",
                            extra={
                                'function': func.__name__,
                                'max_retries': max_retries,
                                'final_error': str(e)
                            }
                        )
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def log_api_request(endpoint: str, method: str, status_code: int, duration: float = None):
    """
    Log API request with structured information
    
    Args:
        endpoint: API endpoint path
        method: HTTP method
        status_code: HTTP status code
        duration: Request duration in seconds
    """
    timestamp = datetime.utcnow().isoformat()
    
    log_data = {
        'timestamp': timestamp,
        'endpoint': endpoint,
        'method': method,
        'status_code': status_code
    }
    
    if duration is not None:
        log_data['duration_seconds'] = duration
    
    logger.info("API request processed", extra=log_data)
