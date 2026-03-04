"""
Tests for error handling and logging
Feature: quantum-energy-load-balancing
"""
import pytest
import logging
import asyncio
from hypothesis import given, strategies as st, settings
from src.utils.error_handlers import (
    DataLoadingError,
    PreprocessingError,
    ModelError,
    OptimizationError,
    DatabaseError,
    APIError,
    handle_error,
    retry_on_failure,
    log_api_request
)
from src.utils.logging_config import setup_logging, get_logger


class TestErrorLogging:
    """Tests for error logging completeness"""
    
    def setup_method(self):
        """Setup logging for tests"""
        setup_logging("DEBUG")
        self.logger = get_logger(__name__)
    
    @given(
        error_type=st.sampled_from(['data_loading', 'model_training', 'optimization'])
    )
    @settings(max_examples=10)
    def test_error_logging_completeness(self, error_type, caplog):
        """
        Property 15: Error Logging Completeness
        For any error occurring during data loading, model training, or optimization,
        the system should log an entry containing the error message and a timestamp
        Feature: quantum-energy-load-balancing, Property 15: Error Logging Completeness
        Validates: Requirements 9.1, 9.2, 9.3
        """
        with caplog.at_level(logging.ERROR):
            test_error = Exception("Test error message")
            
            try:
                handle_error(test_error, error_type, reraise=True)
            except Exception:
                pass
            
            # Check that error was logged
            assert len(caplog.records) > 0
            
            # Check log contains error message
            log_messages = [record.message for record in caplog.records]
            assert any("error" in msg.lower() for msg in log_messages)
            
            # Check log contains timestamp (in extra fields)
            assert any(hasattr(record, 'timestamp') or 'timestamp' in str(record.__dict__) 
                      for record in caplog.records)
    
    def test_data_loading_error_logging(self, caplog):
        """Test data loading errors are logged"""
        with caplog.at_level(logging.ERROR):
            error = DataLoadingError("Failed to load CSV file")
            
            try:
                handle_error(error, 'data_loading')
            except:
                pass
            
            assert len(caplog.records) > 0
            assert "data loading" in caplog.records[0].message.lower()
    
    def test_model_training_error_logging(self, caplog):
        """Test model training errors are logged"""
        with caplog.at_level(logging.ERROR):
            error = ModelError("Training convergence failed")
            
            try:
                handle_error(error, 'model_training')
            except:
                pass
            
            assert len(caplog.records) > 0
            assert "model training" in caplog.records[0].message.lower()
    
    def test_optimization_error_logging(self, caplog):
        """Test optimization errors are logged"""
        with caplog.at_level(logging.ERROR):
            error = OptimizationError("QAOA execution timeout")
            
            try:
                handle_error(error, 'optimization')
            except:
                pass
            
            assert len(caplog.records) > 0
            assert "optimization" in caplog.records[0].message.lower()


class TestDatabaseRetry:
    """Tests for database retry behavior"""
    
    @pytest.mark.asyncio
    async def test_database_retry_behavior(self):
        """
        Property 16: Database Retry Behavior
        For any database operation that fails, the system should retry exactly 3 times
        Feature: quantum-energy-load-balancing, Property 16: Database Retry Behavior
        Validates: Requirements 9.4
        """
        attempt_count = [0]
        
        @retry_on_failure(max_retries=3, delay=0.01, backoff=1.0)
        async def failing_db_operation():
            attempt_count[0] += 1
            raise DatabaseError("Connection failed")
        
        with pytest.raises(DatabaseError):
            await failing_db_operation()
        
        # Should have tried 1 initial + 3 retries = 4 total attempts
        assert attempt_count[0] == 4
    
    @pytest.mark.asyncio
    async def test_successful_retry(self):
        """Test that retry succeeds if operation eventually works"""
        attempt_count = [0]
        
        @retry_on_failure(max_retries=3, delay=0.01, backoff=1.0)
        async def eventually_successful_operation():
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise DatabaseError("Temporary failure")
            return "success"
        
        result = await eventually_successful_operation()
        
        assert result == "success"
        assert attempt_count[0] == 3
    
    def test_sync_retry_behavior(self):
        """Test retry behavior for synchronous functions"""
        attempt_count = [0]
        
        @retry_on_failure(max_retries=3, delay=0.01, backoff=1.0)
        def failing_operation():
            attempt_count[0] += 1
            raise DatabaseError("Operation failed")
        
        with pytest.raises(DatabaseError):
            failing_operation()
        
        assert attempt_count[0] == 4


class TestAPIRequestLogging:
    """Tests for API request logging"""
    
    @given(
        endpoint=st.text(min_size=1, max_size=50),
        method=st.sampled_from(['GET', 'POST', 'PUT', 'DELETE']),
        status_code=st.integers(min_value=200, max_value=599)
    )
    @settings(max_examples=20)
    def test_api_request_logging(self, endpoint, method, status_code, caplog):
        """
        Property 17: API Request Logging
        For any API request (successful or failed), the system should log an entry
        containing the timestamp, endpoint path, and HTTP status code
        Feature: quantum-energy-load-balancing, Property 17: API Request Logging
        Validates: Requirements 9.5
        """
        with caplog.at_level(logging.INFO):
            log_api_request(endpoint, method, status_code, duration=0.123)
            
            # Check that request was logged
            assert len(caplog.records) > 0
            
            # Check log contains endpoint
            log_record = caplog.records[0]
            assert hasattr(log_record, 'endpoint') or 'endpoint' in str(log_record.__dict__)
            
            # Check log contains status code
            assert hasattr(log_record, 'status_code') or 'status_code' in str(log_record.__dict__)
            
            # Check log contains timestamp
            assert hasattr(log_record, 'timestamp') or 'timestamp' in str(log_record.__dict__)
    
    def test_api_request_with_duration(self, caplog):
        """Test API request logging includes duration"""
        with caplog.at_level(logging.INFO):
            log_api_request("/api/forecast", "POST", 200, duration=1.234)
            
            assert len(caplog.records) > 0
            log_record = caplog.records[0]
            assert hasattr(log_record, 'duration_seconds') or 'duration_seconds' in str(log_record.__dict__)


class TestCustomExceptions:
    """Tests for custom exception classes"""
    
    def test_data_loading_error(self):
        """Test DataLoadingError can be raised and caught"""
        with pytest.raises(DataLoadingError):
            raise DataLoadingError("Test error")
    
    def test_preprocessing_error(self):
        """Test PreprocessingError can be raised and caught"""
        with pytest.raises(PreprocessingError):
            raise PreprocessingError("Test error")
    
    def test_model_error(self):
        """Test ModelError can be raised and caught"""
        with pytest.raises(ModelError):
            raise ModelError("Test error")
    
    def test_optimization_error(self):
        """Test OptimizationError can be raised and caught"""
        with pytest.raises(OptimizationError):
            raise OptimizationError("Test error")
    
    def test_database_error(self):
        """Test DatabaseError can be raised and caught"""
        with pytest.raises(DatabaseError):
            raise DatabaseError("Test error")
    
    def test_api_error(self):
        """Test APIError can be raised and caught"""
        with pytest.raises(APIError):
            raise APIError("Test error")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
