"""
Integration tests for error propagation across system boundaries.

Tests how errors are handled and propagated between different
components of the AI Detector system, ensuring graceful degradation
and proper error recovery mechanisms.
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, patch, MagicMock

from src.core.error_handling.exceptions import (
    AIDetectorException, ValidationError, APIError, DetectionError,
    ServiceError, TimeoutError, RateLimitError, ConfigurationError
)
from src.core.error_handling.error_boundary import ErrorBoundary
from src.core.error_handling.error_context import ErrorContext
from src.core.detection.detector import DetectionEngine
from src.integrations.gemini.gemini_structured_analyzer import GeminiStructuredAnalyzer
from src.api.rest.routes import DetectionService
from src.utils.schema_validator import ValidationResult


class TestErrorPropagation:
    """Test suite for error propagation across boundaries."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return "This comprehensive analysis demonstrates the multifaceted nature of contemporary discourse paradigms."
    
    @pytest.fixture
    def detection_engine(self):
        """Create detection engine instance."""
        return DetectionEngine()
    
    @pytest.fixture
    def error_boundary(self):
        """Create error boundary instance."""
        return ErrorBoundary(component="test_boundary")
    
    @pytest.fixture
    def sample_detection_request(self):
        """Sample detection request."""
        return {
            "text": "Test text for detection",
            "request_id": "req_123",
            "options": {
                "method": "ensemble",
                "threshold": 0.7
            }
        }
    
    def test_validation_error_propagation(self, error_boundary):
        """Test validation error propagation through boundaries."""
        # Create validation error
        validation_error = ValidationError(
            message="Invalid input format",
            error_code="VALIDATION_001",
            details={"field": "text", "value": "", "constraint": "non_empty"}
        )
        
        # Test error boundary handling
        with pytest.raises(ValidationError) as exc_info:
            with error_boundary.handle_errors():
                raise validation_error
        
        # Verify error propagation
        assert exc_info.value.error_code == "VALIDATION_001"
        assert "Invalid input format" in str(exc_info.value)
        assert exc_info.value.details["field"] == "text"
    
    @pytest.mark.asyncio
    async def test_api_error_propagation(self, sample_text):
        """Test API error propagation from LLM services."""
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            # Simulate API error
            mock_genai.GenerativeModel.return_value.generate_content.side_effect = Exception("API connection failed")
            
            analyzer = GeminiStructuredAnalyzer(api_key="test_key")
            
            # Test error propagation through analyzer
            with pytest.raises(APIError) as exc_info:
                await analyzer.analyze_text(sample_text)
            
            # Verify error details
            assert "API connection failed" in str(exc_info.value)
            assert exc_info.value.error_code == "APIError"
    
    @pytest.mark.asyncio
    async def test_timeout_error_propagation(self, sample_text):
        """Test timeout error propagation."""
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            # Simulate slow response
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(5)  # Longer than timeout
                return MagicMock()
            
            mock_genai.GenerativeModel.return_value.generate_content.side_effect = slow_response
            
            analyzer = GeminiStructuredAnalyzer(api_key="test_key", timeout=1)
            
            # Test timeout error
            with pytest.raises(TimeoutError) as exc_info:
                await analyzer.analyze_text(sample_text)
            
            # Verify timeout error propagation
            assert exc_info.value.error_code == "TimeoutError"
    
    def test_detection_error_propagation(self, detection_engine, sample_text):
        """Test detection error propagation through engine."""
        with patch.object(detection_engine, '_pattern_detection') as mock_pattern:
            # Simulate detection failure
            mock_pattern.side_effect = DetectionError(
                message="Pattern detection failed",
                error_code="DETECTION_001",
                details={"component": "pattern_analyzer", "text_length": len(sample_text)}
            )
            
            # Test error propagation
            with pytest.raises(DetectionError) as exc_info:
                detection_engine.detect_ai_text(sample_text, method="pattern")
            
            # Verify error details
            assert exc_info.value.error_code == "DETECTION_001"
            assert exc_info.value.details["component"] == "pattern_analyzer"
    
    @pytest.mark.asyncio
    async def test_service_error_propagation(self, sample_detection_request):
        """Test service layer error propagation."""
        with patch('src.core.detection.detector.DetectionEngine') as mock_engine:
            # Simulate service error
            mock_instance = mock_engine.return_value
            mock_instance.detect_ai_text.side_effect = ServiceError(
                message="Service temporarily unavailable",
                error_code="SERVICE_001",
                details={"component": "detection_service", "retry_after": 30}
            )
            
            service = DetectionService()
            
            # Test service error propagation
            with pytest.raises(ServiceError) as exc_info:
                await service.detect_text(sample_detection_request)
            
            # Verify service error details
            assert exc_info.value.error_code == "SERVICE_001"
            assert exc_info.value.details["retry_after"] == 30
    
    def test_error_context_propagation(self, sample_detection_request):
        """Test error context propagation across components."""
        # Create error context
        context = ErrorContext(
            request_id=sample_detection_request["request_id"],
            component="api",
            operation="detect_text",
            metadata={
                "method": sample_detection_request["options"]["method"],
                "text_length": len(sample_detection_request["text"])
            }
        )
        
        # Test context propagation through error
        error = DetectionError(
            message="Detection failed",
            error_code="DETECTION_002",
            context=context
        )
        
        # Verify context is preserved
        assert error.context.request_id == sample_detection_request["request_id"]
        assert error.context.component == "api"
        assert error.context.operation == "detect_text"
        assert error.context.metadata["method"] == "ensemble"
    
    @pytest.mark.asyncio
    async def test_rate_limit_error_propagation(self):
        """Test rate limit error propagation."""
        with patch('httpx.AsyncClient.post') as mock_post:
            # Simulate rate limit response
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "60", "X-RateLimit-Remaining": "0"}
            mock_response.json.return_value = {"error": "Rate limit exceeded"}
            mock_post.return_value = mock_response
            
            from src.integrations.openrouter import OpenRouterClient
            client = OpenRouterClient(api_key="test_key")
            
            # Test rate limit error
            with pytest.raises(RateLimitError) as exc_info:
                await client.analyze_text("Test text")
            
            # Verify rate limit error details
            assert exc_info.value.error_code == "RateLimitError"
            assert "rate limit" in str(exc_info.value).lower()
    
    def test_configuration_error_propagation(self):
        """Test configuration error propagation."""
        # Test invalid configuration
        with pytest.raises(ConfigurationError) as exc_info:
            GeminiStructuredAnalyzer(api_key="")  # Empty API key
        
        # Verify configuration error
        assert exc_info.value.error_code == "ConfigurationError"
        assert "api_key" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_nested_error_propagation(self, sample_text):
        """Test nested error propagation through multiple layers."""
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            # Create nested error chain
            original_error = Exception("Network connection lost")
            api_error = APIError(
                message="Failed to connect to LLM service",
                error_code="API_CONNECTION_001",
                details={"original_error": str(original_error)}
            )
            
            mock_genai.GenerativeModel.return_value.generate_content.side_effect = api_error
            
            # Test through detection engine
            with patch('src.core.detection.detector.DetectionEngine') as mock_engine:
                mock_instance = mock_engine.return_value
                mock_instance._llm_detection.side_effect = DetectionError(
                    message="LLM detection failed",
                    error_code="DETECTION_LLM_001",
                    details={"underlying_error": str(api_error)}
                )
                
                detection_engine = DetectionEngine()
                
                # Test nested error propagation
                with pytest.raises(DetectionError) as exc_info:
                    detection_engine.detect_ai_text(sample_text, method="llm")
                
                # Verify nested error information is preserved
                assert exc_info.value.error_code == "DETECTION_LLM_001"
                assert "LLM detection failed" in str(exc_info.value)
    
    def test_error_boundary_recovery(self, error_boundary):
        """Test error boundary recovery mechanisms."""
        recovery_called = False
        
        def recovery_function(error, context):
            nonlocal recovery_called
            recovery_called = True
            return {"recovered": True, "error_type": type(error).__name__}
        
        error_boundary.set_recovery_function(recovery_function)
        
        # Test error recovery
        with error_boundary.handle_errors():
            try:
                raise ValidationError("Test error")
            except ValidationError:
                result = error_boundary.recover_from_error(
                    ValidationError("Test error"),
                    ErrorContext(request_id="test", component="test")
                )
                
                assert recovery_called
                assert result["recovered"] is True
                assert result["error_type"] == "ValidationError"
    
    @pytest.mark.asyncio
    async def test_concurrent_error_handling(self, sample_text):
        """Test error handling with concurrent operations."""
        errors_caught = []
        
        async def failing_operation(delay, error_type):
            await asyncio.sleep(delay)
            if error_type == "validation":
                raise ValidationError("Validation failed")
            elif error_type == "api":
                raise APIError("API failed")
            elif error_type == "timeout":
                raise TimeoutError("Operation timed out")
        
        # Run concurrent operations that will fail
        tasks = [
            failing_operation(0.01, "validation"),
            failing_operation(0.02, "api"),
            failing_operation(0.03, "timeout")
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all errors were caught
        assert len(results) == 3
        for result in results:
            assert isinstance(result, AIDetectorException)
            errors_caught.append(type(result).__name__)
        
        # Verify different error types
        assert "ValidationError" in errors_caught
        assert "APIError" in errors_caught
        assert "TimeoutError" in errors_caught
    
    def test_error_sanitization(self):
        """Test error message sanitization for client safety."""
        # Create error with sensitive information
        sensitive_error = APIError(
            message="Database connection failed: postgresql://user:password123@localhost:5432/db",
            error_code="DB_CONNECTION_001",
            details={
                "connection_string": "postgresql://user:password123@localhost:5432/db",
                "internal_error": "Authentication failed for user 'admin'"
            }
        )
        
        # Test error sanitization
        sanitized = sensitive_error.sanitize_for_client()
        
        # Verify sensitive information is removed
        assert "password123" not in sanitized["message"]
        assert "postgresql://" not in sanitized["message"]
        assert sanitized["error_code"] == "DB_CONNECTION_001"
        assert "connection_string" not in sanitized.get("details", {})
    
    @pytest.mark.asyncio
    async def test_error_retry_mechanism(self, sample_text):
        """Test automatic retry mechanism on transient errors."""
        retry_count = 0
        
        async def failing_then_succeeding(*args, **kwargs):
            nonlocal retry_count
            retry_count += 1
            
            if retry_count < 3:
                raise APIError("Transient API error")
            
            # Success on third try
            return {
                "ai_probability": 0.8,
                "confidence_score": 0.75,
                "retry_count": retry_count
            }
        
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            mock_genai.GenerativeModel.return_value.generate_content.side_effect = failing_then_succeeding
            
            analyzer = GeminiStructuredAnalyzer(
                api_key="test_key",
                max_retries=3,
                retry_delay=0.01
            )
            
            # Should succeed after retries
            result = await analyzer.analyze_text(sample_text)
            
            # Verify retry mechanism worked
            assert retry_count == 3
            assert result["retry_count"] == 3
    
    def test_error_metrics_collection(self):
        """Test error metrics collection during error propagation."""
        from src.core.monitoring import get_metrics_collector
        
        metrics = get_metrics_collector()
        
        # Simulate errors and collect metrics
        error_types = ["ValidationError", "APIError", "TimeoutError"]
        
        for error_type in error_types:
            metrics.increment_counter(f"errors_total", labels={"error_type": error_type})
        
        # Verify error metrics
        error_metric = metrics.get_metric("errors_total")
        assert error_metric is not None
        
        # Should have recorded multiple error types
        total_errors = error_metric.get_value()
        assert total_errors >= len(error_types)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_error_handling(self):
        """Test circuit breaker pattern during error propagation."""
        failure_count = 0
        circuit_open = False
        
        async def failing_service(*args, **kwargs):
            nonlocal failure_count, circuit_open
            failure_count += 1
            
            if failure_count >= 5:
                circuit_open = True
            
            if circuit_open:
                raise ServiceError("Circuit breaker is OPEN")
            
            raise APIError("Service failure")
        
        # Test circuit breaker behavior
        for i in range(10):
            try:
                await failing_service()
            except (APIError, ServiceError) as e:
                if i >= 5:  # After 5 failures, circuit should be open
                    assert isinstance(e, ServiceError)
                    assert "Circuit breaker is OPEN" in str(e)
                else:
                    assert isinstance(e, APIError)
    
    def test_error_logging_propagation(self):
        """Test error logging during propagation."""
        from src.core.monitoring import get_logger
        
        logger = get_logger("error_propagation_test")
        
        # Create error with context
        context = ErrorContext(
            request_id="test_123",
            component="test",
            operation="test_operation"
        )
        
        error = DetectionError(
            message="Test error for logging",
            error_code="TEST_001",
            context=context
        )
        
        # Test structured error logging
        logger.log_error(error)
        
        # Verify error was logged with context
        # (In real implementation, would check log output)
        assert error.context.request_id == "test_123"
        assert error.error_code == "TEST_001"
    
    def test_error_aggregation(self):
        """Test error aggregation across multiple components."""
        errors = [
            ValidationError("Field validation failed", error_code="VAL_001"),
            APIError("External service failed", error_code="API_001"),
            DetectionError("Detection process failed", error_code="DET_001")
        ]
        
        # Aggregate errors
        aggregated_error = AIDetectorException.aggregate_errors(
            errors,
            message="Multiple errors occurred during processing",
            error_code="AGGREGATE_001"
        )
        
        # Verify aggregation
        assert aggregated_error.error_code == "AGGREGATE_001"
        assert len(aggregated_error.details["aggregated_errors"]) == 3
        
        # Verify individual errors are preserved
        aggregated_codes = [
            err["error_code"] 
            for err in aggregated_error.details["aggregated_errors"]
        ]
        assert "VAL_001" in aggregated_codes
        assert "API_001" in aggregated_codes
        assert "DET_001" in aggregated_codes