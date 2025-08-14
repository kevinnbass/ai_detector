"""
Integration tests for fallback mechanisms across the AI Detector system.

Tests how the system gracefully handles failures by falling back to
alternative methods, services, or degraded functionality modes.
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, patch, MagicMock

from src.core.detection.detector import DetectionEngine
from src.core.error_handling.exceptions import APIError, TimeoutError, ServiceError
from src.integrations.gemini.gemini_structured_analyzer import GeminiStructuredAnalyzer
from src.integrations.openrouter import OpenRouterClient
from src.api.rest.routes import DetectionService
from src.core.monitoring import get_metrics_collector


class TestFallbackMechanisms:
    """Test suite for system fallback mechanisms."""
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return {
            "ai_formal": "This comprehensive analysis demonstrates the multifaceted nature of contemporary discourse paradigms.",
            "ai_academic": "Furthermore, it should be noted that the implementation of sophisticated algorithms requires meticulous attention.",
            "human_casual": "hey just grabbed coffee and it's amazing! ☕ totally recommend this place",
            "human_personal": "omg my cat just did the funniest thing 😂 knocked over my water bottle"
        }
    
    @pytest.fixture
    def detection_engine(self):
        """Create detection engine with fallback configuration."""
        engine = DetectionEngine()
        engine.config.update({
            "enable_fallback": True,
            "fallback_chain": ["llm", "ml", "pattern"],
            "fallback_timeout": 5.0,
            "max_fallback_attempts": 3
        })
        return engine
    
    @pytest.fixture
    def mock_pattern_result(self):
        """Mock pattern detection result."""
        return {
            "is_ai_generated": True,
            "confidence_score": 0.7,
            "processing_time_ms": 50,
            "method_used": "pattern",
            "detected_patterns": ["formal_language", "complex_structure"]
        }
    
    @pytest.fixture
    def mock_ml_result(self):
        """Mock ML detection result."""
        return {
            "is_ai_generated": True,
            "confidence_score": 0.82,
            "processing_time_ms": 150,
            "method_used": "ml",
            "feature_importance": {"formality": 0.8, "complexity": 0.75}
        }
    
    @pytest.fixture
    def mock_llm_result(self):
        """Mock LLM detection result."""
        return {
            "is_ai_generated": True,
            "confidence_score": 0.9,
            "processing_time_ms": 800,
            "method_used": "llm",
            "analysis_details": {"writing_style": 0.9, "authenticity": 0.1}
        }
    
    def test_llm_to_ml_fallback(self, detection_engine, sample_texts, mock_ml_result):
        """Test fallback from LLM to ML detection when LLM fails."""
        with patch.object(detection_engine, '_llm_detection') as mock_llm, \
             patch.object(detection_engine, '_ml_detection') as mock_ml:
            
            # LLM fails
            mock_llm.side_effect = APIError("LLM service unavailable")
            
            # ML succeeds
            mock_ml.return_value = mock_ml_result
            
            # Test detection with fallback
            result = detection_engine.detect_ai_text(
                sample_texts["ai_formal"], 
                method="llm"
            )
            
            # Verify fallback occurred
            assert result["method_used"] == "ml"
            assert result["confidence_score"] == 0.82
            assert "fallback_info" in result
            assert result["fallback_info"]["original_method"] == "llm"
            assert result["fallback_info"]["fallback_reason"] == "LLM service unavailable"
    
    def test_ml_to_pattern_fallback(self, detection_engine, sample_texts, mock_pattern_result):
        """Test fallback from ML to pattern detection when ML fails."""
        with patch.object(detection_engine, '_llm_detection') as mock_llm, \
             patch.object(detection_engine, '_ml_detection') as mock_ml, \
             patch.object(detection_engine, '_pattern_detection') as mock_pattern:
            
            # LLM fails
            mock_llm.side_effect = APIError("LLM service unavailable")
            
            # ML fails
            mock_ml.side_effect = ServiceError("ML model loading failed")
            
            # Pattern succeeds
            mock_pattern.return_value = mock_pattern_result
            
            # Test detection with double fallback
            result = detection_engine.detect_ai_text(
                sample_texts["ai_formal"],
                method="llm"
            )
            
            # Verify double fallback
            assert result["method_used"] == "pattern"
            assert result["confidence_score"] == 0.7
            assert "fallback_info" in result
            assert result["fallback_info"]["fallback_chain"] == ["llm", "ml", "pattern"]
    
    def test_complete_fallback_failure(self, detection_engine, sample_texts):
        """Test behavior when all fallback methods fail."""
        with patch.object(detection_engine, '_llm_detection') as mock_llm, \
             patch.object(detection_engine, '_ml_detection') as mock_ml, \
             patch.object(detection_engine, '_pattern_detection') as mock_pattern:
            
            # All methods fail
            mock_llm.side_effect = APIError("LLM service unavailable")
            mock_ml.side_effect = ServiceError("ML model unavailable")
            mock_pattern.side_effect = ServiceError("Pattern engine failed")
            
            # Should raise exception when all fallbacks fail
            with pytest.raises(ServiceError) as exc_info:
                detection_engine.detect_ai_text(
                    sample_texts["ai_formal"],
                    method="llm"
                )
            
            # Verify error indicates complete failure
            assert "all fallback methods failed" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_api_service_fallback(self, sample_texts):
        """Test API service fallback between different LLM providers."""
        # Test Gemini to OpenRouter fallback
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_gemini, \
             patch('httpx.AsyncClient.post') as mock_openrouter:
            
            # Gemini fails
            mock_gemini.GenerativeModel.return_value.generate_content.side_effect = APIError("Gemini API failed")
            
            # OpenRouter succeeds
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "choices": [{
                    "message": {
                        "content": json.dumps({
                            "is_ai_generated": True,
                            "confidence": 0.85,
                            "reasoning": "Formal academic writing style"
                        })
                    }
                }]
            }
            mock_openrouter.return_value = mock_response
            
            # Create service with fallback configuration
            service = DetectionService()
            service.config["llm_fallback_providers"] = ["gemini", "openrouter"]
            
            # Test fallback
            result = await service.analyze_with_fallback(sample_texts["ai_formal"])
            
            # Verify fallback to OpenRouter
            assert result["provider_used"] == "openrouter"
            assert result["is_ai_generated"] is True
    
    def test_timeout_fallback(self, detection_engine, sample_texts, mock_pattern_result):
        """Test fallback when method times out."""
        with patch.object(detection_engine, '_llm_detection') as mock_llm, \
             patch.object(detection_engine, '_pattern_detection') as mock_pattern:
            
            # LLM times out
            mock_llm.side_effect = TimeoutError("LLM request timed out")
            
            # Pattern succeeds quickly
            mock_pattern.return_value = mock_pattern_result
            
            # Test timeout fallback
            result = detection_engine.detect_ai_text(
                sample_texts["ai_formal"],
                method="llm"
            )
            
            # Verify fallback due to timeout
            assert result["method_used"] == "pattern"
            assert result["fallback_info"]["fallback_reason"] == "LLM request timed out"
    
    def test_ensemble_partial_fallback(self, detection_engine, sample_texts):
        """Test ensemble method with partial fallback."""
        with patch.object(detection_engine, '_llm_detection') as mock_llm, \
             patch.object(detection_engine, '_ml_detection') as mock_ml, \
             patch.object(detection_engine, '_pattern_detection') as mock_pattern:
            
            # LLM fails
            mock_llm.side_effect = APIError("LLM unavailable")
            
            # ML and Pattern succeed
            mock_ml.return_value = {
                "is_ai_generated": True,
                "confidence_score": 0.8,
                "method_used": "ml"
            }
            mock_pattern.return_value = {
                "is_ai_generated": True,
                "confidence_score": 0.75,
                "method_used": "pattern"
            }
            
            # Test ensemble with partial fallback
            result = detection_engine.detect_ai_text(
                sample_texts["ai_formal"],
                method="ensemble"
            )
            
            # Verify ensemble worked with available methods
            assert result["method_used"] == "ensemble"
            assert "ensemble_details" in result
            assert len(result["ensemble_details"]["successful_methods"]) == 2
            assert "ml" in result["ensemble_details"]["successful_methods"]
            assert "pattern" in result["ensemble_details"]["successful_methods"]
            assert "llm" in result["ensemble_details"]["failed_methods"]
    
    def test_degraded_mode_fallback(self, detection_engine, sample_texts):
        """Test degraded mode when advanced features fail."""
        with patch.object(detection_engine, '_llm_detection') as mock_llm, \
             patch.object(detection_engine, '_ml_detection') as mock_ml:
            
            # Advanced methods fail
            mock_llm.side_effect = ServiceError("LLM service down")
            mock_ml.side_effect = ServiceError("ML service down")
            
            # Enable degraded mode
            detection_engine.config["enable_degraded_mode"] = True
            
            # Test degraded mode detection
            result = detection_engine.detect_ai_text(
                sample_texts["ai_formal"],
                method="ensemble"
            )
            
            # Verify degraded mode
            assert result["mode"] == "degraded"
            assert result["method_used"] == "pattern"  # Only basic method available
            assert result["confidence_score"] < 0.8  # Lower confidence in degraded mode
    
    @pytest.mark.asyncio
    async def test_cache_fallback_mechanism(self, sample_texts):
        """Test fallback to cached results when services fail."""
        from src.core.cache import get_cache_manager
        
        cache_manager = get_cache_manager()
        
        # Pre-populate cache
        cached_result = {
            "is_ai_generated": True,
            "confidence_score": 0.85,
            "from_cache": True,
            "cached_at": time.time()
        }
        
        text_hash = cache_manager.generate_hash(sample_texts["ai_formal"])
        cache_manager.set(f"detection:{text_hash}", cached_result, ttl=3600)
        
        # Mock service failure
        with patch('src.core.detection.detector.DetectionEngine.detect_ai_text') as mock_detect:
            mock_detect.side_effect = ServiceError("All detection services failed")
            
            # Service with cache fallback
            service = DetectionService()
            service.config["enable_cache_fallback"] = True
            
            # Test cache fallback
            result = await service.detect_with_cache_fallback(sample_texts["ai_formal"])
            
            # Verify cache fallback
            assert result["from_cache"] is True
            assert result["is_ai_generated"] is True
    
    def test_circuit_breaker_fallback(self, detection_engine, sample_texts):
        """Test circuit breaker pattern with fallback."""
        # Mock circuit breaker
        circuit_breaker = MagicMock()
        circuit_breaker.state = "OPEN"  # Circuit is open
        
        detection_engine.circuit_breakers = {
            "llm": circuit_breaker
        }
        
        with patch.object(detection_engine, '_pattern_detection') as mock_pattern:
            mock_pattern.return_value = {
                "is_ai_generated": True,
                "confidence_score": 0.7,
                "method_used": "pattern"
            }
            
            # Test circuit breaker fallback
            result = detection_engine.detect_ai_text(
                sample_texts["ai_formal"],
                method="llm"
            )
            
            # Verify circuit breaker triggered fallback
            assert result["method_used"] == "pattern"
            assert "circuit_breaker" in result["fallback_info"]["fallback_reason"]
    
    @pytest.mark.asyncio
    async def test_load_balancer_fallback(self, sample_texts):
        """Test load balancer fallback between service instances."""
        service_endpoints = [
            "http://api1.example.com/detect",
            "http://api2.example.com/detect", 
            "http://api3.example.com/detect"
        ]
        
        with patch('httpx.AsyncClient.post') as mock_post:
            # First two endpoints fail
            mock_post.side_effect = [
                APIError("Service 1 unavailable"),
                APIError("Service 2 unavailable"),
                MagicMock(status_code=200, json=lambda: {
                    "is_ai_generated": True,
                    "confidence_score": 0.85
                })
            ]
            
            # Load balancer with fallback
            load_balancer = LoadBalancer(service_endpoints)
            
            # Test load balancer fallback
            result = await load_balancer.detect_with_fallback(sample_texts["ai_formal"])
            
            # Verify fallback to third endpoint
            assert result["is_ai_generated"] is True
            assert mock_post.call_count == 3  # Tried all endpoints
    
    def test_graceful_degradation_levels(self, detection_engine, sample_texts):
        """Test multiple levels of graceful degradation."""
        degradation_levels = [
            "full_featured",    # All methods available
            "limited",          # Only fast methods
            "basic",           # Only pattern matching
            "minimal"          # Basic heuristics only
        ]
        
        for level in degradation_levels:
            detection_engine.set_degradation_level(level)
            
            result = detection_engine.detect_ai_text(
                sample_texts["ai_formal"],
                method="auto"  # Auto-select based on degradation level
            )
            
            # Verify appropriate method for degradation level
            if level == "full_featured":
                assert result["method_used"] in ["llm", "ensemble"]
            elif level == "limited":
                assert result["method_used"] in ["ml", "pattern"]
            elif level == "basic":
                assert result["method_used"] == "pattern"
            elif level == "minimal":
                assert result["method_used"] == "heuristic"
            
            assert result["degradation_level"] == level
    
    def test_fallback_metrics_collection(self, detection_engine, sample_texts):
        """Test metrics collection during fallback operations."""
        metrics = get_metrics_collector()
        
        with patch.object(detection_engine, '_llm_detection') as mock_llm, \
             patch.object(detection_engine, '_pattern_detection') as mock_pattern:
            
            # LLM fails, triggers fallback
            mock_llm.side_effect = APIError("LLM failed")
            mock_pattern.return_value = {
                "is_ai_generated": True,
                "confidence_score": 0.7,
                "method_used": "pattern"
            }
            
            # Execute with fallback
            detection_engine.detect_ai_text(sample_texts["ai_formal"], method="llm")
            
            # Verify fallback metrics
            fallback_counter = metrics.get_metric("fallback_total")
            if fallback_counter:
                assert fallback_counter.get_value() >= 1
            
            method_fallback_counter = metrics.get_metric("method_fallback_total")
            if method_fallback_counter:
                labels = {"from_method": "llm", "to_method": "pattern"}
                assert method_fallback_counter.get_value(labels) >= 1
    
    @pytest.mark.asyncio
    async def test_async_fallback_chain(self, sample_texts):
        """Test asynchronous fallback chain execution."""
        async def failing_service_1():
            await asyncio.sleep(0.1)
            raise APIError("Service 1 failed")
        
        async def failing_service_2():
            await asyncio.sleep(0.1)
            raise TimeoutError("Service 2 timed out")
        
        async def working_service_3():
            await asyncio.sleep(0.1)
            return {
                "is_ai_generated": True,
                "confidence_score": 0.8,
                "service_used": "service_3"
            }
        
        # Async fallback chain
        services = [failing_service_1, failing_service_2, working_service_3]
        
        # Execute fallback chain
        for i, service in enumerate(services):
            try:
                result = await service()
                break
            except (APIError, TimeoutError) as e:
                if i == len(services) - 1:  # Last service
                    raise
                continue
        
        # Verify successful fallback
        assert result["service_used"] == "service_3"
        assert result["is_ai_generated"] is True
    
    def test_fallback_configuration_validation(self, detection_engine):
        """Test validation of fallback configuration."""
        # Test invalid fallback chain
        with pytest.raises(ValueError):
            detection_engine.config["fallback_chain"] = ["invalid_method"]
        
        # Test invalid timeout
        with pytest.raises(ValueError):
            detection_engine.config["fallback_timeout"] = -1
        
        # Test circular fallback
        with pytest.raises(ValueError):
            detection_engine.config["fallback_chain"] = ["llm", "ml", "llm"]  # Circular
    
    def test_fallback_priority_system(self, detection_engine, sample_texts):
        """Test priority-based fallback system."""
        # Configure method priorities
        detection_engine.config["method_priorities"] = {
            "llm": 1,      # Highest priority
            "ml": 2,       # Medium priority  
            "pattern": 3   # Lowest priority
        }
        
        with patch.object(detection_engine, '_llm_detection') as mock_llm, \
             patch.object(detection_engine, '_ml_detection') as mock_ml, \
             patch.object(detection_engine, '_pattern_detection') as mock_pattern:
            
            # High priority method fails
            mock_llm.side_effect = APIError("LLM failed")
            
            # Medium priority succeeds
            mock_ml.return_value = {
                "is_ai_generated": True,
                "confidence_score": 0.8,
                "method_used": "ml"
            }
            
            # Test priority-based fallback
            result = detection_engine.detect_ai_text(
                sample_texts["ai_formal"],
                method="auto"
            )
            
            # Should fallback to next highest priority
            assert result["method_used"] == "ml"
            assert result["fallback_info"]["priority_order"] == ["llm", "ml", "pattern"]


class LoadBalancer:
    """Mock load balancer for testing."""
    
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
    
    async def detect_with_fallback(self, text: str) -> Dict[str, Any]:
        """Try endpoints in order until one succeeds."""
        for endpoint in self.endpoints:
            try:
                # Simulate API call
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.post(endpoint, json={"text": text})
                    if response.status_code == 200:
                        return response.json()
            except Exception:
                continue
        
        raise ServiceError("All endpoints failed")