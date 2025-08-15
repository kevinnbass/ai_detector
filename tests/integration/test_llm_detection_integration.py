"""
Integration tests for LLM API to detection engine integration.

Tests the integration between LLM services (Gemini, OpenRouter, etc.)
and the detection engine, including fallback mechanisms and error handling.
"""

import pytest
import asyncio
import json
import time
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, patch, MagicMock

from src.integrations.gemini.gemini_structured_analyzer import GeminiStructuredAnalyzer
from src.integrations.openrouter import OpenRouterClient
from src.core.detection.detector import DetectionEngine
from src.core.error_handling import APIError, TimeoutError, ValidationError
from src.utils.schema_validator import validate_llm_analysis, validate_detection_response


class TestLLMDetectionIntegration:
    """Test suite for LLM-Detection engine integration."""
    
    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return {
            "ai_formal": "This comprehensive analysis demonstrates the multifaceted nature of contemporary discourse paradigms, necessitating careful consideration of various methodological approaches.",
            "ai_academic": "Furthermore, it should be noted that the implementation of sophisticated algorithms requires meticulous attention to detail and adherence to established protocols.",
            "human_casual": "hey just grabbed coffee and it's amazing! ☕ totally recommend this place, the barista was super friendly too",
            "human_personal": "omg my cat just did the funniest thing 😂 knocked over my water bottle and then gave me this look like it was MY fault lol",
            "mixed_content": "The weather today is quite nice. Furthermore, it should be mentioned that atmospheric conditions significantly impact daily activities."
        }
    
    @pytest.fixture
    def mock_gemini_response(self):
        """Mock Gemini API response."""
        return {
            "analysis_id": "analysis_123456",
            "input_text": "This comprehensive analysis demonstrates...",
            "ai_probability": 0.85,
            "confidence_score": 0.78,
            "analysis_dimensions": {
                "writing_style": {
                    "formality_level": 0.92,
                    "complexity_score": 0.88,
                    "vocabulary_sophistication": 0.90,
                    "tone_consistency": 0.95
                },
                "language_patterns": {
                    "hedging_frequency": 0.12,
                    "modal_verb_usage": 0.08,
                    "passive_voice_ratio": 0.15,
                    "transition_marker_density": 0.25
                },
                "content_structure": {
                    "logical_flow": 0.85,
                    "topic_coherence": 0.80,
                    "argument_structure": 0.75,
                    "information_density": 0.60
                },
                "authenticity_markers": {
                    "personal_experience": 0.05,
                    "emotional_expression": 0.10,
                    "conversational_elements": 0.08,
                    "spontaneous_language": 0.15
                }
            },
            "detailed_reasoning": {
                "primary_indicators": [
                    {
                        "indicator": "High formality and complexity",
                        "weight": 0.8,
                        "evidence": "Academic language with sophisticated vocabulary"
                    }
                ],
                "conclusion_summary": "The text shows strong AI-generation patterns."
            },
            "metadata": {
                "llm_model": "gemini-2.5-flash-002",
                "processing_time_ms": 1200,
                "text_length": 95
            }
        }
    
    @pytest.fixture
    def mock_openrouter_response(self):
        """Mock OpenRouter API response."""
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "is_ai_generated": True,
                            "confidence": 0.82,
                            "reasoning": "Formal academic writing style with complex vocabulary",
                            "indicators": ["formal_language", "complex_structure", "academic_tone"]
                        })
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 150,
                "completion_tokens": 75,
                "total_tokens": 225
            }
        }
    
    @pytest.fixture
    def detection_engine(self):
        """Create detection engine instance."""
        return DetectionEngine()
    
    @pytest.mark.asyncio
    async def test_gemini_integration_success(self, sample_texts, mock_gemini_response):
        """Test successful Gemini API integration."""
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            # Setup mock
            mock_response = MagicMock()
            mock_response.text = json.dumps(mock_gemini_response)
            mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
            
            # Create analyzer
            analyzer = GeminiStructuredAnalyzer(api_key="test_key")
            
            # Test analysis
            result = await analyzer.analyze_text(sample_texts["ai_formal"])
            
            # Validate response format
            validation_result = validate_llm_analysis(result)
            assert validation_result.is_valid, f"LLM analysis validation failed: {validation_result.errors}"
            
            # Verify content
            assert result["ai_probability"] == 0.85
            assert result["confidence_score"] == 0.78
            assert "analysis_dimensions" in result
            assert "writing_style" in result["analysis_dimensions"]
    
    @pytest.mark.asyncio
    async def test_openrouter_integration_success(self, sample_texts, mock_openrouter_response):
        """Test successful OpenRouter API integration."""
        with patch('httpx.AsyncClient.post') as mock_post:
            # Setup mock
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openrouter_response
            mock_post.return_value = mock_response
            
            # Create client
            client = OpenRouterClient(api_key="test_key")
            
            # Test analysis
            result = await client.analyze_text(sample_texts["ai_formal"])
            
            # Verify response
            assert "is_ai_generated" in result
            assert "confidence" in result
            assert result["is_ai_generated"] is True
    
    @pytest.mark.asyncio
    async def test_detection_engine_llm_integration(self, detection_engine, sample_texts, mock_gemini_response):
        """Test detection engine integration with LLM services."""
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            # Setup Gemini mock
            mock_response = MagicMock()
            mock_response.text = json.dumps(mock_gemini_response)
            mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
            
            # Configure detection engine to use LLM
            detection_engine.config["use_llm"] = True
            detection_engine.config["llm_provider"] = "gemini"
            
            # Test detection
            result = detection_engine.detect_ai_text(
                text=sample_texts["ai_formal"],
                method="llm"
            )
            
            # Validate detection response
            validation_result = validate_detection_response(result)
            assert validation_result.is_valid, f"Detection response validation failed: {validation_result.errors}"
            
            # Verify result structure
            assert "is_ai_generated" in result
            assert "confidence_score" in result
            assert "processing_time_ms" in result
            assert "detection_details" in result
    
    @pytest.mark.asyncio
    async def test_ensemble_method_with_llm(self, detection_engine, sample_texts, mock_gemini_response):
        """Test ensemble detection method including LLM."""
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            # Setup mocks
            mock_response = MagicMock()
            mock_response.text = json.dumps(mock_gemini_response)
            mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
            
            # Mock pattern and ML detection
            with patch.object(detection_engine, '_pattern_detection') as mock_pattern, \
                 patch.object(detection_engine, '_ml_detection') as mock_ml:
                
                mock_pattern.return_value = {
                    "is_ai_generated": True,
                    "confidence_score": 0.75,
                    "detected_patterns": ["formal_language", "complex_structure"]
                }
                
                mock_ml.return_value = {
                    "is_ai_generated": True,
                    "confidence_score": 0.80,
                    "feature_importance": {"formality": 0.8, "complexity": 0.7}
                }
                
                # Test ensemble detection
                result = detection_engine.detect_ai_text(
                    text=sample_texts["ai_formal"],
                    method="ensemble"
                )
                
                # Verify ensemble combines all methods
                assert "detection_details" in result
                details = result["detection_details"]
                assert "individual_scores" in details
                assert "pattern_score" in details["individual_scores"]
                assert "ml_score" in details["individual_scores"]
                assert "llm_score" in details["individual_scores"]
    
    @pytest.mark.asyncio
    async def test_llm_api_error_handling(self, sample_texts):
        """Test LLM API error handling and fallback."""
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            # Simulate API error
            mock_genai.GenerativeModel.return_value.generate_content.side_effect = Exception("API Error")
            
            analyzer = GeminiStructuredAnalyzer(api_key="test_key")
            
            # Test error handling
            with pytest.raises(APIError):
                await analyzer.analyze_text(sample_texts["ai_formal"])
    
    @pytest.mark.asyncio
    async def test_llm_timeout_handling(self, sample_texts):
        """Test LLM API timeout handling."""
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            # Simulate timeout
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(10)  # Longer than timeout
                return MagicMock()
            
            mock_genai.GenerativeModel.return_value.generate_content.side_effect = slow_response
            
            analyzer = GeminiStructuredAnalyzer(api_key="test_key", timeout=1)
            
            # Test timeout handling
            with pytest.raises(TimeoutError):
                await analyzer.analyze_text(sample_texts["ai_formal"])
    
    @pytest.mark.asyncio
    async def test_llm_response_validation(self, sample_texts):
        """Test LLM response validation and error handling."""
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            # Test invalid JSON response
            mock_response = MagicMock()
            mock_response.text = "Invalid JSON response"
            mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
            
            analyzer = GeminiStructuredAnalyzer(api_key="test_key")
            
            with pytest.raises(ValidationError):
                await analyzer.analyze_text(sample_texts["ai_formal"])
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self, detection_engine, sample_texts):
        """Test fallback from LLM to pattern detection on failure."""
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            # Simulate LLM failure
            mock_genai.GenerativeModel.return_value.generate_content.side_effect = Exception("LLM Failed")
            
            # Mock pattern detection as fallback
            with patch.object(detection_engine, '_pattern_detection') as mock_pattern:
                mock_pattern.return_value = {
                    "is_ai_generated": True,
                    "confidence_score": 0.70,
                    "detected_patterns": ["formal_language"]
                }
                
                # Enable fallback
                detection_engine.config["enable_fallback"] = True
                
                # Test detection with fallback
                result = detection_engine.detect_ai_text(
                    text=sample_texts["ai_formal"],
                    method="llm"
                )
                
                # Should fallback to pattern detection
                assert result["is_ai_generated"] is True
                assert result["detection_details"]["method_used"] == "pattern_fallback"
    
    @pytest.mark.asyncio
    async def test_rate_limiting_handling(self, sample_texts):
        """Test rate limiting handling for LLM APIs."""
        with patch('httpx.AsyncClient.post') as mock_post:
            # Simulate rate limit error
            mock_response = MagicMock()
            mock_response.status_code = 429
            mock_response.headers = {"Retry-After": "60"}
            mock_post.return_value = mock_response
            
            client = OpenRouterClient(api_key="test_key")
            
            with pytest.raises(APIError) as exc_info:
                await client.analyze_text(sample_texts["ai_formal"])
            
            # Verify rate limit error
            assert "rate limit" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_llm_requests(self, sample_texts, mock_gemini_response):
        """Test concurrent LLM requests handling."""
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            # Setup mock with delay
            async def delayed_response(*args, **kwargs):
                await asyncio.sleep(0.1)  # Small delay
                mock_response = MagicMock()
                mock_response.text = json.dumps(mock_gemini_response)
                return mock_response
            
            mock_genai.GenerativeModel.return_value.generate_content.side_effect = delayed_response
            
            analyzer = GeminiStructuredAnalyzer(api_key="test_key")
            
            # Create multiple concurrent requests
            tasks = []
            for text in sample_texts.values():
                task = analyzer.analyze_text(text)
                tasks.append(task)
            
            # Wait for all requests
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all requests completed
            assert len(results) == len(sample_texts)
            for result in results:
                if not isinstance(result, Exception):
                    assert "ai_probability" in result
    
    @pytest.mark.asyncio
    async def test_llm_caching_integration(self, sample_texts, mock_gemini_response):
        """Test LLM response caching."""
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            mock_response = MagicMock()
            mock_response.text = json.dumps(mock_gemini_response)
            mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
            
            analyzer = GeminiStructuredAnalyzer(api_key="test_key", enable_caching=True)
            
            # First request
            result1 = await analyzer.analyze_text(sample_texts["ai_formal"])
            
            # Second request with same text (should use cache)
            result2 = await analyzer.analyze_text(sample_texts["ai_formal"])
            
            # Verify both results are identical
            assert result1 == result2
            
            # Verify API was called only once (due to caching)
            assert mock_genai.GenerativeModel.return_value.generate_content.call_count == 1
    
    def test_llm_configuration_validation(self):
        """Test LLM configuration validation."""
        # Test invalid API key
        with pytest.raises(ValueError):
            GeminiStructuredAnalyzer(api_key="")
        
        # Test invalid model
        with pytest.raises(ValueError):
            GeminiStructuredAnalyzer(api_key="valid_key", model="invalid_model")
        
        # Test invalid timeout
        with pytest.raises(ValueError):
            GeminiStructuredAnalyzer(api_key="valid_key", timeout=-1)
    
    @pytest.mark.asyncio
    async def test_batch_llm_processing(self, sample_texts, mock_gemini_response):
        """Test batch processing with LLM."""
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            mock_response = MagicMock()
            mock_response.text = json.dumps(mock_gemini_response)
            mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
            
            analyzer = GeminiStructuredAnalyzer(api_key="test_key")
            
            # Test batch analysis
            texts = list(sample_texts.values())
            results = await analyzer.analyze_batch(texts)
            
            # Verify batch results
            assert len(results) == len(texts)
            for result in results:
                assert "ai_probability" in result
                assert "confidence_score" in result
    
    @pytest.mark.asyncio
    async def test_llm_metrics_collection(self, sample_texts, mock_gemini_response):
        """Test metrics collection for LLM operations."""
        from src.core.monitoring import get_metrics_collector
        
        metrics = get_metrics_collector()
        
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            mock_response = MagicMock()
            mock_response.text = json.dumps(mock_gemini_response)
            mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
            
            analyzer = GeminiStructuredAnalyzer(api_key="test_key")
            
            # Analyze text (should record metrics)
            await analyzer.analyze_text(sample_texts["ai_formal"])
            
            # Verify metrics were recorded
            llm_requests = metrics.get_metric("llm_requests_total")
            if llm_requests:
                assert llm_requests.get_value() >= 1
            
            llm_duration = metrics.get_metric("llm_request_duration_ms")
            if llm_duration:
                duration_stats = llm_duration.get_value()
                assert duration_stats["count"] >= 1
    
    @pytest.mark.asyncio
    async def test_llm_structured_output_validation(self, sample_texts):
        """Test structured output validation from LLM."""
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            # Test with malformed response
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "ai_probability": "invalid_probability",  # Should be number
                "confidence_score": 1.5,  # Should be <= 1
                "missing_required_field": "test"
            })
            mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
            
            analyzer = GeminiStructuredAnalyzer(api_key="test_key")
            
            with pytest.raises(ValidationError):
                await analyzer.analyze_text(sample_texts["ai_formal"])
    
    @pytest.mark.asyncio
    async def test_multiple_llm_provider_integration(self, sample_texts, mock_gemini_response, mock_openrouter_response):
        """Test integration with multiple LLM providers."""
        detection_engine = DetectionEngine()
        
        # Test Gemini
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            mock_response = MagicMock()
            mock_response.text = json.dumps(mock_gemini_response)
            mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
            
            detection_engine.config["llm_provider"] = "gemini"
            result_gemini = detection_engine.detect_ai_text(
                text=sample_texts["ai_formal"],
                method="llm"
            )
            
            assert result_gemini["is_ai_generated"] is True
        
        # Test OpenRouter
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = mock_openrouter_response
            mock_post.return_value = mock_response
            
            detection_engine.config["llm_provider"] = "openrouter"
            result_openrouter = detection_engine.detect_ai_text(
                text=sample_texts["ai_formal"],
                method="llm"
            )
            
            assert result_openrouter["is_ai_generated"] is True
    
    @pytest.mark.asyncio
    async def test_llm_prompt_optimization(self, sample_texts):
        """Test LLM prompt optimization and versioning."""
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            # Track prompts sent to LLM
            sent_prompts = []
            
            def capture_prompt(*args, **kwargs):
                if args:
                    sent_prompts.append(args[0])
                mock_response = MagicMock()
                mock_response.text = json.dumps({"ai_probability": 0.5, "confidence_score": 0.6})
                return mock_response
            
            mock_genai.GenerativeModel.return_value.generate_content.side_effect = capture_prompt
            
            analyzer = GeminiStructuredAnalyzer(api_key="test_key", prompt_version="v2.1")
            
            await analyzer.analyze_text(sample_texts["ai_formal"])
            
            # Verify prompt was sent
            assert len(sent_prompts) >= 1
            # In real implementation, would verify prompt contains version-specific content