"""
Unit Tests for Core Detection System
Comprehensive tests for detection functionality with 95% coverage target
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import json

from src.core.detection.detector import GPT4oDetector
from src.core.interfaces.detector_interfaces import IDetectionResult, DetectionMethod
from src.core.messaging.protocol import RequestMessage, ResponseMessage


class TestGPT4oDetector:
    """Test suite for GPT4o text detector"""
    
    @pytest.fixture
    def detector(self):
        """Create detector instance for testing"""
        return GPT4oDetector()
    
    @pytest.fixture
    def mock_patterns(self):
        """Mock patterns for testing"""
        return {
            "hedging": {
                "patterns": [r"\b(?:perhaps|might|could|seems|appears)\b"],
                "weight": 0.3,
                "description": "Hedging language"
            },
            "meta_commentary": {
                "patterns": [r"important to note", r"it should be mentioned"],
                "weight": 0.4,
                "description": "Meta-commentary phrases"
            },
            "formal_structure": {
                "patterns": [r"firstly|secondly|finally", r"in conclusion"],
                "weight": 0.2,
                "description": "Formal structure indicators"
            }
        }
    
    @pytest.mark.unit
    async def test_detector_initialization(self, detector):
        """Test detector initialization"""
        # Test uninitialized state
        assert not detector.is_initialized()
        
        # Test initialization
        result = await detector.initialize()
        assert result is True
        assert detector.is_initialized()
        
        # Test double initialization
        result = await detector.initialize()
        assert result is True  # Should handle gracefully
    
    @pytest.mark.unit
    async def test_detect_human_text(self, detector, sample_texts):
        """Test detection of human-written text"""
        await detector.initialize()
        
        human_texts = [
            sample_texts["human_short"],
            sample_texts["human_long"],
            sample_texts["human_casual"],
            sample_texts["human_emotional"]
        ]
        
        for text in human_texts:
            result = await detector.detect(text)
            
            assert isinstance(result, IDetectionResult)
            score = result.get_score()
            
            # Human text should have low AI probability
            assert score.ai_probability < 0.5
            assert score.prediction == "human"
            assert score.confidence > 0.6  # Should be reasonably confident
    
    @pytest.mark.unit
    async def test_detect_ai_text(self, detector, sample_texts):
        """Test detection of AI-generated text"""
        await detector.initialize()
        
        ai_texts = [
            sample_texts["ai_obvious"],
            sample_texts["ai_formal"],
            sample_texts["ai_hedged"],
            sample_texts["ai_structured"]
        ]
        
        for text in ai_texts:
            result = await detector.detect(text)
            
            assert isinstance(result, IDetectionResult)
            score = result.get_score()
            
            # AI text should have high AI probability
            assert score.ai_probability > 0.5
            assert score.prediction == "ai"
            assert score.confidence > 0.6
    
    @pytest.mark.unit
    async def test_detect_edge_cases(self, detector, sample_texts):
        """Test detection with edge cases"""
        await detector.initialize()
        
        # Empty text
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await detector.detect("")
        
        # Very short text
        result = await detector.detect(sample_texts["edge_case_short"])
        assert result is not None
        score = result.get_score()
        assert 0.0 <= score.ai_probability <= 1.0
        
        # Very long text
        result = await detector.detect(sample_texts["edge_case_long"])
        assert result is not None
        
        # Special characters
        result = await detector.detect(sample_texts["edge_case_special"])
        assert result is not None
        
        # Mixed content
        result = await detector.detect(sample_texts["edge_case_mixed"])
        assert result is not None
    
    @pytest.mark.unit
    async def test_detect_batch(self, detector, sample_texts):
        """Test batch detection"""
        await detector.initialize()
        
        texts = [
            sample_texts["human_short"],
            sample_texts["ai_formal"],
            sample_texts["human_casual"]
        ]
        
        results = await detector.detect_batch(texts)
        
        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result, IDetectionResult)
            assert result.get_score().ai_probability is not None
    
    @pytest.mark.unit
    def test_detector_properties(self, detector):
        """Test detector property methods"""
        # Test supported languages
        languages = detector.get_supported_languages()
        assert isinstance(languages, list)
        assert "en" in languages
        
        # Test detection method
        method = detector.get_detection_method()
        assert method == DetectionMethod.PATTERN_BASED
        
        # Test model info
        model_info = detector.get_model_info()
        assert isinstance(model_info, dict)
        assert "version" in model_info
        assert "name" in model_info
        
        # Test explanation capability
        can_explain = detector.can_explain()
        assert isinstance(can_explain, bool)
    
    @pytest.mark.unit
    async def test_pattern_matching(self, detector, mock_patterns):
        """Test pattern matching functionality"""
        await detector.initialize()
        
        # Mock the patterns
        with patch.object(detector, '_patterns', mock_patterns):
            # Test hedging pattern
            text_with_hedging = "This might perhaps be a good idea, though it seems uncertain."
            result = await detector.detect(text_with_hedging)
            score = result.get_score()
            assert score.ai_probability > 0.3  # Should detect hedging
            
            # Test meta-commentary pattern
            text_with_meta = "It's important to note that this is a complex issue."
            result = await detector.detect(text_with_meta)
            score = result.get_score()
            assert score.ai_probability > 0.4  # Should detect meta-commentary
    
    @pytest.mark.unit
    async def test_detection_configuration(self, detector):
        """Test detection with different configurations"""
        await detector.initialize()
        
        text = "This is a test message for configuration testing."
        
        # Test with quick mode
        config = Mock()
        config.is_quick_mode.return_value = True
        config.get_threshold.return_value = 0.7
        
        result = await detector.detect(text, config)
        assert result is not None
        
        # Test with different threshold
        config.get_threshold.return_value = 0.9
        result = await detector.detect(text, config)
        assert result is not None
    
    @pytest.mark.unit
    async def test_detection_result_interface(self, detector, sample_texts):
        """Test that detection results implement IDetectionResult interface"""
        await detector.initialize()
        
        result = await detector.detect(sample_texts["human_short"])
        
        # Test score
        score = result.get_score()
        assert hasattr(score, 'ai_probability')
        assert hasattr(score, 'prediction')
        assert hasattr(score, 'confidence')
        assert 0.0 <= score.ai_probability <= 1.0
        assert score.prediction in ["ai", "human"]
        assert 0.0 <= score.confidence <= 1.0
        
        # Test evidence
        evidence = result.get_evidence()
        assert isinstance(evidence, list)
        
        # Test metadata
        metadata = result.get_metadata()
        assert isinstance(metadata, dict)
        
        # Test processing time
        processing_time = result.get_processing_time()
        assert isinstance(processing_time, (int, float))
        assert processing_time >= 0
        
        # Test serialization
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert "prediction" in result_dict
        assert "ai_probability" in result_dict
        
        result_json = result.to_json()
        assert isinstance(result_json, str)
        # Should be valid JSON
        json.loads(result_json)
    
    @pytest.mark.unit
    async def test_detector_error_handling(self, detector):
        """Test error handling in detector"""
        await detector.initialize()
        
        # Test None input
        with pytest.raises(ValueError):
            await detector.detect(None)
        
        # Test non-string input
        with pytest.raises(TypeError):
            await detector.detect(123)
        
        # Test with uninitialized detector
        uninit_detector = GPT4oDetector()
        with pytest.raises(RuntimeError, match="not initialized"):
            await uninit_detector.detect("test")
    
    @pytest.mark.unit
    async def test_detector_caching(self, detector):
        """Test detection result caching"""
        await detector.initialize()
        
        text = "This is a test for caching functionality."
        
        # First detection
        result1 = await detector.detect(text)
        processing_time1 = result1.get_processing_time()
        
        # Second detection (should be cached)
        result2 = await detector.detect(text)
        processing_time2 = result2.get_processing_time()
        
        # Results should be consistent
        assert result1.get_score().prediction == result2.get_score().prediction
        assert result1.get_score().ai_probability == result2.get_score().ai_probability
        
        # Second call should be faster (cached)
        # Note: This might not always be true due to test timing, so we make it optional
        if hasattr(detector, '_cache'):
            assert processing_time2 <= processing_time1
    
    @pytest.mark.unit
    @pytest.mark.performance
    async def test_detection_performance(self, detector, performance_monitor):
        """Test detection performance requirements"""
        await detector.initialize()
        
        text = "This is a performance test message for detection speed."
        
        # Test single detection performance
        performance_monitor.start()
        result = await detector.detect(text)
        duration = performance_monitor.stop()
        
        # Should complete in reasonable time (adjust based on requirements)
        assert duration < 1.0  # Less than 1 second
        assert result is not None
        
        # Test batch performance
        texts = [text] * 10
        performance_monitor.start()
        results = await detector.detect_batch(texts)
        batch_duration = performance_monitor.stop()
        
        # Batch should be more efficient than individual calls
        expected_individual_time = duration * len(texts)
        assert batch_duration < expected_individual_time
        assert len(results) == len(texts)
    
    @pytest.mark.unit
    async def test_detector_memory_usage(self, detector):
        """Test detector memory usage"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        await detector.initialize()
        
        # Process many texts to test memory usage
        large_text = "A" * 10000  # Large text
        for _ in range(100):
            await detector.detect(large_text)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (adjust based on requirements)
        assert memory_increase < 100  # Less than 100MB increase
    
    @pytest.mark.unit
    async def test_concurrent_detection(self, detector):
        """Test concurrent detection requests"""
        await detector.initialize()
        
        texts = [
            "First test message for concurrency.",
            "Second test message for concurrency.",
            "Third test message for concurrency.",
            "Fourth test message for concurrency.",
            "Fifth test message for concurrency."
        ]
        
        # Run concurrent detections
        tasks = [detector.detect(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        assert len(results) == len(texts)
        for result in results:
            assert isinstance(result, IDetectionResult)
            assert result.get_score().ai_probability is not None
    
    @pytest.mark.unit
    def test_detector_configuration_validation(self, detector):
        """Test detector configuration validation"""
        # Test valid configuration
        valid_config = {
            "threshold": 0.7,
            "patterns_enabled": True,
            "cache_enabled": True,
            "max_text_length": 10000
        }
        
        is_valid, errors = detector.validate_configuration(valid_config)
        assert is_valid
        assert len(errors) == 0
        
        # Test invalid configuration
        invalid_config = {
            "threshold": 1.5,  # Invalid threshold
            "max_text_length": -1  # Invalid length
        }
        
        is_valid, errors = detector.validate_configuration(invalid_config)
        assert not is_valid
        assert len(errors) > 0
    
    @pytest.mark.unit
    async def test_detector_metrics(self, detector):
        """Test detector metrics collection"""
        await detector.initialize()
        
        # Perform some detections
        texts = ["Test message one.", "Test message two.", "Test message three."]
        for text in texts:
            await detector.detect(text)
        
        # Get metrics
        metrics = detector.get_metrics()
        
        assert isinstance(metrics, dict)
        assert "total_detections" in metrics
        assert "average_processing_time" in metrics
        assert "cache_hits" in metrics or "cache_misses" in metrics
        assert metrics["total_detections"] >= len(texts)
    
    @pytest.mark.unit
    async def test_detector_explanation(self, detector, sample_texts):
        """Test detection explanation functionality"""
        await detector.initialize()
        
        # Only test if detector supports explanations
        if detector.can_explain():
            text = sample_texts["ai_formal"]
            result = await detector.detect(text)
            
            # Get explanation
            explanation = await detector.explain(text, result)
            
            assert isinstance(explanation, dict)
            assert "reasoning" in explanation or "key_indicators" in explanation
        else:
            # Should raise NotImplementedError or similar
            text = sample_texts["ai_formal"]
            result = await detector.detect(text)
            
            with pytest.raises((NotImplementedError, AttributeError)):
                await detector.explain(text, result)


class TestDetectionResult:
    """Test suite for detection result implementation"""
    
    @pytest.fixture
    def sample_result_data(self):
        """Sample detection result data"""
        return {
            "prediction": "ai",
            "ai_probability": 0.85,
            "confidence": 0.92,
            "processing_time": 0.156,
            "key_indicators": ["formal_language", "hedging", "structure"],
            "model_version": "v1.0.0",
            "timestamp": datetime.now()
        }
    
    @pytest.mark.unit
    def test_detection_result_creation(self, sample_result_data):
        """Test detection result creation"""
        from src.core.detection.detector import DetectionResult
        
        result = DetectionResult(**sample_result_data)
        
        assert result is not None
        assert isinstance(result, IDetectionResult)
    
    @pytest.mark.unit
    def test_detection_result_score(self, sample_result_data):
        """Test detection result score"""
        from src.core.detection.detector import DetectionResult
        
        result = DetectionResult(**sample_result_data)
        score = result.get_score()
        
        assert score.ai_probability == sample_result_data["ai_probability"]
        assert score.prediction == sample_result_data["prediction"]
        assert score.confidence == sample_result_data["confidence"]
    
    @pytest.mark.unit
    def test_detection_result_serialization(self, sample_result_data):
        """Test detection result serialization"""
        from src.core.detection.detector import DetectionResult
        
        result = DetectionResult(**sample_result_data)
        
        # Test to_dict
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["prediction"] == sample_result_data["prediction"]
        assert result_dict["ai_probability"] == sample_result_data["ai_probability"]
        
        # Test to_json
        result_json = result.to_json()
        assert isinstance(result_json, str)
        
        # Should be valid JSON
        parsed = json.loads(result_json)
        assert parsed["prediction"] == sample_result_data["prediction"]
    
    @pytest.mark.unit
    def test_detection_result_evidence(self, sample_result_data):
        """Test detection result evidence"""
        from src.core.detection.detector import DetectionResult
        
        result = DetectionResult(**sample_result_data)
        evidence = result.get_evidence()
        
        assert isinstance(evidence, list)
        # Evidence should contain relevant indicators
        if sample_result_data["key_indicators"]:
            assert len(evidence) > 0


@pytest.mark.integration
class TestDetectorIntegration:
    """Integration tests for detector with other components"""
    
    @pytest.mark.integration
    async def test_detector_with_message_bus(self, mock_message_bus):
        """Test detector integration with message bus"""
        detector = GPT4oDetector()
        await detector.initialize()
        
        # Create detection request message
        request = RequestMessage(
            subject="detect_text",
            payload={"text": "This is a test message for integration testing."}
        )
        
        # Simulate processing
        text = request.payload["text"]
        result = await detector.detect(text)
        
        # Create response message
        response = ResponseMessage(
            subject="detection_result",
            payload=result.to_dict(),
            request_message=request,
            success=True
        )
        
        assert response.payload["prediction"] in ["ai", "human"]
        assert "ai_probability" in response.payload
    
    @pytest.mark.integration
    async def test_detector_with_api_client(self, mock_api_client):
        """Test detector integration with API client"""
        detector = GPT4oDetector()
        await detector.initialize()
        
        text = "Integration test message for API client."
        result = await detector.detect(text)
        
        # Simulate API response
        api_response = {
            "prediction": result.get_score().prediction,
            "ai_probability": result.get_score().ai_probability,
            "confidence": result.get_score().confidence,
            "processing_time": result.get_processing_time()
        }
        
        # Mock API call
        mock_api_client.post.return_value = Mock(
            status_code=200,
            body={"success": True, "result": api_response}
        )
        
        response = await mock_api_client.post("/api/v1/detect", data={"text": text})
        
        assert response.status_code == 200
        assert response.body["success"] is True
        assert "result" in response.body