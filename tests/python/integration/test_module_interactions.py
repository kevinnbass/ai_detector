"""
Integration Tests for Module Interactions
Tests for interactions between different system modules and components
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.core.detection.detector import GPT4oDetector
from src.core.data.collectors import ManualDataCollector, TwitterDataCollector
from src.core.data.processors import TextPreprocessor, FeatureExtractor
from src.core.data.validators import DataValidator
from src.core.data.stores import FileDataStore, DatabaseDataStore
from src.core.data.pipeline import DataPipeline
from src.core.services.detection_service import DetectionService
from src.core.services.data_service import DataService
from src.core.services.service_registry import ServiceRegistry
from src.core.api_client.unified_client import UnifiedAPIClient
from src.core.messaging.protocol import MessageProtocol, RequestMessage, ResponseMessage
from src.core.interfaces.data_interfaces import DataSample, DataBatch


class TestDetectionDataIntegration:
    """Test integration between detection and data modules"""
    
    @pytest.fixture
    async def integrated_detection_pipeline(self, temp_dir):
        """Create integrated detection and data pipeline"""
        # Initialize components
        detector = GPT4oDetector()
        collector = ManualDataCollector()
        preprocessor = TextPreprocessor()
        validator = DataValidator()
        store = FileDataStore(base_path=str(temp_dir))
        
        # Initialize all components
        await detector.initialize()
        await collector.initialize()
        await preprocessor.initialize()
        await validator.initialize()
        await store.initialize()
        
        return {
            "detector": detector,
            "collector": collector,
            "preprocessor": preprocessor,
            "validator": validator,
            "store": store
        }
    
    @pytest.mark.integration
    async def test_data_collection_to_detection_flow(self, integrated_detection_pipeline):
        """Test flow from data collection to detection"""
        components = integrated_detection_pipeline
        
        # 1. Collect data manually
        samples = [
            DataSample(
                id="integration_1",
                content="This is important to note that this text demonstrates careful consideration.",
                label="ai"
            ),
            DataSample(
                id="integration_2", 
                content="hey just grabbed coffee and it's amazing! 😊",
                label="human"
            )
        ]
        
        for sample in samples:
            await components["collector"].add_sample(sample)
        
        # 2. Retrieve collected data
        collected_samples = await components["collector"].get_samples()
        assert len(collected_samples) == 2
        
        # 3. Preprocess the text
        preprocessed_samples = []
        for sample in collected_samples:
            preprocessed = await components["preprocessor"].process_text(sample.content)
            preprocessed_samples.append((sample, preprocessed))
        
        # 4. Validate the data
        batch = DataBatch(samples=collected_samples, batch_id="integration_test")
        validation_result = await components["validator"].validate_batch(batch)
        assert validation_result.is_valid
        
        # 5. Run detection on preprocessed text
        detection_results = []
        for sample, preprocessed in preprocessed_samples:
            result = await components["detector"].detect(preprocessed.processed_text)
            detection_results.append((sample, result))
        
        # 6. Store results
        await components["store"].store_batch(batch)
        
        # Verify the flow worked
        assert len(detection_results) == 2
        for sample, result in detection_results:
            assert result is not None
            score = result.get_score()
            assert 0.0 <= score.ai_probability <= 1.0
            assert score.prediction in ["ai", "human"]
    
    @pytest.mark.integration
    async def test_detection_feedback_loop(self, integrated_detection_pipeline):
        """Test detection results feeding back into data collection"""
        components = integrated_detection_pipeline
        
        # Start with uncertain text
        uncertain_text = "This might perhaps be considered somewhat formal in nature."
        
        # Get detection result
        result = await components["detector"].detect(uncertain_text)
        score = result.get_score()
        
        # If confidence is low, add to manual review collection
        if score.confidence < 0.8:
            review_sample = DataSample(
                id="manual_review_1",
                content=uncertain_text,
                label="pending_review",
                metadata={
                    "ai_probability": score.ai_probability,
                    "confidence": score.confidence,
                    "requires_manual_review": True
                }
            )
            
            await components["collector"].add_sample(review_sample)
            
            # Simulate manual review and labeling
            review_sample.label = "ai"  # Manual reviewer decides it's AI
            review_sample.metadata["manually_reviewed"] = True
            
            # Store the manually reviewed sample
            await components["store"].store_sample(review_sample)
            
            # Verify the feedback loop
            stored_samples = await components["store"].list_samples()
            review_samples = [s for s in stored_samples if s.metadata.get("manually_reviewed")]
            assert len(review_samples) == 1
            assert review_samples[0].label == "ai"


class TestServiceAPIIntegration:
    """Test integration between services and API layer"""
    
    @pytest.fixture
    async def service_api_setup(self, temp_dir):
        """Set up services with API integration"""
        # Mock HTTP client for API calls
        mock_http_client = Mock()
        mock_http_client.request = AsyncMock()
        
        # Create API client
        api_client = UnifiedAPIClient(
            base_url="https://api.test.com",
            http_client=mock_http_client
        )
        await api_client.initialize()
        
        # Create services
        registry = ServiceRegistry()
        
        # Mock detector and store
        mock_detector = Mock()
        mock_detector.detect = AsyncMock()
        mock_detector.initialize = AsyncMock(return_value=True)
        mock_detector.is_initialized.return_value = True
        
        mock_store = Mock()
        mock_store.store_sample = AsyncMock(return_value=True)
        mock_store.initialize = AsyncMock(return_value=True)
        
        detection_service = DetectionService(detector=mock_detector)
        data_service = DataService(data_store=mock_store)
        
        registry.register_service("detection", detection_service)
        registry.register_service("data", data_service)
        
        await registry.start_all_services()
        
        return {
            "api_client": api_client,
            "registry": registry,
            "detection_service": detection_service,
            "data_service": data_service,
            "mock_http_client": mock_http_client,
            "mock_detector": mock_detector,
            "mock_store": mock_store
        }
    
    @pytest.mark.integration
    async def test_api_to_service_request_flow(self, service_api_setup):
        """Test API request flowing through services"""
        setup = service_api_setup
        
        # Configure detection mock
        mock_result = Mock()
        mock_result.to_dict.return_value = {
            "prediction": "ai",
            "ai_probability": 0.85,
            "confidence": 0.92
        }
        setup["mock_detector"].detect.return_value = mock_result
        
        # Simulate API request for detection
        text_to_analyze = "This text requires careful analysis and consideration."
        
        # 1. API receives request
        api_request_data = {
            "text": text_to_analyze,
            "store_result": True
        }
        
        # 2. Service processes detection
        detection_result = await setup["detection_service"].detect_text(text_to_analyze)
        
        # 3. Store the result if requested
        if api_request_data.get("store_result"):
            sample = DataSample(
                id="api_request_1",
                content=text_to_analyze,
                label=detection_result["prediction"]
            )
            await setup["data_service"].store_sample(sample)
        
        # Verify the flow
        assert detection_result["prediction"] == "ai"
        setup["mock_detector"].detect.assert_called_once_with(text_to_analyze)
        setup["mock_store"].store_sample.assert_called_once()
    
    @pytest.mark.integration
    async def test_service_to_api_notification_flow(self, service_api_setup):
        """Test services making API calls for notifications"""
        setup = service_api_setup
        
        # Configure API client mock response
        from src.core.interfaces.api_interfaces import APIResponse
        mock_response = APIResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body={"success": True, "notification_id": "notif_123"}
        )
        setup["mock_http_client"].request.return_value = mock_response
        
        # Simulate service needing to send notification
        notification_data = {
            "event": "high_confidence_ai_detected",
            "text": "Detected AI text with high confidence",
            "confidence": 0.95,
            "timestamp": datetime.now().isoformat()
        }
        
        # Service makes API call
        response = await setup["api_client"].request(
            method="POST",
            endpoint="/notifications",
            data=notification_data
        )
        
        # Verify API call was made
        assert response.status_code == 200
        assert response.body["success"] is True
        setup["mock_http_client"].request.assert_called_once()


class TestMessageProtocolIntegration:
    """Test message protocol integration across components"""
    
    @pytest.mark.integration
    async def test_request_response_message_flow(self):
        """Test complete request-response message flow"""
        # Create request message
        request = RequestMessage(
            subject="analyze_text",
            payload={
                "text": "This text needs analysis for AI detection patterns.",
                "options": {
                    "include_evidence": True,
                    "detailed_analysis": True
                }
            },
            timeout=30.0
        )
        
        # Validate request
        MessageProtocol.validate_message(request)
        
        # Simulate processing the request
        text = request.payload["text"]
        
        # Mock analysis result
        analysis_result = {
            "prediction": "ai",
            "confidence": 0.87,
            "evidence": ["formal_language", "hedging_patterns"],
            "processing_time": 0.245
        }
        
        # Create response
        response = MessageProtocol.create_response(
            request,
            payload=analysis_result,
            success=True
        )
        
        # Validate response
        MessageProtocol.validate_message(response)
        
        # Verify message correlation
        assert response.headers.correlation_id == request.headers.correlation_id
        assert response.payload["prediction"] == "ai"
        assert response.get_custom_header("success") is True
    
    @pytest.mark.integration
    async def test_error_message_propagation(self):
        """Test error message propagation through system"""
        # Create request that will fail
        failing_request = RequestMessage(
            subject="invalid_operation",
            payload={"invalid_data": None}
        )
        
        # Simulate processing error
        try:
            # This would cause an error in real processing
            raise ValueError("Invalid data provided for analysis")
        except Exception as e:
            # Create error response
            error_response = MessageProtocol.create_error_response(
                failing_request,
                e,
                error_code="INVALID_DATA"
            )
        
        # Validate error response
        MessageProtocol.validate_message(error_response)
        
        # Verify error information
        assert error_response.get_custom_header("error_code") == "INVALID_DATA"
        assert "Invalid data" in error_response.payload["error"]


class TestDataPipelineIntegration:
    """Test complete data pipeline integration"""
    
    @pytest.fixture
    async def complete_pipeline(self, temp_dir):
        """Set up complete data processing pipeline"""
        # Create all pipeline components
        collector = ManualDataCollector()
        preprocessor = TextPreprocessor()
        extractor = FeatureExtractor()
        validator = DataValidator()
        store = FileDataStore(base_path=str(temp_dir))
        
        # Initialize components
        await collector.initialize()
        await preprocessor.initialize() 
        await extractor.initialize()
        await validator.initialize()
        await store.initialize()
        
        # Create pipeline
        pipeline = DataPipeline()
        pipeline.add_stage("collect", collector)
        pipeline.add_stage("preprocess", preprocessor)
        pipeline.add_stage("extract", extractor)
        pipeline.add_stage("validate", validator)
        pipeline.add_stage("store", store)
        
        return pipeline, {
            "collector": collector,
            "preprocessor": preprocessor,
            "extractor": extractor,
            "validator": validator,
            "store": store
        }
    
    @pytest.mark.integration
    async def test_end_to_end_pipeline_execution(self, complete_pipeline):
        """Test complete end-to-end pipeline execution"""
        pipeline, components = complete_pipeline
        
        # Prepare test data
        test_samples = [
            DataSample(
                id="pipeline_test_1",
                content="It's important to note that this text demonstrates formal academic writing patterns.",
                label="ai"
            ),
            DataSample(
                id="pipeline_test_2",
                content="omg this is so cool! can't wait to try it out 🎉",
                label="human"
            )
        ]
        
        # Add samples to collector first
        for sample in test_samples:
            await components["collector"].add_sample(sample)
        
        # Execute pipeline
        pipeline_config = {
            "max_samples": 10,
            "validation_strict": True,
            "store_results": True
        }
        
        result = await pipeline.execute(pipeline_config)
        
        # Verify pipeline execution
        assert result.success is True
        assert result.metadata["stages_completed"] > 0
        
        # Verify data was processed through all stages
        stored_samples = await components["store"].list_samples()
        assert len(stored_samples) >= 2
    
    @pytest.mark.integration
    async def test_pipeline_error_recovery(self, complete_pipeline):
        """Test pipeline error handling and recovery"""
        pipeline, components = complete_pipeline
        
        # Add invalid sample that will cause validation to fail
        invalid_sample = DataSample(
            id="invalid",
            content="",  # Empty content
            label="human"
        )
        await components["collector"].add_sample(invalid_sample)
        
        # Execute pipeline with error handling
        result = await pipeline.execute({
            "continue_on_error": True,
            "max_samples": 5
        })
        
        # Pipeline should handle errors gracefully
        assert result.success is False  # Due to validation error
        assert result.error is not None
        assert "validation" in str(result.error).lower()


class TestCrossModuleCommunication:
    """Test communication patterns between different modules"""
    
    @pytest.mark.integration
    async def test_detector_to_data_store_integration(self, temp_dir):
        """Test integration between detector and data storage"""
        # Set up components
        detector = GPT4oDetector()
        store = FileDataStore(base_path=str(temp_dir))
        
        await detector.initialize()
        await store.initialize()
        
        # Test text
        test_text = "This formal text demonstrates patterns typical of AI generation."
        
        # 1. Detect
        detection_result = await detector.detect(test_text)
        
        # 2. Create sample with detection results
        sample = DataSample(
            id="detector_integration",
            content=test_text,
            label=detection_result.get_score().prediction,
            metadata={
                "ai_probability": detection_result.get_score().ai_probability,
                "confidence": detection_result.get_score().confidence,
                "processing_time": detection_result.get_processing_time(),
                "detection_method": "gpt4o_detector"
            }
        )
        
        # 3. Store sample
        store_result = await store.store_sample(sample)
        assert store_result is True
        
        # 4. Retrieve and verify
        retrieved_sample = await store.get_sample("detector_integration")
        assert retrieved_sample is not None
        assert retrieved_sample.label == detection_result.get_score().prediction
        assert retrieved_sample.metadata["ai_probability"] == detection_result.get_score().ai_probability
    
    @pytest.mark.integration
    async def test_api_client_to_service_integration(self):
        """Test API client integration with services"""
        # Mock external API
        mock_http_client = Mock()
        
        # Configure mock to return AI analysis result
        from src.core.interfaces.api_interfaces import APIResponse
        mock_response = APIResponse(
            status_code=200,
            headers={"Content-Type": "application/json"},
            body={
                "prediction": "ai",
                "confidence": 0.91,
                "analysis": {
                    "patterns": ["formal_language", "hedging"],
                    "indicators": ["important to note", "should be considered"]
                }
            }
        )
        mock_http_client.request = AsyncMock(return_value=mock_response)
        
        # Set up API client
        api_client = UnifiedAPIClient(
            base_url="https://ai-analysis-api.com",
            http_client=mock_http_client
        )
        await api_client.initialize()
        
        # Mock service that uses API client
        class EnhancedDetectionService:
            def __init__(self, api_client):
                self.api_client = api_client
            
            async def enhanced_detect(self, text):
                # Use external API for additional analysis
                response = await self.api_client.request(
                    method="POST",
                    endpoint="/analyze",
                    data={"text": text}
                )
                
                return response.body
        
        # Test the integration
        service = EnhancedDetectionService(api_client)
        result = await service.enhanced_detect(
            "This text requires sophisticated analysis techniques."
        )
        
        # Verify API was called and result returned
        assert result["prediction"] == "ai"
        assert result["confidence"] == 0.91
        assert "patterns" in result["analysis"]
        
        mock_http_client.request.assert_called_once()
        await api_client.close()
    
    @pytest.mark.integration
    async def test_message_protocol_service_integration(self):
        """Test message protocol integration with services"""
        # Create mock service that handles messages
        class MessageHandlingService:
            def __init__(self):
                self.processed_messages = []
            
            async def handle_detection_request(self, message):
                # Validate incoming message
                MessageProtocol.validate_message(message)
                
                # Process the request
                text = message.payload["text"]
                
                # Mock detection
                result = {
                    "prediction": "ai" if "formal" in text.lower() else "human",
                    "confidence": 0.85,
                    "text_length": len(text)
                }
                
                # Create response
                response = MessageProtocol.create_response(
                    message,
                    payload=result,
                    success=True
                )
                
                self.processed_messages.append((message, response))
                return response
        
        # Test the service
        service = MessageHandlingService()
        
        # Create request message
        request = RequestMessage(
            subject="detect_text",
            payload={"text": "This formal text needs analysis."}
        )
        
        # Process through service
        response = await service.handle_detection_request(request)
        
        # Verify message handling
        assert len(service.processed_messages) == 1
        assert response.payload["prediction"] == "ai"
        assert response.headers.correlation_id == request.headers.correlation_id


@pytest.mark.integration 
class TestSystemWideIntegration:
    """Test system-wide integration scenarios"""
    
    @pytest.mark.integration
    async def test_complete_detection_workflow(self, temp_dir):
        """Test complete detection workflow from data input to result storage"""
        # Set up all components
        collector = ManualDataCollector()
        detector = GPT4oDetector()
        store = FileDataStore(base_path=str(temp_dir))
        
        await collector.initialize()
        await detector.initialize()
        await store.initialize()
        
        # Create service registry
        registry = ServiceRegistry()
        
        detection_service = DetectionService(detector=detector)
        data_service = DataService(data_store=store)
        
        registry.register_service("detection", detection_service)
        registry.register_service("data", data_service)
        
        await registry.start_all_services()
        
        try:
            # 1. Input: Add samples for detection
            input_texts = [
                "It's important to note that this analysis requires careful consideration of multiple factors.",
                "hey just saw the coolest movie ever! 🍿 totally recommend it",
                "The methodology employed in this study demonstrates rigorous academic standards.",
                "lol that's hilarious 😂 made my day"
            ]
            
            results = []
            
            # 2. Process: Detect each text and store results
            for i, text in enumerate(input_texts):
                # Detect
                detection_result = await detection_service.detect_text(text)
                
                # Create sample with results
                sample = DataSample(
                    id=f"workflow_test_{i}",
                    content=text,
                    label=detection_result["prediction"],
                    metadata={
                        "confidence": detection_result.get("confidence", 0.0),
                        "processing_timestamp": datetime.now().isoformat()
                    }
                )
                
                # Store
                await data_service.store_sample(sample)
                results.append(detection_result)
            
            # 3. Verify: Check that all data was processed and stored
            stored_samples = await store.list_samples()
            assert len(stored_samples) == 4
            
            # Check that AI and human texts were classified correctly
            ai_predictions = [r for r in results if r["prediction"] == "ai"]
            human_predictions = [r for r in results if r["prediction"] == "human"]
            
            assert len(ai_predictions) >= 1  # Should detect some AI text
            assert len(human_predictions) >= 1  # Should detect some human text
            
        finally:
            await registry.stop_all_services()
    
    @pytest.mark.integration
    async def test_error_propagation_across_modules(self, temp_dir):
        """Test error propagation and handling across all modules"""
        # Set up components with potential failure points
        collector = ManualDataCollector()
        detector = GPT4oDetector()
        
        # Create store that will fail
        failing_store = Mock()
        failing_store.store_sample = AsyncMock(side_effect=Exception("Storage failure"))
        failing_store.initialize = AsyncMock(return_value=True)
        
        await collector.initialize()
        await detector.initialize()
        
        # Create services
        detection_service = DetectionService(detector=detector)
        data_service = DataService(data_store=failing_store)
        
        await detection_service.start()
        await data_service.start()
        
        try:
            # Process text successfully
            detection_result = await detection_service.detect_text("Test text")
            assert detection_result is not None
            
            # Try to store (this should fail gracefully)
            sample = DataSample(
                id="error_test",
                content="Test text",
                label=detection_result["prediction"]
            )
            
            # This should handle the storage error gracefully
            try:
                await data_service.store_sample(sample)
                assert False, "Should have raised an exception"
            except Exception as e:
                assert "Storage failure" in str(e)
                
            # Services should still be running despite the error
            assert detection_service.get_status().name == "RUNNING"
            assert data_service.get_status().name == "RUNNING"
            
        finally:
            await detection_service.stop()
            await data_service.stop()
    
    @pytest.mark.integration
    async def test_concurrent_module_operations(self, temp_dir):
        """Test concurrent operations across multiple modules"""
        # Set up components
        collector = ManualDataCollector()
        detector = GPT4oDetector()
        store = FileDataStore(base_path=str(temp_dir))
        
        await collector.initialize()
        await detector.initialize()
        await store.initialize()
        
        # Test concurrent operations
        async def process_text_batch(texts, batch_id):
            """Process a batch of texts concurrently"""
            tasks = []
            
            for i, text in enumerate(texts):
                async def process_single_text(text, index):
                    # Detect
                    result = await detector.detect(text)
                    
                    # Store
                    sample = DataSample(
                        id=f"{batch_id}_{index}",
                        content=text,
                        label=result.get_score().prediction
                    )
                    await store.store_sample(sample)
                    
                    return result
                
                tasks.append(process_single_text(text, i))
            
            return await asyncio.gather(*tasks)
        
        # Run multiple batches concurrently
        batch1_texts = [
            "This formal analysis demonstrates academic rigor.",
            "yo that's awesome! love it 👍"
        ]
        
        batch2_texts = [
            "The methodology requires careful consideration.",
            "can't believe how good this is!"
        ]
        
        # Process batches concurrently
        batch1_task = process_text_batch(batch1_texts, "batch1")
        batch2_task = process_text_batch(batch2_texts, "batch2")
        
        batch1_results, batch2_results = await asyncio.gather(batch1_task, batch2_task)
        
        # Verify all texts were processed
        assert len(batch1_results) == 2
        assert len(batch2_results) == 2
        
        # Verify all samples were stored
        all_samples = await store.list_samples()
        assert len(all_samples) == 4
        
        # Verify batch isolation (samples have correct batch IDs)
        batch1_samples = [s for s in all_samples if s.id.startswith("batch1_")]
        batch2_samples = [s for s in all_samples if s.id.startswith("batch2_")]
        
        assert len(batch1_samples) == 2
        assert len(batch2_samples) == 2