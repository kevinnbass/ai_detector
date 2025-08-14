"""
Unit Tests for Service Layer Components
Comprehensive tests for service implementations with high coverage
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, call
from datetime import datetime, timedelta
import json

from src.core.interfaces.service_interfaces import (
    IBaseService, ServiceStatus, IDetectionService, IDataService,
    IAnalysisService, INotificationService
)
from src.core.services.detection_service import DetectionService
from src.core.services.data_service import DataService  
from src.core.services.analysis_service import AnalysisService
from src.core.services.notification_service import NotificationService
from src.core.services.service_registry import ServiceRegistry
from src.core.interfaces.detector_interfaces import IDetector, DetectionMethod
from src.core.interfaces.data_interfaces import DataSample, DataBatch


class TestServiceRegistry:
    """Test suite for ServiceRegistry"""
    
    @pytest.fixture
    def service_registry(self):
        """Create ServiceRegistry instance"""
        return ServiceRegistry()
    
    @pytest.mark.unit
    def test_service_registration(self, service_registry):
        """Test service registration and discovery"""
        # Create mock service
        mock_service = Mock(spec=IBaseService)
        mock_service.get_service_name.return_value = "test_service"
        mock_service.get_status.return_value = ServiceStatus.RUNNING
        
        # Register service
        service_registry.register_service("test_service", mock_service)
        
        # Verify registration
        assert service_registry.has_service("test_service")
        retrieved = service_registry.get_service("test_service")
        assert retrieved == mock_service
        
        # Test list services
        services = service_registry.list_services()
        assert "test_service" in services
    
    @pytest.mark.unit
    def test_service_unregistration(self, service_registry):
        """Test service unregistration"""
        mock_service = Mock(spec=IBaseService)
        mock_service.get_service_name.return_value = "temp_service"
        
        # Register and then unregister
        service_registry.register_service("temp_service", mock_service)
        assert service_registry.has_service("temp_service")
        
        service_registry.unregister_service("temp_service")
        assert not service_registry.has_service("temp_service")
    
    @pytest.mark.unit
    def test_service_registry_error_handling(self, service_registry):
        """Test service registry error handling"""
        # Test getting non-existent service
        with pytest.raises(KeyError):
            service_registry.get_service("nonexistent")
        
        # Test duplicate registration
        mock_service = Mock(spec=IBaseService)
        service_registry.register_service("duplicate", mock_service)
        
        with pytest.raises(ValueError, match="already registered"):
            service_registry.register_service("duplicate", mock_service)
    
    @pytest.mark.unit
    async def test_service_registry_lifecycle(self, service_registry):
        """Test service registry lifecycle management"""
        # Create mock services
        services = []
        for i in range(3):
            service = Mock(spec=IBaseService)
            service.get_service_name.return_value = f"service_{i}"
            service.start = AsyncMock()
            service.stop = AsyncMock()
            service.get_status.return_value = ServiceStatus.STOPPED
            services.append(service)
            service_registry.register_service(f"service_{i}", service)
        
        # Start all services
        await service_registry.start_all_services()
        
        # Verify all services were started
        for service in services:
            service.start.assert_called_once()
        
        # Stop all services
        await service_registry.stop_all_services()
        
        # Verify all services were stopped
        for service in services:
            service.stop.assert_called_once()


class TestDetectionService:
    """Test suite for DetectionService"""
    
    @pytest.fixture
    def mock_detector(self):
        """Create mock detector"""
        detector = Mock(spec=IDetector)
        
        async def mock_detect(text):
            result = Mock()
            result.get_score.return_value = Mock(
                ai_probability=0.75,
                prediction="ai",
                confidence=0.85
            )
            result.get_processing_time.return_value = 0.15
            result.to_dict.return_value = {
                "prediction": "ai",
                "ai_probability": 0.75,
                "confidence": 0.85
            }
            return result
        
        detector.detect = AsyncMock(side_effect=mock_detect)
        detector.detect_batch = AsyncMock()
        detector.initialize = AsyncMock(return_value=True)
        detector.is_initialized.return_value = True
        detector.get_detection_method.return_value = DetectionMethod.PATTERN_BASED
        
        return detector
    
    @pytest.fixture
    def detection_service(self, mock_detector):
        """Create DetectionService instance"""
        return DetectionService(detector=mock_detector)
    
    @pytest.mark.unit
    async def test_detection_service_initialization(self, detection_service):
        """Test DetectionService initialization"""
        assert detection_service.get_status() == ServiceStatus.STOPPED
        
        await detection_service.start()
        assert detection_service.get_status() == ServiceStatus.RUNNING
        
        await detection_service.stop()
        assert detection_service.get_status() == ServiceStatus.STOPPED
    
    @pytest.mark.unit
    async def test_detection_service_single_detection(self, detection_service):
        """Test single text detection"""
        await detection_service.start()
        
        result = await detection_service.detect_text("This is a test message")
        
        assert result is not None
        assert result["prediction"] == "ai"
        assert result["ai_probability"] == 0.75
        assert result["confidence"] == 0.85
    
    @pytest.mark.unit
    async def test_detection_service_batch_detection(self, detection_service, mock_detector):
        """Test batch text detection"""
        await detection_service.start()
        
        # Configure batch mock
        mock_detector.detect_batch.return_value = [
            Mock(to_dict=lambda: {"prediction": "human", "ai_probability": 0.2}),
            Mock(to_dict=lambda: {"prediction": "ai", "ai_probability": 0.8})
        ]
        
        texts = ["Human text here", "AI generated text"]
        results = await detection_service.detect_batch(texts)
        
        assert len(results) == 2
        assert results[0]["prediction"] == "human"
        assert results[1]["prediction"] == "ai"
        
        mock_detector.detect_batch.assert_called_once_with(texts)
    
    @pytest.mark.unit
    async def test_detection_service_error_handling(self, detection_service, mock_detector):
        """Test detection service error handling"""
        await detection_service.start()
        
        # Configure detector to fail
        mock_detector.detect.side_effect = Exception("Detection failed")
        
        result = await detection_service.detect_text("test")
        
        # Should return error result
        assert result is not None
        assert "error" in result
        assert "Detection failed" in result["error"]
    
    @pytest.mark.unit
    async def test_detection_service_metrics(self, detection_service):
        """Test detection service metrics collection"""
        await detection_service.start()
        
        # Perform some detections
        await detection_service.detect_text("Test 1")
        await detection_service.detect_text("Test 2")
        
        metrics = detection_service.get_metrics()
        assert metrics["total_detections"] >= 2
        assert "average_processing_time" in metrics
        assert "success_rate" in metrics
    
    @pytest.mark.unit
    async def test_detection_service_health_check(self, detection_service, mock_detector):
        """Test detection service health check"""
        await detection_service.start()
        
        # Healthy detector
        health = await detection_service.check_health()
        assert health["status"] == "healthy"
        assert health["detector_initialized"] is True
        
        # Unhealthy detector
        mock_detector.is_initialized.return_value = False
        health = await detection_service.check_health()
        assert health["status"] == "unhealthy"


class TestDataService:
    """Test suite for DataService"""
    
    @pytest.fixture
    def mock_data_store(self):
        """Create mock data store"""
        store = Mock()
        
        # Mock data storage
        stored_samples = []
        
        async def mock_store_sample(sample):
            stored_samples.append(sample)
            return True
        
        async def mock_get_samples(limit=None, label_filter=None):
            samples = stored_samples
            if label_filter:
                samples = [s for s in samples if s.label == label_filter]
            if limit:
                samples = samples[:limit]
            return samples
        
        store.store_sample = AsyncMock(side_effect=mock_store_sample)
        store.get_samples = AsyncMock(side_effect=mock_get_samples)
        store.store_batch = AsyncMock(return_value=True)
        store.get_statistics = AsyncMock(return_value={"total_samples": len(stored_samples)})
        store.initialize = AsyncMock(return_value=True)
        
        return store
    
    @pytest.fixture
    def data_service(self, mock_data_store):
        """Create DataService instance"""
        return DataService(data_store=mock_data_store)
    
    @pytest.mark.unit
    async def test_data_service_initialization(self, data_service):
        """Test DataService initialization"""
        assert data_service.get_status() == ServiceStatus.STOPPED
        
        await data_service.start()
        assert data_service.get_status() == ServiceStatus.RUNNING
    
    @pytest.mark.unit
    async def test_data_service_sample_operations(self, data_service):
        """Test data service sample operations"""
        await data_service.start()
        
        # Store sample
        sample = DataSample(
            id="test_sample",
            content="Test content",
            label="human"
        )
        
        result = await data_service.store_sample(sample)
        assert result is True
        
        # Retrieve samples
        samples = await data_service.get_samples()
        assert len(samples) == 1
        assert samples[0].id == "test_sample"
    
    @pytest.mark.unit
    async def test_data_service_batch_operations(self, data_service, mock_data_store):
        """Test data service batch operations"""
        await data_service.start()
        
        # Create batch
        samples = [
            DataSample(id=f"batch_{i}", content=f"Content {i}", label="human")
            for i in range(3)
        ]
        batch = DataBatch(samples=samples, batch_id="test_batch")
        
        result = await data_service.store_batch(batch)
        assert result is True
        
        mock_data_store.store_batch.assert_called_once_with(batch)
    
    @pytest.mark.unit
    async def test_data_service_filtering(self, data_service):
        """Test data service filtering capabilities"""
        await data_service.start()
        
        # Store samples with different labels
        human_sample = DataSample(id="human", content="Human text", label="human")
        ai_sample = DataSample(id="ai", content="AI text", label="ai")
        
        await data_service.store_sample(human_sample)
        await data_service.store_sample(ai_sample)
        
        # Filter by label
        human_samples = await data_service.get_samples(label_filter="human")
        ai_samples = await data_service.get_samples(label_filter="ai")
        
        assert len(human_samples) == 1
        assert len(ai_samples) == 1
        assert human_samples[0].label == "human"
        assert ai_samples[0].label == "ai"
    
    @pytest.mark.unit
    async def test_data_service_statistics(self, data_service, mock_data_store):
        """Test data service statistics"""
        await data_service.start()
        
        stats = await data_service.get_statistics()
        assert isinstance(stats, dict)
        
        mock_data_store.get_statistics.assert_called_once()


class TestAnalysisService:
    """Test suite for AnalysisService"""
    
    @pytest.fixture
    def mock_analyzer(self):
        """Create mock analyzer"""
        analyzer = Mock()
        
        async def mock_analyze(data):
            return {
                "patterns_detected": ["hedging", "formal_language"],
                "confidence_score": 0.82,
                "key_indicators": ["important to note", "should be considered"],
                "analysis_time": 0.25
            }
        
        analyzer.analyze = AsyncMock(side_effect=mock_analyze)
        analyzer.analyze_batch = AsyncMock()
        analyzer.initialize = AsyncMock(return_value=True)
        
        return analyzer
    
    @pytest.fixture
    def analysis_service(self, mock_analyzer):
        """Create AnalysisService instance"""
        return AnalysisService(analyzer=mock_analyzer)
    
    @pytest.mark.unit
    async def test_analysis_service_single_analysis(self, analysis_service):
        """Test single text analysis"""
        await analysis_service.start()
        
        result = await analysis_service.analyze_text("Test text for analysis")
        
        assert result["patterns_detected"] == ["hedging", "formal_language"]
        assert result["confidence_score"] == 0.82
        assert "key_indicators" in result
    
    @pytest.mark.unit
    async def test_analysis_service_batch_analysis(self, analysis_service, mock_analyzer):
        """Test batch text analysis"""
        await analysis_service.start()
        
        # Configure batch mock
        mock_analyzer.analyze_batch.return_value = [
            {"patterns_detected": ["casual"], "confidence_score": 0.7},
            {"patterns_detected": ["formal"], "confidence_score": 0.9}
        ]
        
        texts = ["Casual text", "Formal text"]
        results = await analysis_service.analyze_batch(texts)
        
        assert len(results) == 2
        assert results[0]["patterns_detected"] == ["casual"]
        assert results[1]["patterns_detected"] == ["formal"]
    
    @pytest.mark.unit
    async def test_analysis_service_performance_tracking(self, analysis_service):
        """Test analysis service performance tracking"""
        await analysis_service.start()
        
        # Perform analyses
        await analysis_service.analyze_text("First analysis")
        await analysis_service.analyze_text("Second analysis")
        
        metrics = analysis_service.get_metrics()
        assert metrics["total_analyses"] >= 2
        assert "average_analysis_time" in metrics


class TestNotificationService:
    """Test suite for NotificationService"""
    
    @pytest.fixture
    def mock_notification_channels(self):
        """Create mock notification channels"""
        email_channel = Mock()
        email_channel.send_notification = AsyncMock(return_value=True)
        email_channel.get_channel_type.return_value = "email"
        
        webhook_channel = Mock()
        webhook_channel.send_notification = AsyncMock(return_value=True)
        webhook_channel.get_channel_type.return_value = "webhook"
        
        return {
            "email": email_channel,
            "webhook": webhook_channel
        }
    
    @pytest.fixture
    def notification_service(self, mock_notification_channels):
        """Create NotificationService instance"""
        service = NotificationService()
        for channel_type, channel in mock_notification_channels.items():
            service.add_channel(channel_type, channel)
        return service
    
    @pytest.mark.unit
    async def test_notification_service_send_notification(self, notification_service):
        """Test sending notifications"""
        await notification_service.start()
        
        notification = {
            "subject": "Test Alert",
            "message": "This is a test notification",
            "severity": "warning",
            "metadata": {"source": "test"}
        }
        
        result = await notification_service.send_notification(
            notification, 
            channels=["email"]
        )
        
        assert result is True
    
    @pytest.mark.unit
    async def test_notification_service_broadcast(self, notification_service, mock_notification_channels):
        """Test broadcasting to all channels"""
        await notification_service.start()
        
        notification = {
            "subject": "Broadcast Alert",
            "message": "This goes to all channels",
            "severity": "critical"
        }
        
        result = await notification_service.broadcast_notification(notification)
        assert result is True
        
        # Verify all channels received notification
        for channel in mock_notification_channels.values():
            channel.send_notification.assert_called_once()
    
    @pytest.mark.unit
    async def test_notification_service_filtering(self, notification_service):
        """Test notification filtering"""
        await notification_service.start()
        
        # Set severity filter
        notification_service.set_severity_filter("error")
        
        # Low severity notification should be filtered
        low_severity = {
            "subject": "Info",
            "message": "Low priority info",
            "severity": "info"
        }
        
        result = await notification_service.send_notification(low_severity, channels=["email"])
        assert result is False  # Should be filtered
        
        # High severity notification should go through
        high_severity = {
            "subject": "Error",
            "message": "Critical error occurred",
            "severity": "error"
        }
        
        result = await notification_service.send_notification(high_severity, channels=["email"])
        assert result is True
    
    @pytest.mark.unit
    async def test_notification_service_retry_on_failure(self, notification_service, mock_notification_channels):
        """Test notification retry on failure"""
        await notification_service.start()
        
        # Configure channel to fail first attempt
        call_count = 0
        
        async def failing_send(notification):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Network error")
            return True
        
        mock_notification_channels["email"].send_notification = AsyncMock(side_effect=failing_send)
        
        notification = {
            "subject": "Retry Test",
            "message": "Testing retry mechanism",
            "severity": "warning"
        }
        
        result = await notification_service.send_notification(notification, channels=["email"])
        assert result is True
        assert call_count == 2  # Should have retried
    
    @pytest.mark.unit
    async def test_notification_service_metrics(self, notification_service):
        """Test notification service metrics"""
        await notification_service.start()
        
        # Send several notifications
        for i in range(3):
            notification = {
                "subject": f"Test {i}",
                "message": f"Message {i}",
                "severity": "info"
            }
            await notification_service.send_notification(notification, channels=["email"])
        
        metrics = notification_service.get_metrics()
        assert metrics["total_notifications"] >= 3
        assert "success_rate" in metrics
        assert "notifications_by_channel" in metrics


@pytest.mark.integration
class TestServiceIntegration:
    """Integration tests for service layer"""
    
    @pytest.mark.integration
    async def test_service_orchestration(self):
        """Test orchestration of multiple services"""
        # Create service registry
        registry = ServiceRegistry()
        
        # Create and register services
        mock_detector = Mock(spec=IDetector)
        mock_detector.detect = AsyncMock(return_value=Mock(
            to_dict=lambda: {"prediction": "ai", "confidence": 0.8}
        ))
        mock_detector.initialize = AsyncMock(return_value=True)
        mock_detector.is_initialized.return_value = True
        
        detection_service = DetectionService(detector=mock_detector)
        
        mock_store = Mock()
        mock_store.store_sample = AsyncMock(return_value=True)
        mock_store.initialize = AsyncMock(return_value=True)
        
        data_service = DataService(data_store=mock_store)
        
        # Register services
        registry.register_service("detection", detection_service)
        registry.register_service("data", data_service)
        
        # Start all services
        await registry.start_all_services()
        
        try:
            # Test service interaction
            detection_result = await detection_service.detect_text("Test text")
            assert detection_result["prediction"] == "ai"
            
            # Store detection result
            sample = DataSample(
                id="integration_test",
                content="Test text",
                label="ai"
            )
            store_result = await data_service.store_sample(sample)
            assert store_result is True
            
            # Verify services are healthy
            detection_health = await detection_service.check_health()
            assert detection_health["status"] == "healthy"
            
        finally:
            # Clean up
            await registry.stop_all_services()
    
    @pytest.mark.integration
    async def test_service_failure_recovery(self):
        """Test service failure and recovery scenarios"""
        # Create service that can fail and recover
        mock_detector = Mock(spec=IDetector)
        
        # Configure to fail initially, then recover
        call_count = 0
        
        async def mock_detect(text):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Service temporarily unavailable")
            return Mock(to_dict=lambda: {"prediction": "human", "confidence": 0.9})
        
        mock_detector.detect = AsyncMock(side_effect=mock_detect)
        mock_detector.initialize = AsyncMock(return_value=True)
        mock_detector.is_initialized.return_value = True
        
        detection_service = DetectionService(detector=mock_detector)
        await detection_service.start()
        
        # First attempts should handle failures gracefully
        result1 = await detection_service.detect_text("Test 1")
        result2 = await detection_service.detect_text("Test 2")
        
        assert "error" in result1
        assert "error" in result2
        
        # Third attempt should succeed
        result3 = await detection_service.detect_text("Test 3")
        assert result3["prediction"] == "human"
        assert result3["confidence"] == 0.9
        
        await detection_service.stop()
    
    @pytest.mark.integration
    async def test_cross_service_communication(self):
        """Test communication between services"""
        # Set up services with dependencies
        registry = ServiceRegistry()
        
        # Mock components
        mock_detector = Mock(spec=IDetector)
        mock_detector.detect = AsyncMock(return_value=Mock(
            to_dict=lambda: {"prediction": "ai", "confidence": 0.85}
        ))
        mock_detector.initialize = AsyncMock(return_value=True)
        mock_detector.is_initialized.return_value = True
        
        mock_store = Mock()
        mock_store.store_sample = AsyncMock(return_value=True)
        mock_store.initialize = AsyncMock(return_value=True)
        
        mock_notification_channel = Mock()
        mock_notification_channel.send_notification = AsyncMock(return_value=True)
        mock_notification_channel.get_channel_type.return_value = "test"
        
        # Create services
        detection_service = DetectionService(detector=mock_detector)
        data_service = DataService(data_store=mock_store)
        notification_service = NotificationService()
        notification_service.add_channel("test", mock_notification_channel)
        
        # Register services
        registry.register_service("detection", detection_service)
        registry.register_service("data", data_service)
        registry.register_service("notification", notification_service)
        
        await registry.start_all_services()
        
        try:
            # Simulate workflow: detect -> store -> notify
            text = "This is a test message for cross-service communication"
            
            # 1. Detect
            detection_result = await detection_service.detect_text(text)
            assert detection_result["prediction"] == "ai"
            
            # 2. Store
            sample = DataSample(
                id="cross_service_test",
                content=text,
                label=detection_result["prediction"]
            )
            store_result = await data_service.store_sample(sample)
            assert store_result is True
            
            # 3. Notify
            notification = {
                "subject": "AI Content Detected",
                "message": f"AI content detected with {detection_result['confidence']} confidence",
                "severity": "warning"
            }
            notify_result = await notification_service.send_notification(
                notification, 
                channels=["test"]
            )
            assert notify_result is True
            
            # Verify all services were called
            mock_detector.detect.assert_called_once()
            mock_store.store_sample.assert_called_once()
            mock_notification_channel.send_notification.assert_called_once()
            
        finally:
            await registry.stop_all_services()