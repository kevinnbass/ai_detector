"""
Unit Tests for Data Processing System
Comprehensive tests for data processors with 85% coverage target
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, call
from datetime import datetime, timedelta
import json
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np

from src.core.interfaces.data_interfaces import (
    IDataCollector, IDataProcessor, IDataValidator, IDataStore,
    DataSample, DataBatch, ValidationResult, ProcessingResult
)
from src.core.data.collectors import TwitterDataCollector, ManualDataCollector
from src.core.data.processors import TextPreprocessor, FeatureExtractor, DataEnricher
from src.core.data.validators import DataValidator, SchemaValidator
from src.core.data.stores import FileDataStore, DatabaseDataStore
from src.core.data.pipeline import DataPipeline, PipelineStage


class TestDataSample:
    """Test suite for DataSample class"""
    
    @pytest.mark.unit
    def test_data_sample_creation(self):
        """Test DataSample creation and properties"""
        sample = DataSample(
            id="test_001",
            content="This is test content",
            label="human",
            metadata={"source": "test", "category": "example"},
            timestamp=datetime(2024, 1, 1, 12, 0, 0)
        )
        
        assert sample.id == "test_001"
        assert sample.content == "This is test content"
        assert sample.label == "human"
        assert sample.metadata["source"] == "test"
        assert sample.timestamp.year == 2024
    
    @pytest.mark.unit
    def test_data_sample_validation(self):
        """Test DataSample validation"""
        # Valid sample
        valid_sample = DataSample(
            id="valid_001",
            content="Valid content here",
            label="ai"
        )
        assert valid_sample.is_valid()
        
        # Invalid samples
        invalid_samples = [
            DataSample(id="", content="content", label="human"),  # Empty ID
            DataSample(id="test", content="", label="human"),     # Empty content
            DataSample(id="test", content="content", label="invalid"),  # Invalid label
        ]
        
        for sample in invalid_samples:
            assert not sample.is_valid()
    
    @pytest.mark.unit
    def test_data_sample_serialization(self):
        """Test DataSample serialization/deserialization"""
        original = DataSample(
            id="serialize_test",
            content="Content for serialization test",
            label="human",
            metadata={"test": True, "number": 42},
            timestamp=datetime.now()
        )
        
        # Test to_dict
        sample_dict = original.to_dict()
        assert isinstance(sample_dict, dict)
        assert sample_dict["id"] == "serialize_test"
        assert sample_dict["content"] == "Content for serialization test"
        assert sample_dict["metadata"]["test"] is True
        
        # Test from_dict
        restored = DataSample.from_dict(sample_dict)
        assert restored.id == original.id
        assert restored.content == original.content
        assert restored.label == original.label
        assert restored.metadata == original.metadata
    
    @pytest.mark.unit
    def test_data_sample_hashing(self):
        """Test DataSample hashing for deduplication"""
        sample1 = DataSample(id="test", content="same content", label="human")
        sample2 = DataSample(id="test", content="same content", label="human")
        sample3 = DataSample(id="test", content="different content", label="human")
        
        # Same content should have same hash
        assert sample1.get_content_hash() == sample2.get_content_hash()
        # Different content should have different hash
        assert sample1.get_content_hash() != sample3.get_content_hash()


class TestDataBatch:
    """Test suite for DataBatch class"""
    
    @pytest.fixture
    def sample_data_samples(self):
        """Create sample DataSample objects"""
        return [
            DataSample(id="1", content="First sample", label="human"),
            DataSample(id="2", content="Second sample", label="ai"),
            DataSample(id="3", content="Third sample", label="human")
        ]
    
    @pytest.mark.unit
    def test_data_batch_creation(self, sample_data_samples):
        """Test DataBatch creation"""
        batch = DataBatch(
            samples=sample_data_samples,
            batch_id="test_batch",
            metadata={"created_by": "test"}
        )
        
        assert batch.batch_id == "test_batch"
        assert len(batch.samples) == 3
        assert batch.size() == 3
        assert batch.metadata["created_by"] == "test"
    
    @pytest.mark.unit
    def test_data_batch_filtering(self, sample_data_samples):
        """Test DataBatch filtering capabilities"""
        batch = DataBatch(samples=sample_data_samples, batch_id="filter_test")
        
        # Filter by label
        human_samples = batch.filter_by_label("human")
        assert len(human_samples) == 2
        
        ai_samples = batch.filter_by_label("ai")
        assert len(ai_samples) == 1
        
        # Filter by metadata
        sample_data_samples[0].metadata = {"category": "social"}
        sample_data_samples[1].metadata = {"category": "news"}
        
        batch = DataBatch(samples=sample_data_samples, batch_id="meta_test")
        social_samples = batch.filter_by_metadata("category", "social")
        assert len(social_samples) == 1
    
    @pytest.mark.unit
    def test_data_batch_statistics(self, sample_data_samples):
        """Test DataBatch statistics calculation"""
        batch = DataBatch(samples=sample_data_samples, batch_id="stats_test")
        
        stats = batch.get_statistics()
        assert stats["total_samples"] == 3
        assert stats["label_distribution"]["human"] == 2
        assert stats["label_distribution"]["ai"] == 1
        assert "average_content_length" in stats
    
    @pytest.mark.unit
    def test_data_batch_splitting(self, sample_data_samples):
        """Test DataBatch splitting functionality"""
        batch = DataBatch(samples=sample_data_samples * 10, batch_id="split_test")  # 30 samples
        
        # Split by ratio
        train_batch, test_batch = batch.split(train_ratio=0.8)
        assert train_batch.size() == 24
        assert test_batch.size() == 6
        
        # Split by count
        small_batches = batch.split_by_count(10)
        assert len(small_batches) == 3
        assert all(b.size() == 10 for b in small_batches)


class TestTwitterDataCollector:
    """Test suite for TwitterDataCollector"""
    
    @pytest.fixture
    def mock_twitter_api(self):
        """Mock Twitter API responses"""
        api = Mock()
        api.get_tweets.return_value = [
            {
                "id": "tweet_1",
                "text": "This is a tweet about AI detection",
                "user": {"username": "user1"},
                "created_at": "2024-01-01T12:00:00Z",
                "metrics": {"retweet_count": 5, "like_count": 10}
            },
            {
                "id": "tweet_2", 
                "text": "Another tweet for testing purposes",
                "user": {"username": "user2"},
                "created_at": "2024-01-01T13:00:00Z",
                "metrics": {"retweet_count": 2, "like_count": 7}
            }
        ]
        return api
    
    @pytest.mark.unit
    async def test_twitter_collector_initialization(self):
        """Test TwitterDataCollector initialization"""
        collector = TwitterDataCollector(api_key="test_key", api_secret="test_secret")
        
        assert not collector.is_initialized()
        
        with patch.object(collector, '_setup_api', return_value=True):
            await collector.initialize()
            assert collector.is_initialized()
    
    @pytest.mark.unit
    async def test_twitter_collector_data_collection(self, mock_twitter_api):
        """Test TwitterDataCollector data collection"""
        collector = TwitterDataCollector(api_key="test_key", api_secret="test_secret")
        
        with patch.object(collector, '_api', mock_twitter_api):
            with patch.object(collector, 'is_initialized', return_value=True):
                samples = await collector.collect_data(query="AI detection", max_samples=10)
                
                assert len(samples) == 2
                assert all(isinstance(s, DataSample) for s in samples)
                assert samples[0].content == "This is a tweet about AI detection"
                assert samples[0].metadata["source"] == "twitter"
                assert "user" in samples[0].metadata
    
    @pytest.mark.unit
    async def test_twitter_collector_rate_limiting(self, mock_twitter_api):
        """Test TwitterDataCollector rate limiting"""
        collector = TwitterDataCollector(
            api_key="test_key", 
            api_secret="test_secret",
            rate_limit=2  # Very low limit for testing
        )
        
        with patch.object(collector, '_api', mock_twitter_api):
            with patch.object(collector, 'is_initialized', return_value=True):
                start_time = datetime.now()
                
                # Make multiple requests
                for _ in range(3):
                    await collector.collect_data(query="test", max_samples=1)
                
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                
                # Should take some time due to rate limiting
                assert duration > 0.5  # At least some delay
    
    @pytest.mark.unit
    async def test_twitter_collector_error_handling(self):
        """Test TwitterDataCollector error handling"""
        collector = TwitterDataCollector(api_key="invalid", api_secret="invalid")
        
        # Test uninitialized collector
        with pytest.raises(RuntimeError, match="not initialized"):
            await collector.collect_data(query="test")
        
        # Test API errors
        mock_api = Mock()
        mock_api.get_tweets.side_effect = Exception("API Error")
        
        with patch.object(collector, '_api', mock_api):
            with patch.object(collector, 'is_initialized', return_value=True):
                samples = await collector.collect_data(query="test", max_samples=5)
                assert len(samples) == 0  # Should return empty list on error


class TestManualDataCollector:
    """Test suite for ManualDataCollector"""
    
    @pytest.mark.unit
    async def test_manual_collector_add_sample(self):
        """Test ManualDataCollector adding samples"""
        collector = ManualDataCollector()
        await collector.initialize()
        
        # Add individual sample
        sample = DataSample(
            id="manual_1",
            content="Manually added content",
            label="human"
        )
        
        result = await collector.add_sample(sample)
        assert result is True
        
        # Retrieve samples
        samples = await collector.get_samples()
        assert len(samples) == 1
        assert samples[0].content == "Manually added content"
    
    @pytest.mark.unit
    async def test_manual_collector_batch_operations(self):
        """Test ManualDataCollector batch operations"""
        collector = ManualDataCollector()
        await collector.initialize()
        
        # Add batch of samples
        samples = [
            DataSample(id=f"batch_{i}", content=f"Content {i}", label="human" if i % 2 == 0 else "ai")
            for i in range(5)
        ]
        
        result = await collector.add_batch(samples)
        assert result is True
        
        # Retrieve and verify
        retrieved = await collector.get_samples()
        assert len(retrieved) == 5
        
        # Test filtering
        human_samples = await collector.get_samples(label_filter="human")
        assert len(human_samples) == 3  # 0, 2, 4
    
    @pytest.mark.unit
    async def test_manual_collector_export_import(self, temp_dir):
        """Test ManualDataCollector export/import functionality"""
        collector = ManualDataCollector()
        await collector.initialize()
        
        # Add test data
        samples = [
            DataSample(id=f"export_{i}", content=f"Export content {i}", label="human")
            for i in range(3)
        ]
        await collector.add_batch(samples)
        
        # Export to file
        export_file = temp_dir / "manual_export.json"
        await collector.export_to_file(str(export_file))
        
        assert export_file.exists()
        
        # Create new collector and import
        new_collector = ManualDataCollector()
        await new_collector.initialize()
        await new_collector.import_from_file(str(export_file))
        
        imported_samples = await new_collector.get_samples()
        assert len(imported_samples) == 3
        assert imported_samples[0].content == "Export content 0"


class TestTextPreprocessor:
    """Test suite for TextPreprocessor"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create TextPreprocessor instance"""
        return TextPreprocessor()
    
    @pytest.mark.unit
    async def test_text_preprocessing_basic(self, preprocessor):
        """Test basic text preprocessing"""
        await preprocessor.initialize()
        
        input_text = "  Hello WORLD! This is a TEST message with @mentions #hashtags and links: https://example.com  "
        
        result = await preprocessor.process_text(input_text)
        
        # Should normalize whitespace, handle special characters
        assert result.processed_text.strip() != input_text.strip()
        assert "hello world" in result.processed_text.lower()
        assert result.metadata["original_length"] == len(input_text)
        assert result.metadata["processed_length"] == len(result.processed_text)
    
    @pytest.mark.unit
    async def test_text_preprocessing_advanced(self, preprocessor):
        """Test advanced text preprocessing features"""
        await preprocessor.initialize()
        
        # Configure advanced options
        preprocessor.set_config({
            "remove_urls": True,
            "remove_mentions": True,
            "remove_hashtags": True,
            "normalize_unicode": True,
            "remove_extra_whitespace": True
        })
        
        input_text = "Check out this link https://example.com @user #trending 🔥 multiple   spaces"
        
        result = await preprocessor.process_text(input_text)
        
        # Should remove URLs, mentions, hashtags
        assert "https://example.com" not in result.processed_text
        assert "@user" not in result.processed_text
        assert "#trending" not in result.processed_text
        assert "multiple   spaces" not in result.processed_text  # Extra spaces normalized
    
    @pytest.mark.unit
    async def test_text_preprocessing_batch(self, preprocessor, sample_texts):
        """Test batch text preprocessing"""
        await preprocessor.initialize()
        
        texts = [
            sample_texts["human_casual"],
            sample_texts["ai_formal"],
            sample_texts["edge_case_special"]
        ]
        
        results = await preprocessor.process_batch(texts)
        
        assert len(results) == 3
        for result in results:
            assert hasattr(result, 'processed_text')
            assert hasattr(result, 'metadata')
            assert result.metadata["processing_time"] > 0
    
    @pytest.mark.unit
    def test_text_preprocessing_configuration(self, preprocessor):
        """Test TextPreprocessor configuration"""
        # Test default configuration
        default_config = preprocessor.get_config()
        assert isinstance(default_config, dict)
        assert "remove_urls" in default_config
        
        # Test setting custom configuration
        custom_config = {
            "remove_urls": False,
            "lowercase": True,
            "remove_punctuation": False
        }
        
        preprocessor.set_config(custom_config)
        updated_config = preprocessor.get_config()
        
        assert updated_config["remove_urls"] is False
        assert updated_config["lowercase"] is True


class TestFeatureExtractor:
    """Test suite for FeatureExtractor"""
    
    @pytest.fixture
    def extractor(self):
        """Create FeatureExtractor instance"""
        return FeatureExtractor()
    
    @pytest.mark.unit
    async def test_feature_extraction_basic(self, extractor, sample_texts):
        """Test basic feature extraction"""
        await extractor.initialize()
        
        text = sample_texts["ai_formal"]
        features = await extractor.extract_features(text)
        
        assert isinstance(features, dict)
        assert "text_length" in features
        assert "word_count" in features
        assert "sentence_count" in features
        assert "avg_word_length" in features
        assert "punctuation_ratio" in features
    
    @pytest.mark.unit
    async def test_feature_extraction_linguistic(self, extractor, sample_texts):
        """Test linguistic feature extraction"""
        await extractor.initialize()
        
        # Enable linguistic features
        extractor.set_config({"include_linguistic": True})
        
        text = sample_texts["ai_hedged"]
        features = await extractor.extract_features(text)
        
        # Should include linguistic features
        assert "hedging_words" in features
        assert "certainty_words" in features
        assert "formal_words" in features
        assert "readability_score" in features
        assert "sentiment_score" in features
    
    @pytest.mark.unit
    async def test_feature_extraction_patterns(self, extractor, sample_texts):
        """Test pattern-based feature extraction"""
        await extractor.initialize()
        
        # Configure pattern detection
        extractor.set_config({
            "detect_patterns": True,
            "patterns": {
                "meta_commentary": [r"important to note", r"it should be mentioned"],
                "hedging": [r"perhaps", r"might", r"seems"],
                "formal_structure": [r"firstly", r"secondly", r"in conclusion"]
            }
        })
        
        text = "It's important to note that this might perhaps be considered formal."
        features = await extractor.extract_features(text)
        
        assert "pattern_meta_commentary" in features
        assert "pattern_hedging" in features
        assert features["pattern_meta_commentary"] > 0
        assert features["pattern_hedging"] > 0
    
    @pytest.mark.unit
    async def test_feature_extraction_batch(self, extractor, sample_texts):
        """Test batch feature extraction"""
        await extractor.initialize()
        
        texts = [
            sample_texts["human_casual"],
            sample_texts["ai_formal"],
            sample_texts["human_emotional"]
        ]
        
        batch_features = await extractor.extract_features_batch(texts)
        
        assert len(batch_features) == 3
        for features in batch_features:
            assert isinstance(features, dict)
            assert "text_length" in features
            assert "word_count" in features
    
    @pytest.mark.unit
    async def test_feature_extraction_performance(self, extractor, performance_monitor):
        """Test feature extraction performance"""
        await extractor.initialize()
        
        text = "A" * 1000  # Large text
        
        performance_monitor.start()
        features = await extractor.extract_features(text)
        duration = performance_monitor.stop()
        
        # Should complete reasonably quickly
        assert duration < 1.0  # Less than 1 second
        assert features["text_length"] == 1000


class TestDataValidator:
    """Test suite for DataValidator"""
    
    @pytest.fixture
    def validator(self):
        """Create DataValidator instance"""
        return DataValidator()
    
    @pytest.mark.unit
    async def test_data_validation_basic(self, validator):
        """Test basic data validation"""
        await validator.initialize()
        
        # Valid sample
        valid_sample = DataSample(
            id="valid_001",
            content="This is valid content for testing",
            label="human"
        )
        
        result = await validator.validate_sample(valid_sample)
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0
    
    @pytest.mark.unit
    async def test_data_validation_errors(self, validator):
        """Test data validation with errors"""
        await validator.initialize()
        
        # Invalid samples
        invalid_samples = [
            DataSample(id="", content="content", label="human"),  # Empty ID
            DataSample(id="test", content="", label="human"),     # Empty content
            DataSample(id="test", content="x", label="human"),    # Too short
            DataSample(id="test", content="content", label="invalid"),  # Invalid label
        ]
        
        for sample in invalid_samples:
            result = await validator.validate_sample(sample)
            assert result.is_valid is False
            assert len(result.errors) > 0
    
    @pytest.mark.unit
    async def test_data_validation_batch(self, validator, sample_data_samples):
        """Test batch data validation"""
        await validator.initialize()
        
        # Add some invalid samples
        invalid_sample = DataSample(id="invalid", content="", label="human")
        all_samples = sample_data_samples + [invalid_sample]
        
        batch = DataBatch(samples=all_samples, batch_id="validation_test")
        result = await validator.validate_batch(batch)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is False  # Due to invalid sample
        assert len(result.errors) > 0
        assert result.metadata["total_samples"] == len(all_samples)
        assert result.metadata["valid_samples"] == len(sample_data_samples)
        assert result.metadata["invalid_samples"] == 1
    
    @pytest.mark.unit
    async def test_data_validation_custom_rules(self, validator):
        """Test custom validation rules"""
        await validator.initialize()
        
        # Add custom validation rule
        def check_content_quality(sample):
            if len(sample.content.split()) < 3:
                return False, "Content must have at least 3 words"
            return True, None
        
        validator.add_custom_rule("content_quality", check_content_quality)
        
        # Test with content that fails custom rule
        sample = DataSample(
            id="custom_test",
            content="Too short",  # Only 2 words
            label="human"
        )
        
        result = await validator.validate_sample(sample)
        assert result.is_valid is False
        assert any("at least 3 words" in error for error in result.errors)


class TestSchemaValidator:
    """Test suite for SchemaValidator"""
    
    @pytest.fixture
    def schema_validator(self):
        """Create SchemaValidator instance"""
        return SchemaValidator()
    
    @pytest.mark.unit
    def test_schema_validation_json(self, schema_validator):
        """Test JSON schema validation"""
        # Define schema
        schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "content": {"type": "string", "minLength": 5},
                "label": {"type": "string", "enum": ["human", "ai"]}
            },
            "required": ["id", "content", "label"]
        }
        
        schema_validator.set_schema(schema)
        
        # Valid data
        valid_data = {
            "id": "test_001",
            "content": "Valid content here",
            "label": "human"
        }
        
        result = schema_validator.validate(valid_data)
        assert result.is_valid is True
        
        # Invalid data
        invalid_data = {
            "id": "",  # Too short
            "content": "Short",  # Too short
            "label": "invalid"  # Not in enum
        }
        
        result = schema_validator.validate(invalid_data)
        assert result.is_valid is False
        assert len(result.errors) > 0


class TestDataStores:
    """Test suite for data storage implementations"""
    
    @pytest.mark.unit
    async def test_file_data_store(self, temp_dir):
        """Test FileDataStore operations"""
        store = FileDataStore(base_path=str(temp_dir))
        await store.initialize()
        
        # Test storing data
        sample = DataSample(
            id="file_test",
            content="Test content for file store",
            label="human"
        )
        
        result = await store.store_sample(sample)
        assert result is True
        
        # Test retrieving data
        retrieved = await store.get_sample("file_test")
        assert retrieved is not None
        assert retrieved.content == "Test content for file store"
        
        # Test listing samples
        all_samples = await store.list_samples()
        assert len(all_samples) == 1
        assert all_samples[0].id == "file_test"
    
    @pytest.mark.unit
    async def test_file_data_store_batch(self, temp_dir):
        """Test FileDataStore batch operations"""
        store = FileDataStore(base_path=str(temp_dir))
        await store.initialize()
        
        # Create batch of samples
        samples = [
            DataSample(id=f"batch_{i}", content=f"Batch content {i}", label="human")
            for i in range(5)
        ]
        
        batch = DataBatch(samples=samples, batch_id="file_batch_test")
        
        # Store batch
        result = await store.store_batch(batch)
        assert result is True
        
        # Retrieve batch
        retrieved_batch = await store.get_batch("file_batch_test")
        assert retrieved_batch is not None
        assert retrieved_batch.size() == 5
    
    @pytest.mark.unit
    async def test_database_data_store(self, mock_database):
        """Test DatabaseDataStore operations"""
        store = DatabaseDataStore(database=mock_database)
        await store.initialize()
        
        # Test storing sample
        sample = DataSample(
            id="db_test",
            content="Test content for database",
            label="ai"
        )
        
        result = await store.store_sample(sample)
        assert result is True
        
        # Verify database was called
        mock_database.insert.assert_called()
        
        # Test querying
        samples = await store.query_samples(label="ai")
        assert isinstance(samples, list)


class TestDataPipeline:
    """Test suite for DataPipeline"""
    
    @pytest.fixture
    def mock_pipeline_stages(self):
        """Create mock pipeline stages"""
        collector = Mock(spec=IDataCollector)
        processor = Mock(spec=IDataProcessor)
        validator = Mock(spec=IDataValidator)
        store = Mock(spec=IDataStore)
        
        # Configure mocks
        collector.collect_data = AsyncMock(return_value=[
            DataSample(id="pipeline_1", content="Pipeline test 1", label="human"),
            DataSample(id="pipeline_2", content="Pipeline test 2", label="ai")
        ])
        
        processor.process_batch = AsyncMock(return_value=ProcessingResult(
            success=True,
            processed_count=2,
            metadata={"processing_time": 0.5}
        ))
        
        validator.validate_batch = AsyncMock(return_value=ValidationResult(
            is_valid=True,
            errors=[],
            metadata={"valid_samples": 2}
        ))
        
        store.store_batch = AsyncMock(return_value=True)
        
        return {
            "collector": collector,
            "processor": processor, 
            "validator": validator,
            "store": store
        }
    
    @pytest.mark.unit
    async def test_pipeline_execution(self, mock_pipeline_stages):
        """Test complete pipeline execution"""
        pipeline = DataPipeline()
        
        # Add stages
        pipeline.add_stage("collect", mock_pipeline_stages["collector"])
        pipeline.add_stage("process", mock_pipeline_stages["processor"])
        pipeline.add_stage("validate", mock_pipeline_stages["validator"])
        pipeline.add_stage("store", mock_pipeline_stages["store"])
        
        # Execute pipeline
        result = await pipeline.execute({
            "query": "test query",
            "max_samples": 10
        })
        
        assert result.success is True
        assert result.metadata["stages_completed"] == 4
        
        # Verify all stages were called
        mock_pipeline_stages["collector"].collect_data.assert_called_once()
        mock_pipeline_stages["processor"].process_batch.assert_called_once()
        mock_pipeline_stages["validator"].validate_batch.assert_called_once()
        mock_pipeline_stages["store"].store_batch.assert_called_once()
    
    @pytest.mark.unit
    async def test_pipeline_error_handling(self, mock_pipeline_stages):
        """Test pipeline error handling"""
        pipeline = DataPipeline()
        
        # Configure one stage to fail
        mock_pipeline_stages["processor"].process_batch = AsyncMock(
            side_effect=Exception("Processing failed")
        )
        
        pipeline.add_stage("collect", mock_pipeline_stages["collector"])
        pipeline.add_stage("process", mock_pipeline_stages["processor"])
        pipeline.add_stage("validate", mock_pipeline_stages["validator"])
        
        # Execute pipeline
        result = await pipeline.execute({"query": "test"})
        
        assert result.success is False
        assert "Processing failed" in str(result.error)
        
        # Collector should have been called, but not validator
        mock_pipeline_stages["collector"].collect_data.assert_called_once()
        mock_pipeline_stages["validator"].validate_batch.assert_not_called()
    
    @pytest.mark.unit
    async def test_pipeline_stage_configuration(self):
        """Test pipeline stage configuration"""
        pipeline = DataPipeline()
        
        # Test adding stages
        mock_stage = Mock()
        pipeline.add_stage("test_stage", mock_stage, {"param1": "value1"})
        
        assert "test_stage" in pipeline.get_stages()
        stage_info = pipeline.get_stage_info("test_stage")
        assert stage_info["config"]["param1"] == "value1"
        
        # Test removing stages
        pipeline.remove_stage("test_stage")
        assert "test_stage" not in pipeline.get_stages()


@pytest.mark.integration
class TestDataProcessorIntegration:
    """Integration tests for data processors"""
    
    @pytest.mark.integration
    async def test_full_data_processing_pipeline(self, temp_dir):
        """Test complete data processing workflow"""
        # Initialize components
        collector = ManualDataCollector()
        preprocessor = TextPreprocessor()
        extractor = FeatureExtractor()
        validator = DataValidator()
        store = FileDataStore(base_path=str(temp_dir))
        
        # Initialize all components
        for component in [collector, preprocessor, extractor, validator, store]:
            await component.initialize()
        
        # Add sample data
        samples = [
            DataSample(
                id="integration_1",
                content="This is a formal text that demonstrates important considerations.",
                label="ai"
            ),
            DataSample(
                id="integration_2", 
                content="lol this is so funny 😂 can't stop laughing",
                label="human"
            )
        ]
        
        await collector.add_batch(samples)
        
        # Process through pipeline
        collected_samples = await collector.get_samples()
        
        # Preprocess
        preprocessed_results = []
        for sample in collected_samples:
            result = await preprocessor.process_text(sample.content)
            preprocessed_results.append(result)
        
        # Extract features
        feature_results = []
        for result in preprocessed_results:
            features = await extractor.extract_features(result.processed_text)
            feature_results.append(features)
        
        # Validate
        batch = DataBatch(samples=collected_samples, batch_id="integration_test")
        validation_result = await validator.validate_batch(batch)
        
        # Store
        if validation_result.is_valid:
            store_result = await store.store_batch(batch)
            assert store_result is True
        
        # Verify results
        assert len(preprocessed_results) == 2
        assert len(feature_results) == 2
        assert validation_result.is_valid is True
        
        # Verify features were extracted correctly
        for features in feature_results:
            assert "text_length" in features
            assert "word_count" in features
            assert features["text_length"] > 0
            assert features["word_count"] > 0
    
    @pytest.mark.integration
    async def test_error_propagation_in_pipeline(self):
        """Test error handling across pipeline components"""
        # Create components with intentional failures
        collector = ManualDataCollector()
        await collector.initialize()
        
        # Add invalid sample
        invalid_sample = DataSample(id="invalid", content="", label="human")
        await collector.add_sample(invalid_sample)
        
        validator = DataValidator()
        await validator.initialize()
        
        # Process invalid data
        samples = await collector.get_samples()
        batch = DataBatch(samples=samples, batch_id="error_test")
        
        validation_result = await validator.validate_batch(batch)
        
        # Should properly report validation errors
        assert validation_result.is_valid is False
        assert len(validation_result.errors) > 0
        assert validation_result.metadata["invalid_samples"] > 0