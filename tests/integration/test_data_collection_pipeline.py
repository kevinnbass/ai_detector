"""
Integration tests for data collection to analysis pipeline.

Tests the end-to-end flow from data collection through
training data preparation to model training and analysis.
"""

import pytest
import asyncio
import json
import tempfile
import os
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch, MagicMock

from src.data.collectors.data_collector import DataCollector
from src.data.processors.data_augmenter import DataAugmenter
from src.training.trainers.trainer import ModelTrainer
from src.integrations.gemini.gemini_structured_analyzer import GeminiStructuredAnalyzer
from src.utils.schema_validator import validate_training_data


class TestDataCollectionPipeline:
    """Test suite for data collection to analysis pipeline."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def sample_tweets(self):
        """Sample tweet data for testing."""
        return [
            {
                "id": "tweet_1",
                "text": "This comprehensive analysis demonstrates the multifaceted nature of contemporary discourse paradigms.",
                "user_id": "user_1",
                "timestamp": "2024-01-15T10:00:00Z",
                "platform": "twitter",
                "metadata": {
                    "retweet_count": 5,
                    "like_count": 12,
                    "verified_user": False
                }
            },
            {
                "id": "tweet_2", 
                "text": "just grabbed coffee and it's amazing! ☕ totally recommend this place",
                "user_id": "user_2",
                "timestamp": "2024-01-15T10:15:00Z",
                "platform": "twitter",
                "metadata": {
                    "retweet_count": 2,
                    "like_count": 8,
                    "verified_user": False
                }
            },
            {
                "id": "tweet_3",
                "text": "Furthermore, it should be mentioned that careful consideration of multiple factors is required.",
                "user_id": "user_3",
                "timestamp": "2024-01-15T10:30:00Z",
                "platform": "twitter",
                "metadata": {
                    "retweet_count": 1,
                    "like_count": 3,
                    "verified_user": True
                }
            }
        ]
    
    @pytest.fixture
    def sample_labels(self):
        """Sample labels for training data."""
        return {
            "tweet_1": {
                "label": "ai_generated",
                "confidence": 0.95,
                "annotator": "expert_1",
                "reasoning": "High formality, academic language patterns"
            },
            "tweet_2": {
                "label": "human_written",
                "confidence": 0.9,
                "annotator": "expert_1", 
                "reasoning": "Casual language, emoji usage, personal experience"
            },
            "tweet_3": {
                "label": "ai_generated",
                "confidence": 0.85,
                "annotator": "expert_2",
                "reasoning": "Formal transitional language, academic style"
            }
        }
    
    @pytest.fixture
    def data_collector(self, temp_data_dir):
        """Create data collector instance."""
        return DataCollector(storage_path=temp_data_dir)
    
    @pytest.fixture
    def data_augmenter(self):
        """Create data augmenter instance."""
        return DataAugmenter()
    
    @pytest.fixture
    def model_trainer(self, temp_data_dir):
        """Create model trainer instance."""
        return ModelTrainer(model_path=temp_data_dir / "models")
    
    def test_data_collection_storage(self, data_collector, sample_tweets, temp_data_dir):
        """Test data collection and storage."""
        # Collect sample data
        for tweet in sample_tweets:
            data_collector.collect_text_sample(
                text=tweet["text"],
                source=tweet["platform"],
                metadata=tweet["metadata"],
                sample_id=tweet["id"]
            )
        
        # Verify data was stored
        collected_data = data_collector.get_collected_data()
        assert len(collected_data) == 3
        
        # Verify data structure
        for item in collected_data:
            assert "text" in item
            assert "source" in item
            assert "metadata" in item
            assert "sample_id" in item
            assert "collected_at" in item
    
    def test_manual_labeling_integration(self, data_collector, sample_tweets, sample_labels):
        """Test manual labeling integration with collected data."""
        # Collect data first
        for tweet in sample_tweets:
            data_collector.collect_text_sample(
                text=tweet["text"],
                source=tweet["platform"],
                metadata=tweet["metadata"],
                sample_id=tweet["id"]
            )
        
        # Add manual labels
        for tweet_id, label_info in sample_labels.items():
            data_collector.add_manual_label(
                sample_id=tweet_id,
                label=label_info["label"],
                confidence=label_info["confidence"],
                annotator=label_info["annotator"],
                reasoning=label_info["reasoning"]
            )
        
        # Verify labeled data
        labeled_data = data_collector.get_labeled_data()
        assert len(labeled_data) == 3
        
        for item in labeled_data:
            assert item["label"] in ["ai_generated", "human_written"]
            assert "confidence" in item
            assert "annotator" in item
    
    def test_training_data_generation(self, data_collector, sample_tweets, sample_labels):
        """Test training data generation from collected and labeled data."""
        # Setup data
        for tweet in sample_tweets:
            data_collector.collect_text_sample(
                text=tweet["text"],
                source=tweet["platform"],
                metadata=tweet["metadata"],
                sample_id=tweet["id"]
            )
        
        for tweet_id, label_info in sample_labels.items():
            data_collector.add_manual_label(
                sample_id=tweet_id,
                label=label_info["label"],
                confidence=label_info["confidence"],
                annotator=label_info["annotator"]
            )
        
        # Generate training data
        training_data = data_collector.generate_training_dataset(
            name="test_dataset",
            version="1.0.0"
        )
        
        # Validate training data format
        result = validate_training_data(training_data)
        assert result.is_valid, f"Training data validation failed: {result.errors}"
        
        # Verify structure
        assert "dataset_info" in training_data
        assert "samples" in training_data
        assert len(training_data["samples"]) == 3
        
        # Check sample structure
        for sample in training_data["samples"]:
            assert "id" in sample
            assert "text" in sample
            assert "label" in sample
            assert "confidence" in sample
    
    @pytest.mark.asyncio
    async def test_data_augmentation_pipeline(self, data_augmenter, sample_tweets):
        """Test data augmentation in the pipeline."""
        # Test text augmentation
        original_text = sample_tweets[0]["text"]
        
        # Test different augmentation methods
        augmented_texts = []
        
        # Synonym replacement
        with patch.object(data_augmenter, 'synonym_replacement') as mock_synonym:
            mock_synonym.return_value = "This comprehensive examination demonstrates the multifaceted nature of contemporary discourse paradigms."
            aug_text = data_augmenter.synonym_replacement(original_text)
            augmented_texts.append(aug_text)
        
        # Random insertion
        with patch.object(data_augmenter, 'random_insertion') as mock_insertion:
            mock_insertion.return_value = "This quite comprehensive analysis clearly demonstrates the multifaceted nature of contemporary discourse paradigms."
            aug_text = data_augmenter.random_insertion(original_text)
            augmented_texts.append(aug_text)
        
        # Verify augmentation
        assert len(augmented_texts) == 2
        for aug_text in augmented_texts:
            assert aug_text != original_text
            assert len(aug_text) > 0
    
    @pytest.mark.asyncio
    async def test_llm_analysis_integration(self, sample_tweets):
        """Test LLM analysis integration in the pipeline."""
        with patch('src.integrations.gemini.gemini_structured_analyzer.genai') as mock_genai:
            # Mock LLM response
            mock_response = MagicMock()
            mock_response.text = json.dumps({
                "analysis_id": "analysis_123",
                "input_text": sample_tweets[0]["text"],
                "ai_probability": 0.85,
                "confidence_score": 0.78,
                "analysis_dimensions": {
                    "writing_style": {
                        "formality_level": 0.92,
                        "complexity_score": 0.88,
                        "vocabulary_sophistication": 0.90
                    },
                    "language_patterns": {
                        "hedging_frequency": 0.12,
                        "modal_verb_usage": 0.08,
                        "passive_voice_ratio": 0.15
                    },
                    "content_structure": {
                        "logical_flow": 0.85,
                        "topic_coherence": 0.80,
                        "argument_structure": 0.75
                    },
                    "authenticity_markers": {
                        "personal_experience": 0.05,
                        "emotional_expression": 0.10,
                        "conversational_elements": 0.08
                    }
                }
            })
            
            mock_genai.configure.return_value = None
            mock_genai.GenerativeModel.return_value.generate_content.return_value = mock_response
            
            # Create analyzer
            analyzer = GeminiStructuredAnalyzer(api_key="test_key")
            
            # Analyze text
            result = await analyzer.analyze_text(sample_tweets[0]["text"])
            
            # Verify analysis results
            assert result["ai_probability"] == 0.85
            assert result["confidence_score"] == 0.78
            assert "analysis_dimensions" in result
            assert "writing_style" in result["analysis_dimensions"]
    
    def test_feature_extraction_pipeline(self, sample_tweets):
        """Test feature extraction from collected data."""
        with patch('src.utils.ml_utils.FeatureExtractor') as mock_extractor:
            # Mock feature extraction
            mock_instance = mock_extractor.return_value
            mock_instance.extract_features.return_value = {
                "char_count": 95,
                "word_count": 12,
                "sentence_count": 1,
                "avg_word_length": 7.9,
                "formality_score": 0.9,
                "complexity_score": 0.85,
                "sentiment_score": 0.1
            }
            
            # Extract features for each tweet
            features = []
            for tweet in sample_tweets:
                feature_vector = mock_instance.extract_features(tweet["text"])
                features.append(feature_vector)
            
            # Verify features
            assert len(features) == 3
            for feature_vector in features:
                assert "char_count" in feature_vector
                assert "word_count" in feature_vector
                assert "formality_score" in feature_vector
    
    @pytest.mark.asyncio
    async def test_model_training_pipeline(self, model_trainer, temp_data_dir):
        """Test model training with collected data."""
        # Create mock training data
        training_data = {
            "dataset_info": {
                "name": "test_dataset",
                "version": "1.0.0",
                "sample_count": 100,
                "ai_samples": 50,
                "human_samples": 50
            },
            "samples": [
                {
                    "id": "sample_1",
                    "text": "Formal academic text with sophisticated vocabulary.",
                    "label": "ai_generated",
                    "confidence": 0.9
                },
                {
                    "id": "sample_2", 
                    "text": "hey this is casual text lol 😊",
                    "label": "human_written",
                    "confidence": 0.85
                }
            ]
        }
        
        # Save training data
        data_file = temp_data_dir / "training_data.json"
        with open(data_file, 'w') as f:
            json.dump(training_data, f)
        
        with patch.object(model_trainer, 'train_model') as mock_train:
            mock_train.return_value = {
                "model_id": "model_123",
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.94,
                "f1_score": 0.91
            }
            
            # Train model
            result = model_trainer.train_model(str(data_file))
            
            # Verify training results
            assert result["accuracy"] > 0.9
            assert result["model_id"] == "model_123"
            mock_train.assert_called_once()
    
    def test_data_quality_validation(self, data_collector, sample_tweets):
        """Test data quality validation in the pipeline."""
        # Add data with quality issues
        problematic_data = [
            {"text": "", "id": "empty_text"},  # Empty text
            {"text": "a", "id": "too_short"},  # Too short
            {"text": "A" * 10000, "id": "too_long"},  # Too long
            {"text": "Normal length text for testing", "id": "good_text"}  # Good text
        ]
        
        for item in problematic_data:
            try:
                data_collector.collect_text_sample(
                    text=item["text"],
                    source="test",
                    sample_id=item["id"]
                )
            except ValueError:
                # Expected for invalid data
                pass
        
        # Verify only valid data was collected
        collected = data_collector.get_collected_data()
        valid_items = [item for item in collected if len(item["text"]) > 10 and len(item["text"]) < 5000]
        assert len(valid_items) >= 1  # At least the good text should be there
    
    def test_pipeline_error_handling(self, data_collector):
        """Test error handling throughout the pipeline."""
        # Test invalid data collection
        with pytest.raises(ValueError):
            data_collector.collect_text_sample(
                text="",  # Empty text should raise error
                source="test"
            )
        
        # Test invalid label
        data_collector.collect_text_sample(text="Valid text", source="test", sample_id="test_id")
        
        with pytest.raises(ValueError):
            data_collector.add_manual_label(
                sample_id="test_id",
                label="invalid_label",  # Should be ai_generated or human_written
                confidence=0.9,
                annotator="test"
            )
    
    def test_data_export_import(self, data_collector, sample_tweets, sample_labels, temp_data_dir):
        """Test data export and import functionality."""
        # Setup data
        for tweet in sample_tweets:
            data_collector.collect_text_sample(
                text=tweet["text"],
                source=tweet["platform"],
                sample_id=tweet["id"]
            )
        
        for tweet_id, label_info in sample_labels.items():
            data_collector.add_manual_label(
                sample_id=tweet_id,
                label=label_info["label"],
                confidence=label_info["confidence"],
                annotator=label_info["annotator"]
            )
        
        # Export data
        export_file = temp_data_dir / "exported_data.json"
        data_collector.export_data(str(export_file))
        
        # Verify export file exists
        assert export_file.exists()
        
        # Import data into new collector
        new_collector = DataCollector(storage_path=temp_data_dir / "imported")
        new_collector.import_data(str(export_file))
        
        # Verify imported data
        imported_data = new_collector.get_labeled_data()
        original_data = data_collector.get_labeled_data()
        
        assert len(imported_data) == len(original_data)
    
    @pytest.mark.asyncio
    async def test_real_time_analysis_pipeline(self, sample_tweets):
        """Test real-time analysis pipeline integration."""
        with patch('src.core.detection.detector.DetectionEngine') as mock_engine:
            # Mock real-time detection
            mock_instance = mock_engine.return_value
            mock_instance.detect_ai_text.return_value = {
                "is_ai_generated": True,
                "confidence_score": 0.85,
                "processing_time_ms": 150,
                "method_used": "ensemble"
            }
            
            # Simulate real-time analysis
            results = []
            for tweet in sample_tweets:
                result = mock_instance.detect_ai_text(tweet["text"])
                results.append({
                    "tweet_id": tweet["id"],
                    "analysis": result
                })
            
            # Verify real-time results
            assert len(results) == 3
            for result in results:
                assert "tweet_id" in result
                assert "analysis" in result
                assert result["analysis"]["processing_time_ms"] < 500  # Fast processing
    
    def test_performance_monitoring_integration(self):
        """Test performance monitoring throughout the pipeline."""
        from src.core.monitoring import get_metrics_collector
        
        metrics = get_metrics_collector()
        
        # Test pipeline performance metrics
        with metrics.get_metric("data_collection_duration_ms").time():
            # Simulate data collection
            import time
            time.sleep(0.01)  # 10ms simulation
        
        # Test collection rate
        metrics.record_rate("data_samples_per_second", count=10)
        
        # Verify metrics
        collection_metric = metrics.get_metric("data_collection_duration_ms")
        assert collection_metric is not None
        
        rate_metric = metrics.get_metric("data_samples_per_second")
        assert rate_metric is not None
        assert rate_metric.get_value() > 0