"""
Performance tests for high-throughput data processing system.

Tests to ensure data processing achieves >1000 tweets/min throughput
while maintaining accuracy and system stability.
"""

import pytest
import asyncio
import time
import statistics
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

from src.data.processing.high_throughput_processor import (
    HighThroughputProcessor,
    ProcessingConfig,
    ProcessingStats,
    VectorizedFeatureExtractor,
    VectorizedClassifier,
    StreamProcessingPipeline
)


class TestHighThroughputProcessor:
    
    @pytest.fixture
    def processor_config(self):
        """Configure processor for high throughput testing."""
        return ProcessingConfig(
            max_workers=mp.cpu_count(),
            batch_size=100,
            queue_size=10000,
            processing_timeout=30.0,
            vectorization_enabled=True,
            parallel_feature_extraction=True,
            stream_processing=True,
            memory_optimization=True
        )
    
    @pytest.fixture
    def processor(self, processor_config):
        """Create processor instance for testing."""
        return HighThroughputProcessor(processor_config)
    
    @pytest.fixture
    def sample_tweets(self):
        """Generate sample tweets for testing."""
        tweets = []
        for i in range(2000):  # Generate 2000 tweets for testing
            tweets.append({
                'id': f'tweet_{i}',
                'text': f'This is test tweet number {i}. It contains enough text to be processed effectively. '
                       f'The content includes various patterns and indicators that the AI detection system '
                       f'can analyze for classification purposes. Tweet #{i} demonstrates typical social media content.',
                'timestamp': time.time(),
                'user_id': f'user_{i % 100}',
                'metadata': {
                    'source': 'test',
                    'language': 'en'
                }
            })
        return tweets
    
    @pytest.mark.asyncio
    async def test_throughput_target_achievement(self, processor, sample_tweets):
        """Test that processor achieves >1000 tweets/min target."""
        # Process 1000 tweets and measure time
        test_tweets = sample_tweets[:1000]
        
        start_time = time.time()
        results = await processor.process_tweets_async(test_tweets)
        end_time = time.time()
        
        # Calculate throughput
        duration_minutes = (end_time - start_time) / 60
        tweets_per_minute = len(test_tweets) / duration_minutes
        
        # Verify target achievement
        assert tweets_per_minute >= 1000, f"Throughput {tweets_per_minute:.0f} tweets/min below target"
        assert len(results) == len(test_tweets), "All tweets should be processed"
        
        # Verify processing stats
        stats = processor.get_processing_stats()
        assert stats.tweets_per_minute >= 1000
        assert stats.throughput_efficiency >= 0.8
    
    @pytest.mark.asyncio
    async def test_sustained_high_throughput(self, processor, sample_tweets):
        """Test sustained high throughput over multiple batches."""
        batch_size = 500
        num_batches = 4
        throughput_measurements = []
        
        for batch_num in range(num_batches):
            batch_start = batch_num * batch_size
            batch_end = batch_start + batch_size
            batch_tweets = sample_tweets[batch_start:batch_end]
            
            start_time = time.time()
            results = await processor.process_tweets_async(batch_tweets)
            end_time = time.time()
            
            duration_minutes = (end_time - start_time) / 60
            tweets_per_minute = len(batch_tweets) / duration_minutes
            throughput_measurements.append(tweets_per_minute)
            
            assert len(results) == len(batch_tweets)
        
        # Verify sustained performance
        avg_throughput = statistics.mean(throughput_measurements)
        min_throughput = min(throughput_measurements)
        
        assert avg_throughput >= 1000, f"Average throughput {avg_throughput:.0f} below target"
        assert min_throughput >= 800, f"Minimum throughput {min_throughput:.0f} too low"
        
        # Performance should not degrade significantly
        first_batch_throughput = throughput_measurements[0]
        last_batch_throughput = throughput_measurements[-1]
        degradation = (first_batch_throughput - last_batch_throughput) / first_batch_throughput
        
        assert degradation < 0.2, f"Performance degradation {degradation:.2%} too high"
    
    @pytest.mark.asyncio
    async def test_vectorized_processing_performance(self, processor, sample_tweets):
        """Test that vectorized processing provides performance gains."""
        test_tweets = sample_tweets[:1000]
        
        # Test with vectorization enabled
        processor.config.vectorization_enabled = True
        start_time = time.time()
        vectorized_results = await processor.process_tweets_async(test_tweets)
        vectorized_duration = time.time() - start_time
        
        # Test with vectorization disabled
        processor.config.vectorization_enabled = False
        start_time = time.time()
        sequential_results = await processor.process_tweets_async(test_tweets)
        sequential_duration = time.time() - start_time
        
        # Vectorized processing should be faster
        speedup = sequential_duration / vectorized_duration
        assert speedup >= 1.5, f"Vectorized processing speedup {speedup:.2f}x insufficient"
        
        # Results should be consistent
        assert len(vectorized_results) == len(sequential_results)
    
    @pytest.mark.asyncio
    async def test_stream_processing_efficiency(self, processor, sample_tweets):
        """Test stream processing efficiency and throughput."""
        test_tweets = sample_tweets[:1500]
        
        # Test stream processing
        processor.config.stream_processing = True
        start_time = time.time()
        stream_results = await processor.process_tweets_async(test_tweets)
        stream_duration = time.time() - start_time
        
        # Test batch processing
        processor.config.stream_processing = False
        start_time = time.time()
        batch_results = await processor.process_tweets_async(test_tweets)
        batch_duration = time.time() - start_time
        
        # Calculate throughput for both methods
        stream_throughput = (len(test_tweets) / stream_duration) * 60
        batch_throughput = (len(test_tweets) / batch_duration) * 60
        
        # Both should achieve target, stream should be more efficient
        assert stream_throughput >= 1000, f"Stream throughput {stream_throughput:.0f} below target"
        assert batch_throughput >= 1000, f"Batch throughput {batch_throughput:.0f} below target"
        
        # Results should be consistent
        assert len(stream_results) == len(batch_results)
    
    def test_parallel_processing_scaling(self, processor_config, sample_tweets):
        """Test that parallel processing scales with worker count."""
        test_tweets = sample_tweets[:1000]
        worker_counts = [1, 2, 4, mp.cpu_count()]
        throughput_by_workers = {}
        
        for worker_count in worker_counts:
            if worker_count > mp.cpu_count():
                continue
                
            config = ProcessingConfig(
                max_workers=worker_count,
                batch_size=100,
                vectorization_enabled=True,
                stream_processing=True
            )
            
            processor = HighThroughputProcessor(config)
            
            start_time = time.time()
            results = processor.process_tweets_sync(test_tweets)
            duration = time.time() - start_time
            
            throughput = (len(test_tweets) / duration) * 60
            throughput_by_workers[worker_count] = throughput
            
            assert len(results) == len(test_tweets)
        
        # Throughput should generally increase with more workers
        if len(throughput_by_workers) >= 2:
            single_worker = throughput_by_workers[1]
            max_workers = throughput_by_workers[max(throughput_by_workers.keys())]
            
            scaling_factor = max_workers / single_worker
            assert scaling_factor >= 1.5, f"Parallel scaling factor {scaling_factor:.2f}x insufficient"
    
    def test_memory_efficiency_under_load(self, processor, sample_tweets):
        """Test memory efficiency during high-throughput processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Process large batches
        large_batches = [sample_tweets[i:i+500] for i in range(0, 2000, 500)]
        
        for batch in large_batches:
            results = processor.process_tweets_sync(batch)
            assert len(results) == len(batch)
            
            # Check memory usage
            current_memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_growth = current_memory - baseline_memory
            
            # Memory growth should be controlled
            assert memory_growth < 200, f"Memory growth {memory_growth:.1f}MB too high"
        
        # Final memory check
        final_memory = process.memory_info().rss / (1024 * 1024)  # MB
        total_growth = final_memory - baseline_memory
        
        assert total_growth < 300, f"Total memory growth {total_growth:.1f}MB excessive"
    
    @pytest.mark.asyncio
    async def test_error_handling_under_load(self, processor, sample_tweets):
        """Test error handling during high-throughput processing."""
        # Create tweets with some problematic content
        test_tweets = sample_tweets[:1000]
        
        # Add some tweets that might cause issues
        problematic_tweets = [
            {'id': 'error_1', 'text': ''},  # Empty text
            {'id': 'error_2', 'text': 'x' * 10000},  # Very long text
            {'id': 'error_3'},  # Missing text field
            {'id': 'error_4', 'text': None}  # None text
        ]
        
        test_tweets.extend(problematic_tweets)
        
        # Processing should handle errors gracefully
        results = await processor.process_tweets_async(test_tweets)
        
        # Should process valid tweets despite errors
        assert len(results) >= len(sample_tweets[:1000])
        
        # Error rate should be tracked
        stats = processor.get_processing_stats()
        assert hasattr(stats, 'error_rate')
    
    def test_continuous_processing_performance(self, processor, sample_tweets):
        """Test continuous processing mode performance."""
        import threading
        import queue
        
        input_queue = queue.Queue()
        output_queue = queue.Queue()
        
        # Add tweets to input queue
        for tweet in sample_tweets[:1000]:
            input_queue.put(tweet)
        
        # Start continuous processing
        def input_stream():
            while not input_queue.empty():
                yield input_queue.get()
        
        start_time = time.time()
        processor.start_continuous_processing(input_stream())
        
        # Wait for processing to complete
        processed_count = 0
        while processed_count < 1000 and time.time() - start_time < 60:
            try:
                result = processor.output_queue.get(timeout=1)
                processed_count += 1
            except:
                break
        
        processing_duration = time.time() - start_time
        processor.stop_continuous_processing()
        
        # Calculate throughput
        throughput = (processed_count / processing_duration) * 60
        
        assert throughput >= 1000, f"Continuous processing throughput {throughput:.0f} below target"
        assert processed_count >= 900, f"Only processed {processed_count}/1000 tweets"
    
    def test_throughput_report_accuracy(self, processor, sample_tweets):
        """Test accuracy of throughput reporting."""
        test_tweets = sample_tweets[:1000]
        
        # Process tweets
        start_time = time.time()
        results = processor.process_tweets_sync(test_tweets)
        actual_duration = time.time() - start_time
        
        # Calculate actual throughput
        actual_throughput = (len(test_tweets) / actual_duration) * 60
        
        # Get reported throughput
        report = processor.get_throughput_report()
        reported_throughput = report['current_throughput']['tweets_per_minute']
        
        # Reports should be reasonably accurate
        accuracy = abs(reported_throughput - actual_throughput) / actual_throughput
        assert accuracy < 0.1, f"Throughput reporting accuracy {accuracy:.2%} insufficient"
        
        # Report should indicate target achievement
        target_achieved = report['current_throughput']['target_achieved']
        if actual_throughput >= 1000:
            assert target_achieved, "Report should indicate target achievement"
    
    @pytest.mark.asyncio
    async def test_batch_size_optimization(self, processor_config, sample_tweets):
        """Test optimal batch size for maximum throughput."""
        test_tweets = sample_tweets[:1000]
        batch_sizes = [50, 100, 200, 500]
        throughput_by_batch_size = {}
        
        for batch_size in batch_sizes:
            config = ProcessingConfig(
                max_workers=processor_config.max_workers,
                batch_size=batch_size,
                vectorization_enabled=True,
                stream_processing=True
            )
            
            processor = HighThroughputProcessor(config)
            
            start_time = time.time()
            results = await processor.process_tweets_async(test_tweets)
            duration = time.time() - start_time
            
            throughput = (len(test_tweets) / duration) * 60
            throughput_by_batch_size[batch_size] = throughput
            
            assert len(results) == len(test_tweets)
        
        # Find optimal batch size
        optimal_batch_size = max(throughput_by_batch_size.keys(), 
                               key=lambda k: throughput_by_batch_size[k])
        optimal_throughput = throughput_by_batch_size[optimal_batch_size]
        
        # Optimal configuration should achieve target
        assert optimal_throughput >= 1000, f"Optimal throughput {optimal_throughput:.0f} below target"
        
        print(f"Optimal batch size: {optimal_batch_size} ({optimal_throughput:.0f} tweets/min)")


class TestVectorizedComponents:
    
    def test_vectorized_feature_extractor_performance(self):
        """Test vectorized feature extraction performance."""
        extractor = VectorizedFeatureExtractor()
        
        # Generate test texts
        texts = [f"This is test text number {i} with various patterns and content." for i in range(1000)]
        
        # Measure performance
        start_time = time.time()
        features = extractor.extract_batch(texts)
        duration = time.time() - start_time
        
        # Should process quickly
        throughput = (len(texts) / duration) * 60
        assert throughput >= 5000, f"Feature extraction throughput {throughput:.0f} too low"
        
        # Should return correct number of results
        assert len(features) == len(texts)
        
        # Features should have expected structure
        for feature_set in features[:10]:  # Check first 10
            assert 'char_count' in feature_set
            assert 'word_count' in feature_set
            assert 'ai_indicators' in feature_set
            assert isinstance(feature_set['char_count'], (int, float))
    
    def test_vectorized_classifier_performance(self):
        """Test vectorized classification performance."""
        classifier = VectorizedClassifier()
        
        # Generate test features
        features = []
        for i in range(1000):
            features.append({
                'char_count': 50 + (i % 100),
                'word_count': 10 + (i % 20),
                'avg_word_length': 5.0 + (i % 5),
                'formal_words': i % 3,
                'ai_indicators': i % 2,
                'human_indicators': (i + 1) % 2,
                'text_complexity': 0.5 + (i % 5) * 0.1,
                'readability_score': 0.3 + (i % 7) * 0.1
            })
        
        # Measure performance
        start_time = time.time()
        classifications = classifier.classify_batch(features)
        duration = time.time() - start_time
        
        # Should classify quickly
        throughput = (len(features) / duration) * 60
        assert throughput >= 10000, f"Classification throughput {throughput:.0f} too low"
        
        # Should return correct number of results
        assert len(classifications) == len(features)
        
        # Classifications should have expected structure
        for classification in classifications[:10]:  # Check first 10
            assert 'is_ai_generated' in classification
            assert 'confidence_score' in classification
            assert isinstance(classification['is_ai_generated'], bool)
            assert 0 <= classification['confidence_score'] <= 1


class TestStreamProcessingPipeline:
    
    @pytest.mark.asyncio
    async def test_stream_pipeline_throughput(self):
        """Test stream processing pipeline throughput."""
        pipeline = StreamProcessingPipeline(batch_size=100, max_workers=4)
        
        # Generate test tweets
        tweets = []
        for i in range(1000):
            tweets.append({
                'id': f'stream_tweet_{i}',
                'text': f'Stream processing test tweet {i} with content for analysis.'
            })
        
        # Process through stream
        results = []
        start_time = time.time()
        
        async for batch_result in pipeline.process_stream(tweets):
            results.extend(batch_result)
        
        duration = time.time() - start_time
        
        # Calculate throughput
        throughput = (len(results) / duration) * 60
        
        assert throughput >= 2000, f"Stream pipeline throughput {throughput:.0f} too low"
        assert len(results) == len(tweets)


@pytest.mark.performance
class TestPerformanceUnderLoad:
    
    def test_memory_pressure_handling(self, processor_config, sample_tweets):
        """Test processor behavior under memory pressure."""
        # Force memory optimization
        config = ProcessingConfig(
            max_workers=processor_config.max_workers,
            batch_size=50,  # Smaller batches under pressure
            memory_optimization=True,
            vectorization_enabled=True
        )
        
        processor = HighThroughputProcessor(config)
        
        # Process in multiple rounds to simulate pressure
        for round_num in range(5):
            batch_start = round_num * 400
            batch_end = batch_start + 400
            batch_tweets = sample_tweets[batch_start:batch_end]
            
            start_time = time.time()
            results = processor.process_tweets_sync(batch_tweets)
            duration = time.time() - start_time
            
            throughput = (len(batch_tweets) / duration) * 60
            
            # Should maintain reasonable throughput even under pressure
            assert throughput >= 800, f"Round {round_num} throughput {throughput:.0f} too low under pressure"
            assert len(results) == len(batch_tweets)
    
    def test_concurrent_processing_safety(self, processor, sample_tweets):
        """Test thread safety during concurrent processing."""
        import threading
        
        test_batches = [
            sample_tweets[0:250],
            sample_tweets[250:500],
            sample_tweets[500:750],
            sample_tweets[750:1000]
        ]
        
        results = [None] * len(test_batches)
        threads = []
        
        def process_batch(batch_index, batch_tweets):
            results[batch_index] = processor.process_tweets_sync(batch_tweets)
        
        # Start concurrent processing
        start_time = time.time()
        for i, batch in enumerate(test_batches):
            thread = threading.Thread(target=process_batch, args=(i, batch))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        duration = time.time() - start_time
        
        # Verify all batches processed
        total_processed = sum(len(result) for result in results if result)
        total_tweets = sum(len(batch) for batch in test_batches)
        
        assert total_processed == total_tweets, "Concurrent processing lost tweets"
        
        # Calculate overall throughput
        throughput = (total_tweets / duration) * 60
        assert throughput >= 1000, f"Concurrent throughput {throughput:.0f} below target"


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short"])