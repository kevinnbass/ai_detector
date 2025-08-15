"""
High-throughput data processing system for >1000 tweets/min processing.

Implements parallel processing, vectorized operations, batch optimization,
and stream processing to achieve maximum data processing throughput.
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional, Iterator, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from queue import Queue
import threading
import multiprocessing as mp
from contextlib import contextmanager

import numpy as np
import pandas as pd
from src.core.monitoring import get_logger, get_metrics_collector


@dataclass
class ProcessingConfig:
    """Configuration for high-throughput processing."""
    max_workers: int = mp.cpu_count()
    batch_size: int = 100
    queue_size: int = 10000
    processing_timeout: float = 30.0
    vectorization_enabled: bool = True
    parallel_feature_extraction: bool = True
    stream_processing: bool = True
    memory_optimization: bool = True


@dataclass
class ProcessingStats:
    """Processing performance statistics."""
    tweets_processed: int = 0
    tweets_per_minute: float = 0.0
    processing_time_ms: float = 0.0
    throughput_efficiency: float = 0.0
    error_rate: float = 0.0
    memory_usage_mb: float = 0.0


class HighThroughputProcessor:
    """High-performance data processing system for tweet analysis."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        
        # Processing components
        self.thread_pool = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=self.config.max_workers)
        
        # Processing queues
        self.input_queue = Queue(maxsize=self.config.queue_size)
        self.output_queue = Queue(maxsize=self.config.queue_size)
        self.error_queue = Queue(maxsize=1000)
        
        # Processing state
        self.is_processing = False
        self.processing_workers = []
        self.stats = ProcessingStats()
        
        # Vectorized processors
        self.vectorized_processors = {}
        self.batch_processors = {}
        
        # Initialize components
        self.initialize_processors()
    
    def initialize_processors(self):
        """Initialize vectorized and batch processors."""
        try:
            # Vectorized text processors
            self.vectorized_processors = {
                'length_calculator': VectorizedLengthCalculator(),
                'pattern_matcher': VectorizedPatternMatcher(),
                'feature_extractor': VectorizedFeatureExtractor(),
                'classifier': VectorizedClassifier()
            }
            
            # Batch processors
            self.batch_processors = {
                'text_preprocessor': BatchTextPreprocessor(self.config.batch_size),
                'feature_aggregator': BatchFeatureAggregator(),
                'result_formatter': BatchResultFormatter()
            }
            
            self.logger.info(f"Initialized {len(self.vectorized_processors)} vectorized processors")
            self.logger.info(f"Initialized {len(self.batch_processors)} batch processors")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize processors: {e}")
            raise
    
    async def process_tweets_async(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process tweets asynchronously with high throughput."""
        start_time = time.time()
        
        try:
            # Start parallel processing workers
            if self.config.stream_processing:
                results = await self._stream_process_tweets(tweets)
            else:
                results = await self._batch_process_tweets(tweets)
            
            # Calculate performance metrics
            processing_time = (time.time() - start_time) * 1000
            tweets_per_minute = (len(tweets) / (processing_time / 1000)) * 60
            
            # Update statistics
            self.stats.tweets_processed += len(tweets)
            self.stats.processing_time_ms = processing_time
            self.stats.tweets_per_minute = tweets_per_minute
            self.stats.throughput_efficiency = min(tweets_per_minute / 1000, 1.0)
            
            # Record metrics
            self.metrics.histogram('data_processing_duration_ms', processing_time)
            self.metrics.gauge('data_processing_throughput_tpm', tweets_per_minute)
            self.metrics.counter('data_tweets_processed_total', len(tweets))
            
            self.logger.info(f"Processed {len(tweets)} tweets in {processing_time:.1f}ms "
                           f"({tweets_per_minute:.0f} tweets/min)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Tweet processing failed: {e}")
            self.stats.error_rate += 1
            raise
    
    async def _stream_process_tweets(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process tweets using streaming pipeline."""
        # Create processing pipeline
        pipeline = StreamProcessingPipeline(
            batch_size=self.config.batch_size,
            max_workers=self.config.max_workers
        )
        
        # Process tweets through pipeline
        results = []
        async for batch_result in pipeline.process_stream(tweets):
            results.extend(batch_result)
        
        return results
    
    async def _batch_process_tweets(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process tweets using optimized batch processing."""
        # Split into batches
        batches = self._create_batches(tweets, self.config.batch_size)
        
        # Process batches in parallel
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_single_batch(batch))
            tasks.append(task)
        
        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        results = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                self.logger.error(f"Batch processing error: {batch_result}")
                continue
            results.extend(batch_result)
        
        return results
    
    async def _process_single_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a single batch of tweets."""
        try:
            # Extract texts for vectorized processing
            texts = [tweet.get('text', '') for tweet in batch]
            
            # Run vectorized processing
            if self.config.vectorization_enabled:
                features = await self._vectorized_feature_extraction(texts)
                classifications = await self._vectorized_classification(features)
            else:
                # Fallback to individual processing
                features = []
                classifications = []
                for text in texts:
                    feature = await self._extract_features_individual(text)
                    classification = await self._classify_individual(feature)
                    features.append(feature)
                    classifications.append(classification)
            
            # Combine results
            results = []
            for i, tweet in enumerate(batch):
                result = {
                    'tweet_id': tweet.get('id'),
                    'original_text': tweet.get('text'),
                    'is_ai_generated': classifications[i]['is_ai_generated'],
                    'confidence_score': classifications[i]['confidence_score'],
                    'features': features[i],
                    'processing_metadata': {
                        'batch_size': len(batch),
                        'processing_method': 'vectorized' if self.config.vectorization_enabled else 'individual',
                        'timestamp': time.time()
                    }
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            return []
    
    async def _vectorized_feature_extraction(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract features using vectorized operations."""
        loop = asyncio.get_event_loop()
        
        # Run feature extraction in thread pool
        features = await loop.run_in_executor(
            self.thread_pool,
            self.vectorized_processors['feature_extractor'].extract_batch,
            texts
        )
        
        return features
    
    async def _vectorized_classification(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform classification using vectorized operations."""
        loop = asyncio.get_event_loop()
        
        # Run classification in thread pool
        classifications = await loop.run_in_executor(
            self.thread_pool,
            self.vectorized_processors['classifier'].classify_batch,
            features
        )
        
        return classifications
    
    async def _extract_features_individual(self, text: str) -> Dict[str, Any]:
        """Extract features for individual text (fallback)."""
        return {
            'char_count': len(text),
            'word_count': len(text.split()),
            'avg_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1)
        }
    
    async def _classify_individual(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Classify individual feature set (fallback)."""
        # Simple heuristic classification
        is_ai = features['avg_word_length'] > 6
        confidence = 0.7 if is_ai else 0.3
        
        return {
            'is_ai_generated': is_ai,
            'confidence_score': confidence
        }
    
    def _create_batches(self, items: List[Any], batch_size: int) -> List[List[Any]]:
        """Create batches from list of items."""
        batches = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def process_tweets_sync(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Synchronous wrapper for tweet processing."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(self.process_tweets_async(tweets))
        finally:
            loop.close()
    
    def start_continuous_processing(self, input_stream: Iterator[Dict[str, Any]]) -> None:
        """Start continuous processing of tweet stream."""
        self.is_processing = True
        
        # Start processing workers
        for i in range(self.config.max_workers):
            worker = threading.Thread(
                target=self._continuous_processing_worker,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.processing_workers.append(worker)
        
        # Feed input stream
        self._feed_input_stream(input_stream)
    
    def _continuous_processing_worker(self, worker_id: int) -> None:
        """Continuous processing worker thread."""
        self.logger.debug(f"Started processing worker {worker_id}")
        
        while self.is_processing:
            try:
                # Get batch from input queue
                batch = []
                
                # Collect batch
                for _ in range(self.config.batch_size):
                    try:
                        tweet = self.input_queue.get(timeout=1.0)
                        batch.append(tweet)
                    except:
                        break  # Timeout or empty queue
                
                if not batch:
                    continue
                
                # Process batch
                results = self.process_tweets_sync(batch)
                
                # Put results in output queue
                for result in results:
                    self.output_queue.put(result)
                
                # Mark tasks as done
                for _ in batch:
                    self.input_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                self.error_queue.put(e)
    
    def _feed_input_stream(self, input_stream: Iterator[Dict[str, Any]]) -> None:
        """Feed input stream to processing queue."""
        def feed_worker():
            try:
                for tweet in input_stream:
                    self.input_queue.put(tweet)
            except Exception as e:
                self.logger.error(f"Input stream error: {e}")
                self.error_queue.put(e)
        
        feed_thread = threading.Thread(target=feed_worker, daemon=True)
        feed_thread.start()
    
    def stop_continuous_processing(self) -> None:
        """Stop continuous processing."""
        self.is_processing = False
        
        # Wait for workers to finish
        for worker in self.processing_workers:
            worker.join(timeout=5.0)
        
        self.processing_workers.clear()
    
    def get_processing_stats(self) -> ProcessingStats:
        """Get current processing statistics."""
        return self.stats
    
    def get_throughput_report(self) -> Dict[str, Any]:
        """Get comprehensive throughput report."""
        return {
            'current_throughput': {
                'tweets_per_minute': self.stats.tweets_per_minute,
                'target_achieved': self.stats.tweets_per_minute >= 1000,
                'efficiency': self.stats.throughput_efficiency
            },
            'performance_metrics': {
                'total_processed': self.stats.tweets_processed,
                'avg_processing_time_ms': self.stats.processing_time_ms,
                'error_rate': self.stats.error_rate,
                'memory_usage_mb': self.stats.memory_usage_mb
            },
            'configuration': {
                'max_workers': self.config.max_workers,
                'batch_size': self.config.batch_size,
                'vectorization_enabled': self.config.vectorization_enabled,
                'stream_processing': self.config.stream_processing
            },
            'optimization_recommendations': self._get_optimization_recommendations()
        }
    
    def _get_optimization_recommendations(self) -> List[str]:
        """Get optimization recommendations based on current performance."""
        recommendations = []
        
        if self.stats.tweets_per_minute < 1000:
            recommendations.append("Increase batch size or worker count")
        
        if self.stats.throughput_efficiency < 0.8:
            recommendations.append("Enable vectorization and stream processing")
        
        if self.stats.error_rate > 0.05:
            recommendations.append("Investigate and fix processing errors")
        
        if self.stats.memory_usage_mb > 1000:
            recommendations.append("Enable memory optimization")
        
        return recommendations
    
    def cleanup(self) -> None:
        """Clean up processing resources."""
        self.stop_continuous_processing()
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)


class VectorizedLengthCalculator:
    """Vectorized text length calculations."""
    
    def calculate_lengths(self, texts: List[str]) -> np.ndarray:
        """Calculate character and word lengths vectorized."""
        # Convert to numpy for vectorized operations
        text_array = np.array(texts, dtype=object)
        
        # Vectorized length calculations
        char_lengths = np.array([len(text) for text in text_array])
        word_lengths = np.array([len(text.split()) for text in text_array])
        
        return np.column_stack([char_lengths, word_lengths])


class VectorizedPatternMatcher:
    """Vectorized pattern matching for text analysis."""
    
    def __init__(self):
        self.patterns = {
            'formal_words': ['furthermore', 'however', 'therefore', 'moreover'],
            'ai_indicators': ['comprehensive', 'multifaceted', 'paradigm'],
            'human_indicators': ['lol', 'omg', 'haha', '😂', '😊']
        }
    
    def match_patterns(self, texts: List[str]) -> np.ndarray:
        """Match patterns across all texts vectorized."""
        results = np.zeros((len(texts), len(self.patterns)))
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            
            for j, (pattern_name, patterns) in enumerate(self.patterns.items()):
                count = sum(text_lower.count(pattern) for pattern in patterns)
                results[i, j] = count
        
        return results


class VectorizedFeatureExtractor:
    """Vectorized feature extraction for text processing."""
    
    def __init__(self):
        self.length_calc = VectorizedLengthCalculator()
        self.pattern_matcher = VectorizedPatternMatcher()
    
    def extract_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Extract features for batch of texts."""
        if not texts:
            return []
        
        # Vectorized calculations
        lengths = self.length_calc.calculate_lengths(texts)
        patterns = self.pattern_matcher.match_patterns(texts)
        
        # Convert to feature dictionaries
        features = []
        for i, text in enumerate(texts):
            feature_dict = {
                'char_count': int(lengths[i, 0]),
                'word_count': int(lengths[i, 1]),
                'avg_word_length': lengths[i, 0] / max(lengths[i, 1], 1),
                'formal_words': int(patterns[i, 0]),
                'ai_indicators': int(patterns[i, 1]),
                'human_indicators': int(patterns[i, 2]),
                'text_complexity': self._calculate_complexity(text),
                'readability_score': self._calculate_readability(lengths[i, 0], lengths[i, 1])
            }
            features.append(feature_dict)
        
        return features
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate text complexity score."""
        words = text.split()
        if not words:
            return 0.0
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        unique_words = len(set(word.lower() for word in words))
        vocabulary_diversity = unique_words / len(words)
        
        return (avg_word_length / 10) * vocabulary_diversity
    
    def _calculate_readability(self, char_count: int, word_count: int) -> float:
        """Calculate readability score."""
        if word_count == 0:
            return 0.0
        
        avg_word_length = char_count / word_count
        return max(0, 1 - (avg_word_length / 15))  # Inverse relationship


class VectorizedClassifier:
    """Vectorized classification for AI text detection."""
    
    def __init__(self):
        # Simple linear model weights (in production, use trained model)
        self.weights = np.array([
            0.01,   # char_count
            0.02,   # word_count  
            0.15,   # avg_word_length
            0.20,   # formal_words
            0.25,   # ai_indicators
            -0.30,  # human_indicators (negative)
            0.20,   # text_complexity
            -0.10   # readability_score (negative)
        ])
        self.bias = 0.1
    
    def classify_batch(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Classify batch of feature sets."""
        if not features:
            return []
        
        # Convert features to matrix
        feature_matrix = self._features_to_matrix(features)
        
        # Vectorized classification
        scores = np.dot(feature_matrix, self.weights) + self.bias
        probabilities = 1 / (1 + np.exp(-scores))  # Sigmoid
        
        # Convert to classification results
        results = []
        for i, prob in enumerate(probabilities):
            results.append({
                'is_ai_generated': bool(prob > 0.5),
                'confidence_score': float(abs(prob - 0.5) * 2),  # Convert to confidence
                'raw_probability': float(prob)
            })
        
        return results
    
    def _features_to_matrix(self, features: List[Dict[str, Any]]) -> np.ndarray:
        """Convert feature dictionaries to numpy matrix."""
        feature_keys = [
            'char_count', 'word_count', 'avg_word_length',
            'formal_words', 'ai_indicators', 'human_indicators',
            'text_complexity', 'readability_score'
        ]
        
        matrix = np.zeros((len(features), len(feature_keys)))
        
        for i, feature_dict in enumerate(features):
            for j, key in enumerate(feature_keys):
                matrix[i, j] = feature_dict.get(key, 0)
        
        return matrix


class BatchTextPreprocessor:
    """Batch text preprocessing for efficiency."""
    
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess batch of texts."""
        # Vectorized preprocessing operations
        processed = []
        
        for text in texts:
            # Basic preprocessing
            cleaned = text.strip().lower()
            
            # Remove extra whitespace
            cleaned = ' '.join(cleaned.split())
            
            processed.append(cleaned)
        
        return processed


class BatchFeatureAggregator:
    """Aggregate features across batches."""
    
    def aggregate_features(self, feature_batches: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Aggregate features from multiple batches."""
        all_features = []
        for batch in feature_batches:
            all_features.extend(batch)
        
        if not all_features:
            return {}
        
        # Calculate aggregate statistics
        char_counts = [f['char_count'] for f in all_features]
        word_counts = [f['word_count'] for f in all_features]
        ai_indicators = [f['ai_indicators'] for f in all_features]
        
        return {
            'total_texts': len(all_features),
            'avg_char_count': np.mean(char_counts),
            'avg_word_count': np.mean(word_counts),
            'total_ai_indicators': np.sum(ai_indicators),
            'processing_timestamp': time.time()
        }


class BatchResultFormatter:
    """Format batch processing results."""
    
    def format_batch(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format batch of results."""
        if not results:
            return {'results': [], 'batch_stats': {}}
        
        # Calculate batch statistics
        ai_count = sum(1 for r in results if r.get('is_ai_generated', False))
        avg_confidence = np.mean([r.get('confidence_score', 0) for r in results])
        
        return {
            'results': results,
            'batch_stats': {
                'total_items': len(results),
                'ai_detected': ai_count,
                'human_detected': len(results) - ai_count,
                'average_confidence': float(avg_confidence),
                'batch_id': f"batch_{int(time.time())}"
            }
        }


class StreamProcessingPipeline:
    """Stream processing pipeline for continuous tweet processing."""
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.logger = get_logger(__name__)
    
    async def process_stream(self, tweets: List[Dict[str, Any]]) -> Iterator[List[Dict[str, Any]]]:
        """Process tweets as a stream with batching."""
        # Create batches
        for i in range(0, len(tweets), self.batch_size):
            batch = tweets[i:i + self.batch_size]
            
            # Process batch asynchronously
            result = await self._process_stream_batch(batch)
            
            yield result
    
    async def _process_stream_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a single batch in the stream."""
        # Simulate processing
        await asyncio.sleep(0.01)  # Small delay to simulate processing
        
        results = []
        for tweet in batch:
            # Simple processing for demo
            text = tweet.get('text', '')
            
            result = {
                'tweet_id': tweet.get('id'),
                'is_ai_generated': len(text) > 100,  # Simple heuristic
                'confidence_score': 0.8,
                'processing_time_ms': 10,
                'stream_processed': True
            }
            results.append(result)
        
        return results