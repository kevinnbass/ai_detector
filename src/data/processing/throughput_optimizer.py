"""
Throughput optimization system for achieving >1000 tweets/min processing.

Combines multiple optimization strategies including GPU acceleration,
multiprocessing, vectorization, and adaptive batching to maximize
data processing throughput.
"""

import time
import asyncio
import statistics
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
import threading
import queue

from src.data.processing.high_throughput_processor import HighThroughputProcessor, ProcessingConfig
from src.data.processing.gpu_accelerated_processor import GPUAcceleratedProcessor, GPUConfig, HybridProcessor
from src.core.monitoring import get_logger, get_metrics_collector


@dataclass
class ThroughputTarget:
    """Target throughput configuration."""
    tweets_per_minute: int = 1000
    max_processing_time_ms: float = 100.0
    min_accuracy: float = 0.85
    max_memory_mb: float = 2000.0


@dataclass
class OptimizationStrategy:
    """Optimization strategy configuration."""
    use_gpu: bool = True
    use_multiprocessing: bool = True
    use_vectorization: bool = True
    use_adaptive_batching: bool = True
    use_stream_processing: bool = True
    dynamic_scaling: bool = True


class ThroughputOptimizer:
    """Main throughput optimization system."""
    
    def __init__(self, 
                 target: Optional[ThroughputTarget] = None,
                 strategy: Optional[OptimizationStrategy] = None):
        self.target = target or ThroughputTarget()
        self.strategy = strategy or OptimizationStrategy()
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        
        # Processing systems
        self.high_throughput_processor = None
        self.gpu_processor = None
        self.hybrid_processor = None
        
        # Performance tracking
        self.performance_history = []
        self.current_config = None
        self.benchmark_results = {}
        
        # Dynamic optimization
        self.optimization_lock = threading.Lock()
        self.last_optimization = 0
        self.optimization_interval = 60  # seconds
        
        self.initialize_processors()
    
    def initialize_processors(self):
        """Initialize all processing systems."""
        try:
            # High-throughput processor
            processing_config = ProcessingConfig(
                max_workers=mp.cpu_count(),
                batch_size=100,
                vectorization_enabled=self.strategy.use_vectorization,
                stream_processing=self.strategy.use_stream_processing,
                parallel_feature_extraction=True,
                memory_optimization=True
            )
            self.high_throughput_processor = HighThroughputProcessor(processing_config)
            
            # GPU processor if enabled
            if self.strategy.use_gpu:
                gpu_config = GPUConfig(
                    use_gpu=True,
                    batch_size_gpu=200,
                    fallback_to_cpu=True
                )
                self.gpu_processor = GPUAcceleratedProcessor(gpu_config)
                self.hybrid_processor = HybridProcessor(gpu_config)
            
            self.logger.info("Throughput optimization processors initialized")
            
        except Exception as e:
            self.logger.error(f"Processor initialization failed: {e}")
            raise
    
    async def optimize_throughput(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimize throughput for processing tweets."""
        start_time = time.time()
        
        try:
            # Choose optimal processing method
            processor, method = await self._select_optimal_processor(tweets)
            
            # Process with chosen method
            results = await self._process_with_optimization(processor, tweets, method)
            
            # Track performance
            duration = time.time() - start_time
            throughput = (len(tweets) / duration) * 60
            
            self._record_performance(method, len(tweets), duration, throughput)
            
            # Check if target achieved
            target_achieved = throughput >= self.target.tweets_per_minute
            
            self.metrics.gauge('throughput_optimizer_tpm', throughput)
            self.metrics.counter('throughput_optimizer_batches_processed')
            
            if target_achieved:
                self.metrics.counter('throughput_optimizer_target_achieved')
            else:
                self.metrics.counter('throughput_optimizer_target_missed')
            
            self.logger.info(f"Processed {len(tweets)} tweets in {duration:.3f}s "
                           f"({throughput:.0f} tpm) using {method} - "
                           f"Target {'ACHIEVED' if target_achieved else 'MISSED'}")
            
            # Trigger dynamic optimization if needed
            if self.strategy.dynamic_scaling:
                await self._trigger_dynamic_optimization(throughput)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Throughput optimization failed: {e}")
            # Fallback to basic processing
            return await self._fallback_processing(tweets)
    
    async def _select_optimal_processor(self, tweets: List[Dict[str, Any]]) -> tuple:
        """Select optimal processor based on current conditions."""
        batch_size = len(tweets)
        
        # Simple heuristics for processor selection
        if self.hybrid_processor and batch_size >= 500:
            return self.hybrid_processor, 'hybrid'
        elif self.gpu_processor and batch_size >= 200:
            return self.gpu_processor, 'gpu'
        elif batch_size >= 100:
            return self.high_throughput_processor, 'high_throughput'
        else:
            return self.high_throughput_processor, 'standard'
    
    async def _process_with_optimization(self, processor, tweets: List[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
        """Process tweets with specific optimization method."""
        if method == 'hybrid':
            return processor.process_tweets_optimal(tweets)
        elif method == 'gpu':
            return processor.process_tweets_gpu(tweets)
        elif method == 'high_throughput':
            return await processor.process_tweets_async(tweets)
        else:
            return await processor.process_tweets_async(tweets)
    
    def _record_performance(self, method: str, tweet_count: int, duration: float, throughput: float):
        """Record performance metrics for analysis."""
        performance_record = {
            'timestamp': time.time(),
            'method': method,
            'tweet_count': tweet_count,
            'duration': duration,
            'throughput': throughput,
            'target_achieved': throughput >= self.target.tweets_per_minute
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history
        cutoff_time = time.time() - 3600  # 1 hour
        self.performance_history = [
            record for record in self.performance_history 
            if record['timestamp'] > cutoff_time
        ]
    
    async def _trigger_dynamic_optimization(self, current_throughput: float):
        """Trigger dynamic optimization if conditions are met."""
        now = time.time()
        
        if now - self.last_optimization < self.optimization_interval:
            return
        
        if current_throughput < self.target.tweets_per_minute * 0.8:  # 80% of target
            with self.optimization_lock:
                await self._perform_dynamic_optimization()
                self.last_optimization = now
    
    async def _perform_dynamic_optimization(self):
        """Perform dynamic optimization of processing parameters."""
        self.logger.info("Performing dynamic optimization")
        
        try:
            # Analyze recent performance
            recent_performance = self._analyze_recent_performance()
            
            # Adjust processing configurations
            if recent_performance['avg_throughput'] < self.target.tweets_per_minute:
                await self._scale_up_processing()
            
            # Update configurations
            await self._update_processor_configs()
            
        except Exception as e:
            self.logger.error(f"Dynamic optimization failed: {e}")
    
    def _analyze_recent_performance(self) -> Dict[str, Any]:
        """Analyze recent performance for optimization decisions."""
        if not self.performance_history:
            return {'avg_throughput': 0, 'method_performance': {}}
        
        recent_records = self.performance_history[-10:]  # Last 10 records
        
        avg_throughput = statistics.mean([r['throughput'] for r in recent_records])
        
        method_performance = {}
        for record in recent_records:
            method = record['method']
            if method not in method_performance:
                method_performance[method] = []
            method_performance[method].append(record['throughput'])
        
        # Calculate average for each method
        for method, throughputs in method_performance.items():
            method_performance[method] = statistics.mean(throughputs)
        
        return {
            'avg_throughput': avg_throughput,
            'method_performance': method_performance,
            'total_records': len(recent_records)
        }
    
    async def _scale_up_processing(self):
        """Scale up processing capabilities."""
        if self.high_throughput_processor:
            # Increase batch size
            current_batch_size = self.high_throughput_processor.config.batch_size
            new_batch_size = min(current_batch_size * 1.2, 200)
            self.high_throughput_processor.config.batch_size = int(new_batch_size)
            
            # Increase workers if possible
            current_workers = self.high_throughput_processor.config.max_workers
            max_workers = mp.cpu_count() * 2
            new_workers = min(current_workers + 1, max_workers)
            self.high_throughput_processor.config.max_workers = new_workers
            
            self.logger.info(f"Scaled up: batch_size={new_batch_size}, workers={new_workers}")
    
    async def _update_processor_configs(self):
        """Update processor configurations based on optimization."""
        # This would update configurations dynamically
        # Implementation depends on specific processor architecture
        pass
    
    async def _fallback_processing(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback processing when optimization fails."""
        self.logger.warning("Using fallback processing")
        
        if self.high_throughput_processor:
            return await self.high_throughput_processor.process_tweets_async(tweets)
        
        # Simple fallback
        return [
            {
                'tweet_id': tweet.get('id'),
                'original_text': tweet.get('text', ''),
                'is_ai_generated': False,
                'confidence_score': 0.5,
                'processing_method': 'fallback',
                'error': True
            }
            for tweet in tweets
        ]
    
    async def benchmark_all_methods(self, sample_tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark all available processing methods."""
        self.logger.info("Starting comprehensive benchmark")
        
        test_size = min(1000, len(sample_tweets))
        test_tweets = sample_tweets[:test_size]
        
        benchmark_results = {}
        
        # Benchmark each method
        methods_to_test = [
            ('high_throughput', self.high_throughput_processor),
            ('gpu', self.gpu_processor),
            ('hybrid', self.hybrid_processor)
        ]
        
        for method_name, processor in methods_to_test:
            if processor is None:
                continue
                
            try:
                # Run benchmark
                start_time = time.time()
                
                if method_name == 'high_throughput':
                    results = await processor.process_tweets_async(test_tweets)
                elif method_name == 'gpu':
                    results = processor.process_tweets_gpu(test_tweets)
                elif method_name == 'hybrid':
                    results = processor.process_tweets_optimal(test_tweets)
                
                duration = time.time() - start_time
                throughput = (len(test_tweets) / duration) * 60
                
                benchmark_results[method_name] = {
                    'throughput_tpm': throughput,
                    'duration_seconds': duration,
                    'results_count': len(results),
                    'target_achieved': throughput >= self.target.tweets_per_minute,
                    'efficiency_ratio': throughput / self.target.tweets_per_minute
                }
                
                self.logger.info(f"{method_name}: {throughput:.0f} tpm ({duration:.3f}s)")
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {method_name}: {e}")
                benchmark_results[method_name] = {
                    'error': str(e),
                    'throughput_tpm': 0,
                    'target_achieved': False
                }
        
        # Find best method
        best_method = max(
            [k for k in benchmark_results.keys() if 'error' not in benchmark_results[k]],
            key=lambda k: benchmark_results[k]['throughput_tpm'],
            default=None
        )
        
        benchmark_results['best_method'] = best_method
        benchmark_results['benchmark_timestamp'] = time.time()
        
        self.benchmark_results = benchmark_results
        return benchmark_results
    
    async def continuous_optimization(self, tweet_stream, output_callback: Callable = None):
        """Continuously optimize throughput for streaming data."""
        self.logger.info("Starting continuous throughput optimization")
        
        batch_queue = queue.Queue(maxsize=100)
        results_queue = queue.Queue()
        
        # Start processing worker
        processing_task = asyncio.create_task(
            self._continuous_processing_worker(batch_queue, results_queue)
        )
        
        # Start result handler
        if output_callback:
            result_task = asyncio.create_task(
                self._result_handler_worker(results_queue, output_callback)
            )
        
        # Process stream
        batch = []
        batch_size = 100
        
        try:
            for tweet in tweet_stream:
                batch.append(tweet)
                
                if len(batch) >= batch_size:
                    if not batch_queue.full():
                        batch_queue.put(batch)
                        batch = []
                    else:
                        # Queue full, increase batch size temporarily
                        batch_size = min(batch_size * 1.2, 200)
            
            # Process remaining tweets
            if batch:
                batch_queue.put(batch)
            
        except Exception as e:
            self.logger.error(f"Continuous optimization error: {e}")
        finally:
            # Signal completion
            batch_queue.put(None)
            await processing_task
            
            if output_callback:
                results_queue.put(None)
                await result_task
    
    async def _continuous_processing_worker(self, batch_queue: queue.Queue, results_queue: queue.Queue):
        """Worker for continuous batch processing."""
        while True:
            try:
                batch = batch_queue.get(timeout=1.0)
                if batch is None:  # Termination signal
                    break
                
                # Process batch with optimization
                results = await self.optimize_throughput(batch)
                results_queue.put(results)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Processing worker error: {e}")
    
    async def _result_handler_worker(self, results_queue: queue.Queue, callback: Callable):
        """Worker for handling processed results."""
        while True:
            try:
                results = results_queue.get(timeout=1.0)
                if results is None:  # Termination signal
                    break
                
                # Call output callback
                await callback(results)
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Result handler error: {e}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        if not self.performance_history:
            return {'error': 'No performance data available'}
        
        recent_records = self.performance_history[-20:]
        
        # Calculate statistics
        throughputs = [r['throughput'] for r in recent_records]
        avg_throughput = statistics.mean(throughputs)
        max_throughput = max(throughputs)
        min_throughput = min(throughputs)
        
        # Target achievement rate
        achievements = [r['target_achieved'] for r in recent_records]
        achievement_rate = sum(achievements) / len(achievements)
        
        # Method performance
        method_stats = {}
        for record in recent_records:
            method = record['method']
            if method not in method_stats:
                method_stats[method] = []
            method_stats[method].append(record['throughput'])
        
        for method, throughputs in method_stats.items():
            method_stats[method] = {
                'avg_throughput': statistics.mean(throughputs),
                'max_throughput': max(throughputs),
                'samples': len(throughputs)
            }
        
        return {
            'target_tweets_per_minute': self.target.tweets_per_minute,
            'current_performance': {
                'avg_throughput': avg_throughput,
                'max_throughput': max_throughput,
                'min_throughput': min_throughput,
                'target_achievement_rate': achievement_rate
            },
            'method_performance': method_stats,
            'optimization_strategy': {
                'use_gpu': self.strategy.use_gpu,
                'use_multiprocessing': self.strategy.use_multiprocessing,
                'use_vectorization': self.strategy.use_vectorization,
                'dynamic_scaling': self.strategy.dynamic_scaling
            },
            'benchmark_results': self.benchmark_results,
            'total_samples': len(recent_records),
            'report_timestamp': time.time()
        }
    
    def cleanup(self):
        """Clean up all optimization resources."""
        try:
            if self.high_throughput_processor:
                self.high_throughput_processor.cleanup()
            
            if self.gpu_processor:
                self.gpu_processor.cleanup()
            
            if self.hybrid_processor:
                self.hybrid_processor.cleanup()
                
            self.logger.info("Throughput optimizer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


# Factory function for easy initialization
def create_throughput_optimizer(target_tpm: int = 1000, enable_gpu: bool = True) -> ThroughputOptimizer:
    """Create optimized throughput processor with specified target."""
    target = ThroughputTarget(tweets_per_minute=target_tpm)
    strategy = OptimizationStrategy(use_gpu=enable_gpu)
    
    return ThroughputOptimizer(target, strategy)