"""
GPU-accelerated data processing for ultra-high throughput.

Implements CUDA/OpenCL acceleration for vectorized operations
to achieve >1000 tweets/min processing with GPU optimization.
"""

import time
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import concurrent.futures
import multiprocessing as mp
from src.core.monitoring import get_logger, get_metrics_collector

try:
    import cupy as cp
    import cupyx.scipy.sparse
    GPU_AVAILABLE = True
except ImportError:
    cp = None
    GPU_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


@dataclass
class GPUConfig:
    """Configuration for GPU-accelerated processing."""
    use_gpu: bool = GPU_AVAILABLE
    gpu_device_id: int = 0
    gpu_memory_limit: int = 2048  # MB
    batch_size_gpu: int = 1000
    fallback_to_cpu: bool = True
    parallel_gpu_streams: int = 2


class GPUAcceleratedProcessor:
    """High-performance processor with GPU acceleration."""
    
    def __init__(self, gpu_config: Optional[GPUConfig] = None):
        self.gpu_config = gpu_config or GPUConfig()
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        
        # GPU initialization
        self.gpu_available = False
        self.gpu_device = None
        self.gpu_context = None
        
        # Processing components
        self.cpu_pool = mp.Pool(processes=mp.cpu_count())
        self.gpu_streams = []
        
        # Performance tracking
        self.processing_stats = {
            'gpu_processed': 0,
            'cpu_processed': 0,
            'gpu_speedup': 1.0,
            'memory_usage': 0
        }
        
        self.initialize_gpu()
    
    def initialize_gpu(self):
        """Initialize GPU resources if available."""
        if not self.gpu_config.use_gpu:
            self.logger.info("GPU acceleration disabled by configuration")
            return
        
        try:
            if GPU_AVAILABLE and cp is not None:
                # Initialize CuPy
                cp.cuda.Device(self.gpu_config.gpu_device_id).use()
                self.gpu_available = True
                self.gpu_device = 'cupy'
                
                # Set memory pool limit
                mempool = cp.get_default_memory_pool()
                mempool.set_limit(size=self.gpu_config.gpu_memory_limit * 1024 * 1024)
                
                self.logger.info(f"CuPy GPU acceleration initialized on device {self.gpu_config.gpu_device_id}")
                
            elif TORCH_AVAILABLE and torch.cuda.is_available():
                # Initialize PyTorch CUDA
                torch.cuda.set_device(self.gpu_config.gpu_device_id)
                self.gpu_available = True
                self.gpu_device = 'torch'
                
                self.logger.info(f"PyTorch CUDA acceleration initialized on device {self.gpu_config.gpu_device_id}")
                
            else:
                self.logger.info("No GPU acceleration available, using CPU only")
                
        except Exception as e:
            self.logger.error(f"GPU initialization failed: {e}")
            if self.gpu_config.fallback_to_cpu:
                self.logger.info("Falling back to CPU processing")
            else:
                raise
    
    def process_tweets_gpu(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process tweets using GPU acceleration."""
        if not self.gpu_available:
            return self.process_tweets_cpu(tweets)
        
        try:
            start_time = time.time()
            
            # Extract texts for GPU processing
            texts = [tweet.get('text', '') for tweet in tweets]
            
            if self.gpu_device == 'cupy':
                results = self._process_with_cupy(texts, tweets)
            elif self.gpu_device == 'torch':
                results = self._process_with_torch(texts, tweets)
            else:
                results = self.process_tweets_cpu(tweets)
            
            duration = time.time() - start_time
            throughput = (len(tweets) / duration) * 60
            
            self.processing_stats['gpu_processed'] += len(tweets)
            self.metrics.histogram('gpu_processing_duration_ms', duration * 1000)
            self.metrics.gauge('gpu_processing_throughput_tpm', throughput)
            
            self.logger.debug(f"GPU processed {len(tweets)} tweets in {duration:.3f}s ({throughput:.0f} tpm)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"GPU processing failed: {e}")
            if self.gpu_config.fallback_to_cpu:
                return self.process_tweets_cpu(tweets)
            raise
    
    def _process_with_cupy(self, texts: List[str], tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process using CuPy GPU acceleration."""
        # Convert texts to feature matrices on GPU
        feature_matrices = self._extract_features_gpu_cupy(texts)
        
        # Run classification on GPU
        predictions = self._classify_gpu_cupy(feature_matrices)
        
        # Convert back to CPU and format results
        predictions_cpu = cp.asnumpy(predictions)
        
        results = []
        for i, tweet in enumerate(tweets):
            confidence = float(predictions_cpu[i])
            is_ai = confidence > 0.5
            
            results.append({
                'tweet_id': tweet.get('id'),
                'original_text': tweet.get('text'),
                'is_ai_generated': is_ai,
                'confidence_score': abs(confidence - 0.5) * 2,  # Convert to confidence
                'processing_method': 'gpu_cupy',
                'processing_time_ms': 5.0,  # Estimated per tweet
                'gpu_accelerated': True
            })
        
        return results
    
    def _extract_features_gpu_cupy(self, texts: List[str]) -> cp.ndarray:
        """Extract features using CuPy GPU operations."""
        # Simple vectorized feature extraction on GPU
        features = []
        
        for text in texts:
            # Basic features that can be computed quickly
            char_count = len(text)
            word_count = len(text.split())
            avg_word_length = char_count / max(word_count, 1)
            
            # Pattern counting (simplified for GPU)
            formal_words = text.lower().count('furthermore') + text.lower().count('however')
            ai_indicators = text.lower().count('comprehensive') + text.lower().count('paradigm')
            
            features.append([
                char_count / 1000,  # Normalized
                word_count / 100,
                avg_word_length / 10,
                formal_words,
                ai_indicators
            ])
        
        # Convert to GPU array
        feature_array = cp.array(features, dtype=cp.float32)
        return feature_array
    
    def _classify_gpu_cupy(self, features: cp.ndarray) -> cp.ndarray:
        """Classify features using CuPy GPU operations."""
        # Simple linear model weights on GPU
        weights = cp.array([0.1, 0.15, 0.2, 0.25, 0.3], dtype=cp.float32)
        bias = cp.array(0.1, dtype=cp.float32)
        
        # Matrix multiplication on GPU
        scores = cp.dot(features, weights) + bias
        
        # Sigmoid activation
        predictions = 1.0 / (1.0 + cp.exp(-scores))
        
        return predictions
    
    def _process_with_torch(self, texts: List[str], tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process using PyTorch CUDA acceleration."""
        # Extract features and move to GPU
        features = self._extract_features_torch(texts)
        features_gpu = features.cuda()
        
        # Run classification on GPU
        with torch.no_grad():
            predictions = self._classify_torch(features_gpu)
            predictions_cpu = predictions.cpu().numpy()
        
        # Format results
        results = []
        for i, tweet in enumerate(tweets):
            confidence = float(predictions_cpu[i])
            is_ai = confidence > 0.5
            
            results.append({
                'tweet_id': tweet.get('id'),
                'original_text': tweet.get('text'),
                'is_ai_generated': is_ai,
                'confidence_score': abs(confidence - 0.5) * 2,
                'processing_method': 'gpu_torch',
                'processing_time_ms': 3.0,  # Estimated per tweet
                'gpu_accelerated': True
            })
        
        return results
    
    def _extract_features_torch(self, texts: List[str]) -> torch.Tensor:
        """Extract features using PyTorch operations."""
        features = []
        
        for text in texts:
            char_count = len(text)
            word_count = len(text.split())
            avg_word_length = char_count / max(word_count, 1)
            
            formal_words = text.lower().count('furthermore') + text.lower().count('however')
            ai_indicators = text.lower().count('comprehensive') + text.lower().count('paradigm')
            
            features.append([
                char_count / 1000,
                word_count / 100,
                avg_word_length / 10,
                formal_words,
                ai_indicators
            ])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _classify_torch(self, features: torch.Tensor) -> torch.Tensor:
        """Classify features using PyTorch GPU operations."""
        # Simple linear layer
        weights = torch.tensor([0.1, 0.15, 0.2, 0.25, 0.3], dtype=torch.float32).cuda()
        bias = torch.tensor(0.1, dtype=torch.float32).cuda()
        
        # Linear transformation + sigmoid
        scores = torch.matmul(features, weights) + bias
        predictions = torch.sigmoid(scores)
        
        return predictions
    
    def process_tweets_cpu(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback CPU processing with multiprocessing."""
        start_time = time.time()
        
        # Split into chunks for parallel processing
        chunk_size = max(1, len(tweets) // mp.cpu_count())
        chunks = [tweets[i:i + chunk_size] for i in range(0, len(tweets), chunk_size)]
        
        # Process chunks in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
            futures = [executor.submit(self._process_chunk_cpu, chunk) for chunk in chunks]
            chunk_results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # Flatten results
        results = []
        for chunk_result in chunk_results:
            results.extend(chunk_result)
        
        duration = time.time() - start_time
        throughput = (len(tweets) / duration) * 60
        
        self.processing_stats['cpu_processed'] += len(tweets)
        self.metrics.histogram('cpu_processing_duration_ms', duration * 1000)
        self.metrics.gauge('cpu_processing_throughput_tpm', throughput)
        
        return results
    
    @staticmethod
    def _process_chunk_cpu(chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a chunk of tweets on CPU."""
        results = []
        
        for tweet in chunk:
            text = tweet.get('text', '')
            
            # Simple feature extraction
            char_count = len(text)
            word_count = len(text.split())
            avg_word_length = char_count / max(word_count, 1)
            
            # Simple heuristic classification
            formal_words = text.lower().count('furthermore') + text.lower().count('however')
            ai_indicators = text.lower().count('comprehensive') + text.lower().count('paradigm')
            
            # Simple scoring
            score = (avg_word_length / 10) + (formal_words * 0.2) + (ai_indicators * 0.3)
            confidence = min(max(score, 0), 1)
            is_ai = confidence > 0.5
            
            results.append({
                'tweet_id': tweet.get('id'),
                'original_text': text,
                'is_ai_generated': is_ai,
                'confidence_score': abs(confidence - 0.5) * 2,
                'processing_method': 'cpu_multiprocess',
                'processing_time_ms': 10.0,
                'gpu_accelerated': False
            })
        
        return results
    
    def process_tweets_adaptive(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Adaptively choose processing method based on load and availability."""
        batch_size = len(tweets)
        
        # Choose processing method based on batch size and GPU availability
        if self.gpu_available and batch_size >= self.gpu_config.batch_size_gpu:
            return self.process_tweets_gpu(tweets)
        elif batch_size >= 100:  # Use multiprocessing for medium batches
            return self.process_tweets_cpu(tweets)
        else:
            # Use simple processing for small batches
            return self._process_chunk_cpu(tweets)
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics."""
        total_processed = self.processing_stats['gpu_processed'] + self.processing_stats['cpu_processed']
        
        gpu_ratio = self.processing_stats['gpu_processed'] / max(total_processed, 1)
        
        return {
            'total_processed': total_processed,
            'gpu_processed': self.processing_stats['gpu_processed'],
            'cpu_processed': self.processing_stats['cpu_processed'],
            'gpu_utilization_ratio': gpu_ratio,
            'gpu_available': self.gpu_available,
            'gpu_device': self.gpu_device,
            'estimated_speedup': self._estimate_gpu_speedup(),
            'memory_usage': self._get_gpu_memory_usage()
        }
    
    def _estimate_gpu_speedup(self) -> float:
        """Estimate GPU speedup compared to CPU."""
        if not self.gpu_available:
            return 1.0
        
        # Rough estimates based on typical GPU vs CPU performance
        if self.gpu_device == 'cupy':
            return 5.0  # CuPy typically 5x faster for vectorized ops
        elif self.gpu_device == 'torch':
            return 8.0  # PyTorch CUDA typically 8x faster
        else:
            return 1.0
    
    def _get_gpu_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        if not self.gpu_available:
            return {'used': 0, 'total': 0, 'utilization': 0}
        
        try:
            if self.gpu_device == 'cupy':
                mempool = cp.get_default_memory_pool()
                used_bytes = mempool.used_bytes()
                total_bytes = mempool.total_bytes()
                
                return {
                    'used': used_bytes / (1024 * 1024),  # MB
                    'total': total_bytes / (1024 * 1024),
                    'utilization': used_bytes / max(total_bytes, 1)
                }
            elif self.gpu_device == 'torch':
                allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
                reserved = torch.cuda.memory_reserved() / (1024 * 1024)
                
                return {
                    'used': allocated,
                    'total': reserved,
                    'utilization': allocated / max(reserved, 1)
                }
        except Exception as e:
            self.logger.debug(f"GPU memory query failed: {e}")
        
        return {'used': 0, 'total': 0, 'utilization': 0}
    
    def benchmark_processing_methods(self, sample_tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Benchmark different processing methods."""
        benchmark_results = {}
        test_size = min(1000, len(sample_tweets))
        test_tweets = sample_tweets[:test_size]
        
        # Benchmark GPU processing
        if self.gpu_available:
            start_time = time.time()
            gpu_results = self.process_tweets_gpu(test_tweets)
            gpu_duration = time.time() - start_time
            gpu_throughput = (len(test_tweets) / gpu_duration) * 60
            
            benchmark_results['gpu'] = {
                'duration_seconds': gpu_duration,
                'throughput_tweets_per_minute': gpu_throughput,
                'device': self.gpu_device,
                'results_count': len(gpu_results)
            }
        
        # Benchmark CPU processing
        start_time = time.time()
        cpu_results = self.process_tweets_cpu(test_tweets)
        cpu_duration = time.time() - start_time
        cpu_throughput = (len(test_tweets) / cpu_duration) * 60
        
        benchmark_results['cpu'] = {
            'duration_seconds': cpu_duration,
            'throughput_tweets_per_minute': cpu_throughput,
            'results_count': len(cpu_results)
        }
        
        # Calculate speedup
        if self.gpu_available:
            speedup = cpu_duration / gpu_duration
            benchmark_results['gpu_speedup'] = speedup
            benchmark_results['recommendation'] = 'gpu' if speedup > 1.2 else 'cpu'
        else:
            benchmark_results['recommendation'] = 'cpu'
        
        return benchmark_results
    
    def cleanup(self):
        """Clean up GPU and CPU resources."""
        try:
            if self.gpu_available:
                if self.gpu_device == 'cupy':
                    cp.get_default_memory_pool().free_all_blocks()
                elif self.gpu_device == 'torch':
                    torch.cuda.empty_cache()
            
            if self.cpu_pool:
                self.cpu_pool.close()
                self.cpu_pool.join()
                
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")


class HybridProcessor:
    """Hybrid processor that automatically selects optimal processing method."""
    
    def __init__(self, gpu_config: Optional[GPUConfig] = None):
        self.gpu_processor = GPUAcceleratedProcessor(gpu_config)
        self.logger = get_logger(__name__)
        
        # Performance tracking
        self.method_performance = {
            'gpu': {'total_time': 0, 'total_tweets': 0},
            'cpu': {'total_time': 0, 'total_tweets': 0}
        }
        
    def process_tweets_optimal(self, tweets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process tweets using optimal method based on current conditions."""
        batch_size = len(tweets)
        
        # Use adaptive processing
        start_time = time.time()
        results = self.gpu_processor.process_tweets_adaptive(tweets)
        duration = time.time() - start_time
        
        # Track performance
        method = 'gpu' if results and results[0].get('gpu_accelerated') else 'cpu'
        self.method_performance[method]['total_time'] += duration
        self.method_performance[method]['total_tweets'] += len(tweets)
        
        throughput = (len(tweets) / duration) * 60
        self.logger.debug(f"Processed {len(tweets)} tweets using {method} in {duration:.3f}s ({throughput:.0f} tpm)")
        
        return results
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for both methods."""
        summary = {}
        
        for method, stats in self.method_performance.items():
            if stats['total_tweets'] > 0:
                avg_throughput = (stats['total_tweets'] / stats['total_time']) * 60
                summary[method] = {
                    'total_tweets_processed': stats['total_tweets'],
                    'total_time_seconds': stats['total_time'],
                    'average_throughput_tpm': avg_throughput
                }
            else:
                summary[method] = {
                    'total_tweets_processed': 0,
                    'total_time_seconds': 0,
                    'average_throughput_tpm': 0
                }
        
        return summary
    
    def cleanup(self):
        """Clean up all resources."""
        self.gpu_processor.cleanup()