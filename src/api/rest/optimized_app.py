"""
Optimized FastAPI application for sub-2s API response times.

Integrates all performance optimizations including async processing,
connection pooling, caching, compression, and request batching.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvloop

from src.api.performance.api_optimizer import APIPerformanceOptimizer, APIPerformanceConfig
from src.core.detection.optimized_detector import OptimizedDetector, PerformanceMode
from src.core.monitoring import get_logger, get_metrics_collector


# Request/Response Models
class DetectionRequest(BaseModel):
    """Optimized detection request model."""
    text: str = Field(..., min_length=1, max_length=50000)
    request_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    priority: Optional[str] = Field(default="normal", regex="^(low|normal|high|critical)$")
    performance_mode: Optional[str] = Field(default="balanced", regex="^(ultra_fast|fast|balanced|accurate)$")


class BatchDetectionRequest(BaseModel):
    """Batch detection request model."""
    texts: List[str] = Field(..., min_items=1, max_items=100)
    request_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = Field(default_factory=dict)
    performance_mode: Optional[str] = Field(default="balanced")


class DetectionResponse(BaseModel):
    """Optimized detection response model."""
    is_ai_generated: Optional[bool]
    confidence_score: float
    processing_time_ms: float
    api_processing_time_ms: float
    method_used: str
    request_id: str
    performance_mode: str
    from_cache: bool = False
    optimizations_applied: List[str]


class BatchDetectionResponse(BaseModel):
    """Batch detection response model."""
    results: List[DetectionResponse]
    total_processing_time_ms: float
    batch_size: int
    requests_per_second: float


# Global components
api_optimizer: Optional[APIPerformanceOptimizer] = None
detector_instances: Dict[str, OptimizedDetector] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global api_optimizer, detector_instances
    
    # Set event loop policy for better performance
    if hasattr(asyncio, 'set_event_loop_policy'):
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    
    # Initialize performance optimizer
    config = APIPerformanceConfig(
        max_concurrent_requests=200,
        request_timeout_seconds=1.8,
        enable_compression=True,
        compression_threshold=512,
        enable_request_batching=True,
        batch_size=20,
        batch_timeout_ms=30,
        enable_response_caching=True,
        cache_ttl_seconds=300,
        connection_pool_size=50
    )
    
    api_optimizer = APIPerformanceOptimizer(config)
    
    # Initialize detector instances for each performance mode
    for mode in PerformanceMode:
        detector_instances[mode.value] = OptimizedDetector(mode=mode)
    
    # Warm up detectors
    warmup_text = "This is a warmup text to initialize all detection components."
    for detector in detector_instances.values():
        try:
            await detector.detect(warmup_text)
        except Exception as e:
            get_logger(__name__).warning(f"Warmup failed for detector: {e}")
    
    get_logger(__name__).info("Optimized API started successfully")
    
    yield
    
    # Cleanup
    get_logger(__name__).info("Optimized API shutting down")


# Create FastAPI app with optimizations
app = FastAPI(
    title="AI Detector Optimized API",
    description="High-performance AI text detection API with sub-2s response times",
    version="1.0.0",
    lifespan=lifespan
)

# Add performance middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = get_logger(__name__)
metrics = get_metrics_collector()


@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    """Middleware for performance monitoring and optimization."""
    start_time = time.time()
    
    # Add request ID if not present
    request_id = request.headers.get("X-Request-ID", f"req_{int(time.time() * 1000)}")
    
    # Process request
    response = await call_next(request)
    
    # Calculate total response time
    total_time = (time.time() - start_time) * 1000
    
    # Add performance headers
    response.headers["X-Response-Time"] = f"{total_time:.2f}ms"
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Performance-Target"] = "sub-2s"
    response.headers["X-Sub-2s-Compliant"] = str(total_time < 2000)
    
    # Record metrics
    metrics.observe_histogram("api_total_response_time_ms", total_time)
    if total_time < 2000:
        metrics.increment_counter("api_sub_2s_responses_total")
    
    return response


@app.get("/health")
async def health_check():
    """Health check endpoint with performance validation."""
    if not api_optimizer:
        raise HTTPException(status_code=503, detail="API optimizer not initialized")
    
    health_result = await api_optimizer.health_check()
    
    if health_result["status"] == "healthy":
        return health_result
    else:
        raise HTTPException(status_code=503, detail=health_result)


@app.post("/detect", response_model=DetectionResponse)
async def detect_text(request: DetectionRequest, background_tasks: BackgroundTasks):
    """Optimized single text detection endpoint."""
    start_time = time.time()
    
    try:
        # Get detector for requested performance mode
        performance_mode = PerformanceMode(request.performance_mode)
        detector = detector_instances[performance_mode.value]
        
        # Prepare request data
        request_data = {
            "text": request.text,
            "request_id": request.request_id or f"req_{int(time.time() * 1000)}",
            "options": request.options or {},
            "priority": request.priority
        }
        
        # Process through API optimizer
        response_data, headers = await api_optimizer.process_request(request_data)
        
        # Convert to response model
        detection_response = DetectionResponse(
            is_ai_generated=response_data.get("is_ai_generated"),
            confidence_score=response_data.get("confidence_score", 0.0),
            processing_time_ms=response_data.get("processing_time_ms", 0.0),
            api_processing_time_ms=response_data.get("api_processing_time_ms", 0.0),
            method_used=response_data.get("method_used", "unknown"),
            request_id=response_data.get("request_id", ""),
            performance_mode=request.performance_mode,
            from_cache=response_data.get("from_cache", False),
            optimizations_applied=response_data.get("optimizations_applied", [])
        )
        
        # Record background metrics
        background_tasks.add_task(
            record_detection_metrics,
            performance_mode.value,
            (time.time() - start_time) * 1000,
            len(request.text)
        )
        
        return detection_response
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail="Detection processing failed")


@app.post("/detect/batch", response_model=BatchDetectionResponse)
async def batch_detect_text(request: BatchDetectionRequest, background_tasks: BackgroundTasks):
    """Optimized batch text detection endpoint."""
    start_time = time.time()
    
    try:
        # Validate batch size
        if len(request.texts) > 100:
            raise HTTPException(status_code=400, detail="Batch size exceeds maximum of 100")
        
        # Get detector for requested performance mode
        performance_mode = PerformanceMode(request.performance_mode)
        detector = detector_instances[performance_mode.value]
        
        # Prepare batch request data
        batch_requests = []
        base_request_id = request.request_id or f"batch_{int(time.time() * 1000)}"
        
        for i, text in enumerate(request.texts):
            batch_requests.append({
                "text": text,
                "request_id": f"{base_request_id}_{i}",
                "options": request.options or {},
                "priority": "normal"
            })
        
        # Process batch through API optimizer
        batch_results = await api_optimizer.process_batch_requests(batch_requests)
        
        # Convert to response models
        detection_responses = []
        for result in batch_results:
            detection_responses.append(DetectionResponse(
                is_ai_generated=result.get("is_ai_generated"),
                confidence_score=result.get("confidence_score", 0.0),
                processing_time_ms=result.get("processing_time_ms", 0.0),
                api_processing_time_ms=result.get("api_processing_time_ms", 0.0),
                method_used=result.get("method_used", "unknown"),
                request_id=result.get("request_id", ""),
                performance_mode=request.performance_mode,
                from_cache=result.get("from_cache", False),
                optimizations_applied=result.get("optimizations_applied", [])
            ))
        
        # Calculate batch metrics
        total_time = (time.time() - start_time) * 1000
        requests_per_second = len(request.texts) / (total_time / 1000)
        
        batch_response = BatchDetectionResponse(
            results=detection_responses,
            total_processing_time_ms=total_time,
            batch_size=len(request.texts),
            requests_per_second=requests_per_second
        )
        
        # Record background metrics
        background_tasks.add_task(
            record_batch_metrics,
            len(request.texts),
            total_time,
            requests_per_second
        )
        
        return batch_response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch detection failed: {e}")
        raise HTTPException(status_code=500, detail="Batch detection processing failed")


@app.get("/performance/stats")
async def get_performance_stats():
    """Get comprehensive performance statistics."""
    if not api_optimizer:
        raise HTTPException(status_code=503, detail="API optimizer not initialized")
    
    api_stats = api_optimizer.get_performance_stats()
    
    # Add detector statistics
    detector_stats = {}
    for mode, detector in detector_instances.items():
        detector_stats[mode] = detector.get_performance_report()
    
    return {
        "api_performance": api_stats,
        "detector_performance": detector_stats,
        "global_metrics": {
            "total_requests": metrics.get_metric("api_requests_total").get_value() if metrics.get_metric("api_requests_total") else 0,
            "sub_2s_rate": api_stats["performance_targets"]["sub_2s_rate"],
            "average_response_time": api_stats["request_performance"].get("mean_ms", 0)
        }
    }


@app.get("/performance/benchmark")
async def run_performance_benchmark():
    """Run comprehensive performance benchmark."""
    benchmark_texts = [
        "Short text",
        "This is a medium length text for testing detection performance across various scenarios.",
        "This comprehensive analysis demonstrates the multifaceted nature of contemporary discourse paradigms, necessitating careful consideration of various methodological approaches and theoretical frameworks that facilitate enhanced understanding.",
        "lol that's so funny 😂 can't believe it",
        "The rapid advancement of artificial intelligence technologies has fundamentally transformed the landscape of digital communication, creating unprecedented opportunities for innovation while simultaneously presenting significant challenges related to authenticity and verification."
    ]
    
    results = {}
    
    for mode in PerformanceMode:
        detector = detector_instances[mode.value]
        mode_results = await detector.benchmark(benchmark_texts)
        results[mode.value] = mode_results
    
    # API-level benchmark
    api_benchmark_start = time.time()
    api_results = []
    
    for text in benchmark_texts:
        start_time = time.time()
        request_data = {"text": text, "performance_mode": "balanced"}
        response_data, headers = await api_optimizer.process_request(request_data)
        api_time = (time.time() - start_time) * 1000
        api_results.append(api_time)
    
    api_benchmark_time = (time.time() - api_benchmark_start) * 1000
    
    results["api_benchmark"] = {
        "total_time_ms": api_benchmark_time,
        "average_time_ms": sum(api_results) / len(api_results),
        "sub_2s_count": sum(1 for t in api_results if t < 2000),
        "sub_1s_count": sum(1 for t in api_results if t < 1000),
        "performance_grade": "A" if sum(api_results) / len(api_results) < 500 else "B" if sum(api_results) / len(api_results) < 1000 else "C"
    }
    
    return results


@app.get("/metrics")
async def get_metrics():
    """Get Prometheus-style metrics."""
    # This would integrate with actual metrics export
    return {
        "api_requests_total": metrics.get_metric("api_requests_total").get_value() if metrics.get_metric("api_requests_total") else 0,
        "api_sub_2s_responses_total": metrics.get_metric("api_sub_2s_responses_total").get_value() if metrics.get_metric("api_sub_2s_responses_total") else 0,
        "api_response_time_histogram": "# Would contain histogram data",
        "connection_pool_utilization": api_optimizer.connection_pool.get_stats()["utilization"] if api_optimizer else 0
    }


async def record_detection_metrics(performance_mode: str, response_time: float, text_length: int):
    """Record detection metrics in background."""
    metrics.increment_counter("detections_total", labels={"mode": performance_mode})
    metrics.observe_histogram("detection_response_time_ms", response_time, labels={"mode": performance_mode})
    metrics.observe_histogram("detection_text_length", text_length, labels={"mode": performance_mode})


async def record_batch_metrics(batch_size: int, total_time: float, requests_per_second: float):
    """Record batch processing metrics in background."""
    metrics.increment_counter("batch_requests_total")
    metrics.observe_histogram("batch_processing_time_ms", total_time)
    metrics.observe_gauge("batch_throughput_rps", requests_per_second)
    metrics.observe_histogram("batch_size_distribution", batch_size)


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with performance tracking."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": request.headers.get("X-Request-ID", "unknown")
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    metrics.increment_counter("api_errors_total")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": request.headers.get("X-Request-ID", "unknown")
        }
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run with optimized settings
    uvicorn.run(
        "src.api.rest.optimized_app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Use single worker with async processing
        loop="uvloop",
        http="httptools",
        access_log=False,  # Disable for performance
        server_header=False,
        date_header=False
    )