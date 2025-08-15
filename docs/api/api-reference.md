# AI Detector API Reference

Complete technical reference for the AI Detector API endpoints, including detailed request/response schemas, authentication methods, and integration patterns.

## Base Information

- **Base URL**: `http://localhost:8000/api`
- **Protocol**: HTTP/HTTPS
- **Content-Type**: `application/json`
- **API Version**: `v1`

## Authentication

### API Key Authentication (Recommended)

```http
GET /api/detect HTTP/1.1
Host: localhost:8000
X-API-Key: your-api-key-here
Content-Type: application/json
```

### Bearer Token Authentication

```http
GET /api/detect HTTP/1.1
Host: localhost:8000
Authorization: Bearer your-jwt-token-here
Content-Type: application/json
```

### Request ID Tracking

All requests support optional request ID tracking:

```http
POST /api/detect HTTP/1.1
X-Request-ID: custom-request-id-123
```

## Core Detection API

### Single Text Detection

#### `POST /api/detect`

Analyze a single text for AI-generated content.

**Request Schema:**

```typescript
interface DetectionRequest {
  text: string;                    // 1-50,000 characters
  request_id?: string;             // Optional unique identifier
  options?: {
    detection_method?: 'pattern' | 'ml' | 'llm' | 'ensemble';
    confidence_threshold?: number; // 0.0 - 1.0
    return_features?: boolean;
    language?: string;             // ISO 639-1 code
    timeout?: number;              // Seconds (max 30)
  };
  metadata?: {
    source?: string;
    user_id?: string;
    timestamp?: string;            // ISO 8601
    custom_fields?: Record<string, any>;
  };
}
```

**Response Schema:**

```typescript
interface DetectionResponse {
  request_id: string;
  is_ai_generated: boolean;
  confidence_score: number;        // 0.0 - 1.0
  detection_method: string;
  processing_time_ms: number;
  metadata: {
    text_length: number;
    word_count: number;
    detected_patterns?: string[];
    language: string;
    complexity_score?: number;
  };
  features?: {
    pattern_features?: PatternFeatures;
    ml_features?: MLFeatures;
    llm_features?: LLMFeatures;
  };
  timestamp: string;               // ISO 8601
}

interface PatternFeatures {
  formal_word_count: number;
  transition_phrases: number;
  academic_indicators: number;
  repetitive_structures: number;
  avg_sentence_length: number;
}

interface MLFeatures {
  feature_vector: number[];
  model_version: string;
  prediction_confidence: number;
  feature_importance: Record<string, number>;
}

interface LLMFeatures {
  model_used: string;
  reasoning: string;
  alternative_confidence: number;
  context_analysis: string;
}
```

**Example Request:**

```json
{
  "text": "Furthermore, this comprehensive analysis demonstrates the multifaceted paradigm inherent in contemporary discourse.",
  "request_id": "analysis_001",
  "options": {
    "detection_method": "ensemble",
    "confidence_threshold": 0.7,
    "return_features": true,
    "language": "en"
  },
  "metadata": {
    "source": "academic_paper",
    "user_id": "researcher_123"
  }
}
```

**Example Response:**

```json
{
  "request_id": "analysis_001",
  "is_ai_generated": true,
  "confidence_score": 0.87,
  "detection_method": "ensemble",
  "processing_time_ms": 234.5,
  "metadata": {
    "text_length": 142,
    "word_count": 15,
    "detected_patterns": ["furthermore", "comprehensive", "multifaceted", "paradigm"],
    "language": "en",
    "complexity_score": 0.73
  },
  "features": {
    "pattern_features": {
      "formal_word_count": 4,
      "transition_phrases": 1,
      "academic_indicators": 3,
      "repetitive_structures": 0,
      "avg_sentence_length": 15.0
    },
    "ml_features": {
      "feature_vector": [0.23, 0.67, 0.89, 0.45, 0.12],
      "model_version": "v2.1.0",
      "prediction_confidence": 0.82,
      "feature_importance": {
        "vocabulary_sophistication": 0.34,
        "sentence_structure": 0.28,
        "coherence_score": 0.23,
        "stylistic_patterns": 0.15
      }
    },
    "llm_features": {
      "model_used": "gpt-4-turbo",
      "reasoning": "High presence of academic jargon and complex sentence structures typical of AI generation",
      "alternative_confidence": 0.91,
      "context_analysis": "Academic writing style with elevated vocabulary"
    }
  },
  "timestamp": "2025-01-15T14:30:45.123Z"
}
```

### Batch Detection

#### `POST /api/detect/batch`

Analyze multiple texts simultaneously with optimized processing.

**Request Schema:**

```typescript
interface BatchDetectionRequest {
  texts: TextInput[];
  options?: {
    detection_method?: 'pattern' | 'ml' | 'llm' | 'ensemble';
    confidence_threshold?: number;
    parallel_processing?: boolean;
    max_concurrency?: number;       // Default: 10
    return_features?: boolean;
    timeout?: number;               // Seconds per text
  };
  metadata?: {
    batch_source?: string;
    user_id?: string;
    priority?: 'low' | 'normal' | 'high';
  };
}

interface TextInput {
  id: string;                      // Unique within batch
  text: string;
  metadata?: Record<string, any>;
}
```

**Response Schema:**

```typescript
interface BatchDetectionResponse {
  batch_id: string;
  total_texts: number;
  successful_detections: number;
  failed_detections: number;
  processing_time_ms: number;
  results: BatchResult[];
  statistics: {
    ai_detected: number;
    human_detected: number;
    average_confidence: number;
    confidence_distribution: {
      high: number;                // > 0.8
      medium: number;              // 0.5 - 0.8
      low: number;                 // < 0.5
    };
  };
  performance: {
    texts_per_second: number;
    cache_hit_rate: number;
    average_text_processing_ms: number;
  };
  timestamp: string;
}

interface BatchResult {
  id: string;
  is_ai_generated: boolean;
  confidence_score: number;
  detection_method: string;
  processing_time_ms: number;
  error?: {
    code: string;
    message: string;
  };
}
```

### Stream Detection

#### `POST /api/detect/stream`

Real-time detection for streaming text data.

**Request:** Server-Sent Events (SSE) compatible

```typescript
interface StreamDetectionRequest {
  stream_id: string;
  options?: {
    detection_method?: string;
    confidence_threshold?: number;
    buffer_size?: number;          // Characters to buffer
    flush_interval_ms?: number;    // Auto-flush interval
  };
}
```

**Response:** Event stream

```
event: detection_result
data: {"stream_id":"stream_123","is_ai_generated":true,"confidence_score":0.85}

event: batch_complete
data: {"stream_id":"stream_123","total_processed":50,"ai_detected":23}

event: error
data: {"stream_id":"stream_123","error":"Processing timeout"}
```

## Detection History API

### Get Detection Result

#### `GET /api/detect/{request_id}`

Retrieve cached results for a previous detection.

**Response:** Same as [DetectionResponse](#single-text-detection)

**HTTP Status Codes:**
- `200`: Result found
- `404`: Request ID not found
- `410`: Result expired from cache

### List Detection History

#### `GET /api/detect/history`

Get paginated list of recent detections.

**Query Parameters:**
- `page`: Page number (default: 1)
- `limit`: Results per page (max: 100, default: 20)
- `user_id`: Filter by user ID
- `start_date`: Start date (ISO 8601)
- `end_date`: End date (ISO 8601)
- `ai_only`: Boolean, only AI detections
- `min_confidence`: Minimum confidence score

**Response:**

```typescript
interface HistoryResponse {
  page: number;
  limit: number;
  total_results: number;
  total_pages: number;
  results: HistoryItem[];
}

interface HistoryItem {
  request_id: string;
  is_ai_generated: boolean;
  confidence_score: number;
  text_length: number;
  detection_method: string;
  processing_time_ms: number;
  timestamp: string;
  metadata?: Record<string, any>;
}
```

## System Status API

### Health Check

#### `GET /api/health`

Check overall system health and availability.

**Response:**

```typescript
interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  uptime_seconds: number;
  checks: {
    database: 'healthy' | 'unhealthy';
    llm_service: 'healthy' | 'degraded' | 'unhealthy';
    ml_models: 'healthy' | 'degraded' | 'unhealthy';
    memory_usage: 'normal' | 'high' | 'critical';
    disk_usage: 'normal' | 'high' | 'critical';
  };
  performance: {
    avg_response_time_ms: number;
    p95_response_time_ms: number;
    requests_per_minute: number;
    cache_hit_rate: number;
    error_rate: number;
  };
  dependencies: {
    openrouter_api: 'available' | 'degraded' | 'unavailable';
    gemini_api: 'available' | 'degraded' | 'unavailable';
  };
}
```

### Detailed Status

#### `GET /api/status`

Get comprehensive system status and metrics.

**Response:**

```typescript
interface StatusResponse {
  system: {
    version: string;
    environment: 'development' | 'staging' | 'production';
    uptime_seconds: number;
    memory_usage_mb: number;
    cpu_usage_percent: number;
  };
  api: {
    total_requests: number;
    requests_today: number;
    requests_this_hour: number;
    average_response_time_ms: number;
    error_rate_percent: number;
  };
  detection: {
    models_loaded: string[];
    cache_size: number;
    queue_size: number;
    active_connections: number;
  };
  performance: {
    throughput_tpm: number;         // Texts per minute
    concurrent_requests: number;
    cache_hit_rate: number;
    avg_processing_time_ms: number;
  };
}
```

### Metrics Endpoint

#### `GET /api/metrics`

Prometheus-compatible metrics endpoint.

**Response Format:** Prometheus text format

```
# HELP api_requests_total Total number of API requests
# TYPE api_requests_total counter
api_requests_total{method="POST",endpoint="/detect"} 10543

# HELP api_request_duration_seconds Request duration
# TYPE api_request_duration_seconds histogram
api_request_duration_seconds_bucket{le="0.1"} 8234
api_request_duration_seconds_bucket{le="0.5"} 9876
api_request_duration_seconds_bucket{le="1.0"} 10234

# HELP detection_confidence_score Distribution of confidence scores
# TYPE detection_confidence_score histogram
detection_confidence_score_bucket{le="0.5"} 2341
detection_confidence_score_bucket{le="0.8"} 7823
detection_confidence_score_bucket{le="1.0"} 10543
```

## Configuration API

### Get Configuration

#### `GET /api/config`

Retrieve current API configuration.

**Response:**

```typescript
interface ConfigResponse {
  detection_methods: {
    pattern: {
      enabled: boolean;
      confidence_weight: number;
      patterns_count: number;
    };
    ml: {
      enabled: boolean;
      model_version: string;
      confidence_weight: number;
      features_count: number;
    };
    llm: {
      enabled: boolean;
      provider: string;
      model: string;
      confidence_weight: number;
      max_tokens: number;
    };
    ensemble: {
      enabled: boolean;
      voting_strategy: 'weighted' | 'majority' | 'consensus';
      weights: Record<string, number>;
    };
  };
  performance: {
    max_text_length: number;
    max_batch_size: number;
    timeout_seconds: number;
    cache_ttl_seconds: number;
    rate_limit_per_minute: number;
  };
  features: {
    caching_enabled: boolean;
    monitoring_enabled: boolean;
    rate_limiting_enabled: boolean;
    batch_processing_enabled: boolean;
    stream_processing_enabled: boolean;
  };
}
```

### Update Configuration

#### `PUT /api/config`

Update API configuration (admin authentication required).

**Request:**

```typescript
interface ConfigUpdateRequest {
  detection_methods?: {
    pattern?: {
      enabled?: boolean;
      confidence_weight?: number;
    };
    ml?: {
      enabled?: boolean;
      confidence_weight?: number;
    };
    llm?: {
      enabled?: boolean;
      provider?: string;
      model?: string;
      confidence_weight?: number;
    };
  };
  performance?: {
    max_text_length?: number;
    max_batch_size?: number;
    timeout_seconds?: number;
    cache_ttl_seconds?: number;
  };
  features?: {
    caching_enabled?: boolean;
    rate_limiting_enabled?: boolean;
  };
}
```

## Statistics API

### Usage Statistics

#### `GET /api/stats`

Get comprehensive usage statistics.

**Query Parameters:**
- `period`: Time period ('hour', 'day', 'week', 'month')
- `start_date`: Start date (ISO 8601)
- `end_date`: End date (ISO 8601)

**Response:**

```typescript
interface StatsResponse {
  period: {
    start: string;
    end: string;
    duration_hours: number;
  };
  requests: {
    total: number;
    successful: number;
    failed: number;
    rate_limited: number;
  };
  detection: {
    ai_detected: number;
    human_detected: number;
    ai_detection_rate: number;
    confidence_distribution: {
      very_high: number;    // > 0.9
      high: number;         // 0.8 - 0.9
      medium: number;       // 0.6 - 0.8
      low: number;          // 0.4 - 0.6
      very_low: number;     // < 0.4
    };
  };
  performance: {
    avg_response_time_ms: number;
    p50_response_time_ms: number;
    p95_response_time_ms: number;
    p99_response_time_ms: number;
    throughput_per_minute: number;
    cache_hit_rate: number;
  };
  methods: {
    pattern: {
      usage_count: number;
      avg_confidence: number;
      avg_processing_time_ms: number;
    };
    ml: {
      usage_count: number;
      avg_confidence: number;
      avg_processing_time_ms: number;
    };
    llm: {
      usage_count: number;
      avg_confidence: number;
      avg_processing_time_ms: number;
    };
    ensemble: {
      usage_count: number;
      avg_confidence: number;
      avg_processing_time_ms: number;
    };
  };
}
```

### Real-time Statistics

#### `GET /api/stats/realtime`

Get real-time system statistics via Server-Sent Events.

**Response:** Event stream

```
event: stats_update
data: {"timestamp":"2025-01-15T14:30:45Z","active_requests":23,"queue_size":5,"cache_hit_rate":0.73}

event: performance_update  
data: {"avg_response_time_ms":45.2,"throughput_tpm":1250,"error_rate":0.02}
```

## Error Handling

### Error Response Format

All API errors follow a consistent format:

```typescript
interface ErrorResponse {
  error: {
    code: string;
    message: string;
    details?: any;
    request_id?: string;
    timestamp: string;
  };
  debug_info?: {
    trace_id: string;
    span_id: string;
    user_agent: string;
    ip_address: string;
  };
}
```

### Error Codes Reference

| Code | HTTP Status | Description | Retry Strategy |
|------|-------------|-------------|----------------|
| `VALIDATION_ERROR` | 400 | Request validation failed | Fix request, don't retry |
| `TEXT_TOO_LONG` | 400 | Text exceeds maximum length | Reduce text size |
| `TEXT_TOO_SHORT` | 400 | Text below minimum length | Increase text size |
| `INVALID_METHOD` | 400 | Unsupported detection method | Use valid method |
| `INVALID_LANGUAGE` | 400 | Unsupported language code | Use supported language |
| `UNAUTHORIZED` | 401 | Authentication required | Provide valid credentials |
| `FORBIDDEN` | 403 | Insufficient permissions | Check authorization |
| `NOT_FOUND` | 404 | Resource not found | Verify resource exists |
| `METHOD_NOT_ALLOWED` | 405 | HTTP method not allowed | Use correct HTTP method |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests | Implement backoff |
| `PROCESSING_TIMEOUT` | 408 | Request processing timeout | Retry with shorter text |
| `MODEL_UNAVAILABLE` | 503 | Detection model unavailable | Try different method |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily down | Retry with exponential backoff |
| `INTERNAL_ERROR` | 500 | Internal server error | Retry with exponential backoff |
| `QUEUE_FULL` | 503 | Processing queue full | Retry later |
| `INSUFFICIENT_RESOURCES` | 503 | Server overloaded | Reduce request rate |

### Rate Limiting

#### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248645
X-RateLimit-Window: 60
X-RateLimit-Scope: ip
```

#### Rate Limit Tiers

| Tier | Requests/Minute | Batch Size | Features |
|------|-----------------|------------|----------|
| **Free** | 100 | 10 | Basic detection |
| **Pro** | 1,000 | 100 | All methods, features |
| **Enterprise** | 10,000 | 1,000 | Priority processing, SLA |

## Request/Response Headers

### Standard Request Headers

```http
Content-Type: application/json
Accept: application/json
User-Agent: YourApp/1.0.0
X-Request-ID: unique-request-id
X-API-Key: your-api-key
Authorization: Bearer jwt-token
```

### Standard Response Headers

```http
Content-Type: application/json
X-Request-ID: unique-request-id
X-Response-Time: 234ms
X-RateLimit-Remaining: 95
Cache-Control: no-cache
```

### Custom Headers

| Header | Description | Example |
|--------|-------------|---------|
| `X-Request-ID` | Unique request identifier | `req_1234567890` |
| `X-Response-Time` | Processing time | `234ms` |
| `X-Cache-Status` | Cache hit/miss status | `HIT`, `MISS`, `BYPASS` |
| `X-Detection-Method` | Method used | `ensemble` |
| `X-Model-Version` | Model version used | `v2.1.0` |
| `X-Processing-Region` | Processing region | `us-east-1` |

## SDK Integration

### Python SDK Usage

```python
from ai_detector_sdk import AIDetectorClient, DetectionOptions

client = AIDetectorClient(
    base_url="http://localhost:8000",
    api_key="your-api-key",
    timeout=30
)

# Single detection
result = client.detect(
    text="Your text here",
    options=DetectionOptions(
        method="ensemble",
        confidence_threshold=0.7,
        return_features=True
    )
)

# Batch detection
texts = [
    {"id": "1", "text": "First text"},
    {"id": "2", "text": "Second text"}
]
batch_result = client.detect_batch(texts)

# Async detection
import asyncio

async def async_detect():
    result = await client.detect_async("Your text here")
    return result
```

### JavaScript SDK Usage

```javascript
import { AIDetectorClient } from 'ai-detector-sdk';

const client = new AIDetectorClient({
    baseURL: 'http://localhost:8000',
    apiKey: 'your-api-key',
    timeout: 30000
});

// Single detection
const result = await client.detect({
    text: 'Your text here',
    options: {
        method: 'ensemble',
        confidenceThreshold: 0.7,
        returnFeatures: true
    }
});

// Batch detection
const texts = [
    { id: '1', text: 'First text' },
    { id: '2', text: 'Second text' }
];
const batchResult = await client.detectBatch(texts);

// Stream detection
const stream = client.detectStream({
    streamId: 'stream_123',
    options: { method: 'ensemble' }
});

stream.on('result', (result) => {
    console.log('Detection result:', result);
});

stream.on('error', (error) => {
    console.error('Stream error:', error);
});
```

## Webhooks

### Webhook Configuration

#### `POST /api/webhooks`

Register a webhook endpoint for detection events.

**Request:**

```typescript
interface WebhookRequest {
  url: string;
  events: string[];              // ['detection.completed', 'batch.completed']
  secret?: string;               // For signature verification
  active: boolean;
  metadata?: Record<string, any>;
}
```

### Webhook Events

#### Detection Completed

```json
{
  "event": "detection.completed",
  "timestamp": "2025-01-15T14:30:45Z",
  "data": {
    "request_id": "req_123",
    "is_ai_generated": true,
    "confidence_score": 0.87,
    "processing_time_ms": 234.5
  }
}
```

#### Batch Completed

```json
{
  "event": "batch.completed",
  "timestamp": "2025-01-15T14:30:45Z", 
  "data": {
    "batch_id": "batch_456",
    "total_texts": 100,
    "ai_detected": 34,
    "processing_time_ms": 5678.9
  }
}
```

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- **JSON**: `GET /api/openapi.json`
- **YAML**: `GET /api/openapi.yaml`
- **Swagger UI**: `GET /api/docs`
- **ReDoc**: `GET /api/redoc`

---

*Last updated: January 15, 2025*