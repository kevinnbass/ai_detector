# AI Detector API Documentation

A comprehensive AI-generated text detection system with RESTful API, Chrome extension integration, and high-throughput data processing capabilities.

## Table of Contents

- [Quick Start](#quick-start)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
- [Request/Response Formats](#requestresponse-formats)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Examples](#examples)
- [SDKs](#sdks)

## Quick Start

### Base URL
```
http://localhost:8000/api
```

### Basic Detection Request
```bash
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a comprehensive analysis of the multifaceted paradigm.",
    "options": {
      "detection_method": "ensemble",
      "confidence_threshold": 0.7
    }
  }'
```

### Response
```json
{
  "request_id": "req_123456789",
  "is_ai_generated": true,
  "confidence_score": 0.85,
  "detection_method": "ensemble",
  "processing_time_ms": 45.2,
  "metadata": {
    "text_length": 62,
    "word_count": 8,
    "detected_patterns": ["comprehensive", "multifaceted", "paradigm"],
    "language": "en"
  },
  "timestamp": "2025-01-15T10:30:45Z"
}
```

## Authentication

Currently, the API operates without authentication in development mode. For production deployment, implement one of the following:

### API Key Authentication
```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/detect
```

### Bearer Token Authentication
```bash
curl -H "Authorization: Bearer your-jwt-token" http://localhost:8000/api/detect
```

## API Endpoints

### Core Detection Endpoints

#### `POST /api/detect`
Detect AI-generated text in a single input.

**Request Body:**
```json
{
  "text": "string (required, 1-50000 characters)",
  "request_id": "string (optional, alphanumeric + hyphens/underscores)",
  "options": {
    "detection_method": "pattern|ml|llm|ensemble (default: ensemble)",
    "confidence_threshold": "number (0-1, default: 0.7)",
    "return_features": "boolean (default: false)",
    "language": "string (default: auto-detect)"
  },
  "metadata": {
    "source": "string (optional)",
    "user_id": "string (optional)",
    "custom_fields": "object (optional)"
  }
}
```

**Response:** [Standard Detection Response](#standard-detection-response)

#### `POST /api/detect/batch`
Detect AI-generated text in multiple inputs simultaneously.

**Request Body:**
```json
{
  "texts": [
    {
      "id": "text_1",
      "text": "First text to analyze..."
    },
    {
      "id": "text_2", 
      "text": "Second text to analyze..."
    }
  ],
  "options": {
    "detection_method": "ensemble",
    "confidence_threshold": 0.7,
    "parallel_processing": true
  }
}
```

**Response:**
```json
{
  "batch_id": "batch_123456789",
  "total_texts": 2,
  "processing_time_ms": 89.4,
  "results": [
    {
      "id": "text_1",
      "is_ai_generated": true,
      "confidence_score": 0.82,
      "processing_time_ms": 43.1
    },
    {
      "id": "text_2", 
      "is_ai_generated": false,
      "confidence_score": 0.35,
      "processing_time_ms": 38.7
    }
  ],
  "statistics": {
    "ai_detected": 1,
    "human_detected": 1,
    "average_confidence": 0.585,
    "total_processing_time_ms": 89.4
  }
}
```

#### `GET /api/detect/{request_id}`
Retrieve results for a previous detection request.

**Response:** [Standard Detection Response](#standard-detection-response)

### Health and Status Endpoints

#### `GET /api/health`
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime_seconds": 3600,
  "checks": {
    "database": "healthy",
    "llm_service": "healthy", 
    "memory_usage": "normal"
  },
  "performance": {
    "avg_response_time_ms": 52.3,
    "requests_per_minute": 145,
    "cache_hit_rate": 0.73
  }
}
```

#### `GET /api/stats`
Get API usage statistics.

**Response:**
```json
{
  "total_requests": 10543,
  "requests_today": 892,
  "detection_statistics": {
    "ai_detected": 3421,
    "human_detected": 7122,
    "ai_detection_rate": 0.325
  },
  "performance_metrics": {
    "average_response_time_ms": 48.7,
    "p95_response_time_ms": 125.3,
    "throughput_per_minute": 150.2
  },
  "method_usage": {
    "pattern": 0.15,
    "ml": 0.25,
    "llm": 0.35,
    "ensemble": 0.25
  }
}
```

### Configuration Endpoints

#### `GET /api/config`
Get current API configuration.

**Response:**
```json
{
  "detection_methods": {
    "pattern": {
      "enabled": true,
      "confidence_weight": 0.2
    },
    "ml": {
      "enabled": true,
      "model_version": "v2.1.0",
      "confidence_weight": 0.3
    },
    "llm": {
      "enabled": true,
      "provider": "openrouter",
      "model": "gpt-4",
      "confidence_weight": 0.5
    }
  },
  "performance": {
    "max_text_length": 50000,
    "timeout_seconds": 30,
    "max_batch_size": 100
  },
  "features": {
    "caching_enabled": true,
    "monitoring_enabled": true,
    "rate_limiting_enabled": false
  }
}
```

#### `PUT /api/config`
Update API configuration (admin only).

**Request Body:**
```json
{
  "detection_methods": {
    "ensemble": {
      "pattern_weight": 0.2,
      "ml_weight": 0.3,
      "llm_weight": 0.5
    }
  },
  "performance": {
    "timeout_seconds": 25,
    "max_batch_size": 50
  }
}
```

## Request/Response Formats

### Standard Detection Response
```json
{
  "request_id": "string",
  "is_ai_generated": "boolean",
  "confidence_score": "number (0-1)",
  "detection_method": "string",
  "processing_time_ms": "number",
  "metadata": {
    "text_length": "number",
    "word_count": "number", 
    "detected_patterns": ["string"],
    "language": "string",
    "complexity_score": "number (0-1)"
  },
  "features": {
    "pattern_features": "object (optional)",
    "ml_features": "object (optional)", 
    "llm_features": "object (optional)"
  },
  "timestamp": "string (ISO 8601)"
}
```

### Error Response Format
```json
{
  "error": {
    "code": "string",
    "message": "string", 
    "details": "object (optional)",
    "request_id": "string (optional)"
  },
  "timestamp": "string (ISO 8601)"
}
```

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid request format or parameters |
| 401 | Unauthorized | Authentication required |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 422 | Unprocessable Entity | Validation errors |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Error Codes

| Code | Description |
|------|-------------|
| `VALIDATION_ERROR` | Request validation failed |
| `TEXT_TOO_LONG` | Text exceeds maximum length |
| `TEXT_TOO_SHORT` | Text below minimum length |
| `INVALID_METHOD` | Unsupported detection method |
| `PROCESSING_TIMEOUT` | Request processing timeout |
| `MODEL_UNAVAILABLE` | Detection model unavailable |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `INTERNAL_ERROR` | Internal server error |

### Error Examples

#### Validation Error
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "field": "text",
      "issue": "Text must be between 1 and 50000 characters",
      "provided_length": 75000
    }
  },
  "timestamp": "2025-01-15T10:30:45Z"
}
```

#### Rate Limit Error
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED", 
    "message": "Too many requests. Please try again later.",
    "details": {
      "limit": 100,
      "window": "1 minute",
      "retry_after": 45
    }
  },
  "timestamp": "2025-01-15T10:30:45Z"
}
```

## Rate Limiting

### Default Limits
- **Per IP**: 100 requests per minute
- **Per API Key**: 1000 requests per minute
- **Batch requests**: 10 requests per minute
- **Text length**: Maximum 50,000 characters per request

### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248645
X-RateLimit-Window: 60
```

### Handling Rate Limits
When rate limited, implement exponential backoff:

```python
import time
import requests

def detect_with_retry(text, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post('/api/detect', json={'text': text})
        
        if response.status_code == 429:
            retry_after = int(response.headers.get('X-RateLimit-Reset', 60))
            time.sleep(min(retry_after, 2 ** attempt))
            continue
            
        return response.json()
    
    raise Exception("Max retries exceeded")
```

## Examples

### Python SDK Example

```python
from ai_detector_sdk import AIDetectorClient

# Initialize client
client = AIDetectorClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"  # Optional
)

# Single detection
result = client.detect(
    text="This is a comprehensive analysis of the paradigm.",
    options={
        "detection_method": "ensemble",
        "confidence_threshold": 0.7
    }
)

print(f"AI Generated: {result.is_ai_generated}")
print(f"Confidence: {result.confidence_score}")

# Batch detection
texts = [
    "First text to analyze...",
    "Second text to analyze..."
]

batch_result = client.detect_batch(texts)
for result in batch_result.results:
    print(f"Text {result.id}: AI={result.is_ai_generated}")
```

### JavaScript/Node.js Example

```javascript
const AIDetectorClient = require('ai-detector-sdk');

const client = new AIDetectorClient({
    baseURL: 'http://localhost:8000',
    apiKey: 'your-api-key'  // Optional
});

// Single detection
async function detectText() {
    try {
        const result = await client.detect({
            text: "This is a comprehensive analysis of the paradigm.",
            options: {
                detection_method: "ensemble",
                confidence_threshold: 0.7
            }
        });
        
        console.log(`AI Generated: ${result.is_ai_generated}`);
        console.log(`Confidence: ${result.confidence_score}`);
    } catch (error) {
        console.error('Detection failed:', error.message);
    }
}

// Batch detection
async function detectBatch() {
    const texts = [
        { id: '1', text: 'First text...' },
        { id: '2', text: 'Second text...' }
    ];
    
    const results = await client.detectBatch(texts);
    results.forEach(result => {
        console.log(`${result.id}: ${result.is_ai_generated}`);
    });
}
```

### Chrome Extension Integration

```javascript
// Background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type === 'DETECT_TEXT') {
        detectText(message.text)
            .then(result => sendResponse(result))
            .catch(error => sendResponse({ error: error.message }));
        return true; // Keep channel open
    }
});

async function detectText(text) {
    const response = await fetch('http://localhost:8000/api/detect', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: text,
            options: {
                detection_method: 'ensemble',
                confidence_threshold: 0.7
            }
        })
    });
    
    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
}

// Content script
function analyzePageText() {
    const textElements = document.querySelectorAll('p, div, article');
    
    textElements.forEach(element => {
        const text = element.textContent.trim();
        if (text.length > 50) {
            chrome.runtime.sendMessage({
                type: 'DETECT_TEXT',
                text: text
            }, (result) => {
                if (result.is_ai_generated && result.confidence_score > 0.8) {
                    addAIIndicator(element, result);
                }
            });
        }
    });
}
```

### cURL Examples

#### Basic Detection
```bash
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d '{
    "text": "Furthermore, this comprehensive analysis demonstrates the multifaceted paradigm.",
    "options": {
      "detection_method": "ensemble",
      "confidence_threshold": 0.8,
      "return_features": true
    }
  }'
```

#### Batch Detection
```bash
curl -X POST http://localhost:8000/api/detect/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      {
        "id": "tweet_1",
        "text": "Just had an amazing coffee this morning! ☕"
      },
      {
        "id": "tweet_2", 
        "text": "This comprehensive analysis elucidates the multifaceted paradigm."
      }
    ],
    "options": {
      "detection_method": "ensemble",
      "parallel_processing": true
    }
  }'
```

#### Health Check
```bash
curl -X GET http://localhost:8000/api/health
```

#### Get Statistics
```bash
curl -X GET http://localhost:8000/api/stats \
  -H "X-API-Key: your-api-key"
```

### Response Parsing Examples

#### Python Response Handling
```python
import requests
import json

def handle_detection_response(response):
    if response.status_code == 200:
        result = response.json()
        
        if result['is_ai_generated']:
            confidence = result['confidence_score']
            method = result['detection_method']
            print(f"AI detected with {confidence:.2%} confidence using {method}")
            
            # Check specific patterns
            patterns = result['metadata'].get('detected_patterns', [])
            if patterns:
                print(f"Detected patterns: {', '.join(patterns)}")
        else:
            print("Human-written text detected")
            
        # Performance metrics
        processing_time = result['processing_time_ms']
        print(f"Processed in {processing_time}ms")
        
    elif response.status_code == 429:
        error = response.json()['error']
        retry_after = error['details'].get('retry_after', 60)
        print(f"Rate limited. Retry after {retry_after} seconds")
        
    else:
        error = response.json()['error']
        print(f"Error {error['code']}: {error['message']}")
```

#### JavaScript Error Handling
```javascript
async function detectWithErrorHandling(text) {
    try {
        const response = await fetch('/api/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            switch (data.error.code) {
                case 'RATE_LIMIT_EXCEEDED':
                    const retryAfter = data.error.details.retry_after;
                    console.log(`Rate limited. Retry after ${retryAfter}s`);
                    break;
                case 'TEXT_TOO_LONG':
                    console.log('Text exceeds maximum length');
                    break;
                default:
                    console.error('API Error:', data.error.message);
            }
            return null;
        }
        
        return data;
        
    } catch (error) {
        console.error('Network error:', error.message);
        return null;
    }
}
```

## SDKs

### Official SDKs

- **Python SDK**: `pip install ai-detector-sdk`
- **JavaScript/Node.js SDK**: `npm install ai-detector-sdk`
- **Chrome Extension Library**: Available in `/src/extension/`

### Community SDKs

- **PHP SDK**: `composer require community/ai-detector-php`
- **Go SDK**: `go get github.com/community/ai-detector-go`
- **Ruby SDK**: `gem install ai-detector-ruby`

### SDK Features

- Automatic retry with exponential backoff
- Request/response validation
- Built-in error handling
- Async/await support
- Batch processing utilities
- Rate limit handling
- Caching support

---

## Support

- **Documentation**: [Full API Docs](./api-reference.md)
- **Examples**: [Code Examples](./examples/)
- **Issues**: [GitHub Issues](https://github.com/kevinnbass/ai_detector/issues)
- **Discord**: [Community Chat](https://discord.gg/ai-detector)

## Version History

- **v1.0.0**: Initial release with core detection
- **v1.1.0**: Added batch processing and performance optimizations
- **v1.2.0**: Chrome extension integration
- **v1.3.0**: High-throughput processing (>1000 tweets/min)

---

*Last updated: January 15, 2025*