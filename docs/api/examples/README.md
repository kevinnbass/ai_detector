# AI Detector API Examples

Comprehensive collection of code examples demonstrating how to integrate with the AI Detector API across different programming languages, frameworks, and use cases.

## Table of Contents

- [Quick Start Examples](#quick-start-examples)
- [Language-Specific Examples](#language-specific-examples)
- [Framework Integrations](#framework-integrations)
- [Advanced Use Cases](#advanced-use-cases)
- [Error Handling Patterns](#error-handling-patterns)
- [Performance Optimization](#performance-optimization)

## Quick Start Examples

### Simple Detection (cURL)

```bash
# Basic text detection
curl -X POST http://localhost:8000/api/detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Furthermore, this comprehensive analysis demonstrates the multifaceted paradigm.",
    "options": {
      "detection_method": "ensemble",
      "confidence_threshold": 0.7
    }
  }'

# Response
{
  "request_id": "req_123456789",
  "is_ai_generated": true,
  "confidence_score": 0.87,
  "detection_method": "ensemble",
  "processing_time_ms": 45.2,
  "metadata": {
    "text_length": 89,
    "word_count": 12,
    "detected_patterns": ["furthermore", "comprehensive", "multifaceted", "paradigm"]
  }
}
```

### Batch Detection (cURL)

```bash
curl -X POST http://localhost:8000/api/detect/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      {
        "id": "social_1",
        "text": "OMG just had the best coffee ever! ☕ #MondayMotivation"
      },
      {
        "id": "academic_1",
        "text": "This comprehensive analysis elucidates the multifaceted paradigm inherent in contemporary discourse."
      }
    ],
    "options": {
      "detection_method": "ensemble",
      "parallel_processing": true
    }
  }'
```

## Language-Specific Examples

### Python Examples

#### Basic Usage

```python
import requests
import json

class AIDetectorClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def detect(self, text, **options):
        """Detect AI-generated content in text."""
        payload = {
            "text": text,
            "options": options
        }
        
        response = requests.post(
            f"{self.base_url}/api/detect",
            headers=self.headers,
            json=payload
        )
        
        response.raise_for_status()
        return response.json()
    
    def detect_batch(self, texts, **options):
        """Detect AI-generated content in multiple texts."""
        payload = {
            "texts": texts,
            "options": options
        }
        
        response = requests.post(
            f"{self.base_url}/api/detect/batch",
            headers=self.headers,
            json=payload
        )
        
        response.raise_for_status()
        return response.json()

# Usage example
client = AIDetectorClient(api_key="your-api-key")

# Single detection
result = client.detect(
    "This comprehensive analysis demonstrates the paradigm.",
    detection_method="ensemble",
    confidence_threshold=0.8
)

print(f"AI Generated: {result['is_ai_generated']}")
print(f"Confidence: {result['confidence_score']:.2%}")

# Batch detection
texts = [
    {"id": "tweet_1", "text": "Just loving this sunny day! 🌞"},
    {"id": "essay_1", "text": "Furthermore, the comprehensive analysis reveals..."}
]

batch_result = client.detect_batch(texts, detection_method="ensemble")

for result in batch_result['results']:
    print(f"{result['id']}: AI={result['is_ai_generated']} ({result['confidence_score']:.2%})")
```

#### Async Python with aiohttp

```python
import aiohttp
import asyncio
import json

class AsyncAIDetectorClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    async def detect(self, session, text, **options):
        """Async detection of AI-generated content."""
        payload = {
            "text": text,
            "options": options
        }
        
        async with session.post(
            f"{self.base_url}/api/detect",
            headers=self.headers,
            json=payload
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def detect_multiple(self, texts, **options):
        """Process multiple texts concurrently."""
        async with aiohttp.ClientSession() as session:
            tasks = [
                self.detect(session, text, **options)
                for text in texts
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return results

# Usage
async def main():
    client = AsyncAIDetectorClient(api_key="your-api-key")
    
    texts = [
        "Hey everyone, just wanted to share this amazing experience!",
        "Furthermore, this comprehensive analysis demonstrates the multifaceted paradigm.",
        "Can't believe how good this pizza is! 🍕",
        "The implementation of this methodology facilitates enhanced outcomes."
    ]
    
    results = await client.detect_multiple(texts, detection_method="ensemble")
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Text {i+1}: Error - {result}")
        else:
            print(f"Text {i+1}: AI={result['is_ai_generated']} ({result['confidence_score']:.2%})")

# Run async example
asyncio.run(main())
```

#### Django Integration

```python
# views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import requests

@csrf_exempt
@require_http_methods(["POST"])
def detect_text(request):
    """Django view for text detection."""
    try:
        data = json.loads(request.body)
        text = data.get('text')
        
        if not text:
            return JsonResponse({'error': 'Text is required'}, status=400)
        
        # Call AI Detector API
        response = requests.post(
            'http://localhost:8000/api/detect',
            json={
                'text': text,
                'options': {
                    'detection_method': 'ensemble',
                    'confidence_threshold': 0.7
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            return JsonResponse({
                'is_ai_generated': result['is_ai_generated'],
                'confidence_score': result['confidence_score'],
                'processing_time_ms': result['processing_time_ms']
            })
        else:
            return JsonResponse(
                {'error': 'Detection service unavailable'}, 
                status=503
            )
            
    except Exception as e:
        return JsonResponse(
            {'error': str(e)}, 
            status=500
        )

# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('detect/', views.detect_text, name='detect_text'),
]
```

### JavaScript/Node.js Examples

#### Basic Node.js Client

```javascript
const axios = require('axios');

class AIDetectorClient {
    constructor(baseURL = 'http://localhost:8000', apiKey = null) {
        this.client = axios.create({
            baseURL,
            headers: {
                'Content-Type': 'application/json',
                ...(apiKey && { 'X-API-Key': apiKey })
            }
        });
        
        // Add response interceptor for error handling
        this.client.interceptors.response.use(
            response => response.data,
            error => {
                if (error.response?.data?.error) {
                    throw new Error(`API Error: ${error.response.data.error.message}`);
                }
                throw error;
            }
        );
    }
    
    async detect(text, options = {}) {
        return await this.client.post('/api/detect', {
            text,
            options
        });
    }
    
    async detectBatch(texts, options = {}) {
        return await this.client.post('/api/detect/batch', {
            texts,
            options
        });
    }
    
    async getHealth() {
        return await this.client.get('/api/health');
    }
}

// Usage
async function example() {
    const client = new AIDetectorClient('http://localhost:8000', 'your-api-key');
    
    try {
        // Single detection
        const result = await client.detect(
            "Furthermore, this comprehensive analysis demonstrates the paradigm.",
            {
                detection_method: 'ensemble',
                confidence_threshold: 0.8,
                return_features: true
            }
        );
        
        console.log(`AI Generated: ${result.is_ai_generated}`);
        console.log(`Confidence: ${(result.confidence_score * 100).toFixed(1)}%`);
        
        if (result.features) {
            console.log('Pattern Features:', result.features.pattern_features);
        }
        
        // Batch detection
        const texts = [
            { id: 'casual', text: 'Hey! Just grabbed some coffee ☕' },
            { id: 'formal', text: 'This comprehensive analysis elucidates the multifaceted paradigm.' }
        ];
        
        const batchResult = await client.detectBatch(texts, {
            detection_method: 'ensemble',
            parallel_processing: true
        });
        
        console.log(`Processed ${batchResult.total_texts} texts in ${batchResult.processing_time_ms}ms`);
        
        batchResult.results.forEach(result => {
            console.log(`${result.id}: ${result.is_ai_generated ? 'AI' : 'Human'} (${(result.confidence_score * 100).toFixed(1)}%)`);
        });
        
    } catch (error) {
        console.error('Detection failed:', error.message);
    }
}

example();
```

#### Express.js API Integration

```javascript
const express = require('express');
const rateLimit = require('express-rate-limit');
const { AIDetectorClient } = require('./ai-detector-client');

const app = express();
app.use(express.json());

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100 // limit each IP to 100 requests per windowMs
});
app.use('/api/', limiter);

// Initialize AI Detector client
const aiDetector = new AIDetectorClient(
    process.env.AI_DETECTOR_URL || 'http://localhost:8000',
    process.env.AI_DETECTOR_API_KEY
);

// Single text detection endpoint
app.post('/api/detect', async (req, res) => {
    try {
        const { text, options = {} } = req.body;
        
        if (!text || text.trim().length === 0) {
            return res.status(400).json({
                error: 'Text is required and cannot be empty'
            });
        }
        
        if (text.length > 50000) {
            return res.status(400).json({
                error: 'Text exceeds maximum length of 50,000 characters'
            });
        }
        
        const result = await aiDetector.detect(text, {
            detection_method: options.method || 'ensemble',
            confidence_threshold: options.threshold || 0.7,
            return_features: options.include_features || false
        });
        
        res.json({
            success: true,
            data: {
                is_ai_generated: result.is_ai_generated,
                confidence_score: result.confidence_score,
                detection_method: result.detection_method,
                processing_time_ms: result.processing_time_ms,
                metadata: result.metadata
            }
        });
        
    } catch (error) {
        console.error('Detection error:', error);
        res.status(500).json({
            error: 'Internal server error',
            message: error.message
        });
    }
});

// Batch detection endpoint
app.post('/api/detect/batch', async (req, res) => {
    try {
        const { texts, options = {} } = req.body;
        
        if (!Array.isArray(texts) || texts.length === 0) {
            return res.status(400).json({
                error: 'Texts array is required and cannot be empty'
            });
        }
        
        if (texts.length > 100) {
            return res.status(400).json({
                error: 'Batch size cannot exceed 100 texts'
            });
        }
        
        // Validate each text entry
        for (const item of texts) {
            if (!item.id || !item.text) {
                return res.status(400).json({
                    error: 'Each text item must have id and text fields'
                });
            }
        }
        
        const result = await aiDetector.detectBatch(texts, {
            detection_method: options.method || 'ensemble',
            confidence_threshold: options.threshold || 0.7,
            parallel_processing: true
        });
        
        res.json({
            success: true,
            data: result
        });
        
    } catch (error) {
        console.error('Batch detection error:', error);
        res.status(500).json({
            error: 'Internal server error',
            message: error.message
        });
    }
});

// Health check endpoint
app.get('/health', async (req, res) => {
    try {
        const health = await aiDetector.getHealth();
        res.json({
            status: 'healthy',
            ai_detector_status: health.status,
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        res.status(503).json({
            status: 'unhealthy',
            error: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
```

#### React Frontend Integration

```jsx
import React, { useState, useCallback } from 'react';
import axios from 'axios';

// Custom hook for AI detection
const useAIDetector = () => {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    
    const detectText = useCallback(async (text, options = {}) => {
        setLoading(true);
        setError(null);
        
        try {
            const response = await axios.post('/api/detect', {
                text,
                options: {
                    method: options.method || 'ensemble',
                    threshold: options.threshold || 0.7,
                    include_features: options.includeFeatures || false
                }
            });
            
            return response.data.data;
        } catch (err) {
            const errorMessage = err.response?.data?.error || 'Detection failed';
            setError(errorMessage);
            throw new Error(errorMessage);
        } finally {
            setLoading(false);
        }
    }, []);
    
    return { detectText, loading, error };
};

// Main component
const AIDetectorComponent = () => {
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);
    const { detectText, loading, error } = useAIDetector();
    
    const handleDetect = async () => {
        if (!text.trim()) {
            alert('Please enter some text to analyze');
            return;
        }
        
        try {
            const detectionResult = await detectText(text, {
                method: 'ensemble',
                threshold: 0.7,
                includeFeatures: true
            });
            setResult(detectionResult);
        } catch (err) {
            console.error('Detection failed:', err.message);
        }
    };
    
    const getResultColor = (isAI, confidence) => {
        if (!isAI) return 'green';
        return confidence > 0.8 ? 'red' : confidence > 0.6 ? 'orange' : 'yellow';
    };
    
    return (
        <div className="ai-detector">
            <h2>AI Text Detector</h2>
            
            <div className="input-section">
                <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Enter text to analyze for AI generation..."
                    rows="6"
                    cols="80"
                    maxLength="50000"
                />
                <br />
                <button 
                    onClick={handleDetect}
                    disabled={loading || !text.trim()}
                >
                    {loading ? 'Analyzing...' : 'Detect AI Content'}
                </button>
            </div>
            
            {error && (
                <div className="error" style={{ color: 'red', marginTop: '10px' }}>
                    Error: {error}
                </div>
            )}
            
            {result && (
                <div className="result-section" style={{ marginTop: '20px' }}>
                    <h3>Detection Result</h3>
                    <div 
                        className="result-badge"
                        style={{ 
                            color: getResultColor(result.is_ai_generated, result.confidence_score),
                            fontWeight: 'bold',
                            fontSize: '18px'
                        }}
                    >
                        {result.is_ai_generated ? '🤖 AI Generated' : '👤 Human Written'}
                    </div>
                    
                    <div className="result-details">
                        <p><strong>Confidence:</strong> {(result.confidence_score * 100).toFixed(1)}%</p>
                        <p><strong>Method:</strong> {result.detection_method}</p>
                        <p><strong>Processing Time:</strong> {result.processing_time_ms}ms</p>
                        
                        {result.metadata?.detected_patterns?.length > 0 && (
                            <div>
                                <strong>Detected Patterns:</strong>
                                <ul>
                                    {result.metadata.detected_patterns.map((pattern, index) => (
                                        <li key={index}>{pattern}</li>
                                    ))}
                                </ul>
                            </div>
                        )}
                    </div>
                </div>
            )}
            
            <div className="info-section" style={{ marginTop: '30px', fontSize: '14px', color: '#666' }}>
                <p>This tool analyzes text patterns to detect AI-generated content.</p>
                <p>Confidence scores above 80% indicate high likelihood of AI generation.</p>
            </div>
        </div>
    );
};

export default AIDetectorComponent;
```

### PHP Examples

#### Basic PHP Client

```php
<?php

class AIDetectorClient {
    private $baseUrl;
    private $apiKey;
    private $httpClient;
    
    public function __construct($baseUrl = 'http://localhost:8000', $apiKey = null) {
        $this->baseUrl = rtrim($baseUrl, '/');
        $this->apiKey = $apiKey;
        
        $this->httpClient = curl_init();
        curl_setopt_array($this->httpClient, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_FOLLOWLOCATION => true,
            CURLOPT_TIMEOUT => 30,
            CURLOPT_HTTPHEADER => [
                'Content-Type: application/json',
                'Accept: application/json',
                ...(($this->apiKey) ? ['X-API-Key: ' . $this->apiKey] : [])
            ]
        ]);
    }
    
    public function detect($text, $options = []) {
        $payload = [
            'text' => $text,
            'options' => $options
        ];
        
        return $this->makeRequest('POST', '/api/detect', $payload);
    }
    
    public function detectBatch($texts, $options = []) {
        $payload = [
            'texts' => $texts,
            'options' => $options
        ];
        
        return $this->makeRequest('POST', '/api/detect/batch', $payload);
    }
    
    public function getHealth() {
        return $this->makeRequest('GET', '/api/health');
    }
    
    private function makeRequest($method, $endpoint, $data = null) {
        $url = $this->baseUrl . $endpoint;
        
        curl_setopt($this->httpClient, CURLOPT_URL, $url);
        curl_setopt($this->httpClient, CURLOPT_CUSTOMREQUEST, $method);
        
        if ($data !== null) {
            curl_setopt($this->httpClient, CURLOPT_POSTFIELDS, json_encode($data));
        }
        
        $response = curl_exec($this->httpClient);
        $httpCode = curl_getinfo($this->httpClient, CURLINFO_HTTP_CODE);
        
        if (curl_error($this->httpClient)) {
            throw new Exception('cURL Error: ' . curl_error($this->httpClient));
        }
        
        $decodedResponse = json_decode($response, true);
        
        if ($httpCode >= 400) {
            $errorMessage = $decodedResponse['error']['message'] ?? 'Unknown error';
            throw new Exception("API Error ($httpCode): $errorMessage");
        }
        
        return $decodedResponse;
    }
    
    public function __destruct() {
        if ($this->httpClient) {
            curl_close($this->httpClient);
        }
    }
}

// Usage example
try {
    $client = new AIDetectorClient('http://localhost:8000', 'your-api-key');
    
    // Single detection
    $result = $client->detect(
        "Furthermore, this comprehensive analysis demonstrates the multifaceted paradigm.",
        [
            'detection_method' => 'ensemble',
            'confidence_threshold' => 0.7,
            'return_features' => true
        ]
    );
    
    echo "AI Generated: " . ($result['is_ai_generated'] ? 'Yes' : 'No') . "\n";
    echo "Confidence: " . round($result['confidence_score'] * 100, 1) . "%\n";
    echo "Processing Time: " . $result['processing_time_ms'] . "ms\n";
    
    // Batch detection
    $texts = [
        ['id' => 'casual', 'text' => 'Hey everyone! Just had an amazing day at the beach! 🏖️'],
        ['id' => 'formal', 'text' => 'This comprehensive analysis elucidates the multifaceted paradigm inherent in contemporary discourse.']
    ];
    
    $batchResult = $client->detectBatch($texts, [
        'detection_method' => 'ensemble',
        'parallel_processing' => true
    ]);
    
    echo "\nBatch Results:\n";
    foreach ($batchResult['results'] as $result) {
        $aiStatus = $result['is_ai_generated'] ? 'AI' : 'Human';
        $confidence = round($result['confidence_score'] * 100, 1);
        echo "{$result['id']}: $aiStatus ($confidence%)\n";
    }
    
} catch (Exception $e) {
    echo "Error: " . $e->getMessage() . "\n";
}
```

#### Laravel Integration

```php
<?php
// app/Services/AIDetectorService.php

namespace App\Services;

use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Log;

class AIDetectorService
{
    protected $baseUrl;
    protected $apiKey;
    protected $timeout;
    
    public function __construct()
    {
        $this->baseUrl = config('services.ai_detector.url', 'http://localhost:8000');
        $this->apiKey = config('services.ai_detector.api_key');
        $this->timeout = config('services.ai_detector.timeout', 30);
    }
    
    public function detect(string $text, array $options = [])
    {
        // Create cache key for text
        $cacheKey = 'ai_detect_' . md5($text . serialize($options));
        
        // Try to get from cache first
        return Cache::remember($cacheKey, 3600, function() use ($text, $options) {
            return $this->makeDetectionRequest($text, $options);
        });
    }
    
    public function detectBatch(array $texts, array $options = [])
    {
        $payload = [
            'texts' => $texts,
            'options' => array_merge([
                'detection_method' => 'ensemble',
                'parallel_processing' => true
            ], $options)
        ];
        
        return $this->makeRequest('POST', '/api/detect/batch', $payload);
    }
    
    protected function makeDetectionRequest(string $text, array $options = [])
    {
        $payload = [
            'text' => $text,
            'options' => array_merge([
                'detection_method' => 'ensemble',
                'confidence_threshold' => 0.7
            ], $options)
        ];
        
        return $this->makeRequest('POST', '/api/detect', $payload);
    }
    
    protected function makeRequest(string $method, string $endpoint, array $data = [])
    {
        try {
            $request = Http::timeout($this->timeout);
            
            if ($this->apiKey) {
                $request->withHeaders(['X-API-Key' => $this->apiKey]);
            }
            
            $response = $request->$method($this->baseUrl . $endpoint, $data);
            
            if ($response->failed()) {
                $error = $response->json('error.message', 'Unknown API error');
                throw new \Exception("AI Detector API Error: $error");
            }
            
            return $response->json();
            
        } catch (\Exception $e) {
            Log::error('AI Detector API Error', [
                'endpoint' => $endpoint,
                'error' => $e->getMessage(),
                'data' => $data
            ]);
            
            throw $e;
        }
    }
    
    public function isHealthy(): bool
    {
        try {
            $response = $this->makeRequest('GET', '/api/health');
            return $response['status'] === 'healthy';
        } catch (\Exception $e) {
            return false;
        }
    }
}

// app/Http/Controllers/AIDetectionController.php

namespace App\Http\Controllers;

use App\Services\AIDetectorService;
use Illuminate\Http\Request;
use Illuminate\Http\JsonResponse;

class AIDetectionController extends Controller
{
    protected $aiDetector;
    
    public function __construct(AIDetectorService $aiDetector)
    {
        $this->aiDetector = $aiDetector;
    }
    
    public function detect(Request $request): JsonResponse
    {
        $request->validate([
            'text' => 'required|string|max:50000',
            'options' => 'array',
            'options.method' => 'string|in:pattern,ml,llm,ensemble',
            'options.threshold' => 'numeric|min:0|max:1'
        ]);
        
        try {
            $result = $this->aiDetector->detect(
                $request->input('text'),
                $request->input('options', [])
            );
            
            return response()->json([
                'success' => true,
                'data' => $result
            ]);
            
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'error' => $e->getMessage()
            ], 500);
        }
    }
    
    public function detectBatch(Request $request): JsonResponse
    {
        $request->validate([
            'texts' => 'required|array|max:100',
            'texts.*.id' => 'required|string',
            'texts.*.text' => 'required|string|max:50000',
            'options' => 'array'
        ]);
        
        try {
            $result = $this->aiDetector->detectBatch(
                $request->input('texts'),
                $request->input('options', [])
            );
            
            return response()->json([
                'success' => true,
                'data' => $result
            ]);
            
        } catch (\Exception $e) {
            return response()->json([
                'success' => false,
                'error' => $e->getMessage()
            ], 500);
        }
    }
}

// config/services.php
return [
    // ... other services
    
    'ai_detector' => [
        'url' => env('AI_DETECTOR_URL', 'http://localhost:8000'),
        'api_key' => env('AI_DETECTOR_API_KEY'),
        'timeout' => env('AI_DETECTOR_TIMEOUT', 30),
    ],
];
```

## Framework Integrations

### Chrome Extension Integration

```javascript
// background.js
class AIDetectorExtension {
    constructor() {
        this.apiUrl = 'http://localhost:8000';
        this.cache = new Map();
        this.setupMessageHandlers();
    }
    
    setupMessageHandlers() {
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
            return true; // Keep channel open for async response
        });
    }
    
    async handleMessage(message, sender, sendResponse) {
        try {
            switch (message.type) {
                case 'DETECT_TEXT':
                    const result = await this.detectText(message.text, message.options);
                    sendResponse({ success: true, data: result });
                    break;
                    
                case 'DETECT_PAGE_CONTENT':
                    const pageResults = await this.detectPageContent(sender.tab.id);
                    sendResponse({ success: true, data: pageResults });
                    break;
                    
                case 'GET_SETTINGS':
                    const settings = await this.getSettings();
                    sendResponse({ success: true, data: settings });
                    break;
                    
                default:
                    sendResponse({ success: false, error: 'Unknown message type' });
            }
        } catch (error) {
            console.error('Message handling error:', error);
            sendResponse({ success: false, error: error.message });
        }
    }
    
    async detectText(text, options = {}) {
        // Check cache first
        const cacheKey = this.generateCacheKey(text, options);
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }
        
        const response = await fetch(`${this.apiUrl}/api/detect`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text,
                options: {
                    detection_method: options.method || 'ensemble',
                    confidence_threshold: options.threshold || 0.7
                }
            })
        });
        
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }
        
        const result = await response.json();
        
        // Cache result for 5 minutes
        this.cache.set(cacheKey, result);
        setTimeout(() => this.cache.delete(cacheKey), 5 * 60 * 1000);
        
        return result;
    }
    
    async detectPageContent(tabId) {
        // Inject content script to extract text
        const results = await chrome.scripting.executeScript({
            target: { tabId },
            function: this.extractPageText
        });
        
        const textElements = results[0].result;
        
        // Process in batches
        const batchSize = 10;
        const batches = this.createBatches(textElements, batchSize);
        const allResults = [];
        
        for (const batch of batches) {
            const batchResult = await this.detectBatch(batch);
            allResults.push(...batchResult.results);
        }
        
        return allResults;
    }
    
    extractPageText() {
        // This function runs in the page context
        const textElements = [];
        const selectors = ['p', 'div', 'article', 'section', 'h1', 'h2', 'h3'];
        
        selectors.forEach(selector => {
            document.querySelectorAll(selector).forEach((element, index) => {
                const text = element.textContent.trim();
                if (text.length > 50 && text.length < 5000) {
                    textElements.push({
                        id: `${selector}_${index}`,
                        text: text,
                        element: element.outerHTML.substring(0, 100) // For identification
                    });
                }
            });
        });
        
        return textElements.slice(0, 50); // Limit to 50 elements
    }
    
    async detectBatch(texts) {
        const response = await fetch(`${this.apiUrl}/api/detect/batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                texts,
                options: {
                    detection_method: 'ensemble',
                    parallel_processing: true
                }
            })
        });
        
        if (!response.ok) {
            throw new Error(`Batch API request failed: ${response.status}`);
        }
        
        return await response.json();
    }
    
    generateCacheKey(text, options) {
        return `${text.substring(0, 50)}_${JSON.stringify(options)}`;
    }
    
    createBatches(items, batchSize) {
        const batches = [];
        for (let i = 0; i < items.length; i += batchSize) {
            batches.push(items.slice(i, i + batchSize));
        }
        return batches;
    }
    
    async getSettings() {
        return new Promise((resolve) => {
            chrome.storage.sync.get([
                'autoDetect',
                'confidenceThreshold',
                'detectionMethod',
                'showIndicators'
            ], (settings) => {
                resolve({
                    autoDetect: settings.autoDetect ?? true,
                    confidenceThreshold: settings.confidenceThreshold ?? 0.7,
                    detectionMethod: settings.detectionMethod ?? 'ensemble',
                    showIndicators: settings.showIndicators ?? true
                });
            });
        });
    }
}

// Initialize extension
const aiDetectorExtension = new AIDetectorExtension();

// content.js
class AIDetectorContent {
    constructor() {
        this.indicators = new Map();
        this.settings = null;
        this.init();
    }
    
    async init() {
        this.settings = await this.getSettings();
        
        if (this.settings.autoDetect) {
            this.startAutoDetection();
        }
        
        this.setupObserver();
    }
    
    async getSettings() {
        return new Promise((resolve) => {
            chrome.runtime.sendMessage({ type: 'GET_SETTINGS' }, (response) => {
                resolve(response.success ? response.data : {});
            });
        });
    }
    
    startAutoDetection() {
        const textElements = document.querySelectorAll('p, div, article');
        const textsToAnalyze = [];
        
        textElements.forEach((element, index) => {
            const text = element.textContent.trim();
            if (text.length > 50 && text.length < 2000) {
                textsToAnalyze.push({
                    element,
                    text,
                    id: `auto_${index}`
                });
            }
        });
        
        // Process in small batches to avoid overwhelming the API
        this.processBatch(textsToAnalyze.slice(0, 20));
    }
    
    async processBatch(items) {
        const texts = items.map(item => ({
            id: item.id,
            text: item.text
        }));
        
        chrome.runtime.sendMessage({
            type: 'DETECT_TEXT',
            text: texts[0].text, // Single detection for now
            options: {
                method: this.settings.detectionMethod,
                threshold: this.settings.confidenceThreshold
            }
        }, (response) => {
            if (response.success && response.data.is_ai_generated) {
                const item = items[0];
                this.addIndicator(item.element, response.data);
            }
        });
    }
    
    addIndicator(element, result) {
        if (!this.settings.showIndicators) return;
        
        const indicator = document.createElement('div');
        indicator.className = 'ai-detector-indicator';
        indicator.innerHTML = '🤖';
        indicator.title = `AI Detected (${(result.confidence_score * 100).toFixed(1)}% confidence)`;
        
        indicator.style.cssText = `
            position: absolute;
            top: -10px;
            right: -10px;
            width: 20px;
            height: 20px;
            background: ${result.confidence_score > 0.8 ? '#ff4444' : '#ff8844'};
            border-radius: 50%;
            color: white;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            cursor: pointer;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        `;
        
        // Position relative to element
        const rect = element.getBoundingClientRect();
        indicator.style.left = `${rect.right - 10}px`;
        indicator.style.top = `${rect.top - 10}px`;
        
        // Add click handler for details
        indicator.addEventListener('click', () => {
            this.showDetailedResult(result);
        });
        
        document.body.appendChild(indicator);
        this.indicators.set(element, indicator);
        
        // Auto-remove after 30 seconds
        setTimeout(() => {
            this.removeIndicator(element);
        }, 30000);
    }
    
    removeIndicator(element) {
        const indicator = this.indicators.get(element);
        if (indicator && indicator.parentNode) {
            indicator.parentNode.removeChild(indicator);
        }
        this.indicators.delete(element);
    }
    
    showDetailedResult(result) {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: white;
            border: 2px solid #333;
            border-radius: 8px;
            padding: 20px;
            z-index: 10001;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            max-width: 400px;
        `;
        
        modal.innerHTML = `
            <h3>AI Detection Result</h3>
            <p><strong>AI Generated:</strong> ${result.is_ai_generated ? 'Yes' : 'No'}</p>
            <p><strong>Confidence:</strong> ${(result.confidence_score * 100).toFixed(1)}%</p>
            <p><strong>Method:</strong> ${result.detection_method}</p>
            <p><strong>Processing Time:</strong> ${result.processing_time_ms}ms</p>
            ${result.metadata?.detected_patterns ? `
                <p><strong>Patterns:</strong> ${result.metadata.detected_patterns.join(', ')}</p>
            ` : ''}
            <button onclick="this.parentNode.remove()">Close</button>
        `;
        
        document.body.appendChild(modal);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (modal.parentNode) {
                modal.parentNode.removeChild(modal);
            }
        }, 10000);
    }
    
    setupObserver() {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    // New content added, potentially analyze it
                    if (this.settings.autoDetect) {
                        setTimeout(() => this.analyzeNewContent(mutation.addedNodes), 1000);
                    }
                }
            });
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
    
    analyzeNewContent(nodes) {
        const textElements = [];
        
        nodes.forEach(node => {
            if (node.nodeType === Node.ELEMENT_NODE) {
                const texts = node.querySelectorAll('p, div');
                texts.forEach(element => {
                    const text = element.textContent.trim();
                    if (text.length > 50 && text.length < 2000) {
                        textElements.push({ element, text, id: `new_${Date.now()}` });
                    }
                });
            }
        });
        
        if (textElements.length > 0) {
            this.processBatch(textElements.slice(0, 5)); // Limit new content analysis
        }
    }
}

// Initialize content script
const aiDetectorContent = new AIDetectorContent();
```

This comprehensive examples documentation provides developers with practical, ready-to-use code samples for integrating with the AI Detector API across multiple programming languages and frameworks. Each example includes error handling, best practices, and real-world usage patterns.

---

*Last updated: January 15, 2025*