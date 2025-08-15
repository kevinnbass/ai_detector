# AI Detector Integration Guide

Comprehensive guide for integrating the AI Detector system into your applications, services, and workflows. Covers multiple integration patterns, best practices, and real-world examples.

## Table of Contents

- [Integration Overview](#integration-overview)
- [REST API Integration](#rest-api-integration)
- [SDK Integration](#sdk-integration)
- [Chrome Extension Integration](#chrome-extension-integration)
- [Webhook Integration](#webhook-integration)
- [Batch Processing Integration](#batch-processing-integration)
- [Stream Processing Integration](#stream-processing-integration)
- [Third-Party Platform Integration](#third-party-platform-integration)
- [Enterprise Integration](#enterprise-integration)
- [Best Practices](#best-practices)

## Integration Overview

The AI Detector system offers multiple integration approaches to fit different use cases and architectural patterns:

### Integration Methods

| Method | Use Case | Complexity | Real-time | Batch Support |
|--------|----------|------------|-----------|---------------|
| **REST API** | Web apps, mobile apps | Low | Yes | Yes |
| **Python SDK** | Python applications | Low | Yes | Yes |
| **JavaScript SDK** | Node.js, browsers | Low | Yes | Yes |
| **Chrome Extension** | Browser integration | Medium | Yes | Limited |
| **Webhooks** | Event-driven systems | Medium | Yes | Yes |
| **Batch API** | Data processing | Low | No | Yes |
| **Stream API** | Real-time pipelines | High | Yes | Yes |

### Architecture Patterns

```mermaid
graph TB
    subgraph "Your Application"
        APP[Application Layer]
        BIZ[Business Logic]
        DATA[Data Layer]
    end
    
    subgraph "Integration Layer"
        SDK[AI Detector SDK]
        HTTP[HTTP Client]
        WEBHOOK[Webhook Handler]
        STREAM[Stream Processor]
    end
    
    subgraph "AI Detector System"
        API[REST API]
        BATCH[Batch Processor]
        REALTIME[Real-time Engine]
        CACHE[Cache Layer]
    end
    
    APP --> SDK
    APP --> HTTP
    BIZ --> WEBHOOK
    DATA --> STREAM
    
    SDK --> API
    HTTP --> API
    WEBHOOK --> API
    STREAM --> BATCH
    
    API --> REALTIME
    BATCH --> REALTIME
    REALTIME --> CACHE
```

## REST API Integration

### Basic Integration

#### Single Text Detection

```python
import requests
import json

class AIDetectorClient:
    def __init__(self, base_url="http://localhost:8000", api_key=None):
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    def detect_text(self, text, options=None):
        """Detect AI-generated content in text."""
        payload = {
            "text": text,
            "options": options or {}
        }
        
        response = requests.post(
            f"{self.base_url}/api/detect",
            headers=self.headers,
            json=payload,
            timeout=30
        )
        
        response.raise_for_status()
        return response.json()

# Usage example
client = AIDetectorClient(api_key="your-api-key")

result = client.detect_text(
    "This comprehensive analysis demonstrates the multifaceted paradigm.",
    options={
        "detection_method": "ensemble",
        "confidence_threshold": 0.7,
        "return_features": True
    }
)

print(f"AI Generated: {result['is_ai_generated']}")
print(f"Confidence: {result['confidence_score']:.2%}")
```

#### Error Handling and Retry Logic

```python
import time
import random
from typing import Optional

class AIDetectorClientWithRetry:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({"X-API-Key": api_key})
    
    def detect_with_retry(self, text: str, max_retries: int = 3, backoff_factor: float = 0.3):
        """Detect text with exponential backoff retry."""
        for attempt in range(max_retries + 1):
            try:
                response = self.session.post(
                    f"{self.base_url}/api/detect",
                    json={"text": text},
                    timeout=30
                )
                
                if response.status_code == 429:  # Rate limited
                    retry_after = int(response.headers.get('Retry-After', 60))
                    if attempt < max_retries:
                        time.sleep(retry_after)
                        continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.exceptions.RequestException as e:
                if attempt == max_retries:
                    raise e
                
                # Exponential backoff with jitter
                delay = backoff_factor * (2 ** attempt) + random.uniform(0, 1)
                time.sleep(delay)
        
        raise Exception("Max retries exceeded")

# Usage
client = AIDetectorClientWithRetry("http://localhost:8000", "your-api-key")
result = client.detect_with_retry("Text to analyze")
```

### Advanced API Integration

#### Batch Processing

```python
class BatchProcessor:
    def __init__(self, client: AIDetectorClient, batch_size: int = 100):
        self.client = client
        self.batch_size = batch_size
    
    def process_texts(self, texts: list, callback=None):
        """Process multiple texts in batches."""
        results = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            # Prepare batch payload
            batch_payload = {
                "texts": [
                    {"id": str(idx), "text": text}
                    for idx, text in enumerate(batch, start=i)
                ],
                "options": {
                    "detection_method": "ensemble",
                    "parallel_processing": True
                }
            }
            
            # Process batch
            batch_result = self.client.session.post(
                f"{self.client.base_url}/api/detect/batch",
                json=batch_payload
            ).json()
            
            results.extend(batch_result["results"])
            
            # Call progress callback
            if callback:
                callback(len(results), len(texts))
        
        return results

# Usage
def progress_callback(processed, total):
    print(f"Processed {processed}/{total} texts ({processed/total:.1%})")

batch_processor = BatchProcessor(client)
texts = ["Text 1", "Text 2", "Text 3", ...]
results = batch_processor.process_texts(texts, progress_callback)
```

#### Async Integration

```python
import asyncio
import aiohttp
from typing import List, Dict, Any

class AsyncAIDetectorClient:
    def __init__(self, base_url: str, api_key: str = None):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
    
    async def detect_text(self, session: aiohttp.ClientSession, text: str) -> Dict[str, Any]:
        """Async text detection."""
        async with session.post(
            f"{self.base_url}/api/detect",
            headers=self.headers,
            json={"text": text}
        ) as response:
            response.raise_for_status()
            return await response.json()
    
    async def detect_multiple(self, texts: List[str], max_concurrent: int = 10) -> List[Dict[str, Any]]:
        """Process multiple texts concurrently."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def detect_with_semaphore(session, text):
            async with semaphore:
                return await self.detect_text(session, text)
        
        async with aiohttp.ClientSession() as session:
            tasks = [detect_with_semaphore(session, text) for text in texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            return [result for result in results if not isinstance(result, Exception)]

# Usage
async def main():
    client = AsyncAIDetectorClient("http://localhost:8000", "your-api-key")
    texts = ["Text 1", "Text 2", "Text 3"]
    results = await client.detect_multiple(texts)
    print(f"Processed {len(results)} texts")

asyncio.run(main())
```

## SDK Integration

### Python SDK

#### Installation and Setup

```bash
pip install ai-detector-sdk
```

```python
from ai_detector_sdk import AIDetectorClient, DetectionOptions

# Initialize client
client = AIDetectorClient(
    base_url="https://api.ai-detector.com",
    api_key="your-api-key"
)

# Configure default options
default_options = DetectionOptions(
    method="ensemble",
    confidence_threshold=0.7,
    return_features=True,
    timeout=30
)

client.set_default_options(default_options)
```

#### Advanced Usage

```python
from ai_detector_sdk import AIDetectorClient, DetectionOptions, BatchOptions
import pandas as pd

class ContentModerationService:
    def __init__(self, api_key: str):
        self.detector = AIDetectorClient(api_key=api_key)
        self.detection_options = DetectionOptions(
            method="ensemble",
            confidence_threshold=0.8,
            return_features=True
        )
    
    def moderate_content(self, content: str, content_type: str = "general") -> dict:
        """Moderate content with AI detection."""
        try:
            result = self.detector.detect(content, self.detection_options)
            
            return {
                "content": content,
                "is_ai_generated": result.is_ai_generated,
                "confidence": result.confidence_score,
                "action": self._determine_action(result, content_type),
                "reasoning": result.features.get("llm_features", {}).get("reasoning", ""),
                "timestamp": result.timestamp
            }
        except Exception as e:
            return {
                "content": content,
                "error": str(e),
                "action": "manual_review"
            }
    
    def _determine_action(self, result, content_type: str) -> str:
        """Determine moderation action based on detection result."""
        if not result.is_ai_generated:
            return "approve"
        
        if result.confidence_score > 0.9:
            return "reject"
        elif result.confidence_score > 0.7:
            return "flag_for_review"
        else:
            return "approve_with_warning"
    
    def moderate_bulk_content(self, content_list: list) -> pd.DataFrame:
        """Moderate multiple content items."""
        batch_options = BatchOptions(
            parallel_processing=True,
            max_concurrency=10
        )
        
        # Prepare batch
        texts = [
            {"id": str(i), "text": content}
            for i, content in enumerate(content_list)
        ]
        
        # Process batch
        results = self.detector.detect_batch(texts, batch_options)
        
        # Convert to DataFrame
        moderation_results = []
        for result in results.results:
            moderation_results.append({
                "content_id": result.id,
                "is_ai_generated": result.is_ai_generated,
                "confidence": result.confidence_score,
                "action": self._determine_action(result, "general")
            })
        
        return pd.DataFrame(moderation_results)

# Usage
moderator = ContentModerationService("your-api-key")

# Single content moderation
result = moderator.moderate_content("This is sample content to moderate.")
print(f"Action: {result['action']}")

# Bulk moderation
contents = ["Content 1", "Content 2", "Content 3"]
results_df = moderator.moderate_bulk_content(contents)
print(results_df.to_string())
```

### JavaScript/Node.js SDK

#### Installation and Setup

```bash
npm install ai-detector-sdk
```

```javascript
const { AIDetectorClient } = require('ai-detector-sdk');

// Initialize client
const client = new AIDetectorClient({
    baseURL: 'https://api.ai-detector.com',
    apiKey: 'your-api-key',
    timeout: 30000
});

// Configure default options
client.setDefaultOptions({
    method: 'ensemble',
    confidenceThreshold: 0.7,
    returnFeatures: true
});
```

#### Express.js Integration

```javascript
const express = require('express');
const { AIDetectorClient } = require('ai-detector-sdk');
const rateLimit = require('express-rate-limit');

const app = express();
app.use(express.json());

// Initialize AI Detector
const aiDetector = new AIDetectorClient({
    baseURL: process.env.AI_DETECTOR_URL,
    apiKey: process.env.AI_DETECTOR_API_KEY
});

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100 // limit each IP to 100 requests per windowMs
});
app.use('/api/moderate', limiter);

// Content moderation endpoint
app.post('/api/moderate', async (req, res) => {
    try {
        const { content, options = {} } = req.body;
        
        if (!content || content.trim().length === 0) {
            return res.status(400).json({
                error: 'Content is required'
            });
        }
        
        // Detect AI content
        const result = await aiDetector.detect(content, {
            method: options.method || 'ensemble',
            confidenceThreshold: options.threshold || 0.7,
            returnFeatures: true
        });
        
        // Determine moderation action
        const action = determineModerationAction(result);
        
        res.json({
            success: true,
            data: {
                isAiGenerated: result.isAiGenerated,
                confidence: result.confidenceScore,
                action: action,
                reasoning: result.features?.llmFeatures?.reasoning,
                processingTimeMs: result.processingTimeMs
            }
        });
        
    } catch (error) {
        console.error('Moderation error:', error);
        res.status(500).json({
            success: false,
            error: 'Internal server error'
        });
    }
});

function determineModerationAction(result) {
    if (!result.isAiGenerated) {
        return 'approve';
    }
    
    if (result.confidenceScore > 0.9) {
        return 'reject';
    } else if (result.confidenceScore > 0.7) {
        return 'flag_for_review';
    } else {
        return 'approve_with_warning';
    }
}

// Batch moderation endpoint
app.post('/api/moderate/batch', async (req, res) => {
    try {
        const { contents, options = {} } = req.body;
        
        if (!Array.isArray(contents) || contents.length === 0) {
            return res.status(400).json({
                error: 'Contents array is required'
            });
        }
        
        if (contents.length > 100) {
            return res.status(400).json({
                error: 'Maximum 100 contents per batch'
            });
        }
        
        // Prepare batch
        const texts = contents.map((content, index) => ({
            id: index.toString(),
            text: content
        }));
        
        // Process batch
        const batchResult = await aiDetector.detectBatch(texts, {
            method: options.method || 'ensemble',
            parallelProcessing: true
        });
        
        // Process results
        const moderationResults = batchResult.results.map(result => ({
            contentId: parseInt(result.id),
            isAiGenerated: result.isAiGenerated,
            confidence: result.confidenceScore,
            action: determineModerationAction(result)
        }));
        
        res.json({
            success: true,
            data: {
                totalProcessed: batchResult.totalTexts,
                processingTimeMs: batchResult.processingTimeMs,
                results: moderationResults
            }
        });
        
    } catch (error) {
        console.error('Batch moderation error:', error);
        res.status(500).json({
            success: false,
            error: 'Internal server error'
        });
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Content moderation service running on port ${PORT}`);
});
```

## Chrome Extension Integration

### Manifest Configuration

```json
{
  "manifest_version": 3,
  "name": "AI Content Detector",
  "version": "1.0.0",
  "description": "Detect AI-generated content on web pages",
  
  "permissions": [
    "storage",
    "activeTab",
    "scripting"
  ],
  
  "host_permissions": [
    "https://*.ai-detector.com/*"
  ],
  
  "background": {
    "service_worker": "background.js",
    "type": "module"
  },
  
  "content_scripts": [
    {
      "matches": ["<all_urls>"],
      "js": ["content.js"],
      "run_at": "document_idle"
    }
  ],
  
  "action": {
    "default_popup": "popup.html",
    "default_title": "AI Content Detector"
  }
}
```

### Background Script Integration

```javascript
// background.js
import { AIDetectorClient } from './lib/ai-detector.js';

class AIDetectorExtension {
    constructor() {
        this.apiClient = new AIDetectorClient({
            baseURL: 'https://api.ai-detector.com',
            apiKey: this.getApiKey()
        });
        
        this.cache = new Map();
        this.setupMessageHandlers();
    }
    
    async getApiKey() {
        const result = await chrome.storage.sync.get(['apiKey']);
        return result.apiKey;
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
                    
                case 'UPDATE_SETTINGS':
                    await this.updateSettings(message.settings);
                    sendResponse({ success: true });
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
        
        // Make API request
        const result = await this.apiClient.detect(text, {
            method: options.method || 'ensemble',
            confidenceThreshold: options.threshold || 0.7
        });
        
        // Cache result for 5 minutes
        this.cache.set(cacheKey, result);
        setTimeout(() => this.cache.delete(cacheKey), 5 * 60 * 1000);
        
        return result;
    }
    
    async detectPageContent(tabId) {
        // Extract text from page
        const [result] = await chrome.scripting.executeScript({
            target: { tabId },
            function: this.extractPageText
        });
        
        const textElements = result.result;
        
        // Process in batches to avoid overwhelming API
        const batchSize = 10;
        const results = [];
        
        for (let i = 0; i < textElements.length; i += batchSize) {
            const batch = textElements.slice(i, i + batchSize);
            const batchResults = await this.processBatch(batch);
            results.push(...batchResults);
        }
        
        return results;
    }
    
    extractPageText() {
        // This function runs in the page context
        const textElements = [];
        const selectors = ['p', 'div', 'article', 'h1', 'h2', 'h3'];
        
        selectors.forEach(selector => {
            document.querySelectorAll(selector).forEach((element, index) => {
                const text = element.textContent.trim();
                if (text.length > 100 && text.length < 2000) {
                    textElements.push({
                        id: `${selector}_${index}`,
                        text: text,
                        selector: `${selector}:nth-of-type(${index + 1})`
                    });
                }
            });
        });
        
        return textElements.slice(0, 20); // Limit to 20 elements
    }
    
    async processBatch(textElements) {
        const texts = textElements.map(element => ({
            id: element.id,
            text: element.text
        }));
        
        const batchResult = await this.apiClient.detectBatch(texts);
        
        // Combine with original element data
        return batchResult.results.map(result => ({
            ...result,
            selector: textElements.find(el => el.id === result.id)?.selector
        }));
    }
    
    generateCacheKey(text, options) {
        return `${text.substring(0, 50)}_${JSON.stringify(options)}`;
    }
    
    async getSettings() {
        const result = await chrome.storage.sync.get([
            'autoDetect',
            'confidenceThreshold',
            'detectionMethod',
            'showIndicators',
            'apiKey'
        ]);
        
        return {
            autoDetect: result.autoDetect ?? true,
            confidenceThreshold: result.confidenceThreshold ?? 0.7,
            detectionMethod: result.detectionMethod ?? 'ensemble',
            showIndicators: result.showIndicators ?? true,
            apiKey: result.apiKey || ''
        };
    }
    
    async updateSettings(settings) {
        await chrome.storage.sync.set(settings);
        
        // Update API key if changed
        if (settings.apiKey) {
            this.apiClient.updateApiKey(settings.apiKey);
        }
    }
}

// Initialize extension
const aiDetectorExtension = new AIDetectorExtension();
```

### Content Script Integration

```javascript
// content.js
class AIDetectorContent {
    constructor() {
        this.indicators = new Map();
        this.settings = null;
        this.observer = null;
        this.init();
    }
    
    async init() {
        this.settings = await this.getSettings();
        
        if (this.settings.autoDetect) {
            this.startAutoDetection();
        }
        
        this.setupObserver();
        this.setupMessageHandlers();
    }
    
    async getSettings() {
        return new Promise((resolve) => {
            chrome.runtime.sendMessage({ type: 'GET_SETTINGS' }, (response) => {
                resolve(response.success ? response.data : {});
            });
        });
    }
    
    startAutoDetection() {
        // Wait for page to be fully loaded
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => {
                setTimeout(() => this.detectPageContent(), 1000);
            });
        } else {
            setTimeout(() => this.detectPageContent(), 1000);
        }
    }
    
    async detectPageContent() {
        chrome.runtime.sendMessage(
            { type: 'DETECT_PAGE_CONTENT' },
            (response) => {
                if (response.success && this.settings.showIndicators) {
                    this.displayResults(response.data);
                }
            }
        );
    }
    
    displayResults(results) {
        results.forEach(result => {
            if (result.isAiGenerated && result.confidenceScore >= this.settings.confidenceThreshold) {
                this.addIndicator(result);
            }
        });
    }
    
    addIndicator(result) {
        try {
            const element = document.querySelector(result.selector);
            if (!element || this.indicators.has(element)) {
                return;
            }
            
            const indicator = this.createIndicator(result);
            this.positionIndicator(indicator, element);
            
            this.indicators.set(element, indicator);
            
            // Auto-remove after 30 seconds
            setTimeout(() => {
                this.removeIndicator(element);
            }, 30000);
            
        } catch (error) {
            console.debug('Failed to add indicator:', error);
        }
    }
    
    createIndicator(result) {
        const indicator = document.createElement('div');
        indicator.className = 'ai-detector-indicator';
        indicator.innerHTML = '🤖';
        indicator.title = `AI Detected (${(result.confidenceScore * 100).toFixed(1)}% confidence)`;
        
        indicator.style.cssText = `
            position: absolute;
            width: 20px;
            height: 20px;
            background: linear-gradient(45deg, #ff4444, #ff6666);
            border: 2px solid white;
            border-radius: 50%;
            color: white;
            font-size: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            cursor: pointer;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            transition: transform 0.2s ease;
        `;
        
        // Add hover effects
        indicator.addEventListener('mouseenter', () => {
            indicator.style.transform = 'scale(1.2)';
        });
        
        indicator.addEventListener('mouseleave', () => {
            indicator.style.transform = 'scale(1)';
        });
        
        // Add click handler for detailed view
        indicator.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.showDetailedResult(result);
        });
        
        return indicator;
    }
    
    positionIndicator(indicator, element) {
        const rect = element.getBoundingClientRect();
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
        
        indicator.style.top = `${rect.top + scrollTop - 10}px`;
        indicator.style.left = `${rect.right + scrollLeft - 10}px`;
        
        document.body.appendChild(indicator);
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
            border: 1px solid #ccc;
            border-radius: 8px;
            padding: 20px;
            max-width: 400px;
            z-index: 10001;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        `;
        
        modal.innerHTML = `
            <div style="margin-bottom: 15px;">
                <h3 style="margin: 0 0 10px 0; color: #333;">AI Detection Result</h3>
                <p style="margin: 5px 0;"><strong>Status:</strong> ${result.isAiGenerated ? 'AI Generated' : 'Human Written'}</p>
                <p style="margin: 5px 0;"><strong>Confidence:</strong> ${(result.confidenceScore * 100).toFixed(1)}%</p>
                <p style="margin: 5px 0;"><strong>Method:</strong> ${result.detectionMethod || 'ensemble'}</p>
            </div>
            <div style="text-align: right;">
                <button id="closeModal" style="
                    background: #007cba;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    cursor: pointer;
                ">Close</button>
            </div>
        `;
        
        modal.querySelector('#closeModal').addEventListener('click', () => {
            document.body.removeChild(modal);
        });
        
        document.body.appendChild(modal);
        
        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (modal.parentNode) {
                document.body.removeChild(modal);
            }
        }, 10000);
    }
    
    setupObserver() {
        this.observer = new MutationObserver((mutations) => {
            if (this.settings.autoDetect) {
                const hasNewContent = mutations.some(mutation => 
                    mutation.type === 'childList' && mutation.addedNodes.length > 0
                );
                
                if (hasNewContent) {
                    // Debounce content detection
                    clearTimeout(this.detectTimeout);
                    this.detectTimeout = setTimeout(() => {
                        this.detectPageContent();
                    }, 2000);
                }
            }
        });
        
        this.observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
    
    setupMessageHandlers() {
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            switch (message.type) {
                case 'SETTINGS_UPDATED':
                    this.settings = message.settings;
                    if (!this.settings.showIndicators) {
                        this.clearAllIndicators();
                    }
                    break;
                    
                case 'CLEAR_INDICATORS':
                    this.clearAllIndicators();
                    break;
                    
                case 'DETECT_CURRENT_PAGE':
                    this.detectPageContent();
                    break;
            }
        });
    }
    
    clearAllIndicators() {
        for (const [element, indicator] of this.indicators) {
            if (indicator.parentNode) {
                indicator.parentNode.removeChild(indicator);
            }
        }
        this.indicators.clear();
    }
    
    destroy() {
        if (this.observer) {
            this.observer.disconnect();
        }
        this.clearAllIndicators();
    }
}

// Initialize content script
const aiDetectorContent = new AIDetectorContent();

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    aiDetectorContent.destroy();
});
```

## Webhook Integration

### Setting Up Webhooks

```python
from flask import Flask, request, jsonify
import hmac
import hashlib
from ai_detector_sdk import AIDetectorClient

app = Flask(__name__)

# Initialize AI Detector client
ai_detector = AIDetectorClient(api_key="your-api-key")

# Webhook configuration
WEBHOOK_SECRET = "your-webhook-secret"

def verify_signature(payload, signature, secret):
    """Verify webhook signature."""
    expected_signature = hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(f"sha256={expected_signature}", signature)

@app.route('/webhook/content-created', methods=['POST'])
def handle_content_created():
    """Handle content creation webhook."""
    try:
        # Verify signature
        signature = request.headers.get('X-Signature-256')
        if not verify_signature(request.data, signature, WEBHOOK_SECRET):
            return jsonify({'error': 'Invalid signature'}), 401
        
        # Parse webhook payload
        data = request.json
        content_id = data.get('content_id')
        content_text = data.get('content')
        content_type = data.get('type', 'post')
        
        if not content_text:
            return jsonify({'error': 'No content provided'}), 400
        
        # Detect AI content
        result = ai_detector.detect(content_text, {
            'method': 'ensemble',
            'confidence_threshold': 0.7,
            'return_features': True
        })
        
        # Determine action based on result
        action = determine_moderation_action(result, content_type)
        
        # Update content status (pseudo-code)
        update_content_status(content_id, {
            'ai_detected': result.is_ai_generated,
            'confidence': result.confidence_score,
            'action': action,
            'reasoning': result.features.get('llm_features', {}).get('reasoning', '')
        })
        
        # Send notification if flagged
        if action in ['reject', 'flag_for_review']:
            send_moderation_notification(content_id, action, result)
        
        return jsonify({
            'success': True,
            'content_id': content_id,
            'action': action,
            'ai_detected': result.is_ai_generated,
            'confidence': result.confidence_score
        })
        
    except Exception as e:
        app.logger.error(f"Webhook processing error: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def determine_moderation_action(result, content_type):
    """Determine moderation action based on detection result and content type."""
    if not result.is_ai_generated:
        return 'approve'
    
    # Stricter rules for certain content types
    if content_type in ['review', 'testimonial', 'news']:
        if result.confidence_score > 0.6:
            return 'reject'
        else:
            return 'flag_for_review'
    
    # Standard rules
    if result.confidence_score > 0.9:
        return 'reject'
    elif result.confidence_score > 0.7:
        return 'flag_for_review'
    else:
        return 'approve_with_warning'

def update_content_status(content_id, moderation_data):
    """Update content status in database."""
    # Implementation depends on your database setup
    pass

def send_moderation_notification(content_id, action, result):
    """Send notification to moderation team."""
    # Implementation depends on your notification system
    pass

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

### Registering Webhooks

```python
import requests

def register_webhook(webhook_url, events, secret):
    """Register webhook with AI Detector service."""
    response = requests.post(
        'https://api.ai-detector.com/api/webhooks',
        headers={
            'Authorization': 'Bearer your-api-key',
            'Content-Type': 'application/json'
        },
        json={
            'url': webhook_url,
            'events': events,
            'secret': secret,
            'active': True
        }
    )
    
    return response.json()

# Register webhook for detection events
webhook_config = register_webhook(
    webhook_url='https://your-domain.com/webhook/ai-detection',
    events=['detection.completed', 'batch.completed'],
    secret='your-webhook-secret'
)

print(f"Webhook registered with ID: {webhook_config['id']}")
```

## Batch Processing Integration

### Large-Scale Content Processing

```python
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ai_detector_sdk import AIDetectorClient

class BatchContentProcessor:
    def __init__(self, api_key, max_workers=10, batch_size=100):
        self.client = AIDetectorClient(api_key=api_key)
        self.max_workers = max_workers
        self.batch_size = batch_size
    
    def process_csv_file(self, file_path, text_column, output_path=None):
        """Process content from CSV file."""
        # Read CSV
        df = pd.read_csv(file_path)
        
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in CSV")
        
        # Process in batches
        results = []
        for i in range(0, len(df), self.batch_size):
            batch_df = df.iloc[i:i + self.batch_size]
            batch_results = self._process_batch(batch_df, text_column)
            results.extend(batch_results)
            
            print(f"Processed {min(i + self.batch_size, len(df))}/{len(df)} rows")
        
        # Add results to dataframe
        result_df = df.copy()
        result_df['ai_detected'] = [r['is_ai_generated'] for r in results]
        result_df['confidence_score'] = [r['confidence_score'] for r in results]
        result_df['detection_method'] = [r['detection_method'] for r in results]
        
        # Save results
        if output_path:
            result_df.to_csv(output_path, index=False)
        
        return result_df
    
    def _process_batch(self, batch_df, text_column):
        """Process a batch of texts."""
        texts = [
            {"id": str(idx), "text": str(text)}
            for idx, text in batch_df[text_column].items()
            if pd.notna(text) and len(str(text).strip()) > 0
        ]
        
        if not texts:
            return []
        
        batch_result = self.client.detect_batch(texts, {
            'method': 'ensemble',
            'parallel_processing': True
        })
        
        return batch_result.results
    
    async def process_async_batch(self, texts):
        """Process batch asynchronously for better performance."""
        loop = asyncio.get_event_loop()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            # Split into smaller batches for parallel processing
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                future = loop.run_in_executor(executor, self._process_batch_sync, batch)
                futures.append(future)
            
            # Wait for all batches to complete
            batch_results = await asyncio.gather(*futures)
            
            # Flatten results
            all_results = []
            for batch_result in batch_results:
                all_results.extend(batch_result)
            
            return all_results
    
    def _process_batch_sync(self, texts):
        """Synchronous batch processing for thread executor."""
        batch_texts = [
            {"id": str(i), "text": text}
            for i, text in enumerate(texts)
        ]
        
        result = self.client.detect_batch(batch_texts)
        return result.results

# Usage example
processor = BatchContentProcessor(
    api_key="your-api-key",
    max_workers=5,
    batch_size=50
)

# Process CSV file
result_df = processor.process_csv_file(
    file_path="content_data.csv",
    text_column="content",
    output_path="content_with_ai_detection.csv"
)

print(f"Processed {len(result_df)} rows")
print(f"AI content detected: {result_df['ai_detected'].sum()}")
```

### Database Integration

```python
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

class DatabaseBatchProcessor:
    def __init__(self, database_url, ai_detector_client):
        self.engine = sa.create_engine(database_url)
        self.Session = sessionmaker(bind=self.engine)
        self.client = ai_detector_client
    
    @contextmanager
    def get_session(self):
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def process_pending_content(self, batch_size=100):
        """Process content that hasn't been analyzed yet."""
        with self.get_session() as session:
            # Query pending content
            pending_content = session.execute(
                sa.text("""
                    SELECT id, content_text
                    FROM content_table
                    WHERE ai_analysis_status = 'pending'
                    LIMIT :batch_size
                """),
                {"batch_size": batch_size}
            ).fetchall()
            
            if not pending_content:
                return 0
            
            # Prepare batch for AI detection
            texts = [
                {"id": str(item.id), "text": item.content_text}
                for item in pending_content
            ]
            
            # Process batch
            results = self.client.detect_batch(texts)
            
            # Update database with results
            for result in results.results:
                content_id = int(result.id)
                session.execute(
                    sa.text("""
                        UPDATE content_table
                        SET ai_analysis_status = 'completed',
                            is_ai_generated = :is_ai,
                            ai_confidence = :confidence,
                            detection_method = :method,
                            analyzed_at = NOW()
                        WHERE id = :content_id
                    """),
                    {
                        "content_id": content_id,
                        "is_ai": result.is_ai_generated,
                        "confidence": result.confidence_score,
                        "method": result.detection_method
                    }
                )
            
            return len(results.results)
    
    def process_all_pending(self, batch_size=100):
        """Process all pending content in batches."""
        total_processed = 0
        
        while True:
            processed = self.process_pending_content(batch_size)
            if processed == 0:
                break
            
            total_processed += processed
            print(f"Processed {processed} items (total: {total_processed})")
        
        return total_processed

# Usage
from ai_detector_sdk import AIDetectorClient

client = AIDetectorClient(api_key="your-api-key")
processor = DatabaseBatchProcessor(
    database_url="postgresql://user:pass@localhost/db",
    ai_detector_client=client
)

total = processor.process_all_pending(batch_size=50)
print(f"Total items processed: {total}")
```

This comprehensive integration guide provides detailed examples for all major integration patterns with the AI Detector system. Each section includes production-ready code examples, error handling, and best practices for different use cases and platforms.

---

*Last updated: January 15, 2025*