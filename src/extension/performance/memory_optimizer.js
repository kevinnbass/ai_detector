/**
 * Memory optimizer for Chrome extension to maintain <50MB memory usage.
 * 
 * Implements memory monitoring, cache management, DOM optimization,
 * and garbage collection strategies for efficient extension operation.
 */

class MemoryOptimizer {
    constructor() {
        this.logger = new Logger('MemoryOptimizer');
        this.metrics = new MetricsCollector();
        
        // Memory configuration
        this.config = {
            maxMemoryMB: 45, // Target below 50MB limit
            cacheMaxItems: 1000,
            cacheMaxAgeMins: 30,
            gcIntervalMins: 5,
            memoryCheckIntervalMins: 2,
            domObserverThrottleMs: 1000,
            messageQueueMaxSize: 100
        };
        
        // Memory tracking
        this.memoryStats = {
            peak: 0,
            current: 0,
            collections: 0,
            cacheHits: 0,
            cacheMisses: 0
        };
        
        // Cache storage with size limits
        this.caches = {
            detectionResults: new LRUCache(this.config.cacheMaxItems),
            domElements: new WeakMap(),
            apiResponses: new Map(),
            patterns: new Map()
        };
        
        // Memory monitoring
        this.memoryMonitor = null;
        this.gcTimer = null;
        
        // Initialize optimization
        this.initialize();
    }
    
    async initialize() {
        try {
            // Start memory monitoring
            this.startMemoryMonitoring();
            
            // Start garbage collection timer
            this.startGarbageCollection();
            
            // Optimize existing caches
            await this.optimizeCaches();
            
            // Setup memory-efficient event listeners
            this.setupOptimizedListeners();
            
            this.logger.info('Memory optimizer initialized');
        } catch (error) {
            this.logger.error('Failed to initialize memory optimizer:', error);
        }
    }
    
    startMemoryMonitoring() {
        this.memoryMonitor = setInterval(async () => {
            try {
                const memoryInfo = await this.getMemoryUsage();
                this.memoryStats.current = memoryInfo.usedJSHeapSize / (1024 * 1024);
                
                if (this.memoryStats.current > this.memoryStats.peak) {
                    this.memoryStats.peak = this.memoryStats.current;
                }
                
                // Record metrics
                this.metrics.gauge('extension_memory_mb', this.memoryStats.current);
                this.metrics.gauge('extension_memory_peak_mb', this.memoryStats.peak);
                
                // Check if memory usage is too high
                if (this.memoryStats.current > this.config.maxMemoryMB) {
                    this.logger.warn(`Memory usage high: ${this.memoryStats.current.toFixed(1)}MB`);
                    await this.performEmergencyCleanup();
                }
                
                // Log memory status periodically
                if (this.memoryStats.collections % 10 === 0) {
                    this.logger.debug(`Memory: ${this.memoryStats.current.toFixed(1)}MB (peak: ${this.memoryStats.peak.toFixed(1)}MB)`);
                }
                
            } catch (error) {
                this.logger.error('Memory monitoring failed:', error);
            }
        }, this.config.memoryCheckIntervalMins * 60 * 1000);
    }
    
    startGarbageCollection() {
        this.gcTimer = setInterval(async () => {
            try {
                await this.performGarbageCollection();
                this.memoryStats.collections++;
            } catch (error) {
                this.logger.error('Garbage collection failed:', error);
            }
        }, this.config.gcIntervalMins * 60 * 1000);
    }
    
    async getMemoryUsage() {
        if (performance.memory) {
            return {
                usedJSHeapSize: performance.memory.usedJSHeapSize,
                totalJSHeapSize: performance.memory.totalJSHeapSize,
                jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
            };
        }
        
        // Fallback estimation
        return {
            usedJSHeapSize: this.estimateMemoryUsage(),
            totalJSHeapSize: 0,
            jsHeapSizeLimit: 0
        };
    }
    
    estimateMemoryUsage() {
        // Estimate memory usage based on cache sizes and DOM elements
        let estimate = 0;
        
        // Cache memory estimation
        estimate += this.caches.detectionResults.size * 1024; // ~1KB per detection result
        estimate += this.caches.apiResponses.size * 2048; // ~2KB per API response
        estimate += this.caches.patterns.size * 512; // ~512B per pattern
        
        // DOM elements estimation
        const domElements = document.querySelectorAll('*').length;
        estimate += domElements * 100; // ~100B per DOM element tracked
        
        return estimate;
    }
    
    async performGarbageCollection() {
        const startTime = performance.now();
        let memoryFreed = 0;
        
        try {
            // Clean expired cache entries
            memoryFreed += await this.cleanExpiredCaches();
            
            // Clean DOM references
            memoryFreed += await this.cleanDOMReferences();
            
            // Clean message queues
            memoryFreed += await this.cleanMessageQueues();
            
            // Clean event listeners
            memoryFreed += await this.cleanEventListeners();
            
            // Force garbage collection if available
            if (window.gc && typeof window.gc === 'function') {
                window.gc();
            }
            
            const duration = performance.now() - startTime;
            
            this.metrics.histogram('gc_duration_ms', duration);
            this.metrics.counter('gc_collections_total');
            this.metrics.gauge('gc_memory_freed_kb', memoryFreed / 1024);
            
            this.logger.debug(`GC completed: ${memoryFreed / 1024}KB freed in ${duration.toFixed(1)}ms`);
            
        } catch (error) {
            this.logger.error('Garbage collection error:', error);
        }
    }
    
    async cleanExpiredCaches() {
        let memoryFreed = 0;
        const now = Date.now();
        const maxAge = this.config.cacheMaxAgeMins * 60 * 1000;
        
        // Clean detection results cache
        const detectionCache = this.caches.detectionResults;
        const initialSize = detectionCache.size;
        
        for (const [key, value] of detectionCache.entries()) {
            if (now - value.timestamp > maxAge) {
                detectionCache.delete(key);
                memoryFreed += this.estimateObjectSize(value);
            }
        }
        
        // Clean API responses cache
        for (const [key, value] of this.caches.apiResponses) {
            if (now - value.timestamp > maxAge) {
                this.caches.apiResponses.delete(key);
                memoryFreed += this.estimateObjectSize(value);
            }
        }
        
        // Limit cache size if too large
        if (detectionCache.size > this.config.cacheMaxItems) {
            const excess = detectionCache.size - this.config.cacheMaxItems;
            const oldestKeys = Array.from(detectionCache.keys()).slice(0, excess);
            
            for (const key of oldestKeys) {
                const value = detectionCache.get(key);
                detectionCache.delete(key);
                memoryFreed += this.estimateObjectSize(value);
            }
        }
        
        this.logger.debug(`Cache cleanup: ${initialSize - detectionCache.size} items removed`);
        return memoryFreed;
    }
    
    async cleanDOMReferences() {
        let memoryFreed = 0;
        
        // Remove references to detached DOM elements
        const elementsToRemove = [];
        
        for (const [element] of this.caches.domElements) {
            if (!document.contains(element)) {
                elementsToRemove.push(element);
                memoryFreed += 100; // Estimated size per DOM reference
            }
        }
        
        elementsToRemove.forEach(element => {
            this.caches.domElements.delete(element);
        });
        
        // Clean up any stored selectors for non-existent elements
        const selectors = this.getStoredSelectors();
        const validSelectors = selectors.filter(selector => {
            try {
                return document.querySelector(selector) !== null;
            } catch {
                return false;
            }
        });
        
        if (validSelectors.length < selectors.length) {
            this.storeSelectors(validSelectors);
            memoryFreed += (selectors.length - validSelectors.length) * 50;
        }
        
        return memoryFreed;
    }
    
    async cleanMessageQueues() {
        let memoryFreed = 0;
        
        // Clean background script message queue
        if (chrome.runtime && chrome.runtime.getBackgroundPage) {
            try {
                const background = await chrome.runtime.getBackgroundPage();
                if (background && background.messageQueue) {
                    const queue = background.messageQueue;
                    if (queue.length > this.config.messageQueueMaxSize) {
                        const excess = queue.length - this.config.messageQueueMaxSize;
                        const removed = queue.splice(0, excess);
                        memoryFreed += removed.length * 500; // Estimated size per message
                    }
                }
            } catch (error) {
                // Background page might not be available
            }
        }
        
        return memoryFreed;
    }
    
    async cleanEventListeners() {
        let memoryFreed = 0;
        
        // Remove unused event listeners
        const eventTargets = document.querySelectorAll('[data-ai-detector-listener]');
        
        for (const element of eventTargets) {
            if (!this.isElementVisible(element)) {
                // Remove listeners from invisible elements
                element.removeAttribute('data-ai-detector-listener');
                this.removeElementListeners(element);
                memoryFreed += 200; // Estimated size per listener set
            }
        }
        
        return memoryFreed;
    }
    
    async performEmergencyCleanup() {
        this.logger.warn('Performing emergency memory cleanup');
        
        try {
            // Aggressive cache clearing
            this.caches.detectionResults.clear();
            this.caches.apiResponses.clear();
            
            // Clear all non-essential data
            this.clearNonEssentialData();
            
            // Force garbage collection
            await this.performGarbageCollection();
            
            // Reduce monitoring frequency temporarily
            this.reduceMonitoringFrequency();
            
        } catch (error) {
            this.logger.error('Emergency cleanup failed:', error);
        }
    }
    
    async optimizeCaches() {
        // Use more efficient cache structures
        this.caches.detectionResults = new MemoryEfficientCache({
            maxSize: this.config.cacheMaxItems,
            maxAge: this.config.cacheMaxAgeMins * 60 * 1000,
            estimateSize: this.estimateObjectSize.bind(this)
        });
        
        // Pre-populate with essential patterns
        this.preloadEssentialPatterns();
    }
    
    preloadEssentialPatterns() {
        const essentialPatterns = [
            { pattern: /\b(?:furthermore|however|therefore)\b/gi, weight: 0.8 },
            { pattern: /\b(?:comprehensive|multifaceted|paradigm)\b/gi, weight: 0.9 },
            { pattern: /[😀-🙏]/g, weight: -0.9 }
        ];
        
        essentialPatterns.forEach((item, index) => {
            this.caches.patterns.set(`essential_${index}`, item);
        });
    }
    
    setupOptimizedListeners() {
        // Use throttled observers for better memory efficiency
        this.setupThrottledDOMObserver();
        this.setupOptimizedMessageListeners();
        this.setupMemoryEfficientStorageListeners();
    }
    
    setupThrottledDOMObserver() {
        let throttleTimer = null;
        
        const observer = new MutationObserver((mutations) => {
            if (throttleTimer) return;
            
            throttleTimer = setTimeout(() => {
                this.handleDOMChanges(mutations);
                throttleTimer = null;
            }, this.config.domObserverThrottleMs);
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: false,
            characterData: false
        });
        
        // Store weak reference to avoid memory leaks
        this.domObserver = new WeakRef(observer);
    }
    
    setupOptimizedMessageListeners() {
        // Use single message listener with routing
        const messageHandler = (message, sender, sendResponse) => {
            try {
                this.routeMessage(message, sender, sendResponse);
            } catch (error) {
                this.logger.error('Message handling error:', error);
                sendResponse({ error: error.message });
            }
        };
        
        chrome.runtime.onMessage.addListener(messageHandler);
        
        // Store reference for cleanup
        this.messageHandler = messageHandler;
    }
    
    setupMemoryEfficientStorageListeners() {
        // Batch storage operations to reduce memory overhead
        let pendingWrites = new Map();
        let writeTimer = null;
        
        const batchedWrite = () => {
            if (pendingWrites.size > 0) {
                const data = Object.fromEntries(pendingWrites);
                chrome.storage.local.set(data);
                pendingWrites.clear();
            }
            writeTimer = null;
        };
        
        this.queueStorageWrite = (key, value) => {
            pendingWrites.set(key, value);
            
            if (!writeTimer) {
                writeTimer = setTimeout(batchedWrite, 100);
            }
        };
    }
    
    handleDOMChanges(mutations) {
        // Process mutations efficiently
        const addedNodes = [];
        const removedNodes = [];
        
        for (const mutation of mutations) {
            if (mutation.type === 'childList') {
                addedNodes.push(...Array.from(mutation.addedNodes));
                removedNodes.push(...Array.from(mutation.removedNodes));
            }
        }
        
        // Clean up references to removed nodes
        removedNodes.forEach(node => {
            if (node.nodeType === Node.ELEMENT_NODE) {
                this.cleanupElementReferences(node);
            }
        });
        
        // Efficiently process added nodes
        this.processAddedNodes(addedNodes);
    }
    
    cleanupElementReferences(element) {
        // Remove from DOM cache
        this.caches.domElements.delete(element);
        
        // Clean up any stored data
        element.removeAttribute('data-ai-detector-processed');
        element.removeAttribute('data-ai-detector-listener');
        
        // Clean up child elements
        const children = element.querySelectorAll('[data-ai-detector-processed]');
        children.forEach(child => {
            this.caches.domElements.delete(child);
        });
    }
    
    processAddedNodes(nodes) {
        // Process nodes in batches to avoid blocking
        const batchSize = 50;
        let index = 0;
        
        const processBatch = () => {
            const batch = nodes.slice(index, index + batchSize);
            
            batch.forEach(node => {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    this.processElement(node);
                }
            });
            
            index += batchSize;
            
            if (index < nodes.length) {
                // Use requestIdleCallback if available
                if (window.requestIdleCallback) {
                    requestIdleCallback(processBatch);
                } else {
                    setTimeout(processBatch, 0);
                }
            }
        };
        
        processBatch();
    }
    
    processElement(element) {
        // Lightweight element processing
        if (this.isTextElement(element)) {
            // Store minimal reference
            this.caches.domElements.set(element, {
                processed: Date.now(),
                selector: this.generateLightweightSelector(element)
            });
        }
    }
    
    isTextElement(element) {
        const textElements = ['P', 'DIV', 'SPAN', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6'];
        return textElements.includes(element.tagName) && 
               element.textContent.trim().length > 20;
    }
    
    generateLightweightSelector(element) {
        // Generate minimal selector for memory efficiency
        if (element.id) {
            return `#${element.id}`;
        }
        
        if (element.className) {
            const classes = element.className.split(' ').slice(0, 2).join('.');
            return `.${classes}`;
        }
        
        return element.tagName.toLowerCase();
    }
    
    // Memory-efficient cache implementation
    cacheDetectionResult(key, result) {
        // Compress result before caching
        const compressed = this.compressResult(result);
        
        this.caches.detectionResults.set(key, {
            data: compressed,
            timestamp: Date.now(),
            size: this.estimateObjectSize(compressed)
        });
        
        this.memoryStats.cacheMisses++;
    }
    
    getCachedDetectionResult(key) {
        const cached = this.caches.detectionResults.get(key);
        
        if (cached) {
            this.memoryStats.cacheHits++;
            return this.decompressResult(cached.data);
        }
        
        return null;
    }
    
    compressResult(result) {
        // Remove non-essential fields for caching
        return {
            isAI: result.is_ai_generated,
            confidence: Math.round(result.confidence_score * 100) / 100,
            method: result.method_used
        };
    }
    
    decompressResult(compressed) {
        // Reconstruct full result
        return {
            is_ai_generated: compressed.isAI,
            confidence_score: compressed.confidence,
            method_used: compressed.method,
            from_cache: true
        };
    }
    
    estimateObjectSize(obj) {
        // Rough estimation of object memory usage
        if (typeof obj === 'string') {
            return obj.length * 2; // 2 bytes per character
        }
        
        if (typeof obj === 'number') {
            return 8; // 8 bytes for number
        }
        
        if (typeof obj === 'boolean') {
            return 1;
        }
        
        if (Array.isArray(obj)) {
            return obj.reduce((sum, item) => sum + this.estimateObjectSize(item), 0);
        }
        
        if (typeof obj === 'object' && obj !== null) {
            return Object.entries(obj).reduce((sum, [key, value]) => {
                return sum + key.length * 2 + this.estimateObjectSize(value);
            }, 0);
        }
        
        return 0;
    }
    
    isElementVisible(element) {
        const rect = element.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0 && 
               rect.top < window.innerHeight && rect.bottom > 0;
    }
    
    removeElementListeners(element) {
        // Remove all AI detector related listeners
        const events = ['click', 'mouseover', 'focus'];
        events.forEach(event => {
            element.removeEventListener(event, this.handleElementEvent);
        });
    }
    
    getStoredSelectors() {
        // Get selectors from storage (mock implementation)
        return JSON.parse(localStorage.getItem('ai-detector-selectors') || '[]');
    }
    
    storeSelectors(selectors) {
        // Store selectors efficiently
        localStorage.setItem('ai-detector-selectors', JSON.stringify(selectors));
    }
    
    clearNonEssentialData() {
        // Clear all non-essential cached data
        localStorage.removeItem('ai-detector-stats');
        localStorage.removeItem('ai-detector-history');
        
        // Keep only essential patterns
        const essential = new Map();
        for (const [key, value] of this.caches.patterns) {
            if (key.startsWith('essential_')) {
                essential.set(key, value);
            }
        }
        this.caches.patterns = essential;
    }
    
    reduceMonitoringFrequency() {
        // Temporarily reduce monitoring to save memory
        if (this.memoryMonitor) {
            clearInterval(this.memoryMonitor);
            this.memoryMonitor = setInterval(async () => {
                await this.getMemoryUsage();
            }, this.config.memoryCheckIntervalMins * 2 * 60 * 1000);
        }
        
        // Restore normal frequency after 5 minutes
        setTimeout(() => {
            this.startMemoryMonitoring();
        }, 5 * 60 * 1000);
    }
    
    routeMessage(message, sender, sendResponse) {
        // Efficient message routing without creating closures
        switch (message.type) {
            case 'GET_MEMORY_STATS':
                sendResponse(this.getMemoryStats());
                break;
            case 'TRIGGER_GC':
                this.performGarbageCollection().then(() => {
                    sendResponse({ success: true });
                });
                break;
            case 'CLEAR_CACHE':
                this.clearCaches();
                sendResponse({ success: true });
                break;
            default:
                sendResponse({ error: 'Unknown message type' });
        }
    }
    
    getMemoryStats() {
        return {
            ...this.memoryStats,
            cacheStats: {
                detectionResults: this.caches.detectionResults.size,
                apiResponses: this.caches.apiResponses.size,
                patterns: this.caches.patterns.size,
                domElements: this.caches.domElements.size || 0
            },
            config: this.config
        };
    }
    
    clearCaches() {
        this.caches.detectionResults.clear();
        this.caches.apiResponses.clear();
        
        // Keep essential patterns
        const essential = new Map();
        for (const [key, value] of this.caches.patterns) {
            if (key.startsWith('essential_')) {
                essential.set(key, value);
            }
        }
        this.caches.patterns = essential;
    }
    
    destroy() {
        // Clean up all resources
        if (this.memoryMonitor) {
            clearInterval(this.memoryMonitor);
        }
        
        if (this.gcTimer) {
            clearInterval(this.gcTimer);
        }
        
        if (this.domObserver) {
            const observer = this.domObserver.deref();
            if (observer) {
                observer.disconnect();
            }
        }
        
        if (this.messageHandler) {
            chrome.runtime.onMessage.removeListener(this.messageHandler);
        }
        
        this.clearCaches();
    }
}

// Memory-efficient LRU Cache implementation
class LRUCache {
    constructor(maxSize) {
        this.maxSize = maxSize;
        this.cache = new Map();
    }
    
    get(key) {
        if (this.cache.has(key)) {
            // Move to end (most recently used)
            const value = this.cache.get(key);
            this.cache.delete(key);
            this.cache.set(key, value);
            return value;
        }
        return undefined;
    }
    
    set(key, value) {
        if (this.cache.has(key)) {
            // Update existing
            this.cache.delete(key);
        } else if (this.cache.size >= this.maxSize) {
            // Remove least recently used
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        
        this.cache.set(key, value);
    }
    
    delete(key) {
        return this.cache.delete(key);
    }
    
    clear() {
        this.cache.clear();
    }
    
    get size() {
        return this.cache.size;
    }
    
    entries() {
        return this.cache.entries();
    }
    
    keys() {
        return this.cache.keys();
    }
}

// Memory-efficient cache with size estimation
class MemoryEfficientCache extends LRUCache {
    constructor(options) {
        super(options.maxSize);
        this.maxAge = options.maxAge;
        this.estimateSize = options.estimateSize;
        this.currentSize = 0;
        this.maxMemorySize = options.maxMemorySize || 10 * 1024 * 1024; // 10MB default
    }
    
    set(key, value) {
        const now = Date.now();
        const estimatedSize = this.estimateSize(value);
        
        // Check if we need to make space
        while (this.currentSize + estimatedSize > this.maxMemorySize && this.cache.size > 0) {
            this.evictOldest();
        }
        
        const wrappedValue = {
            data: value,
            timestamp: now,
            size: estimatedSize
        };
        
        if (this.cache.has(key)) {
            const old = this.cache.get(key);
            this.currentSize -= old.size;
        }
        
        super.set(key, wrappedValue);
        this.currentSize += estimatedSize;
    }
    
    get(key) {
        const wrapped = super.get(key);
        if (!wrapped) return undefined;
        
        // Check if expired
        if (Date.now() - wrapped.timestamp > this.maxAge) {
            this.delete(key);
            return undefined;
        }
        
        return wrapped.data;
    }
    
    delete(key) {
        const wrapped = this.cache.get(key);
        if (wrapped) {
            this.currentSize -= wrapped.size;
        }
        return super.delete(key);
    }
    
    evictOldest() {
        const firstKey = this.cache.keys().next().value;
        if (firstKey) {
            this.delete(firstKey);
        }
    }
    
    clear() {
        super.clear();
        this.currentSize = 0;
    }
}

// Export for use in extension
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MemoryOptimizer;
} else {
    window.MemoryOptimizer = MemoryOptimizer;
}