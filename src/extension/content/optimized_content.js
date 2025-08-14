/**
 * Optimized content script for minimal memory footprint and DOM impact.
 * 
 * Implements lazy loading, efficient DOM observation, minimal visual indicators,
 * and aggressive cleanup to maintain <50MB total extension memory.
 */

class OptimizedContentScript {
    constructor() {
        this.logger = new Logger('OptimizedContent');
        this.metrics = new MetricsCollector();
        
        // Memory optimization
        this.memoryOptimizer = new MemoryOptimizer();
        
        // Configuration
        this.config = {
            maxProcessedElements: 500,
            observerThrottleMs: 2000,
            batchProcessingSize: 20,
            cleanupIntervalMs: 60000,
            maxVisualIndicators: 50,
            lazyLoadThreshold: 100
        };
        
        // Element tracking with weak references
        this.processedElements = new WeakSet();
        this.visualIndicators = new Map();
        this.pendingElements = [];
        
        // Observers and timers
        this.mutationObserver = null;
        this.intersectionObserver = null;
        this.cleanupTimer = null;
        this.processingTimer = null;
        
        // Throttling
        this.lastProcessTime = 0;
        this.processingQueue = [];
        
        // Initialize optimized content script
        this.initialize();
    }
    
    async initialize() {
        try {
            // Wait for DOM to be ready
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', () => this.initializeAfterLoad());
            } else {
                this.initializeAfterLoad();
            }
            
        } catch (error) {
            this.logger.error('Content script initialization failed:', error);
        }
    }
    
    async initializeAfterLoad() {
        // Setup optimized DOM observation
        this.setupOptimizedObservers();
        
        // Setup message handling
        this.setupMessageHandling();
        
        // Setup periodic cleanup
        this.setupPeriodicCleanup();
        
        // Process existing elements (lazy)
        this.scheduleInitialProcessing();
        
        this.logger.debug('Optimized content script initialized');
    }
    
    setupOptimizedObservers() {
        // Throttled mutation observer
        this.mutationObserver = new MutationObserver((mutations) => {
            this.throttledMutationHandler(mutations);
        });
        
        // Observe with minimal configuration
        this.mutationObserver.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: false,
            characterData: false
        });
        
        // Intersection observer for lazy loading
        this.intersectionObserver = new IntersectionObserver((entries) => {
            this.handleIntersection(entries);
        }, {
            root: null,
            rootMargin: '50px',
            threshold: 0.1
        });
    }
    
    throttledMutationHandler(mutations) {
        const now = performance.now();
        
        if (now - this.lastProcessTime < this.config.observerThrottleMs) {
            return; // Skip if too frequent
        }
        
        this.lastProcessTime = now;
        
        // Extract added nodes efficiently
        const addedNodes = [];
        for (const mutation of mutations) {
            if (mutation.type === 'childList') {
                for (const node of mutation.addedNodes) {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        addedNodes.push(node);
                    }
                }
            }
        }
        
        if (addedNodes.length > 0) {
            this.queueElementsForProcessing(addedNodes);
        }
    }
    
    queueElementsForProcessing(elements) {
        // Filter relevant elements
        const textElements = elements.filter(el => this.isRelevantTextElement(el));
        
        // Add to processing queue
        this.processingQueue.push(...textElements);
        
        // Limit queue size to prevent memory issues
        if (this.processingQueue.length > this.config.maxProcessedElements) {
            this.processingQueue = this.processingQueue.slice(-this.config.maxProcessedElements);
        }
        
        // Schedule processing
        this.scheduleProcessing();
    }
    
    isRelevantTextElement(element) {
        // Quick checks for text-containing elements
        if (!element.textContent || element.textContent.trim().length < 20) {
            return false;
        }
        
        // Check tag types
        const textTags = ['P', 'DIV', 'SPAN', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'ARTICLE', 'SECTION'];
        if (!textTags.includes(element.tagName)) {
            return false;
        }
        
        // Skip already processed
        if (this.processedElements.has(element)) {
            return false;
        }
        
        // Skip hidden elements
        if (element.offsetParent === null && element.style.display !== 'none') {
            return false;
        }
        
        return true;
    }
    
    scheduleProcessing() {
        if (this.processingTimer) return;
        
        this.processingTimer = setTimeout(() => {
            this.processQueuedElements();
            this.processingTimer = null;
        }, 100); // Small delay to batch operations
    }
    
    async processQueuedElements() {
        if (this.processingQueue.length === 0) return;
        
        const batch = this.processingQueue.splice(0, this.config.batchProcessingSize);
        
        // Process batch efficiently
        await this.processBatch(batch);
        
        // Continue processing if more elements remain
        if (this.processingQueue.length > 0) {
            this.scheduleProcessing();
        }
    }
    
    async processBatch(elements) {
        const relevantElements = [];
        
        // Filter and prepare elements
        for (const element of elements) {
            if (this.shouldProcessElement(element)) {
                relevantElements.push({
                    element,
                    text: this.extractText(element),
                    selector: this.generateMinimalSelector(element)
                });
                
                // Mark as processed
                this.processedElements.add(element);
            }
        }
        
        if (relevantElements.length === 0) return;
        
        // Use intersection observer for lazy loading
        for (const item of relevantElements) {
            if (this.isElementInViewport(item.element)) {
                await this.processElementImmediate(item);
            } else {
                this.intersectionObserver.observe(item.element);
            }
        }
    }
    
    shouldProcessElement(element) {
        // Additional checks before processing
        if (!document.contains(element)) return false;
        if (this.processedElements.has(element)) return false;
        
        const text = element.textContent.trim();
        if (text.length < 20 || text.length > 5000) return false;
        
        return true;
    }
    
    extractText(element) {
        // Extract text efficiently
        return element.textContent.trim().substring(0, 1000); // Limit to 1KB
    }
    
    generateMinimalSelector(element) {
        // Generate minimal selector for memory efficiency
        if (element.id) {
            return `#${element.id}`;
        }
        
        if (element.className) {
            const classes = element.className.split(' ')[0];
            return `.${classes}`;
        }
        
        return element.tagName.toLowerCase();
    }
    
    async processElementImmediate(item) {
        try {
            // Send to background for detection
            const result = await this.sendDetectionRequest(item.text, {
                selector: item.selector,
                elementType: item.element.tagName
            });
            
            if (result && result.is_ai_generated && result.confidence_score > 0.7) {
                this.addVisualIndicator(item.element, result);
            }
            
            this.metrics.counter('content_elements_processed');
            
        } catch (error) {
            this.logger.debug('Element processing failed:', error);
            this.metrics.counter('content_processing_errors');
        }
    }
    
    async sendDetectionRequest(text, metadata) {
        return new Promise((resolve) => {
            chrome.runtime.sendMessage({
                type: 'DETECT_TEXT',
                text,
                metadata,
                requestId: `content_${Date.now()}`
            }, (response) => {
                resolve(response);
            });
        });
    }
    
    addVisualIndicator(element, result) {
        // Limit number of visual indicators
        if (this.visualIndicators.size >= this.config.maxVisualIndicators) {
            this.removeOldestIndicator();
        }
        
        // Create minimal visual indicator
        const indicator = this.createMinimalIndicator(result);
        
        // Position indicator efficiently
        this.positionIndicator(indicator, element);
        
        // Store reference
        this.visualIndicators.set(element, indicator);
        
        // Auto-remove after delay
        setTimeout(() => {
            this.removeVisualIndicator(element);
        }, 30000); // 30 seconds
    }
    
    createMinimalIndicator(result) {
        const indicator = document.createElement('div');
        indicator.className = 'ai-detector-indicator';
        
        // Minimal styling
        indicator.style.cssText = `
            position: absolute;
            width: 12px;
            height: 12px;
            background: ${result.confidence_score > 0.8 ? '#ff4444' : '#ff8844'};
            border-radius: 50%;
            z-index: 10000;
            pointer-events: none;
            opacity: 0.8;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        `;
        
        return indicator;
    }
    
    positionIndicator(indicator, element) {
        try {
            const rect = element.getBoundingClientRect();
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            const scrollLeft = window.pageXOffset || document.documentElement.scrollLeft;
            
            indicator.style.top = `${rect.top + scrollTop - 6}px`;
            indicator.style.left = `${rect.right + scrollLeft - 6}px`;
            
            document.body.appendChild(indicator);
            
        } catch (error) {
            // Positioning failed, cleanup
            if (indicator.parentNode) {
                indicator.parentNode.removeChild(indicator);
            }
        }
    }
    
    removeVisualIndicator(element) {
        const indicator = this.visualIndicators.get(element);
        if (indicator && indicator.parentNode) {
            indicator.parentNode.removeChild(indicator);
        }
        this.visualIndicators.delete(element);
    }
    
    removeOldestIndicator() {
        const firstElement = this.visualIndicators.keys().next().value;
        if (firstElement) {
            this.removeVisualIndicator(firstElement);
        }
    }
    
    handleIntersection(entries) {
        for (const entry of entries) {
            if (entry.isIntersecting) {
                const element = entry.target;
                
                // Process element now that it's visible
                const item = {
                    element,
                    text: this.extractText(element),
                    selector: this.generateMinimalSelector(element)
                };
                
                this.processElementImmediate(item);
                
                // Stop observing
                this.intersectionObserver.unobserve(element);
            }
        }
    }
    
    isElementInViewport(element) {
        const rect = element.getBoundingClientRect();
        return (
            rect.top >= 0 &&
            rect.left >= 0 &&
            rect.bottom <= window.innerHeight &&
            rect.right <= window.innerWidth
        );
    }
    
    setupMessageHandling() {
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
            return true; // Keep channel open
        });
    }
    
    async handleMessage(message, sender, sendResponse) {
        try {
            switch (message.type) {
                case 'GET_PAGE_TEXT':
                    sendResponse(this.getPageText());
                    break;
                    
                case 'HIGHLIGHT_ELEMENT':
                    this.highlightElement(message.selector);
                    sendResponse({ success: true });
                    break;
                    
                case 'CLEAR_INDICATORS':
                    this.clearAllIndicators();
                    sendResponse({ success: true });
                    break;
                    
                case 'GET_STATS':
                    sendResponse(this.getContentStats());
                    break;
                    
                case 'CLEANUP':
                    await this.performCleanup();
                    sendResponse({ success: true });
                    break;
                    
                default:
                    sendResponse({ error: 'Unknown message type' });
            }
        } catch (error) {
            this.logger.error('Message handling error:', error);
            sendResponse({ error: error.message });
        }
    }
    
    getPageText() {
        // Extract page text efficiently
        const textElements = document.querySelectorAll('p, div, span, h1, h2, h3, h4, h5, h6');
        const texts = [];
        
        for (const element of textElements) {
            const text = element.textContent.trim();
            if (text.length >= 20) {
                texts.push({
                    text: text.substring(0, 500), // Limit size
                    selector: this.generateMinimalSelector(element)
                });
                
                if (texts.length >= 50) break; // Limit number
            }
        }
        
        return texts;
    }
    
    highlightElement(selector) {
        try {
            const element = document.querySelector(selector);
            if (element) {
                element.style.outline = '2px solid #ff4444';
                setTimeout(() => {
                    element.style.outline = '';
                }, 3000);
            }
        } catch (error) {
            this.logger.debug('Highlight failed:', error);
        }
    }
    
    clearAllIndicators() {
        for (const [element, indicator] of this.visualIndicators) {
            if (indicator.parentNode) {
                indicator.parentNode.removeChild(indicator);
            }
        }
        this.visualIndicators.clear();
    }
    
    getContentStats() {
        return {
            processedElements: this.processedElements.size || 0,
            visualIndicators: this.visualIndicators.size,
            queuedElements: this.processingQueue.length,
            memoryStats: this.memoryOptimizer.getMemoryStats()
        };
    }
    
    setupPeriodicCleanup() {
        this.cleanupTimer = setInterval(() => {
            this.performCleanup();
        }, this.config.cleanupIntervalMs);
    }
    
    async performCleanup() {
        const startTime = performance.now();
        
        try {
            // Clean up detached elements
            this.cleanupDetachedElements();
            
            // Clean up old indicators
            this.cleanupOldIndicators();
            
            // Limit processing queue
            this.limitProcessingQueue();
            
            // Trigger memory optimization
            await this.memoryOptimizer.performGarbageCollection();
            
            const duration = performance.now() - startTime;
            this.metrics.histogram('content_cleanup_duration_ms', duration);
            
            this.logger.debug(`Content cleanup completed in ${duration.toFixed(1)}ms`);
            
        } catch (error) {
            this.logger.error('Cleanup failed:', error);
        }
    }
    
    cleanupDetachedElements() {
        // Clean up indicators for removed elements
        const toRemove = [];
        
        for (const [element] of this.visualIndicators) {
            if (!document.contains(element)) {
                toRemove.push(element);
            }
        }
        
        toRemove.forEach(element => {
            this.removeVisualIndicator(element);
        });
        
        // Clean up processing queue
        this.processingQueue = this.processingQueue.filter(element => 
            document.contains(element)
        );
    }
    
    cleanupOldIndicators() {
        // Remove indicators older than 5 minutes
        const cutoffTime = Date.now() - (5 * 60 * 1000);
        const toRemove = [];
        
        for (const [element, indicator] of this.visualIndicators) {
            if (indicator.dataset.created && 
                parseInt(indicator.dataset.created) < cutoffTime) {
                toRemove.push(element);
            }
        }
        
        toRemove.forEach(element => {
            this.removeVisualIndicator(element);
        });
    }
    
    limitProcessingQueue() {
        if (this.processingQueue.length > this.config.maxProcessedElements) {
            // Keep only the most recent elements
            this.processingQueue = this.processingQueue.slice(-this.config.maxProcessedElements / 2);
        }
    }
    
    scheduleInitialProcessing() {
        // Process initial page elements with delay
        setTimeout(() => {
            const initialElements = document.querySelectorAll('p, div, article, section');
            const relevantElements = Array.from(initialElements)
                .filter(el => this.isRelevantTextElement(el))
                .slice(0, this.config.lazyLoadThreshold); // Limit initial processing
            
            this.queueElementsForProcessing(relevantElements);
        }, 1000); // Delay to let page settle
    }
    
    destroy() {
        // Clean up all resources
        if (this.mutationObserver) {
            this.mutationObserver.disconnect();
        }
        
        if (this.intersectionObserver) {
            this.intersectionObserver.disconnect();
        }
        
        if (this.cleanupTimer) {
            clearInterval(this.cleanupTimer);
        }
        
        if (this.processingTimer) {
            clearTimeout(this.processingTimer);
        }
        
        this.clearAllIndicators();
        this.processingQueue = [];
        
        if (this.memoryOptimizer) {
            this.memoryOptimizer.destroy();
        }
    }
}

// Mock classes for standalone operation (same as background script)
class Logger {
    constructor(name) {
        this.name = name;
    }
    
    info(message, ...args) {
        console.log(`[${this.name}] INFO:`, message, ...args);
    }
    
    warn(message, ...args) {
        console.warn(`[${this.name}] WARN:`, message, ...args);
    }
    
    error(message, ...args) {
        console.error(`[${this.name}] ERROR:`, message, ...args);
    }
    
    debug(message, ...args) {
        if (localStorage.getItem('ai-detector-debug') === 'true') {
            console.debug(`[${this.name}] DEBUG:`, message, ...args);
        }
    }
}

class MetricsCollector {
    constructor() {
        this.counters = new Map();
        this.histograms = new Map();
    }
    
    counter(name, labels = {}) {
        const key = `${name}_${JSON.stringify(labels)}`;
        this.counters.set(key, (this.counters.get(key) || 0) + 1);
    }
    
    histogram(name, value, labels = {}) {
        const key = `${name}_${JSON.stringify(labels)}`;
        const values = this.histograms.get(key) || [];
        values.push(value);
        
        // Keep only recent values
        if (values.length > 100) {
            values.splice(0, values.length - 100);
        }
        
        this.histograms.set(key, values);
    }
}

// Basic MemoryOptimizer for content script
class MemoryOptimizer {
    constructor() {
        this.stats = { current: 0, peak: 0, collections: 0 };
    }
    
    async performGarbageCollection() {
        // Force garbage collection if available
        if (window.gc && typeof window.gc === 'function') {
            window.gc();
        }
        
        this.stats.collections++;
    }
    
    getMemoryStats() {
        if (performance.memory) {
            this.stats.current = performance.memory.usedJSHeapSize / (1024 * 1024);
            if (this.stats.current > this.stats.peak) {
                this.stats.peak = this.stats.current;
            }
        }
        
        return this.stats;
    }
    
    destroy() {
        // Cleanup
        this.stats = { current: 0, peak: 0, collections: 0 };
    }
}

// Initialize content script
const optimizedContent = new OptimizedContentScript();

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    optimizedContent.destroy();
});