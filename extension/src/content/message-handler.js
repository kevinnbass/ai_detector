/**
 * Content Script Message Handler
 * Handles message bus communication for content script
 */

// Load message bus
const script = document.createElement('script');
script.src = chrome.runtime.getURL('shared/message-bus.js');
document.documentElement.appendChild(script);
script.remove();

class ContentMessageHandler {
    constructor() {
        this.initialized = false;
        this.detectionResults = new Map();
        this.observedElements = new WeakSet();
    }

    /**
     * Initialize the message handler
     */
    async initialize() {
        if (this.initialized) return;
        
        console.log('[ContentMessageHandler] Initializing...');
        
        // Wait for message bus to be available
        await this.waitForMessageBus();
        
        // Initialize message bus
        await messageBus.initialize();
        
        // Register handlers
        this.registerHandlers();
        
        // Set up event listeners
        this.setupEventListeners();
        
        this.initialized = true;
        console.log('[ContentMessageHandler] Initialized successfully');
        
        // Emit initialization event
        messageBus.emit('content:initialized', {
            url: window.location.href,
            timestamp: Date.now()
        });
    }

    /**
     * Wait for message bus to be available
     */
    async waitForMessageBus() {
        return new Promise((resolve) => {
            const check = () => {
                if (typeof messageBus !== 'undefined') {
                    resolve();
                } else {
                    setTimeout(check, 100);
                }
            };
            check();
        });
    }

    /**
     * Register message handlers
     */
    registerHandlers() {
        // Handle page scan requests
        messageBus.register('SCAN_PAGE', async (data) => {
            const { options = {} } = data;
            return await this.scanPageForText(options);
        });

        // Handle text analysis requests
        messageBus.register('ANALYZE_TEXT', async (data) => {
            const { text, elementId, options = {} } = data;
            return await this.analyzeText(text, elementId, options);
        });

        // Handle highlighting requests
        messageBus.register('HIGHLIGHT_ELEMENTS', async (data) => {
            const { results } = data;
            return this.highlightElements(results);
        });

        // Handle settings updates
        messageBus.register('UPDATE_SETTINGS', async (data) => {
            const { settings } = data;
            return this.updateSettings(settings);
        });

        // Handle detection result updates
        messageBus.register('DETECTION_RESULT', async (data) => {
            const { elementId, result } = data;
            this.detectionResults.set(elementId, result);
            return this.updateElementUI(elementId, result);
        });

        // Listen to detection events from background
        messageBus.subscribe('detection:completed', (data) => {
            const { result, requestId } = data;
            this.handleDetectionCompleted(result, requestId);
        });

        messageBus.subscribe('detection:failed', (data) => {
            const { error, fallback, requestId } = data;
            this.handleDetectionFailed(error, fallback, requestId);
        });

        // Listen to settings updates
        messageBus.subscribe('settings:updated', (settings) => {
            this.applySettings(settings);
        });

        console.log('[ContentMessageHandler] Registered message handlers');
    }

    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Listen for DOM changes
        const observer = new MutationObserver((mutations) => {
            this.handleDOMChanges(mutations);
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: false
        });

        // Listen for text selection
        document.addEventListener('mouseup', () => {
            const selection = window.getSelection().toString().trim();
            if (selection.length > 10) {
                messageBus.emit('text:selected', {
                    text: selection,
                    timestamp: Date.now()
                });
            }
        });

        // Listen for scroll to trigger lazy analysis
        let scrollTimeout;
        window.addEventListener('scroll', () => {
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {
                messageBus.emit('scroll:stopped', {
                    scrollY: window.scrollY,
                    timestamp: Date.now()
                });
            }, 500);
        });

        console.log('[ContentMessageHandler] Set up event listeners');
    }

    /**
     * Scan page for text elements to analyze
     */
    async scanPageForText(options = {}) {
        const selectors = options.selectors || [
            'p', 'div[role="article"]', '.tweet-text', 
            '[data-testid="tweetText"]', 'blockquote'
        ];

        const elements = [];
        const minLength = options.minLength || 20;

        selectors.forEach(selector => {
            const found = document.querySelectorAll(selector);
            found.forEach(element => {
                if (this.observedElements.has(element)) return;

                const text = this.extractTextFromElement(element);
                if (text && text.length >= minLength) {
                    const elementId = this.generateElementId(element);
                    elements.push({
                        elementId,
                        text,
                        selector,
                        boundingRect: element.getBoundingClientRect()
                    });
                    
                    // Mark as observed
                    this.observedElements.add(element);
                    element.setAttribute('data-ai-detector-id', elementId);
                }
            });
        });

        console.log(`[ContentMessageHandler] Found ${elements.length} text elements to analyze`);

        // Request analysis for found elements
        if (elements.length > 0) {
            this.requestBatchAnalysis(elements, options);
        }

        return {
            elementsFound: elements.length,
            elements: elements.slice(0, 5) // Return first 5 for preview
        };
    }

    /**
     * Request batch analysis from background
     */
    async requestBatchAnalysis(elements, options = {}) {
        // Group elements into batches
        const batchSize = options.batchSize || 3;
        const batches = [];
        
        for (let i = 0; i < elements.length; i += batchSize) {
            batches.push(elements.slice(i, i + batchSize));
        }

        // Process batches with delay
        for (let i = 0; i < batches.length; i++) {
            const batch = batches[i];
            
            setTimeout(async () => {
                for (const element of batch) {
                    try {
                        const result = await messageBus.request('DETECT_TEXT', {
                            text: element.text,
                            options: { ...options, quick: true }
                        }, { timeout: 10000 });

                        if (result.success) {
                            this.handleDetectionResult(element.elementId, result.result);
                        } else {
                            console.warn(`[ContentMessageHandler] Detection failed for element ${element.elementId}:`, result.error);
                        }
                    } catch (error) {
                        console.error(`[ContentMessageHandler] Analysis request failed:`, error);
                    }
                }
            }, i * 1000); // 1 second delay between batches
        }
    }

    /**
     * Extract text from element
     */
    extractTextFromElement(element) {
        // Get text content, excluding child elements with certain classes
        const excludeSelectors = [
            '.ai-detector-overlay', '.ai-detector-indicator',
            'script', 'style', 'noscript'
        ];

        let text = element.textContent || '';
        
        // Remove excluded content
        excludeSelectors.forEach(selector => {
            const excluded = element.querySelectorAll(selector);
            excluded.forEach(el => {
                text = text.replace(el.textContent || '', '');
            });
        });

        return text.trim().replace(/\s+/g, ' ');
    }

    /**
     * Generate unique element ID
     */
    generateElementId(element) {
        // Try to use existing ID or create one
        if (element.id) return element.id;
        
        const text = this.extractTextFromElement(element).substring(0, 50);
        const hash = this.simpleHash(text);
        return `ai-detector-${hash}-${Date.now()}`;
    }

    /**
     * Simple hash function
     */
    simpleHash(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        return Math.abs(hash).toString(36);
    }

    /**
     * Handle detection result
     */
    handleDetectionResult(elementId, result) {
        this.detectionResults.set(elementId, result);
        this.updateElementUI(elementId, result);
        
        // Emit event for other components
        messageBus.emit('element:analyzed', {
            elementId,
            result,
            timestamp: Date.now()
        });
    }

    /**
     * Update element UI based on detection result
     */
    updateElementUI(elementId, result) {
        const element = document.querySelector(`[data-ai-detector-id="${elementId}"]`);
        if (!element) return false;

        // Remove existing indicators
        this.removeElementIndicators(element);

        // Add new indicators based on result
        if (result.prediction === 'ai' && result.ai_probability > 0.6) {
            this.addAIIndicator(element, result);
        }

        // Update element attributes
        element.setAttribute('data-ai-probability', result.ai_probability);
        element.setAttribute('data-ai-prediction', result.prediction);

        return true;
    }

    /**
     * Add AI indicator to element
     */
    addAIIndicator(element, result) {
        // Create indicator element
        const indicator = document.createElement('div');
        indicator.className = 'ai-detector-indicator';
        indicator.innerHTML = `
            <span class="ai-indicator-icon">🤖</span>
            <span class="ai-indicator-text">AI (${Math.round(result.ai_probability * 100)}%)</span>
        `;

        // Style the indicator
        Object.assign(indicator.style, {
            position: 'absolute',
            top: '-8px',
            right: '-8px',
            background: '#ff6b6b',
            color: 'white',
            padding: '2px 6px',
            borderRadius: '12px',
            fontSize: '11px',
            fontWeight: 'bold',
            zIndex: '1000',
            cursor: 'pointer'
        });

        // Position parent relatively if needed
        const computedStyle = getComputedStyle(element);
        if (computedStyle.position === 'static') {
            element.style.position = 'relative';
        }

        // Add click handler for details
        indicator.addEventListener('click', (e) => {
            e.stopPropagation();
            this.showDetectionDetails(result);
        });

        element.appendChild(indicator);
    }

    /**
     * Remove element indicators
     */
    removeElementIndicators(element) {
        const indicators = element.querySelectorAll('.ai-detector-indicator');
        indicators.forEach(indicator => indicator.remove());
    }

    /**
     * Show detection details
     */
    showDetectionDetails(result) {
        messageBus.emit('show:details', {
            result,
            timestamp: Date.now()
        });

        // Could show a tooltip or modal with detailed analysis
        console.log('[ContentMessageHandler] Detection details:', result);
    }

    /**
     * Handle DOM changes
     */
    handleDOMChanges(mutations) {
        let newElements = false;

        mutations.forEach(mutation => {
            mutation.addedNodes.forEach(node => {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    // Check if new text elements were added
                    const textElements = node.querySelectorAll 
                        ? node.querySelectorAll('p, div, span')
                        : [];
                        
                    if (textElements.length > 0) {
                        newElements = true;
                    }
                }
            });
        });

        if (newElements) {
            // Debounce page scanning for performance
            clearTimeout(this.scanTimeout);
            this.scanTimeout = setTimeout(() => {
                messageBus.emit('dom:changed', { timestamp: Date.now() });
                this.scanPageForText({ minLength: 30 });
            }, 1000);
        }
    }

    /**
     * Handle detection completed event
     */
    handleDetectionCompleted(result, requestId) {
        console.log('[ContentMessageHandler] Detection completed:', result);
        
        // Update badge count
        this.updateBadgeCount();
    }

    /**
     * Handle detection failed event
     */
    handleDetectionFailed(error, fallback, requestId) {
        console.warn('[ContentMessageHandler] Detection failed:', error);
        
        // Use fallback if available
        if (fallback) {
            console.log('[ContentMessageHandler] Using fallback analysis:', fallback);
        }
    }

    /**
     * Apply settings
     */
    applySettings(settings) {
        console.log('[ContentMessageHandler] Applying settings:', settings);
        
        // Update UI based on settings
        if (!settings.enabled) {
            this.hideAllIndicators();
        }
        
        if (settings.highlightColor) {
            this.updateIndicatorColors(settings.highlightColor);
        }
    }

    /**
     * Update badge count
     */
    updateBadgeCount() {
        const aiCount = Array.from(this.detectionResults.values())
            .filter(result => result.prediction === 'ai' && result.ai_probability > 0.6)
            .length;

        messageBus.send('UPDATE_BADGE', { count: aiCount });
    }

    /**
     * Hide all indicators
     */
    hideAllIndicators() {
        const indicators = document.querySelectorAll('.ai-detector-indicator');
        indicators.forEach(indicator => {
            indicator.style.display = 'none';
        });
    }

    /**
     * Update indicator colors
     */
    updateIndicatorColors(color) {
        const indicators = document.querySelectorAll('.ai-detector-indicator');
        indicators.forEach(indicator => {
            indicator.style.background = color;
        });
    }
}

// Initialize content message handler
const contentMessageHandler = new ContentMessageHandler();

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        contentMessageHandler.initialize();
    });
} else {
    contentMessageHandler.initialize();
}

// Export for use by other scripts
window.contentMessageHandler = contentMessageHandler;