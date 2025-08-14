/**
 * Message Bus System for Chrome Extension
 * Implements centralized message routing and event handling across extension components
 */

class MessageBus {
    constructor() {
        this.handlers = new Map();
        this.middleware = [];
        this.messageQueue = [];
        this.isInitialized = false;
        this.requestHandlers = new Map();
        this.eventSubscriptions = new Map();
        this.messageId = 0;
        this.pendingRequests = new Map();
        
        // Performance metrics
        this.metrics = {
            messagesProcessed: 0,
            averageProcessingTime: 0,
            totalProcessingTime: 0,
            errorCount: 0
        };
    }

    /**
     * Initialize the message bus
     */
    async initialize() {
        if (this.isInitialized) return;
        
        console.log('[MessageBus] Initializing message bus...');
        
        // Set up Chrome extension message listeners
        if (typeof chrome !== 'undefined' && chrome.runtime) {
            chrome.runtime.onMessage.addListener(this._handleChromeMessage.bind(this));
            
            // Handle connection events
            if (chrome.runtime.onConnect) {
                chrome.runtime.onConnect.addListener(this._handleConnection.bind(this));
            }
        }
        
        // Process queued messages
        await this._processMessageQueue();
        
        this.isInitialized = true;
        console.log('[MessageBus] Message bus initialized successfully');
        
        // Emit initialization event
        this.emit('messageBus:initialized', { timestamp: Date.now() });
    }

    /**
     * Register a message handler
     * @param {string} messageType - Type of message to handle
     * @param {Function} handler - Handler function
     * @param {Object} options - Handler options
     */
    register(messageType, handler, options = {}) {
        if (!this.handlers.has(messageType)) {
            this.handlers.set(messageType, []);
        }
        
        const handlerConfig = {
            handler,
            priority: options.priority || 0,
            once: options.once || false,
            context: options.context || null,
            id: this._generateHandlerId()
        };
        
        this.handlers.get(messageType).push(handlerConfig);
        
        // Sort by priority (higher priority first)
        this.handlers.get(messageType).sort((a, b) => b.priority - a.priority);
        
        console.log(`[MessageBus] Registered handler for '${messageType}'`);
        
        return handlerConfig.id;
    }

    /**
     * Unregister a message handler
     * @param {string} messageType - Type of message
     * @param {string} handlerId - Handler ID to remove
     */
    unregister(messageType, handlerId) {
        if (!this.handlers.has(messageType)) return false;
        
        const handlers = this.handlers.get(messageType);
        const index = handlers.findIndex(h => h.id === handlerId);
        
        if (index !== -1) {
            handlers.splice(index, 1);
            if (handlers.length === 0) {
                this.handlers.delete(messageType);
            }
            console.log(`[MessageBus] Unregistered handler ${handlerId} for '${messageType}'`);
            return true;
        }
        
        return false;
    }

    /**
     * Subscribe to events with pattern matching
     * @param {string|RegExp} pattern - Event pattern to match
     * @param {Function} callback - Callback function
     * @param {Object} options - Subscription options
     */
    subscribe(pattern, callback, options = {}) {
        const subscriptionId = this._generateHandlerId();
        
        if (!this.eventSubscriptions.has(pattern)) {
            this.eventSubscriptions.set(pattern, []);
        }
        
        this.eventSubscriptions.get(pattern).push({
            id: subscriptionId,
            callback,
            once: options.once || false,
            context: options.context || null
        });
        
        console.log(`[MessageBus] Subscribed to pattern '${pattern}'`);
        
        return subscriptionId;
    }

    /**
     * Unsubscribe from events
     * @param {string} subscriptionId - Subscription ID
     */
    unsubscribe(subscriptionId) {
        for (const [pattern, subscriptions] of this.eventSubscriptions) {
            const index = subscriptions.findIndex(s => s.id === subscriptionId);
            if (index !== -1) {
                subscriptions.splice(index, 1);
                if (subscriptions.length === 0) {
                    this.eventSubscriptions.delete(pattern);
                }
                console.log(`[MessageBus] Unsubscribed ${subscriptionId}`);
                return true;
            }
        }
        return false;
    }

    /**
     * Send a message
     * @param {string} messageType - Type of message
     * @param {*} data - Message data
     * @param {Object} options - Send options
     */
    async send(messageType, data = null, options = {}) {
        const message = this._createMessage(messageType, data, options);
        
        if (!this.isInitialized) {
            this.messageQueue.push(message);
            console.log(`[MessageBus] Queued message '${messageType}' (not initialized)`);
            return;
        }
        
        return await this._processMessage(message);
    }

    /**
     * Send a request and wait for response
     * @param {string} messageType - Type of request
     * @param {*} data - Request data
     * @param {Object} options - Request options
     */
    async request(messageType, data = null, options = {}) {
        const requestId = this._generateRequestId();
        const timeout = options.timeout || 5000;
        
        const message = this._createMessage(messageType, data, {
            ...options,
            requestId,
            expectsResponse: true
        });
        
        // Create promise for response
        const responsePromise = new Promise((resolve, reject) => {
            const timeoutId = setTimeout(() => {
                this.pendingRequests.delete(requestId);
                reject(new Error(`Request timeout for ${messageType}`));
            }, timeout);
            
            this.pendingRequests.set(requestId, {
                resolve,
                reject,
                timeoutId,
                messageType
            });
        });
        
        // Send the request
        await this.send(messageType, data, {
            ...options,
            requestId,
            expectsResponse: true
        });
        
        return responsePromise;
    }

    /**
     * Send a response to a request
     * @param {string} requestId - Original request ID
     * @param {*} data - Response data
     * @param {Object} options - Response options
     */
    async respond(requestId, data = null, options = {}) {
        if (!requestId) {
            throw new Error('Request ID is required for responses');
        }
        
        const message = this._createMessage('response', data, {
            ...options,
            requestId,
            isResponse: true
        });
        
        return await this._processMessage(message);
    }

    /**
     * Emit an event
     * @param {string} eventType - Type of event
     * @param {*} data - Event data
     * @param {Object} options - Emit options
     */
    emit(eventType, data = null, options = {}) {
        console.log(`[MessageBus] Emitting event '${eventType}'`);
        
        // Find matching subscriptions
        for (const [pattern, subscriptions] of this.eventSubscriptions) {
            let matches = false;
            
            if (pattern instanceof RegExp) {
                matches = pattern.test(eventType);
            } else if (typeof pattern === 'string') {
                matches = pattern === eventType || eventType.startsWith(pattern + ':');
            }
            
            if (matches) {
                subscriptions.forEach((subscription, index) => {
                    try {
                        if (subscription.context) {
                            subscription.callback.call(subscription.context, data, eventType);
                        } else {
                            subscription.callback(data, eventType);
                        }
                        
                        // Remove one-time subscriptions
                        if (subscription.once) {
                            subscriptions.splice(index, 1);
                        }
                    } catch (error) {
                        console.error(`[MessageBus] Error in event subscription:`, error);
                    }
                });
            }
        }
    }

    /**
     * Add middleware for message processing
     * @param {Function} middleware - Middleware function
     */
    use(middleware) {
        this.middleware.push(middleware);
        console.log(`[MessageBus] Added middleware`);
    }

    /**
     * Send message to specific tab
     * @param {number} tabId - Chrome tab ID
     * @param {string} messageType - Message type
     * @param {*} data - Message data
     */
    async sendToTab(tabId, messageType, data = null) {
        if (typeof chrome === 'undefined' || !chrome.tabs) {
            throw new Error('Chrome tabs API not available');
        }
        
        const message = this._createMessage(messageType, data, { target: 'tab' });
        
        return new Promise((resolve, reject) => {
            chrome.tabs.sendMessage(tabId, message, (response) => {
                if (chrome.runtime.lastError) {
                    reject(new Error(chrome.runtime.lastError.message));
                } else {
                    resolve(response);
                }
            });
        });
    }

    /**
     * Broadcast message to all tabs
     * @param {string} messageType - Message type
     * @param {*} data - Message data
     */
    async broadcast(messageType, data = null) {
        if (typeof chrome === 'undefined' || !chrome.tabs) {
            throw new Error('Chrome tabs API not available');
        }
        
        const tabs = await new Promise((resolve) => {
            chrome.tabs.query({}, resolve);
        });
        
        const message = this._createMessage(messageType, data, { target: 'broadcast' });
        
        const results = await Promise.allSettled(
            tabs.map(tab => this.sendToTab(tab.id, messageType, data))
        );
        
        return results;
    }

    /**
     * Get message bus statistics
     */
    getStats() {
        return {
            ...this.metrics,
            handlersCount: Array.from(this.handlers.values()).reduce((sum, handlers) => sum + handlers.length, 0),
            subscriptionsCount: Array.from(this.eventSubscriptions.values()).reduce((sum, subs) => sum + subs.length, 0),
            queuedMessages: this.messageQueue.length,
            pendingRequests: this.pendingRequests.size
        };
    }

    // Private methods

    _createMessage(type, data, options = {}) {
        return {
            id: this._generateMessageId(),
            type,
            data,
            timestamp: Date.now(),
            source: this._getContext(),
            ...options
        };
    }

    async _processMessage(message) {
        const startTime = performance.now();
        
        try {
            // Apply middleware
            for (const middleware of this.middleware) {
                const result = await middleware(message, this);
                if (result === false) {
                    console.log(`[MessageBus] Message '${message.type}' blocked by middleware`);
                    return false;
                }
                if (result && typeof result === 'object') {
                    Object.assign(message, result);
                }
            }
            
            // Handle responses
            if (message.isResponse && message.requestId) {
                this._handleResponse(message);
                return;
            }
            
            // Process handlers
            const handlers = this.handlers.get(message.type) || [];
            const results = [];
            
            for (const handlerConfig of handlers) {
                try {
                    const result = await handlerConfig.handler(message.data, message);
                    results.push(result);
                    
                    // Remove one-time handlers
                    if (handlerConfig.once) {
                        this.unregister(message.type, handlerConfig.id);
                    }
                } catch (error) {
                    console.error(`[MessageBus] Handler error for '${message.type}':`, error);
                    this.metrics.errorCount++;
                }
            }
            
            // Update metrics
            const processingTime = performance.now() - startTime;
            this._updateMetrics(processingTime);
            
            console.log(`[MessageBus] Processed '${message.type}' in ${processingTime.toFixed(2)}ms`);
            
            return results.length === 1 ? results[0] : results;
            
        } catch (error) {
            console.error(`[MessageBus] Error processing message '${message.type}':`, error);
            this.metrics.errorCount++;
            throw error;
        }
    }

    _handleChromeMessage(message, sender, sendResponse) {
        console.log(`[MessageBus] Received Chrome message:`, message);
        
        // Process the message
        this._processMessage(message).then(result => {
            sendResponse({ success: true, result });
        }).catch(error => {
            sendResponse({ success: false, error: error.message });
        });
        
        return true; // Indicates async response
    }

    _handleConnection(port) {
        console.log(`[MessageBus] New connection:`, port.name);
        
        port.onMessage.addListener((message) => {
            this._processMessage(message).then(result => {
                port.postMessage({ success: true, result });
            }).catch(error => {
                port.postMessage({ success: false, error: error.message });
            });
        });
    }

    _handleResponse(message) {
        const pendingRequest = this.pendingRequests.get(message.requestId);
        
        if (pendingRequest) {
            clearTimeout(pendingRequest.timeoutId);
            this.pendingRequests.delete(message.requestId);
            pendingRequest.resolve(message.data);
        }
    }

    async _processMessageQueue() {
        console.log(`[MessageBus] Processing ${this.messageQueue.length} queued messages`);
        
        const queuedMessages = [...this.messageQueue];
        this.messageQueue = [];
        
        for (const message of queuedMessages) {
            try {
                await this._processMessage(message);
            } catch (error) {
                console.error(`[MessageBus] Error processing queued message:`, error);
            }
        }
    }

    _updateMetrics(processingTime) {
        this.metrics.messagesProcessed++;
        this.metrics.totalProcessingTime += processingTime;
        this.metrics.averageProcessingTime = this.metrics.totalProcessingTime / this.metrics.messagesProcessed;
    }

    _generateMessageId() {
        return `msg_${++this.messageId}_${Date.now()}`;
    }

    _generateRequestId() {
        return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    _generateHandlerId() {
        return `handler_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    _getContext() {
        if (typeof chrome !== 'undefined') {
            if (chrome.runtime && chrome.runtime.getManifest) {
                const manifest = chrome.runtime.getManifest();
                if (manifest.background) {
                    return 'background';
                }
            }
            
            if (window.location && window.location.protocol === 'chrome-extension:') {
                return 'popup';
            }
            
            return 'content';
        }
        
        return 'unknown';
    }
}

// Global message bus instance
const messageBus = new MessageBus();

// Auto-initialize in appropriate contexts
if (typeof chrome !== 'undefined') {
    messageBus.initialize().catch(error => {
        console.error('[MessageBus] Initialization failed:', error);
    });
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MessageBus, messageBus };
} else if (typeof window !== 'undefined') {
    window.MessageBus = MessageBus;
    window.messageBus = messageBus;
}