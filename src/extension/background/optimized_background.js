/**
 * Optimized background script for Chrome extension memory efficiency.
 * 
 * Implements service worker optimization, message queuing, cache management,
 * and resource cleanup to maintain <50MB memory usage.
 */

// Import memory optimizer
importScripts('./performance/memory_optimizer.js');

class OptimizedBackgroundScript {
    constructor() {
        this.logger = new Logger('OptimizedBackground');
        this.metrics = new MetricsCollector();
        
        // Memory optimization
        this.memoryOptimizer = new MemoryOptimizer();
        
        // Message queue with size limits
        this.messageQueue = [];
        this.maxQueueSize = 100;
        
        // Connection pools
        this.apiConnections = new Map();
        this.maxConnections = 5;
        
        // Cache management
        this.cacheManager = new BackgroundCacheManager();
        
        // Performance tracking
        this.performanceTracker = new PerformanceTracker();
        
        // Cleanup timers
        this.cleanupInterval = null;
        this.heartbeatInterval = null;
        
        // Initialize optimized background
        this.initialize();
    }
    
    async initialize() {
        try {
            // Setup optimized message handling
            this.setupMessageHandling();
            
            // Setup cache management
            await this.cacheManager.initialize();
            
            // Setup periodic cleanup
            this.setupPeriodicCleanup();
            
            // Setup heartbeat monitoring
            this.setupHeartbeatMonitoring();
            
            // Setup alarm handlers for background tasks
            this.setupAlarmHandlers();
            
            this.logger.info('Optimized background script initialized');
            
        } catch (error) {
            this.logger.error('Background initialization failed:', error);
        }
    }
    
    setupMessageHandling() {
        // Single message handler to avoid memory leaks
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
            return true; // Keep channel open for async responses
        });
        
        // Handle extension startup
        chrome.runtime.onStartup.addListener(() => {
            this.handleStartup();
        });
        
        // Handle extension installation
        chrome.runtime.onInstalled.addListener((details) => {
            this.handleInstallation(details);
        });
        
        // Handle tab updates efficiently
        chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
            if (changeInfo.status === 'complete') {
                this.handleTabUpdate(tabId, tab);
            }
        });
        
        // Handle tab removal for cleanup
        chrome.tabs.onRemoved.addListener((tabId) => {
            this.handleTabRemoval(tabId);
        });
    }
    
    async handleMessage(message, sender, sendResponse) {
        const startTime = performance.now();
        
        try {
            // Queue message if background is busy
            if (this.messageQueue.length >= this.maxQueueSize) {
                // Remove oldest message
                this.messageQueue.shift();
                this.metrics.counter('background_message_queue_overflow');
            }
            
            // Add to queue with timestamp
            this.messageQueue.push({
                message,
                sender,
                timestamp: Date.now()
            });
            
            // Process message
            const result = await this.processMessage(message, sender);
            
            // Send response
            sendResponse(result);
            
            // Record metrics
            const duration = performance.now() - startTime;
            this.metrics.histogram('background_message_duration_ms', duration);
            this.performanceTracker.recordMessageProcessing(message.type, duration);
            
        } catch (error) {
            this.logger.error('Message handling error:', error);
            sendResponse({ error: error.message });
            this.metrics.counter('background_message_errors');
        } finally {
            // Clean up processed message from queue
            this.messageQueue = this.messageQueue.filter(item => 
                item.timestamp > Date.now() - 60000 // Keep only last minute
            );
        }
    }
    
    async processMessage(message, sender) {
        switch (message.type) {
            case 'DETECT_TEXT':
                return await this.handleDetectText(message);
                
            case 'GET_CACHED_RESULT':
                return await this.handleGetCachedResult(message);
                
            case 'CACHE_RESULT':
                return await this.handleCacheResult(message);
                
            case 'GET_SETTINGS':
                return await this.handleGetSettings();
                
            case 'UPDATE_SETTINGS':
                return await this.handleUpdateSettings(message);
                
            case 'GET_STATISTICS':
                return await this.handleGetStatistics();
                
            case 'CLEAR_CACHE':
                return await this.handleClearCache();
                
            case 'GET_MEMORY_STATS':
                return this.getMemoryStats();
                
            case 'HEALTH_CHECK':
                return this.performHealthCheck();
                
            default:
                throw new Error(`Unknown message type: ${message.type}`);
        }
    }
    
    async handleDetectText(message) {
        const { text, options, requestId } = message;
        
        // Check cache first
        const cacheKey = this.generateCacheKey(text, options);
        const cached = await this.cacheManager.get(cacheKey);
        
        if (cached) {
            this.metrics.counter('background_cache_hits');
            return { ...cached, fromCache: true };
        }
        
        // Get API connection
        const connection = await this.getApiConnection();
        
        try {
            // Make API request with optimization
            const result = await this.makeOptimizedApiRequest(connection, {
                text,
                options,
                requestId
            });
            
            // Cache successful result
            if (result.confidence_score >= 0.7) {
                await this.cacheManager.set(cacheKey, result, 300); // 5 min TTL
            }
            
            this.metrics.counter('background_api_requests');
            return result;
            
        } catch (error) {
            this.logger.error('API request failed:', error);
            this.metrics.counter('background_api_errors');
            
            // Return cached result if available (even expired)
            const expiredCache = await this.cacheManager.getExpired(cacheKey);
            if (expiredCache) {
                return { ...expiredCache, fromCache: true, expired: true };
            }
            
            throw error;
            
        } finally {
            this.releaseApiConnection(connection);
        }
    }
    
    async getApiConnection() {
        // Reuse existing connections
        for (const [id, connection] of this.apiConnections) {
            if (!connection.inUse && connection.isHealthy()) {
                connection.inUse = true;
                return connection;
            }
        }
        
        // Create new connection if under limit
        if (this.apiConnections.size < this.maxConnections) {
            const connection = new OptimizedApiConnection();
            await connection.initialize();
            
            const connectionId = Date.now().toString();
            this.apiConnections.set(connectionId, connection);
            connection.id = connectionId;
            connection.inUse = true;
            
            return connection;
        }
        
        // Wait for available connection
        return await this.waitForAvailableConnection();
    }
    
    async waitForAvailableConnection() {
        return new Promise((resolve) => {
            const checkForConnection = () => {
                for (const [id, connection] of this.apiConnections) {
                    if (!connection.inUse && connection.isHealthy()) {
                        connection.inUse = true;
                        resolve(connection);
                        return;
                    }
                }
                
                // Check again in 100ms
                setTimeout(checkForConnection, 100);
            };
            
            checkForConnection();
        });
    }
    
    releaseApiConnection(connection) {
        if (connection) {
            connection.inUse = false;
            connection.lastUsed = Date.now();
        }
    }
    
    async makeOptimizedApiRequest(connection, payload) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5s timeout
        
        try {
            const response = await fetch(connection.url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Request-ID': payload.requestId
                },
                body: JSON.stringify(payload),
                signal: controller.signal
            });
            
            if (!response.ok) {
                throw new Error(`API request failed: ${response.status}`);
            }
            
            return await response.json();
            
        } finally {
            clearTimeout(timeoutId);
        }
    }
    
    generateCacheKey(text, options) {
        // Generate efficient cache key
        const textHash = this.simpleHash(text);
        const optionsHash = this.simpleHash(JSON.stringify(options || {}));
        return `${textHash}_${optionsHash}`;
    }
    
    simpleHash(str) {
        let hash = 0;
        if (str.length === 0) return hash;
        
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash; // Convert to 32-bit integer
        }
        
        return Math.abs(hash).toString(36);
    }
    
    async handleGetCachedResult(message) {
        const { cacheKey } = message;
        const result = await this.cacheManager.get(cacheKey);
        return result ? { result, fromCache: true } : { result: null };
    }
    
    async handleCacheResult(message) {
        const { cacheKey, result, ttl } = message;
        await this.cacheManager.set(cacheKey, result, ttl || 300);
        return { success: true };
    }
    
    async handleGetSettings() {
        try {
            const settings = await chrome.storage.sync.get([
                'autoDetect',
                'confidenceThreshold',
                'visualIndicators',
                'detectionMethod'
            ]);
            
            return {
                autoDetect: settings.autoDetect ?? true,
                confidenceThreshold: settings.confidenceThreshold ?? 0.7,
                visualIndicators: settings.visualIndicators ?? true,
                detectionMethod: settings.detectionMethod ?? 'balanced'
            };
        } catch (error) {
            this.logger.error('Failed to get settings:', error);
            throw error;
        }
    }
    
    async handleUpdateSettings(message) {
        try {
            const { settings } = message;
            await chrome.storage.sync.set(settings);
            return { success: true };
        } catch (error) {
            this.logger.error('Failed to update settings:', error);
            throw error;
        }
    }
    
    async handleGetStatistics() {
        try {
            const stats = await chrome.storage.local.get(['statistics']);
            return stats.statistics || {
                totalDetections: 0,
                aiDetected: 0,
                humanDetected: 0,
                averageConfidence: 0
            };
        } catch (error) {
            this.logger.error('Failed to get statistics:', error);
            throw error;
        }
    }
    
    async handleClearCache() {
        try {
            await this.cacheManager.clear();
            return { success: true };
        } catch (error) {
            this.logger.error('Failed to clear cache:', error);
            throw error;
        }
    }
    
    getMemoryStats() {
        return {
            memoryOptimizer: this.memoryOptimizer.getMemoryStats(),
            messageQueue: {
                size: this.messageQueue.length,
                maxSize: this.maxQueueSize
            },
            apiConnections: {
                active: this.apiConnections.size,
                maxConnections: this.maxConnections
            },
            cache: this.cacheManager.getStats(),
            performance: this.performanceTracker.getStats()
        };
    }
    
    performHealthCheck() {
        const memoryStats = this.memoryOptimizer.getMemoryStats();
        
        return {
            status: memoryStats.current < 45 ? 'healthy' : 'degraded',
            memory: {
                current: memoryStats.current,
                peak: memoryStats.peak,
                limit: 50
            },
            cache: this.cacheManager.getHealthStatus(),
            connections: {
                active: this.apiConnections.size,
                healthy: Array.from(this.apiConnections.values())
                    .filter(conn => conn.isHealthy()).length
            }
        };
    }
    
    setupPeriodicCleanup() {
        this.cleanupInterval = setInterval(async () => {
            try {
                await this.performCleanup();
            } catch (error) {
                this.logger.error('Cleanup failed:', error);
            }
        }, 5 * 60 * 1000); // Every 5 minutes
    }
    
    async performCleanup() {
        const startTime = performance.now();
        
        // Clean message queue
        this.cleanMessageQueue();
        
        // Clean API connections
        await this.cleanApiConnections();
        
        // Clean cache
        await this.cacheManager.cleanup();
        
        // Trigger memory optimization
        await this.memoryOptimizer.performGarbageCollection();
        
        const duration = performance.now() - startTime;
        this.metrics.histogram('background_cleanup_duration_ms', duration);
        
        this.logger.debug(`Cleanup completed in ${duration.toFixed(1)}ms`);
    }
    
    cleanMessageQueue() {
        const now = Date.now();
        const maxAge = 60 * 1000; // 1 minute
        
        const initialSize = this.messageQueue.length;
        this.messageQueue = this.messageQueue.filter(item => 
            now - item.timestamp < maxAge
        );
        
        const cleaned = initialSize - this.messageQueue.length;
        if (cleaned > 0) {
            this.logger.debug(`Cleaned ${cleaned} old messages from queue`);
        }
    }
    
    async cleanApiConnections() {
        const now = Date.now();
        const maxIdleTime = 10 * 60 * 1000; // 10 minutes
        
        const connectionsToRemove = [];
        
        for (const [id, connection] of this.apiConnections) {
            if (!connection.inUse && 
                (now - connection.lastUsed) > maxIdleTime) {
                connectionsToRemove.push(id);
            }
        }
        
        for (const id of connectionsToRemove) {
            const connection = this.apiConnections.get(id);
            await connection.close();
            this.apiConnections.delete(id);
        }
        
        if (connectionsToRemove.length > 0) {
            this.logger.debug(`Cleaned ${connectionsToRemove.length} idle connections`);
        }
    }
    
    setupHeartbeatMonitoring() {
        this.heartbeatInterval = setInterval(() => {
            this.sendHeartbeat();
        }, 60 * 1000); // Every minute
    }
    
    sendHeartbeat() {
        const stats = this.getMemoryStats();
        
        // Log memory status
        this.logger.debug(`Heartbeat: ${stats.memoryOptimizer.current.toFixed(1)}MB memory`);
        
        // Check for memory issues
        if (stats.memoryOptimizer.current > 45) {
            this.logger.warn('High memory usage detected, triggering cleanup');
            this.performCleanup();
        }
    }
    
    setupAlarmHandlers() {
        // Use alarms for background tasks to avoid keeping service worker alive
        chrome.alarms.onAlarm.addListener((alarm) => {
            switch (alarm.name) {
                case 'cleanup':
                    this.performCleanup();
                    break;
                case 'cache_maintenance':
                    this.cacheManager.performMaintenance();
                    break;
                case 'statistics_update':
                    this.updateStatistics();
                    break;
            }
        });
        
        // Schedule regular alarms
        chrome.alarms.create('cleanup', { periodInMinutes: 5 });
        chrome.alarms.create('cache_maintenance', { periodInMinutes: 15 });
        chrome.alarms.create('statistics_update', { periodInMinutes: 30 });
    }
    
    handleStartup() {
        this.logger.info('Extension startup detected');
        this.performCleanup();
    }
    
    handleInstallation(details) {
        this.logger.info(`Extension ${details.reason}:`, details);
        
        if (details.reason === 'install') {
            // Initialize default settings
            this.initializeDefaultSettings();
        }
    }
    
    async initializeDefaultSettings() {
        const defaultSettings = {
            autoDetect: true,
            confidenceThreshold: 0.7,
            visualIndicators: true,
            detectionMethod: 'balanced'
        };
        
        await chrome.storage.sync.set(defaultSettings);
    }
    
    handleTabUpdate(tabId, tab) {
        // Clean up any tab-specific data
        this.cacheManager.cleanupTabData(tabId);
    }
    
    handleTabRemoval(tabId) {
        // Immediate cleanup for closed tabs
        this.cacheManager.cleanupTabData(tabId);
        
        // Remove any pending messages for this tab
        this.messageQueue = this.messageQueue.filter(item => 
            item.sender?.tab?.id !== tabId
        );
    }
    
    async updateStatistics() {
        try {
            const current = await this.handleGetStatistics();
            const performance = this.performanceTracker.getStats();
            
            const updated = {
                ...current,
                lastUpdated: Date.now(),
                performance: {
                    averageResponseTime: performance.averageResponseTime,
                    cacheHitRate: performance.cacheHitRate,
                    memoryUsage: this.memoryOptimizer.getMemoryStats().current
                }
            };
            
            await chrome.storage.local.set({ statistics: updated });
        } catch (error) {
            this.logger.error('Failed to update statistics:', error);
        }
    }
}

// Optimized API connection class
class OptimizedApiConnection {
    constructor() {
        this.url = 'http://localhost:8000/api/detect';
        this.inUse = false;
        this.lastUsed = Date.now();
        this.healthy = true;
        this.consecutiveErrors = 0;
    }
    
    async initialize() {
        // Test connection health
        try {
            const response = await fetch(this.url + '/health', {
                method: 'GET',
                timeout: 5000
            });
            this.healthy = response.ok;
        } catch (error) {
            this.healthy = false;
        }
    }
    
    isHealthy() {
        return this.healthy && this.consecutiveErrors < 3;
    }
    
    recordError() {
        this.consecutiveErrors++;
        if (this.consecutiveErrors >= 3) {
            this.healthy = false;
        }
    }
    
    recordSuccess() {
        this.consecutiveErrors = 0;
        this.healthy = true;
    }
    
    async close() {
        // Clean up any resources
        this.healthy = false;
    }
}

// Background cache manager
class BackgroundCacheManager {
    constructor() {
        this.cache = new Map();
        this.maxSize = 1000;
        this.maxMemoryMB = 10;
        this.currentMemoryMB = 0;
    }
    
    async initialize() {
        // Load cache from storage if needed
        try {
            const stored = await chrome.storage.local.get(['cache']);
            if (stored.cache) {
                this.loadFromStorage(stored.cache);
            }
        } catch (error) {
            console.warn('Failed to load cache from storage:', error);
        }
    }
    
    loadFromStorage(storedCache) {
        const now = Date.now();
        
        for (const [key, value] of Object.entries(storedCache)) {
            if (value.expires && value.expires > now) {
                this.cache.set(key, value);
                this.currentMemoryMB += this.estimateSize(value) / (1024 * 1024);
            }
        }
    }
    
    async get(key) {
        const item = this.cache.get(key);
        
        if (!item) return null;
        
        if (item.expires && Date.now() > item.expires) {
            this.delete(key);
            return null;
        }
        
        // Update access time
        item.lastAccessed = Date.now();
        return item.data;
    }
    
    async getExpired(key) {
        const item = this.cache.get(key);
        return item ? item.data : null;
    }
    
    async set(key, data, ttlSeconds) {
        const item = {
            data,
            created: Date.now(),
            lastAccessed: Date.now(),
            expires: ttlSeconds ? Date.now() + (ttlSeconds * 1000) : null
        };
        
        const size = this.estimateSize(item);
        
        // Make space if needed
        while (this.currentMemoryMB + (size / (1024 * 1024)) > this.maxMemoryMB && this.cache.size > 0) {
            this.evictLRU();
        }
        
        // Remove old item if exists
        if (this.cache.has(key)) {
            const oldItem = this.cache.get(key);
            this.currentMemoryMB -= this.estimateSize(oldItem) / (1024 * 1024);
        }
        
        this.cache.set(key, item);
        this.currentMemoryMB += size / (1024 * 1024);
        
        // Limit cache size
        if (this.cache.size > this.maxSize) {
            this.evictLRU();
        }
    }
    
    delete(key) {
        const item = this.cache.get(key);
        if (item) {
            this.currentMemoryMB -= this.estimateSize(item) / (1024 * 1024);
            this.cache.delete(key);
        }
    }
    
    evictLRU() {
        let oldestKey = null;
        let oldestTime = Date.now();
        
        for (const [key, item] of this.cache) {
            if (item.lastAccessed < oldestTime) {
                oldestTime = item.lastAccessed;
                oldestKey = key;
            }
        }
        
        if (oldestKey) {
            this.delete(oldestKey);
        }
    }
    
    async clear() {
        this.cache.clear();
        this.currentMemoryMB = 0;
    }
    
    async cleanup() {
        const now = Date.now();
        const toDelete = [];
        
        for (const [key, item] of this.cache) {
            if (item.expires && now > item.expires) {
                toDelete.push(key);
            }
        }
        
        toDelete.forEach(key => this.delete(key));
        
        return toDelete.length;
    }
    
    async performMaintenance() {
        await this.cleanup();
        
        // Persist cache to storage
        try {
            const cacheData = Object.fromEntries(this.cache);
            await chrome.storage.local.set({ cache: cacheData });
        } catch (error) {
            console.warn('Failed to persist cache:', error);
        }
    }
    
    cleanupTabData(tabId) {
        // Remove any tab-specific cache entries
        const toDelete = [];
        
        for (const [key, item] of this.cache) {
            if (key.includes(`tab_${tabId}`)) {
                toDelete.push(key);
            }
        }
        
        toDelete.forEach(key => this.delete(key));
    }
    
    estimateSize(item) {
        return JSON.stringify(item).length * 2; // Rough estimate
    }
    
    getStats() {
        return {
            size: this.cache.size,
            maxSize: this.maxSize,
            memoryMB: this.currentMemoryMB,
            maxMemoryMB: this.maxMemoryMB
        };
    }
    
    getHealthStatus() {
        const utilizationRatio = this.currentMemoryMB / this.maxMemoryMB;
        
        return {
            status: utilizationRatio < 0.8 ? 'healthy' : 'degraded',
            utilization: utilizationRatio,
            itemCount: this.cache.size
        };
    }
}

// Performance tracker
class PerformanceTracker {
    constructor() {
        this.metrics = {
            messageProcessingTimes: [],
            cacheHits: 0,
            cacheMisses: 0,
            apiRequests: 0,
            apiErrors: 0
        };
    }
    
    recordMessageProcessing(type, duration) {
        this.metrics.messageProcessingTimes.push({ type, duration, timestamp: Date.now() });
        
        // Keep only recent data
        const fiveMinutesAgo = Date.now() - (5 * 60 * 1000);
        this.metrics.messageProcessingTimes = this.metrics.messageProcessingTimes
            .filter(entry => entry.timestamp > fiveMinutesAgo);
    }
    
    getStats() {
        const times = this.metrics.messageProcessingTimes.map(entry => entry.duration);
        
        return {
            averageResponseTime: times.length > 0 ? 
                times.reduce((a, b) => a + b, 0) / times.length : 0,
            cacheHitRate: this.metrics.cacheHits / 
                Math.max(this.metrics.cacheHits + this.metrics.cacheMisses, 1),
            apiSuccessRate: this.metrics.apiRequests > 0 ? 
                (this.metrics.apiRequests - this.metrics.apiErrors) / this.metrics.apiRequests : 0,
            recentMessages: this.metrics.messageProcessingTimes.length
        };
    }
}

// Mock classes for standalone operation
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
        console.debug(`[${this.name}] DEBUG:`, message, ...args);
    }
}

class MetricsCollector {
    constructor() {
        this.counters = new Map();
        this.histograms = new Map();
        this.gauges = new Map();
    }
    
    counter(name, labels = {}) {
        const key = `${name}_${JSON.stringify(labels)}`;
        this.counters.set(key, (this.counters.get(key) || 0) + 1);
    }
    
    histogram(name, value, labels = {}) {
        const key = `${name}_${JSON.stringify(labels)}`;
        const values = this.histograms.get(key) || [];
        values.push(value);
        this.histograms.set(key, values);
    }
    
    gauge(name, value, labels = {}) {
        const key = `${name}_${JSON.stringify(labels)}`;
        this.gauges.set(key, value);
    }
}

// Initialize background script
const optimizedBackground = new OptimizedBackgroundScript();