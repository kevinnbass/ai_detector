/**
 * Performance tests for Chrome extension memory optimization.
 * 
 * Tests to ensure extension memory usage stays under 50MB
 * across different usage scenarios and page types.
 */

describe('Extension Memory Performance', () => {
    let memoryOptimizer;
    let backgroundScript;
    let contentScript;
    let mockChrome;
    
    beforeEach(() => {
        // Setup mock Chrome APIs
        mockChrome = setupMockChromeAPIs();
        global.chrome = mockChrome;
        
        // Initialize components
        memoryOptimizer = new MemoryOptimizer();
        backgroundScript = new OptimizedBackgroundScript();
        contentScript = new OptimizedContentScript();
    });
    
    afterEach(() => {
        // Cleanup
        if (memoryOptimizer) memoryOptimizer.destroy();
        if (backgroundScript) backgroundScript.destroy();
        if (contentScript) contentScript.destroy();
    });
    
    describe('Memory Usage Limits', () => {
        test('should maintain total memory under 50MB', async () => {
            // Simulate heavy usage
            await simulateHeavyUsage();
            
            const memoryStats = memoryOptimizer.getMemoryStats();
            expect(memoryStats.current).toBeLessThan(50);
        });
        
        test('should handle memory pressure gracefully', async () => {
            // Force high memory usage
            await forceHighMemoryUsage();
            
            // Should trigger emergency cleanup
            const memoryStats = memoryOptimizer.getMemoryStats();
            expect(memoryStats.current).toBeLessThan(45); // Should be under emergency threshold
        });
        
        test('should maintain memory efficiency over time', async () => {
            const measurements = [];
            
            // Monitor memory over 10 minutes of simulated usage
            for (let i = 0; i < 60; i++) {
                await simulateNormalUsage();
                await sleep(10000); // 10 seconds
                
                const stats = memoryOptimizer.getMemoryStats();
                measurements.push(stats.current);
            }
            
            // Memory should not continuously grow
            const initialMemory = measurements[0];
            const finalMemory = measurements[measurements.length - 1];
            const growth = finalMemory - initialMemory;
            
            expect(growth).toBeLessThan(10); // Less than 10MB growth over 10 minutes
        });
    });
    
    describe('Cache Management', () => {
        test('should limit cache size effectively', async () => {
            const cacheManager = backgroundScript.cacheManager;
            
            // Add many cache entries
            for (let i = 0; i < 2000; i++) {
                await cacheManager.set(`key_${i}`, {
                    result: 'test_result',
                    timestamp: Date.now(),
                    size: 1024 // 1KB
                });
            }
            
            const stats = cacheManager.getStats();
            expect(stats.size).toBeLessThanOrEqual(1000); // Max cache items
            expect(stats.memoryMB).toBeLessThan(10); // Max cache memory
        });
        
        test('should clean expired cache entries', async () => {
            const cacheManager = backgroundScript.cacheManager;
            
            // Add entries with short TTL
            for (let i = 0; i < 100; i++) {
                await cacheManager.set(`temp_${i}`, { data: 'test' }, 1); // 1 second TTL
            }
            
            // Wait for expiration
            await sleep(2000);
            
            // Trigger cleanup
            const cleaned = await cacheManager.cleanup();
            expect(cleaned).toBeGreaterThan(90); // Most entries should be cleaned
        });
        
        test('should use LRU eviction correctly', async () => {
            const cache = new MemoryEfficientCache({
                maxSize: 10,
                maxAge: 60000,
                estimateSize: (obj) => JSON.stringify(obj).length
            });
            
            // Fill cache beyond capacity
            for (let i = 0; i < 15; i++) {
                cache.set(`key_${i}`, { data: `value_${i}` });
            }
            
            expect(cache.size).toBeLessThanOrEqual(10);
            
            // Oldest entries should be evicted
            expect(cache.get('key_0')).toBeUndefined();
            expect(cache.get('key_14')).toBeDefined();
        });
    });
    
    describe('DOM Processing Optimization', () => {
        test('should limit processed elements efficiently', async () => {
            // Create large DOM
            const elements = createMockDOMElements(1000);
            
            // Process elements
            await contentScript.processBatch(elements);
            
            const stats = contentScript.getContentStats();
            expect(stats.processedElements).toBeLessThanOrEqual(500); // Max processed elements
        });
        
        test('should clean up detached elements', async () => {
            const mockElements = createMockDOMElements(100);
            
            // Process elements
            await contentScript.processBatch(mockElements);
            
            // Simulate elements being removed from DOM
            mockElements.forEach(el => el.isConnected = false);
            
            // Trigger cleanup
            await contentScript.performCleanup();
            
            const stats = contentScript.getContentStats();
            expect(stats.visualIndicators).toBe(0); // Should be cleaned up
        });
        
        test('should throttle DOM mutations correctly', async () => {
            const mutationHandler = jest.spyOn(contentScript, 'throttledMutationHandler');
            
            // Simulate rapid mutations
            for (let i = 0; i < 10; i++) {
                contentScript.throttledMutationHandler([{ type: 'childList', addedNodes: [] }]);
            }
            
            // Should be throttled
            expect(mutationHandler).toHaveBeenCalledTimes(1);
        });
    });
    
    describe('Message Queue Management', () => {
        test('should limit message queue size', async () => {
            const background = backgroundScript;
            
            // Flood with messages
            for (let i = 0; i < 200; i++) {
                background.messageQueue.push({
                    message: { type: 'TEST', data: `test_${i}` },
                    timestamp: Date.now()
                });
            }
            
            // Should limit queue size
            expect(background.messageQueue.length).toBeLessThanOrEqual(100);
        });
        
        test('should clean old messages from queue', async () => {
            const background = backgroundScript;
            
            // Add old messages
            const oldTimestamp = Date.now() - (2 * 60 * 1000); // 2 minutes ago
            for (let i = 0; i < 50; i++) {
                background.messageQueue.push({
                    message: { type: 'OLD_TEST' },
                    timestamp: oldTimestamp
                });
            }
            
            // Trigger cleanup
            background.cleanMessageQueue();
            
            expect(background.messageQueue.length).toBe(0);
        });
    });
    
    describe('Connection Pool Optimization', () => {
        test('should limit API connections', async () => {
            const background = backgroundScript;
            
            // Try to create many connections
            const connections = [];
            for (let i = 0; i < 10; i++) {
                const conn = await background.getApiConnection();
                connections.push(conn);
            }
            
            expect(background.apiConnections.size).toBeLessThanOrEqual(5); // Max connections
        });
        
        test('should clean idle connections', async () => {
            const background = backgroundScript;
            
            // Create connections and mark as idle
            for (let i = 0; i < 3; i++) {
                const conn = await background.getApiConnection();
                conn.lastUsed = Date.now() - (15 * 60 * 1000); // 15 minutes ago
                background.releaseApiConnection(conn);
            }
            
            // Trigger cleanup
            await background.cleanApiConnections();
            
            expect(background.apiConnections.size).toBe(0);
        });
    });
    
    describe('Garbage Collection', () => {
        test('should perform regular garbage collection', async () => {
            const initialCollections = memoryOptimizer.memoryStats.collections;
            
            // Wait for GC cycles
            await sleep(6000); // 6 seconds
            
            const finalCollections = memoryOptimizer.memoryStats.collections;
            expect(finalCollections).toBeGreaterThan(initialCollections);
        });
        
        test('should free memory during GC', async () => {
            // Create memory pressure
            const largeData = [];
            for (let i = 0; i < 1000; i++) {
                largeData.push(new Array(1000).fill('test'));
            }
            
            const beforeGC = memoryOptimizer.getMemoryStats().current;
            
            // Clear references and trigger GC
            largeData.length = 0;
            await memoryOptimizer.performGarbageCollection();
            
            const afterGC = memoryOptimizer.getMemoryStats().current;
            expect(afterGC).toBeLessThan(beforeGC);
        });
    });
    
    describe('Visual Indicators Optimization', () => {
        test('should limit visual indicators count', async () => {
            const content = contentScript;
            
            // Try to add many indicators
            const elements = createMockDOMElements(100);
            for (const element of elements) {
                content.addVisualIndicator(element, {
                    is_ai_generated: true,
                    confidence_score: 0.8
                });
            }
            
            expect(content.visualIndicators.size).toBeLessThanOrEqual(50);
        });
        
        test('should auto-remove old indicators', async () => {
            const content = contentScript;
            const element = createMockDOMElements(1)[0];
            
            // Add indicator with auto-removal
            content.addVisualIndicator(element, {
                is_ai_generated: true,
                confidence_score: 0.8
            });
            
            expect(content.visualIndicators.size).toBe(1);
            
            // Wait for auto-removal (mock timing)
            await sleep(31000); // 31 seconds
            
            expect(content.visualIndicators.size).toBe(0);
        });
    });
    
    describe('Performance Under Load', () => {
        test('should handle concurrent operations efficiently', async () => {
            const startTime = performance.now();
            
            // Simulate concurrent operations
            const operations = [];
            for (let i = 0; i < 20; i++) {
                operations.push(simulateDetectionRequest());
                operations.push(simulateCacheOperation());
                operations.push(simulateDOMProcessing());
            }
            
            await Promise.all(operations);
            
            const duration = performance.now() - startTime;
            const memoryStats = memoryOptimizer.getMemoryStats();
            
            expect(duration).toBeLessThan(5000); // Should complete in under 5 seconds
            expect(memoryStats.current).toBeLessThan(50); // Memory should stay under limit
        });
        
        test('should maintain performance during extended usage', async () => {
            const performanceMetrics = [];
            
            // Simulate 30 minutes of usage
            for (let minute = 0; minute < 30; minute++) {
                const startTime = performance.now();
                
                await simulateMinuteOfUsage();
                
                const duration = performance.now() - startTime;
                const memory = memoryOptimizer.getMemoryStats().current;
                
                performanceMetrics.push({ minute, duration, memory });
            }
            
            // Performance should not degrade significantly
            const firstMinute = performanceMetrics[0];
            const lastMinute = performanceMetrics[performanceMetrics.length - 1];
            
            const performanceDegradation = lastMinute.duration / firstMinute.duration;
            expect(performanceDegradation).toBeLessThan(2); // Less than 2x degradation
            
            const memoryGrowth = lastMinute.memory - firstMinute.memory;
            expect(memoryGrowth).toBeLessThan(15); // Less than 15MB growth
        });
    });
    
    describe('Emergency Memory Management', () => {
        test('should trigger emergency cleanup at threshold', async () => {
            const emergency = jest.spyOn(memoryOptimizer, 'performEmergencyCleanup');
            
            // Force memory over threshold
            await forceMemoryOverThreshold();
            
            expect(emergency).toHaveBeenCalled();
        });
        
        test('should reduce memory after emergency cleanup', async () => {
            // Force high memory usage
            await forceHighMemoryUsage();
            
            const beforeCleanup = memoryOptimizer.getMemoryStats().current;
            
            // Trigger emergency cleanup
            await memoryOptimizer.performEmergencyCleanup();
            
            const afterCleanup = memoryOptimizer.getMemoryStats().current;
            expect(afterCleanup).toBeLessThan(beforeCleanup * 0.8); // At least 20% reduction
        });
    });
});

// Helper functions
function setupMockChromeAPIs() {
    return {
        runtime: {
            onMessage: {
                addListener: jest.fn(),
                removeListener: jest.fn()
            },
            sendMessage: jest.fn(),
            onStartup: { addListener: jest.fn() },
            onInstalled: { addListener: jest.fn() }
        },
        tabs: {
            onUpdated: { addListener: jest.fn() },
            onRemoved: { addListener: jest.fn() }
        },
        storage: {
            local: {
                get: jest.fn(() => Promise.resolve({})),
                set: jest.fn(() => Promise.resolve()),
                clear: jest.fn(() => Promise.resolve())
            },
            sync: {
                get: jest.fn(() => Promise.resolve({})),
                set: jest.fn(() => Promise.resolve())
            }
        },
        alarms: {
            create: jest.fn(),
            onAlarm: { addListener: jest.fn() }
        }
    };
}

function createMockDOMElements(count) {
    const elements = [];
    for (let i = 0; i < count; i++) {
        elements.push({
            tagName: 'DIV',
            textContent: `This is test content for element ${i}. It has enough text to be processed.`,
            isConnected: true,
            getBoundingClientRect: () => ({ top: 0, left: 0, width: 100, height: 20 }),
            style: {},
            dataset: {}
        });
    }
    return elements;
}

async function simulateHeavyUsage() {
    // Simulate heavy extension usage
    const operations = [];
    
    for (let i = 0; i < 100; i++) {
        operations.push(simulateDetectionRequest());
    }
    
    for (let i = 0; i < 50; i++) {
        operations.push(simulateCacheOperation());
    }
    
    for (let i = 0; i < 200; i++) {
        operations.push(simulateDOMProcessing());
    }
    
    await Promise.all(operations);
}

async function simulateNormalUsage() {
    const operations = [];
    
    for (let i = 0; i < 10; i++) {
        operations.push(simulateDetectionRequest());
    }
    
    for (let i = 0; i < 5; i++) {
        operations.push(simulateDOMProcessing());
    }
    
    await Promise.all(operations);
}

async function simulateDetectionRequest() {
    return new Promise(resolve => {
        setTimeout(() => {
            // Simulate API request and response
            const result = {
                is_ai_generated: Math.random() > 0.5,
                confidence_score: Math.random(),
                processing_time_ms: Math.random() * 100
            };
            resolve(result);
        }, Math.random() * 100);
    });
}

async function simulateCacheOperation() {
    return new Promise(resolve => {
        // Simulate cache read/write
        const data = { result: 'cached_result', timestamp: Date.now() };
        setTimeout(() => resolve(data), 10);
    });
}

async function simulateDOMProcessing() {
    return new Promise(resolve => {
        // Simulate DOM element processing
        const element = createMockDOMElements(1)[0];
        setTimeout(() => resolve(element), 20);
    });
}

async function simulateMinuteOfUsage() {
    const operations = [];
    
    // Normal usage pattern for one minute
    for (let i = 0; i < 20; i++) {
        operations.push(simulateDetectionRequest());
    }
    
    for (let i = 0; i < 10; i++) {
        operations.push(simulateDOMProcessing());
    }
    
    for (let i = 0; i < 5; i++) {
        operations.push(simulateCacheOperation());
    }
    
    await Promise.all(operations);
}

async function forceHighMemoryUsage() {
    // Create memory pressure
    const largeArrays = [];
    for (let i = 0; i < 100; i++) {
        largeArrays.push(new Array(10000).fill(`large_data_${i}`));
    }
    return largeArrays;
}

async function forceMemoryOverThreshold() {
    // Force memory usage over 45MB threshold
    global.mockMemoryUsage = 47 * 1024 * 1024; // 47MB
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

module.exports = {
    setupMockChromeAPIs,
    createMockDOMElements,
    simulateHeavyUsage,
    simulateNormalUsage
};