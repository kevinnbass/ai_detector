/**
 * Performance Benchmarks for Chrome Extension Components
 * JavaScript performance testing and monitoring for extension functionality
 */

import { performance } from 'perf_hooks';

class PerformanceTimer {
  constructor() {
    this.measurements = new Map();
  }

  start(label) {
    this.measurements.set(label, {
      startTime: performance.now(),
      startMemory: this.getMemoryUsage()
    });
  }

  end(label) {
    const measurement = this.measurements.get(label);
    if (!measurement) {
      throw new Error(`No measurement started for label: ${label}`);
    }

    const endTime = performance.now();
    const endMemory = this.getMemoryUsage();
    
    const result = {
      duration: endTime - measurement.startTime,
      memoryDelta: endMemory - measurement.startMemory,
      timestamp: Date.now()
    };

    this.measurements.delete(label);
    return result;
  }

  getMemoryUsage() {
    if (typeof process !== 'undefined' && process.memoryUsage) {
      return process.memoryUsage().heapUsed / 1024 / 1024; // MB
    }
    return 0;
  }
}

class ExtensionBenchmark {
  constructor() {
    this.timer = new PerformanceTimer();
    this.results = [];
  }

  async measureOperation(name, operation, iterations = 100) {
    const times = [];
    const memoryDeltas = [];
    let errors = 0;

    for (let i = 0; i < iterations; i++) {
      try {
        this.timer.start(`${name}_${i}`);
        await operation();
        const result = this.timer.end(`${name}_${i}`);
        
        times.push(result.duration);
        memoryDeltas.push(result.memoryDelta);
      } catch (error) {
        errors++;
        console.error(`Error in ${name} iteration ${i}:`, error);
      }
    }

    const metrics = this.calculateMetrics(name, times, memoryDeltas, errors);
    this.results.push(metrics);
    return metrics;
  }

  calculateMetrics(name, times, memoryDeltas, errors) {
    if (times.length === 0) {
      return {
        name,
        iterations: 0,
        errors,
        avgTime: 0,
        minTime: 0,
        maxTime: 0,
        p95Time: 0,
        p99Time: 0,
        throughput: 0,
        avgMemoryDelta: 0
      };
    }

    times.sort((a, b) => a - b);
    const avgTime = times.reduce((sum, time) => sum + time, 0) / times.length;
    const minTime = times[0];
    const maxTime = times[times.length - 1];
    const p95Time = times[Math.floor(times.length * 0.95)];
    const p99Time = times[Math.floor(times.length * 0.99)];
    const throughput = 1000 / avgTime; // operations per second
    const avgMemoryDelta = memoryDeltas.reduce((sum, delta) => sum + delta, 0) / memoryDeltas.length;

    return {
      name,
      iterations: times.length,
      errors,
      avgTime,
      minTime,
      maxTime,
      p95Time,
      p99Time,
      throughput,
      avgMemoryDelta,
      timestamp: Date.now()
    };
  }

  printResults() {
    console.log('\n='.repeat(80));
    console.log('EXTENSION PERFORMANCE BENCHMARK RESULTS');
    console.log('='.repeat(80));

    this.results.forEach(result => {
      console.log(`\nOperation: ${result.name}`);
      console.log(`  Iterations: ${result.iterations} successful, ${result.errors} errors`);
      console.log(`  Average Time: ${result.avgTime.toFixed(2)}ms`);
      console.log(`  P95 Time: ${result.p95Time.toFixed(2)}ms`);
      console.log(`  P99 Time: ${result.p99Time.toFixed(2)}ms`);
      console.log(`  Throughput: ${result.throughput.toFixed(2)} ops/sec`);
      console.log(`  Memory Delta: ${result.avgMemoryDelta.toFixed(2)}MB`);

      // Performance warnings
      if (result.avgTime > 100) {
        console.log('  ⚠️  SLOW: Average time > 100ms');
      }
      if (result.p95Time > 500) {
        console.log('  ⚠️  SLOW: P95 time > 500ms');
      }
      if (result.avgMemoryDelta > 10) {
        console.log('  ⚠️  HIGH MEMORY: Average memory delta > 10MB');
      }
      if (result.errors > 0) {
        console.log(`  ❌ ERRORS: ${result.errors} operations failed`);
      }
    });
  }

  exportResults(filename) {
    const exportData = {
      timestamp: new Date().toISOString(),
      userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'Node.js',
      results: this.results
    };

    if (typeof require !== 'undefined') {
      const fs = require('fs');
      const path = require('path');
      
      const outputPath = path.join('test-results', filename || `extension-benchmark-${Date.now()}.json`);
      
      // Ensure directory exists
      const dir = path.dirname(outputPath);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
      
      fs.writeFileSync(outputPath, JSON.stringify(exportData, null, 2));
      console.log(`\nResults exported to: ${outputPath}`);
    }

    return exportData;
  }
}

// Mock Chrome APIs for benchmarking
const mockChromeAPIs = {
  runtime: {
    sendMessage: (message) => {
      return new Promise(resolve => {
        // Simulate network delay
        setTimeout(() => {
          resolve({
            success: true,
            result: {
              prediction: Math.random() > 0.5 ? 'ai' : 'human',
              confidence: 0.7 + Math.random() * 0.3
            }
          });
        }, 10 + Math.random() * 40); // 10-50ms delay
      });
    },
    
    onMessage: {
      addListener: () => {},
      removeListener: () => {}
    }
  },

  storage: {
    local: {
      get: (keys) => {
        return new Promise(resolve => {
          const data = {};
          if (Array.isArray(keys)) {
            keys.forEach(key => {
              data[key] = `mock_value_${key}`;
            });
          } else if (typeof keys === 'object') {
            Object.keys(keys).forEach(key => {
              data[key] = keys[key];
            });
          }
          setTimeout(() => resolve(data), 1 + Math.random() * 5);
        });
      },
      
      set: (data) => {
        return new Promise(resolve => {
          setTimeout(() => resolve(), 1 + Math.random() * 5);
        });
      }
    },

    sync: {
      get: (keys) => mockChromeAPIs.storage.local.get(keys),
      set: (data) => mockChromeAPIs.storage.local.set(data)
    }
  },

  tabs: {
    query: (queryInfo) => {
      return new Promise(resolve => {
        const mockTabs = [
          { id: 1, url: 'https://twitter.com/home', active: true },
          { id: 2, url: 'https://x.com/timeline', active: false }
        ];
        setTimeout(() => resolve(mockTabs), 2 + Math.random() * 8);
      });
    },

    sendMessage: (tabId, message) => {
      return mockChromeAPIs.runtime.sendMessage(message);
    }
  }
};

// Set up global Chrome API
if (typeof global !== 'undefined') {
  global.chrome = mockChromeAPIs;
}

// MessageBus performance benchmarks
class MessageBusBenchmarks {
  constructor() {
    this.benchmark = new ExtensionBenchmark();
  }

  async benchmarkMessageSending() {
    const MessageBus = (await import('../../extension/src/shared/message-bus.js')).MessageBus;
    const messageBus = new MessageBus();

    return await this.benchmark.measureOperation(
      'message_bus_send',
      async () => {
        await messageBus.send('DETECT_TEXT', {
          text: 'This is a test message for performance benchmarking.',
          elementId: 'benchmark_element'
        });
      },
      50
    );
  }

  async benchmarkBatchMessaging() {
    const MessageBus = (await import('../../extension/src/shared/message-bus.js')).MessageBus;
    const messageBus = new MessageBus();

    const batchSize = 10;
    const testTexts = Array(batchSize).fill(0).map((_, i) => 
      `Batch message ${i} for performance testing.`
    );

    return await this.benchmark.measureOperation(
      'message_bus_batch',
      async () => {
        await messageBus.send('DETECT_BATCH', {
          texts: testTexts,
          url: 'https://twitter.com/benchmark'
        });
      },
      20
    );
  }

  async benchmarkConcurrentMessages() {
    const MessageBus = (await import('../../extension/src/shared/message-bus.js')).MessageBus;
    const messageBus = new MessageBus();

    const concurrentCount = 5;

    return await this.benchmark.measureOperation(
      'message_bus_concurrent',
      async () => {
        const promises = Array(concurrentCount).fill(0).map((_, i) =>
          messageBus.send('DETECT_TEXT', {
            text: `Concurrent message ${i} for benchmarking.`,
            elementId: `element_${i}`
          })
        );
        await Promise.all(promises);
      },
      10
    );
  }
}

// Content Script performance benchmarks
class ContentScriptBenchmarks {
  constructor() {
    this.benchmark = new ExtensionBenchmark();
  }

  async benchmarkDOMTraversal() {
    // Mock DOM for testing
    const mockDocument = {
      querySelectorAll: (selector) => {
        // Simulate finding tweet elements
        const elements = Array(50).fill(0).map((_, i) => ({
          textContent: `This is mock tweet content ${i} for DOM traversal benchmarking.`,
          getAttribute: () => `tweet-${i}`,
          classList: {
            add: () => {},
            remove: () => {},
            contains: () => false
          },
          dataset: {}
        }));
        return elements;
      }
    };

    return await this.benchmark.measureOperation(
      'dom_traversal',
      async () => {
        const elements = mockDocument.querySelectorAll('[data-testid="tweetText"]');
        let processed = 0;
        
        for (const element of elements) {
          if (element.textContent && element.textContent.length > 10) {
            // Simulate processing
            element.dataset.processed = 'true';
            processed++;
          }
        }
        
        return processed;
      },
      100
    );
  }

  async benchmarkTextExtraction() {
    const mockElements = Array(20).fill(0).map((_, i) => ({
      textContent: `This is mock tweet ${i} with various lengths and content for text extraction benchmarking. `.repeat(Math.floor(Math.random() * 5) + 1),
      children: [],
      getAttribute: () => `tweet-${i}`
    }));

    return await this.benchmark.measureOperation(
      'text_extraction',
      async () => {
        const extractedTexts = [];
        
        for (const element of mockElements) {
          // Simulate text extraction and cleaning
          let text = element.textContent.trim();
          text = text.replace(/\s+/g, ' '); // Normalize whitespace
          text = text.substring(0, 1000); // Limit length
          
          if (text.length > 10) {
            extractedTexts.push({
              text,
              elementId: element.getAttribute('data-testid'),
              length: text.length
            });
          }
        }
        
        return extractedTexts;
      },
      100
    );
  }

  async benchmarkUIUpdates() {
    const mockElements = Array(10).fill(0).map((_, i) => ({
      classList: {
        add: () => {},
        remove: () => {},
        contains: () => false
      },
      dataset: {},
      appendChild: () => {},
      style: {}
    }));

    return await this.benchmark.measureOperation(
      'ui_updates',
      async () => {
        for (const element of mockElements) {
          // Simulate adding AI detection indicators
          element.dataset.aiPrediction = Math.random() > 0.5 ? 'ai' : 'human';
          element.dataset.aiConfidence = (0.5 + Math.random() * 0.5).toFixed(2);
          
          if (element.dataset.aiPrediction === 'ai') {
            element.classList.add('ai-detected');
          }
          
          // Simulate creating and adding indicator element
          const indicator = {
            className: 'ai-indicator',
            textContent: 'AI',
            style: {}
          };
          element.appendChild(indicator);
        }
      },
      50
    );
  }
}

// Storage performance benchmarks
class StorageBenchmarks {
  constructor() {
    this.benchmark = new ExtensionBenchmark();
  }

  async benchmarkStorageRead() {
    return await this.benchmark.measureOperation(
      'storage_read',
      async () => {
        const data = await chrome.storage.sync.get([
          'settings', 'collectedSamples', 'statistics', 'preferences'
        ]);
        return Object.keys(data).length;
      },
      100
    );
  }

  async benchmarkStorageWrite() {
    return await this.benchmark.measureOperation(
      'storage_write',
      async () => {
        const sampleData = {
          id: `sample_${Date.now()}_${Math.random()}`,
          text: 'This is benchmark sample data for storage testing.',
          label: Math.random() > 0.5 ? 'ai' : 'human',
          timestamp: Date.now(),
          confidence: 0.5 + Math.random() * 0.5
        };
        
        await chrome.storage.local.set({
          [`sample_${sampleData.id}`]: sampleData
        });
      },
      50
    );
  }

  async benchmarkBulkStorageOperations() {
    const bulkData = {};
    for (let i = 0; i < 100; i++) {
      bulkData[`bulk_item_${i}`] = {
        id: i,
        content: `Bulk storage item ${i} for performance testing.`,
        timestamp: Date.now(),
        metadata: { processed: true, version: 1 }
      };
    }

    return await this.benchmark.measureOperation(
      'bulk_storage_write',
      async () => {
        await chrome.storage.local.set(bulkData);
      },
      10
    );
  }
}

// Detection flow benchmarks
class DetectionFlowBenchmarks {
  constructor() {
    this.benchmark = new ExtensionBenchmark();
  }

  async benchmarkFullDetectionFlow() {
    const MessageBus = (await import('../../extension/src/shared/message-bus.js')).MessageBus;
    const messageBus = new MessageBus();

    return await this.benchmark.measureOperation(
      'full_detection_flow',
      async () => {
        // Simulate complete detection flow
        const text = 'This is a comprehensive text for full detection flow benchmarking that includes various patterns and characteristics.';
        
        // 1. Extract text (simulated)
        const extractedText = text.trim();
        
        // 2. Send detection request
        const response = await messageBus.send('DETECT_TEXT', {
          text: extractedText,
          url: 'https://twitter.com/benchmark',
          elementId: 'benchmark_tweet'
        });
        
        // 3. Process response
        if (response.success) {
          const prediction = response.result.prediction;
          const confidence = response.result.confidence;
          
          // 4. Update UI (simulated)
          const uiUpdate = {
            prediction,
            confidence,
            timestamp: Date.now()
          };
          
          // 5. Store result (simulated)
          await chrome.storage.local.set({
            [`detection_${Date.now()}`]: uiUpdate
          });
        }
        
        return response;
      },
      30
    );
  }

  async benchmarkBatchDetectionFlow() {
    const MessageBus = (await import('../../extension/src/shared/message-bus.js')).MessageBus;
    const messageBus = new MessageBus();

    const batchTexts = Array(20).fill(0).map((_, i) => 
      `Batch detection text ${i} for comprehensive flow benchmarking with various content patterns.`
    );

    return await this.benchmark.measureOperation(
      'batch_detection_flow',
      async () => {
        // Simulate batch detection flow
        const response = await messageBus.send('DETECT_BATCH', {
          texts: batchTexts,
          url: 'https://twitter.com/batch_benchmark'
        });
        
        if (response.success && response.results) {
          // Process batch results
          const processedResults = response.results.map((result, index) => ({
            text: batchTexts[index],
            prediction: result.prediction,
            confidence: result.confidence,
            index
          }));
          
          // Bulk storage update
          const storageUpdates = {};
          processedResults.forEach((result, i) => {
            storageUpdates[`batch_result_${Date.now()}_${i}`] = result;
          });
          
          await chrome.storage.local.set(storageUpdates);
        }
        
        return response;
      },
      15
    );
  }
}

// Memory leak detection
class MemoryLeakBenchmarks {
  constructor() {
    this.benchmark = new ExtensionBenchmark();
  }

  async benchmarkMemoryLeaks() {
    const initialMemory = this.benchmark.timer.getMemoryUsage();
    let maxMemory = initialMemory;
    const iterations = 1000;

    return await this.benchmark.measureOperation(
      'memory_leak_detection',
      async () => {
        // Create objects that might leak
        const data = Array(100).fill(0).map((_, i) => ({
          id: i,
          content: `Memory test data ${i}`,
          timestamp: Date.now(),
          references: new Array(50).fill(`ref_${i}`)
        }));
        
        // Process data
        const processed = data.map(item => ({
          ...item,
          processed: true,
          hash: item.content.length
        }));
        
        // Check memory usage
        const currentMemory = this.benchmark.timer.getMemoryUsage();
        if (currentMemory > maxMemory) {
          maxMemory = currentMemory;
        }
        
        // Cleanup (important for preventing actual leaks)
        data.length = 0;
        processed.length = 0;
        
        return {
          currentMemory,
          maxMemory,
          memoryIncrease: currentMemory - initialMemory
        };
      },
      iterations
    );
  }
}

// Main benchmark runner
async function runExtensionBenchmarks() {
  console.log('Starting Chrome Extension Performance Benchmarks...');
  
  const results = [];

  try {
    // Message Bus Benchmarks
    console.log('\n🔄 Running Message Bus Benchmarks...');
    const messageBenchmarks = new MessageBusBenchmarks();
    
    results.push(await messageBenchmarks.benchmarkMessageSending());
    results.push(await messageBenchmarks.benchmarkBatchMessaging());
    results.push(await messageBenchmarks.benchmarkConcurrentMessages());

    // Content Script Benchmarks
    console.log('\n📄 Running Content Script Benchmarks...');
    const contentBenchmarks = new ContentScriptBenchmarks();
    
    results.push(await contentBenchmarks.benchmarkDOMTraversal());
    results.push(await contentBenchmarks.benchmarkTextExtraction());
    results.push(await contentBenchmarks.benchmarkUIUpdates());

    // Storage Benchmarks
    console.log('\n💾 Running Storage Benchmarks...');
    const storageBenchmarks = new StorageBenchmarks();
    
    results.push(await storageBenchmarks.benchmarkStorageRead());
    results.push(await storageBenchmarks.benchmarkStorageWrite());
    results.push(await storageBenchmarks.benchmarkBulkStorageOperations());

    // Detection Flow Benchmarks
    console.log('\n🤖 Running Detection Flow Benchmarks...');
    const detectionBenchmarks = new DetectionFlowBenchmarks();
    
    results.push(await detectionBenchmarks.benchmarkFullDetectionFlow());
    results.push(await detectionBenchmarks.benchmarkBatchDetectionFlow());

    // Memory Leak Benchmarks
    console.log('\n🧠 Running Memory Leak Detection...');
    const memoryBenchmarks = new MemoryLeakBenchmarks();
    
    results.push(await memoryBenchmarks.benchmarkMemoryLeaks());

    // Compile and display results
    const combinedBenchmark = new ExtensionBenchmark();
    combinedBenchmark.results = results;
    
    combinedBenchmark.printResults();
    combinedBenchmark.exportResults();

    // Performance requirement checks
    console.log('\n✅ Performance Requirement Checks:');
    
    const messageResult = results.find(r => r.name === 'message_bus_send');
    if (messageResult && messageResult.p95Time < 100) {
      console.log('✅ Message sending P95 < 100ms');
    } else {
      console.log('❌ Message sending P95 >= 100ms');
    }

    const domResult = results.find(r => r.name === 'dom_traversal');
    if (domResult && domResult.avgTime < 50) {
      console.log('✅ DOM traversal average < 50ms');
    } else {
      console.log('❌ DOM traversal average >= 50ms');
    }

    const memoryResult = results.find(r => r.name === 'memory_leak_detection');
    if (memoryResult && memoryResult.avgMemoryDelta < 5) {
      console.log('✅ Memory usage increase < 5MB');
    } else {
      console.log('❌ Memory usage increase >= 5MB');
    }

  } catch (error) {
    console.error('Error running benchmarks:', error);
  }
}

// Export for testing
export {
  ExtensionBenchmark,
  MessageBusBenchmarks,
  ContentScriptBenchmarks,
  StorageBenchmarks,
  DetectionFlowBenchmarks,
  MemoryLeakBenchmarks,
  runExtensionBenchmarks
};

// Run benchmarks if called directly
if (typeof require !== 'undefined' && require.main === module) {
  runExtensionBenchmarks().catch(console.error);
}