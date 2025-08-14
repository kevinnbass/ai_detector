/**
 * Integration Tests for Chrome Extension Components
 * Tests for interactions between extension components and external systems
 */

import { describe, test, expect, beforeEach, afterEach, jest } from '@jest/globals';
import { MessageBus } from '../../../extension/src/shared/message-bus.js';

// Mock Chrome APIs
global.chrome = {
  runtime: {
    sendMessage: jest.fn(),
    onMessage: {
      addListener: jest.fn(),
      removeListener: jest.fn()
    },
    connect: jest.fn(),
    onConnect: {
      addListener: jest.fn()
    }
  },
  tabs: {
    query: jest.fn(),
    sendMessage: jest.fn(),
    onUpdated: {
      addListener: jest.fn()
    }
  },
  storage: {
    local: {
      get: jest.fn(),
      set: jest.fn(),
      remove: jest.fn()
    },
    sync: {
      get: jest.fn(),
      set: jest.fn()
    }
  },
  scripting: {
    executeScript: jest.fn()
  }
};

describe('Extension Component Integration', () => {
  let messageBus;

  beforeEach(() => {
    // Reset all mocks
    jest.clearAllMocks();
    
    // Create fresh message bus instance
    messageBus = new MessageBus();
  });

  afterEach(() => {
    if (messageBus) {
      messageBus.cleanup();
    }
  });

  describe('Content Script to Background Communication', () => {
    test('should send detection request from content script to background', async () => {
      // Mock successful response
      const mockResponse = {
        success: true,
        result: {
          prediction: 'ai',
          confidence: 0.85,
          ai_probability: 0.82
        }
      };
      
      chrome.runtime.sendMessage.mockResolvedValue(mockResponse);

      // Simulate content script sending detection request
      const response = await messageBus.send('DETECT_TEXT', {
        text: 'This is a formal analysis that demonstrates academic rigor.',
        url: 'https://twitter.com/user/status/123',
        elementId: 'tweet-123'
      });

      expect(chrome.runtime.sendMessage).toHaveBeenCalledWith({
        type: 'DETECT_TEXT',
        data: {
          text: 'This is a formal analysis that demonstrates academic rigor.',
          url: 'https://twitter.com/user/status/123',
          elementId: 'tweet-123'
        },
        timestamp: expect.any(Number),
        id: expect.any(String)
      });

      expect(response.success).toBe(true);
      expect(response.result.prediction).toBe('ai');
    });

    test('should handle detection errors gracefully', async () => {
      // Mock error response
      chrome.runtime.sendMessage.mockRejectedValue(new Error('Detection service unavailable'));

      const response = await messageBus.send('DETECT_TEXT', {
        text: 'Test text'
      });

      expect(response.success).toBe(false);
      expect(response.error).toContain('Detection service unavailable');
    });

    test('should batch multiple detection requests', async () => {
      // Mock batch response
      const mockBatchResponse = {
        success: true,
        results: [
          { prediction: 'human', confidence: 0.9 },
          { prediction: 'ai', confidence: 0.8 }
        ]
      };
      
      chrome.runtime.sendMessage.mockResolvedValue(mockBatchResponse);

      const texts = [
        'hey this is so cool! 😎',
        'It is important to note that this analysis requires consideration.'
      ];

      const response = await messageBus.send('DETECT_BATCH', {
        texts: texts,
        url: 'https://twitter.com/timeline'
      });

      expect(chrome.runtime.sendMessage).toHaveBeenCalledWith({
        type: 'DETECT_BATCH',
        data: {
          texts: texts,
          url: 'https://twitter.com/timeline'
        },
        timestamp: expect.any(Number),
        id: expect.any(String)
      });

      expect(response.success).toBe(true);
      expect(response.results).toHaveLength(2);
    });
  });

  describe('Background to API Server Communication', () => {
    test('should forward detection requests to API server', async () => {
      // Mock fetch for API calls
      global.fetch = jest.fn();
      
      const mockApiResponse = {
        ok: true,
        json: async () => ({
          prediction: 'ai',
          ai_probability: 0.87,
          confidence: 0.91,
          processing_time: 0.234
        })
      };
      
      global.fetch.mockResolvedValue(mockApiResponse);

      // Simulate background script handling detection request
      const handleDetectionRequest = async (message) => {
        const response = await fetch('http://localhost:8000/api/v1/detect', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            text: message.data.text,
            options: {
              include_evidence: true,
              quick_mode: false
            }
          })
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        return await response.json();
      };

      const result = await handleDetectionRequest({
        type: 'DETECT_TEXT',
        data: {
          text: 'This formal text requires analysis.'
        }
      });

      expect(global.fetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/detect',
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            text: 'This formal text requires analysis.',
            options: {
              include_evidence: true,
              quick_mode: false
            }
          })
        }
      );

      expect(result.prediction).toBe('ai');
      expect(result.confidence).toBe(0.91);
    });

    test('should handle API server errors with retry logic', async () => {
      global.fetch = jest.fn();
      
      // First two calls fail, third succeeds
      global.fetch
        .mockRejectedValueOnce(new Error('Network error'))
        .mockRejectedValueOnce(new Error('Server error'))
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ prediction: 'human', confidence: 0.8 })
        });

      const retryApiCall = async (url, options, maxRetries = 3) => {
        let lastError;
        
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
          try {
            const response = await fetch(url, options);
            if (response.ok) {
              return await response.json();
            }
            throw new Error(`HTTP ${response.status}`);
          } catch (error) {
            lastError = error;
            if (attempt < maxRetries) {
              await new Promise(resolve => setTimeout(resolve, 100 * attempt));
            }
          }
        }
        
        throw lastError;
      };

      const result = await retryApiCall('http://localhost:8000/api/v1/detect', {
        method: 'POST',
        body: JSON.stringify({ text: 'test' })
      });

      expect(global.fetch).toHaveBeenCalledTimes(3);
      expect(result.prediction).toBe('human');
    });
  });

  describe('Extension Settings and Storage Integration', () => {
    test('should save and load extension settings', async () => {
      const mockSettings = {
        autoDetect: true,
        threshold: 0.7,
        apiEndpoint: 'http://localhost:8000',
        enableNotifications: true
      };

      // Mock storage.sync.set
      chrome.storage.sync.set.mockResolvedValue(undefined);
      chrome.storage.sync.get.mockResolvedValue({ settings: mockSettings });

      // Simulate saving settings
      await new Promise((resolve) => {
        chrome.storage.sync.set({ settings: mockSettings }, resolve);
      });

      // Simulate loading settings
      const loadedData = await new Promise((resolve) => {
        chrome.storage.sync.get(['settings'], resolve);
      });

      expect(chrome.storage.sync.set).toHaveBeenCalledWith({ settings: mockSettings });
      expect(loadedData.settings).toEqual(mockSettings);
    });

    test('should migrate old settings format', async () => {
      // Mock old settings format
      const oldSettings = {
        enabled: true,
        confidence_threshold: 0.8
      };

      chrome.storage.local.get.mockResolvedValue(oldSettings);
      chrome.storage.sync.set.mockResolvedValue(undefined);
      chrome.storage.local.remove.mockResolvedValue(undefined);

      const migrateSettings = async () => {
        const oldData = await new Promise(resolve => 
          chrome.storage.local.get(null, resolve)
        );

        if (oldData.enabled !== undefined) {
          // Convert to new format
          const newSettings = {
            autoDetect: oldData.enabled,
            threshold: oldData.confidence_threshold || 0.7,
            apiEndpoint: 'http://localhost:8000',
            enableNotifications: true
          };

          // Save new format
          await new Promise(resolve => 
            chrome.storage.sync.set({ settings: newSettings }, resolve)
          );

          // Clean up old format
          await new Promise(resolve => 
            chrome.storage.local.remove(Object.keys(oldData), resolve)
          );

          return newSettings;
        }
      };

      const migratedSettings = await migrateSettings();

      expect(migratedSettings.autoDetect).toBe(true);
      expect(migratedSettings.threshold).toBe(0.8);
      expect(chrome.storage.local.remove).toHaveBeenCalled();
    });
  });

  describe('Tab Management and Page Detection', () => {
    test('should detect when user navigates to supported page', async () => {
      const supportedUrls = [
        'https://twitter.com/*',
        'https://x.com/*'
      ];

      const mockTabs = [
        {
          id: 1,
          url: 'https://twitter.com/home',
          active: true
        }
      ];

      chrome.tabs.query.mockResolvedValue(mockTabs);

      const isPageSupported = (url) => {
        return supportedUrls.some(pattern => {
          const regex = new RegExp(pattern.replace('*', '.*'));
          return regex.test(url);
        });
      };

      const handleTabUpdate = async (tabId, changeInfo, tab) => {
        if (changeInfo.status === 'complete' && tab.url) {
          if (isPageSupported(tab.url)) {
            // Inject content script
            await chrome.scripting.executeScript({
              target: { tabId: tab.id },
              files: ['content-script.js']
            });
          }
        }
      };

      // Simulate tab update
      await handleTabUpdate(1, { status: 'complete' }, mockTabs[0]);

      expect(chrome.scripting.executeScript).toHaveBeenCalledWith({
        target: { tabId: 1 },
        files: ['content-script.js']
      });
    });

    test('should handle multiple tabs with detection', async () => {
      const mockTabs = [
        { id: 1, url: 'https://twitter.com/home' },
        { id: 2, url: 'https://x.com/timeline' },
        { id: 3, url: 'https://facebook.com' } // Not supported
      ];

      chrome.tabs.query.mockResolvedValue(mockTabs);

      const setupDetectionForAllTabs = async () => {
        const tabs = await new Promise(resolve => 
          chrome.tabs.query({}, resolve)
        );

        const supportedTabs = tabs.filter(tab => 
          tab.url && (
            tab.url.includes('twitter.com') || 
            tab.url.includes('x.com')
          )
        );

        // Setup detection for each supported tab
        for (const tab of supportedTabs) {
          await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            files: ['content-script.js']
          });
        }

        return supportedTabs.length;
      };

      const supportedCount = await setupDetectionForAllTabs();

      expect(supportedCount).toBe(2);
      expect(chrome.scripting.executeScript).toHaveBeenCalledTimes(2);
    });
  });

  describe('Real-time Detection Flow', () => {
    test('should handle real-time text detection on page', async () => {
      // Mock DOM elements
      const mockElement = {
        textContent: 'This analysis demonstrates sophisticated reasoning patterns.',
        getAttribute: jest.fn(() => 'tweet-123'),
        classList: {
          add: jest.fn(),
          remove: jest.fn(),
          contains: jest.fn(() => false)
        },
        dataset: {}
      };

      // Mock DOM query
      global.document = {
        querySelectorAll: jest.fn(() => [mockElement]),
        createElement: jest.fn(() => ({
          className: '',
          textContent: '',
          style: {}
        }))
      };

      // Mock successful detection
      chrome.runtime.sendMessage.mockResolvedValue({
        success: true,
        result: {
          prediction: 'ai',
          confidence: 0.89,
          ai_probability: 0.86
        }
      });

      const processTextElements = async () => {
        const elements = document.querySelectorAll('[data-testid="tweetText"]');
        const results = [];

        for (const element of elements) {
          const text = element.textContent.trim();
          if (text.length > 10) { // Only process substantial text
            const response = await messageBus.send('DETECT_TEXT', {
              text: text,
              elementId: element.getAttribute('data-testid')
            });

            if (response.success) {
              // Add visual indicator
              element.dataset.aiPrediction = response.result.prediction;
              element.dataset.aiConfidence = response.result.confidence;
              
              if (response.result.prediction === 'ai' && response.result.confidence > 0.8) {
                element.classList.add('ai-detected');
              }

              results.push({
                element: element,
                result: response.result
              });
            }
          }
        }

        return results;
      };

      const results = await processTextElements();

      expect(results).toHaveLength(1);
      expect(results[0].result.prediction).toBe('ai');
      expect(mockElement.classList.add).toHaveBeenCalledWith('ai-detected');
      expect(mockElement.dataset.aiPrediction).toBe('ai');
    });

    test('should throttle detection requests for performance', async () => {
      const throttledDetect = (() => {
        const cache = new Map();
        const pending = new Map();

        return async (text) => {
          // Check cache first
          if (cache.has(text)) {
            return cache.get(text);
          }

          // Check if request is already pending
          if (pending.has(text)) {
            return pending.get(text);
          }

          // Create new request
          const request = messageBus.send('DETECT_TEXT', { text });
          pending.set(text, request);

          try {
            const result = await request;
            cache.set(text, result);
            return result;
          } finally {
            pending.delete(text);
          }
        };
      })();

      chrome.runtime.sendMessage.mockResolvedValue({
        success: true,
        result: { prediction: 'human', confidence: 0.9 }
      });

      const sameText = 'This is the same text for caching test';

      // Make multiple requests for same text
      const [result1, result2, result3] = await Promise.all([
        throttledDetect(sameText),
        throttledDetect(sameText),
        throttledDetect(sameText)
      ]);

      // Should only make one actual API call
      expect(chrome.runtime.sendMessage).toHaveBeenCalledTimes(1);
      expect(result1).toEqual(result2);
      expect(result2).toEqual(result3);
    });
  });

  describe('Error Handling and Fallbacks', () => {
    test('should gracefully handle API server downtime', async () => {
      // Mock API failure
      chrome.runtime.sendMessage.mockRejectedValue(new Error('API server unavailable'));

      const fallbackDetection = async (text) => {
        try {
          return await messageBus.send('DETECT_TEXT', { text });
        } catch (error) {
          // Fallback to simple pattern matching
          const aiPatterns = [
            /important to note/i,
            /it should be mentioned/i,
            /careful consideration/i
          ];

          const hasAiPatterns = aiPatterns.some(pattern => pattern.test(text));
          
          return {
            success: true,
            result: {
              prediction: hasAiPatterns ? 'ai' : 'human',
              confidence: 0.6, // Lower confidence for fallback
              fallback: true
            }
          };
        }
      };

      const result = await fallbackDetection('It is important to note that this requires consideration');

      expect(result.success).toBe(true);
      expect(result.result.prediction).toBe('ai');
      expect(result.result.fallback).toBe(true);
      expect(result.result.confidence).toBe(0.6);
    });

    test('should handle extension context invalidation', async () => {
      // Simulate context invalidation
      const contextInvalidatedError = new Error('Extension context invalidated');
      chrome.runtime.sendMessage.mockRejectedValue(contextInvalidatedError);

      const safeMessageSend = async (type, data) => {
        try {
          return await messageBus.send(type, data);
        } catch (error) {
          if (error.message.includes('context invalidated')) {
            // Reload the extension or show user message
            console.warn('Extension needs to be reloaded');
            return {
              success: false,
              error: 'Extension needs to be reloaded',
              reload_required: true
            };
          }
          throw error;
        }
      };

      const result = await safeMessageSend('DETECT_TEXT', { text: 'test' });

      expect(result.success).toBe(false);
      expect(result.reload_required).toBe(true);
    });
  });

  describe('Performance Optimization', () => {
    test('should debounce rapid detection requests', async () => {
      let callCount = 0;
      chrome.runtime.sendMessage.mockImplementation(async () => {
        callCount++;
        return { success: true, result: { prediction: 'human' } };
      });

      const debounce = (func, delay) => {
        let timeoutId;
        return (...args) => {
          clearTimeout(timeoutId);
          return new Promise((resolve) => {
            timeoutId = setTimeout(async () => {
              const result = await func(...args);
              resolve(result);
            }, delay);
          });
        };
      };

      const debouncedDetect = debounce(
        (text) => messageBus.send('DETECT_TEXT', { text }),
        100
      );

      // Rapidly fire multiple requests
      const promises = [];
      for (let i = 0; i < 5; i++) {
        promises.push(debouncedDetect(`test text ${i}`));
      }

      await Promise.all(promises);

      // Should have made fewer calls due to debouncing
      expect(callCount).toBeLessThan(5);
    });

    test('should implement request queuing for rate limiting', async () => {
      const requestQueue = [];
      let processing = false;

      chrome.runtime.sendMessage.mockImplementation(async () => {
        // Simulate processing delay
        await new Promise(resolve => setTimeout(resolve, 50));
        return { success: true, result: { prediction: 'human' } };
      });

      const queuedDetect = async (text) => {
        return new Promise((resolve, reject) => {
          requestQueue.push({ text, resolve, reject });
          processQueue();
        });
      };

      const processQueue = async () => {
        if (processing || requestQueue.length === 0) return;
        
        processing = true;
        
        while (requestQueue.length > 0) {
          const { text, resolve, reject } = requestQueue.shift();
          
          try {
            const result = await messageBus.send('DETECT_TEXT', { text });
            resolve(result);
          } catch (error) {
            reject(error);
          }
          
          // Rate limit: wait between requests
          await new Promise(r => setTimeout(r, 10));
        }
        
        processing = false;
      };

      // Queue multiple requests
      const texts = ['text1', 'text2', 'text3', 'text4'];
      const results = await Promise.all(
        texts.map(text => queuedDetect(text))
      );

      expect(results).toHaveLength(4);
      expect(results.every(r => r.success)).toBe(true);
    });
  });
});