/**
 * API Testing Setup
 * Setup for testing API endpoints and integration
 */

import { jest } from '@jest/globals';

// API testing configuration
global.API_TEST_CONFIG = {
  BASE_URL: process.env.API_BASE_URL || 'http://localhost:8000',
  TIMEOUT: 30000,
  RETRY_ATTEMPTS: 3,
  RETRY_DELAY: 1000
};

// Mock HTTP client for API testing
class MockHTTPClient {
  constructor() {
    this.requests = [];
    this.responses = new Map();
    this.defaultResponse = {
      status: 200,
      data: { success: true },
      headers: { 'content-type': 'application/json' }
    };
  }

  // Record all requests
  request(config) {
    this.requests.push({
      ...config,
      timestamp: Date.now()
    });

    const key = `${config.method?.toUpperCase() || 'GET'} ${config.url}`;
    const mockResponse = this.responses.get(key) || this.defaultResponse;

    if (mockResponse.status >= 400) {
      const error = new Error(`HTTP ${mockResponse.status}`);
      error.response = mockResponse;
      return Promise.reject(error);
    }

    return Promise.resolve(mockResponse);
  }

  get(url, config = {}) {
    return this.request({ ...config, method: 'GET', url });
  }

  post(url, data, config = {}) {
    return this.request({ ...config, method: 'POST', url, data });
  }

  put(url, data, config = {}) {
    return this.request({ ...config, method: 'PUT', url, data });
  }

  delete(url, config = {}) {
    return this.request({ ...config, method: 'DELETE', url });
  }

  // Mock response setup
  mockResponse(method, url, response) {
    const key = `${method.toUpperCase()} ${url}`;
    this.responses.set(key, response);
  }

  // Clear mocks
  clearMocks() {
    this.requests = [];
    this.responses.clear();
  }

  // Get request history
  getRequests() {
    return [...this.requests];
  }
}

global.mockHTTPClient = new MockHTTPClient();

// API test utilities
global.API_TEST_HELPERS = {
  /**
   * Create test server response
   */
  createAPIResponse: (data, status = 200, headers = {}) => ({
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    data,
    headers: {
      'content-type': 'application/json',
      ...headers
    }
  }),

  /**
   * Create detection API response
   */
  createDetectionResponse: (prediction = 'human', confidence = 0.85) => ({
    status: 200,
    data: {
      success: true,
      result: {
        prediction,
        ai_probability: prediction === 'ai' ? confidence : 1 - confidence,
        confidence: {
          value: confidence,
          level: confidence > 0.8 ? 'high' : confidence > 0.6 ? 'medium' : 'low'
        },
        key_indicators: prediction === 'ai' ? ['formal_language', 'hedging'] : [],
        processing_time: Math.random() * 0.5,
        model_version: 'v1.0.0'
      }
    }
  }),

  /**
   * Create error API response
   */
  createErrorResponse: (message = 'Internal Server Error', status = 500, code = 'INTERNAL_ERROR') => ({
    status,
    data: {
      success: false,
      error: message,
      error_code: code,
      timestamp: new Date().toISOString()
    }
  }),

  /**
   * Setup common API mocks
   */
  setupCommonMocks: () => {
    // Health check endpoint
    global.mockHTTPClient.mockResponse('GET', '/health', {
      status: 200,
      data: {
        status: 'healthy',
        version: '1.0.0',
        timestamp: new Date().toISOString()
      }
    });

    // Detection endpoint - success case
    global.mockHTTPClient.mockResponse('POST', '/api/v1/detect', 
      global.API_TEST_HELPERS.createDetectionResponse('human', 0.85)
    );

    // Settings endpoint
    global.mockHTTPClient.mockResponse('GET', '/api/v1/settings', {
      status: 200,
      data: {
        detection_threshold: 0.7,
        model_version: 'v1.0.0',
        features_enabled: ['pattern_detection', 'llm_analysis']
      }
    });

    // Statistics endpoint
    global.mockHTTPClient.mockResponse('GET', '/api/v1/stats', {
      status: 200,
      data: {
        total_detections: 1000,
        ai_detected: 250,
        human_detected: 750,
        average_confidence: 0.82
      }
    });
  },

  /**
   * Wait for API response
   */
  waitForResponse: (timeout = 5000) => {
    return new Promise((resolve) => {
      setTimeout(resolve, Math.min(100, timeout));
    });
  },

  /**
   * Simulate network delay
   */
  simulateNetworkDelay: (min = 50, max = 200) => {
    const delay = Math.random() * (max - min) + min;
    return new Promise(resolve => setTimeout(resolve, delay));
  },

  /**
   * Create WebSocket mock
   */
  createWebSocketMock: () => {
    const mockWS = {
      CONNECTING: 0,
      OPEN: 1,
      CLOSING: 2,
      CLOSED: 3,
      readyState: 1,
      url: 'ws://localhost:8000/ws',
      protocol: '',
      
      // Event handlers
      onopen: null,
      onclose: null,
      onmessage: null,
      onerror: null,
      
      // Methods
      send: jest.fn(),
      close: jest.fn(),
      
      // Test helpers
      triggerOpen: function() {
        if (this.onopen) this.onopen({ type: 'open' });
      },
      
      triggerMessage: function(data) {
        if (this.onmessage) {
          this.onmessage({
            type: 'message',
            data: typeof data === 'string' ? data : JSON.stringify(data)
          });
        }
      },
      
      triggerClose: function(code = 1000, reason = '') {
        this.readyState = 3;
        if (this.onclose) {
          this.onclose({ type: 'close', code, reason });
        }
      },
      
      triggerError: function(error = new Error('WebSocket error')) {
        if (this.onerror) {
          this.onerror({ type: 'error', error });
        }
      }
    };

    return mockWS;
  },

  /**
   * Test API endpoint
   */
  testEndpoint: async (method, endpoint, data = null, expectedStatus = 200) => {
    const config = {
      method: method.toUpperCase(),
      url: endpoint,
      ...(data && { data })
    };

    try {
      const response = await global.mockHTTPClient.request(config);
      
      expect(response.status).toBe(expectedStatus);
      
      if (expectedStatus < 400) {
        expect(response.data).toBeDefined();
        if (response.data.success !== undefined) {
          expect(response.data.success).toBe(true);
        }
      }
      
      return response;
    } catch (error) {
      if (expectedStatus >= 400) {
        expect(error.response.status).toBe(expectedStatus);
        return error.response;
      }
      throw error;
    }
  }
};

// Mock fetch for API integration tests
global.fetch = jest.fn((url, options = {}) => {
  const method = options.method || 'GET';
  const config = {
    method,
    url: url.toString(),
    ...(options.body && { data: JSON.parse(options.body) }),
    headers: options.headers || {}
  };

  return global.mockHTTPClient.request(config)
    .then(response => ({
      ok: response.status >= 200 && response.status < 300,
      status: response.status,
      statusText: response.statusText || 'OK',
      headers: new Headers(response.headers),
      json: () => Promise.resolve(response.data),
      text: () => Promise.resolve(JSON.stringify(response.data)),
      blob: () => Promise.resolve(new Blob([JSON.stringify(response.data)]))
    }))
    .catch(error => {
      if (error.response) {
        return Promise.resolve({
          ok: false,
          status: error.response.status,
          statusText: 'Error',
          headers: new Headers(error.response.headers),
          json: () => Promise.resolve(error.response.data),
          text: () => Promise.resolve(JSON.stringify(error.response.data))
        });
      }
      return Promise.reject(error);
    });
});

// Mock WebSocket
global.WebSocket = jest.fn(() => global.API_TEST_HELPERS.createWebSocketMock());

// Setup default mocks before each test
beforeEach(() => {
  global.API_TEST_HELPERS.setupCommonMocks();
});

// Cleanup after each test
afterEach(() => {
  global.mockHTTPClient.clearMocks();
  jest.clearAllMocks();
});

// Test environment validation
beforeAll(() => {
  console.log(`API Tests running with base URL: ${global.API_TEST_CONFIG.BASE_URL}`);
});