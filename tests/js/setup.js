/**
 * Jest Setup File
 * Global test setup and configuration
 */

// Import testing utilities
import 'jest-extended';

// Global test utilities
global.sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

global.createMockFunction = (returnValue) => jest.fn(() => returnValue);

global.createMockPromise = (resolveValue, rejectValue = null) => {
  if (rejectValue) {
    return jest.fn(() => Promise.reject(rejectValue));
  }
  return jest.fn(() => Promise.resolve(resolveValue));
};

// Console suppression for cleaner test output
const originalError = console.error;
const originalWarn = console.warn;

beforeAll(() => {
  // Suppress console errors/warnings during tests unless explicitly needed
  console.error = (...args) => {
    if (!args[0]?.includes?.('TEST_LOG')) {
      // Only show errors that include TEST_LOG marker
      return;
    }
    originalError(...args);
  };
  
  console.warn = (...args) => {
    if (!args[0]?.includes?.('TEST_LOG')) {
      return;
    }
    originalWarn(...args);
  };
});

afterAll(() => {
  // Restore console functions
  console.error = originalError;
  console.warn = originalWarn;
});

// Global test data
global.TEST_DATA = {
  SAMPLE_TEXTS: {
    HUMAN: "Just had the best coffee ever! Can't believe how good it was ☕",
    AI_GPT4: "It's important to note that while this approach has merit, one must consider the broader implications and potential drawbacks that might arise.",
    AI_OBVIOUS: "As an AI language model, I must emphasize that this is a complex topic requiring careful consideration of multiple factors."
  },
  
  SAMPLE_TWEETS: [
    {
      id: "1",
      text: "Loving this new restaurant! 🍕",
      author: "user123",
      timestamp: "2024-01-01T12:00:00Z",
      label: "human"
    },
    {
      id: "2", 
      text: "It's crucial to understand that artificial intelligence represents a paradigm shift in computational capabilities.",
      author: "tech_expert",
      timestamp: "2024-01-01T13:00:00Z",
      label: "ai"
    }
  ],
  
  API_RESPONSES: {
    DETECTION_SUCCESS: {
      success: true,
      data: {
        prediction: "human",
        ai_probability: 0.15,
        confidence: {
          value: 0.85,
          level: "high"
        },
        processing_time: 0.123,
        model_version: "v1.0.0"
      }
    },
    
    DETECTION_AI: {
      success: true,
      data: {
        prediction: "ai",
        ai_probability: 0.89,
        confidence: {
          value: 0.92,
          level: "very_high"
        },
        key_indicators: ["formal_language", "hedging", "structure"],
        processing_time: 0.156
      }
    },
    
    ERROR_RESPONSE: {
      success: false,
      error: "Invalid request",
      message: "Text cannot be empty"
    }
  }
};

// Global test helpers
global.TEST_HELPERS = {
  /**
   * Create a mock Chrome API
   */
  createMockChrome: () => ({
    runtime: {
      sendMessage: jest.fn(),
      onMessage: {
        addListener: jest.fn(),
        removeListener: jest.fn()
      },
      getURL: jest.fn((path) => `chrome-extension://test/${path}`),
      id: 'test-extension-id'
    },
    
    storage: {
      sync: {
        get: jest.fn(),
        set: jest.fn(),
        remove: jest.fn(),
        clear: jest.fn()
      },
      local: {
        get: jest.fn(),
        set: jest.fn(), 
        remove: jest.fn(),
        clear: jest.fn()
      },
      onChanged: {
        addListener: jest.fn()
      }
    },
    
    tabs: {
      query: jest.fn(),
      sendMessage: jest.fn(),
      create: jest.fn(),
      update: jest.fn(),
      onUpdated: {
        addListener: jest.fn()
      }
    },
    
    action: {
      setBadgeText: jest.fn(),
      setBadgeBackgroundColor: jest.fn(),
      openPopup: jest.fn()
    }
  }),
  
  /**
   * Create mock DOM elements
   */
  createMockElement: (tagName = 'div', attributes = {}, textContent = '') => {
    const element = document.createElement(tagName);
    
    Object.entries(attributes).forEach(([key, value]) => {
      element.setAttribute(key, value);
    });
    
    if (textContent) {
      element.textContent = textContent;
    }
    
    // Add common methods
    element.getBoundingClientRect = jest.fn(() => ({
      top: 0,
      left: 0,
      bottom: 100,
      right: 100,
      width: 100,
      height: 100,
      x: 0,
      y: 0
    }));
    
    return element;
  },
  
  /**
   * Create mock fetch response
   */
  createMockResponse: (data, status = 200, headers = {}) => ({
    ok: status >= 200 && status < 300,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    headers: new Headers(headers),
    json: jest.fn(() => Promise.resolve(data)),
    text: jest.fn(() => Promise.resolve(JSON.stringify(data))),
    blob: jest.fn(() => Promise.resolve(new Blob([JSON.stringify(data)]))),
    clone: jest.fn(function() { return this; })
  }),
  
  /**
   * Wait for DOM updates
   */
  waitForDOMUpdate: () => new Promise(resolve => setTimeout(resolve, 0)),
  
  /**
   * Create test message
   */
  createTestMessage: (type = 'request', data = {}) => ({
    id: `test-msg-${Date.now()}`,
    type,
    timestamp: Date.now(),
    data,
    source: 'test'
  })
};

// Mock implementations for common browser APIs
Object.defineProperty(window, 'location', {
  value: {
    href: 'https://x.com/test',
    hostname: 'x.com',
    pathname: '/test',
    search: '',
    hash: ''
  },
  writable: true
});

Object.defineProperty(window, 'chrome', {
  value: global.TEST_HELPERS.createMockChrome(),
  writable: true
});

// Mock fetch globally
global.fetch = jest.fn();

// Mock console methods for testing
global.console = {
  ...console,
  log: jest.fn(console.log),
  error: jest.fn(console.error),
  warn: jest.fn(console.warn),
  info: jest.fn(console.info),
  debug: jest.fn(console.debug)
};

// Setup cleanup
afterEach(() => {
  // Clear all mocks
  jest.clearAllMocks();
  
  // Clear fetch mock
  if (global.fetch && global.fetch.mockClear) {
    global.fetch.mockClear();
  }
  
  // Clear timers
  jest.clearAllTimers();
  
  // Clear DOM
  document.body.innerHTML = '';
  document.head.innerHTML = '';
});