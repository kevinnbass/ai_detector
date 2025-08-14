/**
 * Extension-specific Jest Setup
 * Setup for Chrome extension testing environment
 */

// Enhanced Chrome API mock for extension testing
const createEnhancedChromeMock = () => {
  const mockChrome = {
    // Runtime API
    runtime: {
      id: 'test-extension-id',
      getManifest: jest.fn(() => ({
        name: 'AI Detector Test',
        version: '1.0.0',
        manifest_version: 3
      })),
      getURL: jest.fn((path) => `chrome-extension://test-extension-id/${path}`),
      sendMessage: jest.fn(),
      onMessage: {
        addListener: jest.fn(),
        removeListener: jest.fn(),
        hasListener: jest.fn(() => false)
      },
      onInstalled: {
        addListener: jest.fn()
      },
      onStartup: {
        addListener: jest.fn()
      },
      onSuspend: {
        addListener: jest.fn()
      },
      onConnect: {
        addListener: jest.fn()
      },
      connect: jest.fn(() => ({
        postMessage: jest.fn(),
        onMessage: {
          addListener: jest.fn()
        },
        onDisconnect: {
          addListener: jest.fn()
        }
      })),
      lastError: null
    },
    
    // Storage API
    storage: {
      sync: {
        get: jest.fn((keys, callback) => {
          const mockData = {
            settings: {
              enabled: true,
              threshold: 0.7,
              showOverlay: true
            }
          };
          
          if (callback) {
            callback(mockData);
          }
          return Promise.resolve(mockData);
        }),
        set: jest.fn((items, callback) => {
          if (callback) callback();
          return Promise.resolve();
        }),
        remove: jest.fn((keys, callback) => {
          if (callback) callback();
          return Promise.resolve();
        }),
        clear: jest.fn((callback) => {
          if (callback) callback();
          return Promise.resolve();
        })
      },
      local: {
        get: jest.fn((keys, callback) => {
          const mockData = {
            geminiApiKey: 'test-api-key',
            analysisStats: {
              totalAnalyses: 100,
              aiDetected: 25,
              humanDetected: 75
            }
          };
          
          if (callback) {
            callback(mockData);
          }
          return Promise.resolve(mockData);
        }),
        set: jest.fn((items, callback) => {
          if (callback) callback();
          return Promise.resolve();
        }),
        remove: jest.fn((keys, callback) => {
          if (callback) callback();
          return Promise.resolve();
        }),
        clear: jest.fn((callback) => {
          if (callback) callback();
          return Promise.resolve();
        })
      },
      onChanged: {
        addListener: jest.fn(),
        removeListener: jest.fn()
      }
    },
    
    // Tabs API
    tabs: {
      query: jest.fn((queryInfo, callback) => {
        const mockTabs = [
          {
            id: 1,
            url: 'https://x.com/test',
            title: 'Test Page',
            active: true,
            windowId: 1
          }
        ];
        
        if (callback) {
          callback(mockTabs);
        }
        return Promise.resolve(mockTabs);
      }),
      get: jest.fn((tabId, callback) => {
        const mockTab = {
          id: tabId,
          url: 'https://x.com/test',
          title: 'Test Page'
        };
        
        if (callback) {
          callback(mockTab);
        }
        return Promise.resolve(mockTab);
      }),
      sendMessage: jest.fn((tabId, message, options, callback) => {
        const mockResponse = { success: true, data: 'test response' };
        
        if (callback) {
          callback(mockResponse);
        }
        return Promise.resolve(mockResponse);
      }),
      create: jest.fn((createProperties, callback) => {
        const mockTab = {
          id: Date.now(),
          url: createProperties.url,
          title: 'New Tab'
        };
        
        if (callback) {
          callback(mockTab);
        }
        return Promise.resolve(mockTab);
      }),
      update: jest.fn((tabId, updateProperties, callback) => {
        const mockTab = {
          id: tabId,
          url: updateProperties.url,
          title: 'Updated Tab'
        };
        
        if (callback) {
          callback(mockTab);
        }
        return Promise.resolve(mockTab);
      }),
      onUpdated: {
        addListener: jest.fn(),
        removeListener: jest.fn()
      },
      onActivated: {
        addListener: jest.fn()
      }
    },
    
    // Action API (replaces browserAction in MV3)
    action: {
      setBadgeText: jest.fn((details, callback) => {
        if (callback) callback();
        return Promise.resolve();
      }),
      setBadgeBackgroundColor: jest.fn((details, callback) => {
        if (callback) callback();
        return Promise.resolve();
      }),
      setTitle: jest.fn((details, callback) => {
        if (callback) callback();
        return Promise.resolve();
      }),
      setIcon: jest.fn((details, callback) => {
        if (callback) callback();
        return Promise.resolve();
      }),
      openPopup: jest.fn(),
      onClicked: {
        addListener: jest.fn(),
        removeListener: jest.fn()
      }
    },
    
    // Context Menus API
    contextMenus: {
      create: jest.fn((createProperties, callback) => {
        if (callback) callback();
        return Promise.resolve('menu-item-id');
      }),
      update: jest.fn((id, updateProperties, callback) => {
        if (callback) callback();
        return Promise.resolve();
      }),
      remove: jest.fn((menuItemId, callback) => {
        if (callback) callback();
        return Promise.resolve();
      }),
      removeAll: jest.fn((callback) => {
        if (callback) callback();
        return Promise.resolve();
      }),
      onClicked: {
        addListener: jest.fn(),
        removeListener: jest.fn()
      }
    },
    
    // Web Navigation API
    webNavigation: {
      onCompleted: {
        addListener: jest.fn(),
        removeListener: jest.fn()
      },
      onBeforeNavigate: {
        addListener: jest.fn()
      }
    },
    
    // Scripting API (MV3)
    scripting: {
      executeScript: jest.fn((injection, callback) => {
        const result = [{ result: 'script executed' }];
        if (callback) {
          callback(result);
        }
        return Promise.resolve(result);
      }),
      insertCSS: jest.fn((injection, callback) => {
        if (callback) callback();
        return Promise.resolve();
      }),
      removeCSS: jest.fn((injection, callback) => {
        if (callback) callback();
        return Promise.resolve();
      })
    }
  };
  
  return mockChrome;
};

// Set up global Chrome mock
global.chrome = createEnhancedChromeMock();
window.chrome = global.chrome;

// Mock browser for WebExtensions polyfill compatibility
global.browser = global.chrome;
window.browser = global.chrome;

// Extension-specific test utilities
global.EXTENSION_TEST_HELPERS = {
  /**
   * Simulate extension message passing
   */
  simulateMessage: (message, sender = {}, sendResponse = jest.fn()) => {
    const mockSender = {
      tab: { id: 1, url: 'https://x.com/test' },
      frameId: 0,
      id: 'test-extension-id',
      url: 'chrome-extension://test-extension-id/background.html',
      ...sender
    };
    
    // Trigger message listeners
    const listeners = global.chrome.runtime.onMessage.addListener.mock.calls;
    listeners.forEach(([listener]) => {
      if (typeof listener === 'function') {
        listener(message, mockSender, sendResponse);
      }
    });
    
    return { message, sender: mockSender, sendResponse };
  },
  
  /**
   * Simulate tab update
   */
  simulateTabUpdate: (tabId, changeInfo, tab) => {
    const mockTab = {
      id: tabId,
      url: 'https://x.com/test',
      title: 'Test Page',
      status: 'complete',
      ...tab
    };
    
    const mockChangeInfo = {
      status: 'complete',
      ...changeInfo
    };
    
    // Trigger tab update listeners
    const listeners = global.chrome.tabs.onUpdated.addListener.mock.calls;
    listeners.forEach(([listener]) => {
      if (typeof listener === 'function') {
        listener(tabId, mockChangeInfo, mockTab);
      }
    });
    
    return { tabId, changeInfo: mockChangeInfo, tab: mockTab };
  },
  
  /**
   * Simulate storage change
   */
  simulateStorageChange: (changes, areaName = 'sync') => {
    const mockChanges = {};
    
    Object.entries(changes).forEach(([key, { oldValue, newValue }]) => {
      mockChanges[key] = {
        oldValue,
        newValue
      };
    });
    
    // Trigger storage change listeners
    const listeners = global.chrome.storage.onChanged.addListener.mock.calls;
    listeners.forEach(([listener]) => {
      if (typeof listener === 'function') {
        listener(mockChanges, areaName);
      }
    });
    
    return { changes: mockChanges, areaName };
  },
  
  /**
   * Mock extension installation
   */
  simulateInstallation: (reason = 'install') => {
    const details = {
      reason,
      previousVersion: reason === 'update' ? '0.9.0' : undefined
    };
    
    // Trigger installation listeners
    const listeners = global.chrome.runtime.onInstalled.addListener.mock.calls;
    listeners.forEach(([listener]) => {
      if (typeof listener === 'function') {
        listener(details);
      }
    });
    
    return details;
  },
  
  /**
   * Create mock content script environment
   */
  createContentScriptEnvironment: () => {
    // Mock DOM elements commonly found on X/Twitter
    const tweetText = global.TEST_HELPERS.createMockElement('div', {
      'data-testid': 'tweetText',
      'class': 'tweet-text'
    }, 'This is a sample tweet text for testing');
    
    const article = global.TEST_HELPERS.createMockElement('article', {
      'role': 'article',
      'data-testid': 'tweet'
    });
    
    article.appendChild(tweetText);
    document.body.appendChild(article);
    
    // Mock page title and URL for X/Twitter
    Object.defineProperty(document, 'title', {
      value: 'Test Tweet / X',
      writable: true
    });
    
    Object.defineProperty(window.location, 'href', {
      value: 'https://x.com/test/status/123456789',
      writable: true
    });
    
    return { tweetText, article };
  }
};

// Clean up after each test
afterEach(() => {
  // Reset Chrome API mocks
  Object.values(global.chrome).forEach(api => {
    if (api && typeof api === 'object') {
      Object.values(api).forEach(method => {
        if (method && method.mockClear) {
          method.mockClear();
        } else if (method && typeof method === 'object') {
          Object.values(method).forEach(subMethod => {
            if (subMethod && subMethod.mockClear) {
              subMethod.mockClear();
            }
          });
        }
      });
    }
  });
  
  // Clear last error
  global.chrome.runtime.lastError = null;
});