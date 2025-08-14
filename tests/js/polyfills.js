/**
 * Test Environment Polyfills
 * Polyfills for browser APIs not available in test environment
 */

// TextEncoder/TextDecoder polyfill
import { TextEncoder, TextDecoder } from 'util';

global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

// Performance API polyfill
global.performance = global.performance || {
  now: () => Date.now(),
  mark: () => {},
  measure: () => {},
  getEntriesByType: () => [],
  getEntriesByName: () => []
};

// RequestAnimationFrame polyfill
global.requestAnimationFrame = global.requestAnimationFrame || ((callback) => {
  setTimeout(callback, 16);
});

global.cancelAnimationFrame = global.cancelAnimationFrame || ((id) => {
  clearTimeout(id);
});

// IntersectionObserver polyfill
global.IntersectionObserver = global.IntersectionObserver || class IntersectionObserver {
  constructor(callback, options = {}) {
    this.callback = callback;
    this.options = options;
    this.observedElements = new Set();
  }

  observe(element) {
    this.observedElements.add(element);
    // Simulate immediate intersection
    setTimeout(() => {
      this.callback([{
        target: element,
        isIntersecting: true,
        intersectionRatio: 1,
        boundingClientRect: element.getBoundingClientRect(),
        intersectionRect: element.getBoundingClientRect(),
        rootBounds: null,
        time: performance.now()
      }]);
    }, 0);
  }

  unobserve(element) {
    this.observedElements.delete(element);
  }

  disconnect() {
    this.observedElements.clear();
  }
};

// MutationObserver polyfill
global.MutationObserver = global.MutationObserver || class MutationObserver {
  constructor(callback) {
    this.callback = callback;
    this.observedElements = new Set();
  }

  observe(element, config) {
    this.observedElements.add({ element, config });
  }

  disconnect() {
    this.observedElements.clear();
  }

  takeRecords() {
    return [];
  }

  // Test helper to trigger mutations
  triggerMutation(element, type = 'childList') {
    const mutation = {
      type,
      target: element,
      addedNodes: [],
      removedNodes: [],
      previousSibling: null,
      nextSibling: null,
      attributeName: null,
      attributeNamespace: null,
      oldValue: null
    };
    
    this.callback([mutation]);
  }
};

// ResizeObserver polyfill
global.ResizeObserver = global.ResizeObserver || class ResizeObserver {
  constructor(callback) {
    this.callback = callback;
    this.observedElements = new Set();
  }

  observe(element) {
    this.observedElements.add(element);
    // Simulate resize event
    setTimeout(() => {
      this.callback([{
        target: element,
        contentRect: {
          x: 0,
          y: 0,
          width: 100,
          height: 100,
          top: 0,
          right: 100,
          bottom: 100,
          left: 0
        }
      }]);
    }, 0);
  }

  unobserve(element) {
    this.observedElements.delete(element);
  }

  disconnect() {
    this.observedElements.clear();
  }
};

// URL polyfill for older environments
if (!global.URL) {
  global.URL = class URL {
    constructor(url, base = '') {
      const fullUrl = base ? `${base}/${url}` : url;
      const parts = fullUrl.match(/^(https?:)\/\/([^\/]+)(\/.*)?$/);
      
      if (parts) {
        this.protocol = parts[1];
        this.host = parts[2];
        this.pathname = parts[3] || '/';
        this.href = fullUrl;
      } else {
        this.protocol = 'https:';
        this.host = 'localhost';
        this.pathname = '/';
        this.href = 'https://localhost/';
      }
      
      this.origin = `${this.protocol}//${this.host}`;
    }
    
    toString() {
      return this.href;
    }
  };
}

// URLSearchParams polyfill
if (!global.URLSearchParams) {
  global.URLSearchParams = class URLSearchParams {
    constructor(init = '') {
      this.params = new Map();
      
      if (typeof init === 'string') {
        init.replace(/^\?/, '').split('&').forEach(pair => {
          const [key, value] = pair.split('=');
          if (key) {
            this.params.set(decodeURIComponent(key), decodeURIComponent(value || ''));
          }
        });
      }
    }
    
    get(name) {
      return this.params.get(name);
    }
    
    set(name, value) {
      this.params.set(name, value);
    }
    
    has(name) {
      return this.params.has(name);
    }
    
    delete(name) {
      this.params.delete(name);
    }
    
    toString() {
      const pairs = [];
      for (const [key, value] of this.params) {
        pairs.push(`${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
      }
      return pairs.join('&');
    }
  };
}

// Blob polyfill for file handling
if (!global.Blob) {
  global.Blob = class Blob {
    constructor(parts = [], options = {}) {
      this.parts = parts;
      this.type = options.type || '';
      this.size = parts.reduce((size, part) => {
        return size + (typeof part === 'string' ? part.length : part.byteLength || 0);
      }, 0);
    }
    
    text() {
      return Promise.resolve(this.parts.join(''));
    }
    
    arrayBuffer() {
      const text = this.parts.join('');
      const buffer = new ArrayBuffer(text.length);
      const view = new Uint8Array(buffer);
      for (let i = 0; i < text.length; i++) {
        view[i] = text.charCodeAt(i);
      }
      return Promise.resolve(buffer);
    }
  };
}

// File polyfill
if (!global.File) {
  global.File = class File extends global.Blob {
    constructor(parts, name, options = {}) {
      super(parts, options);
      this.name = name;
      this.lastModified = options.lastModified || Date.now();
    }
  };
}

// FormData polyfill
if (!global.FormData) {
  global.FormData = class FormData {
    constructor() {
      this.data = new Map();
    }
    
    append(name, value, filename) {
      const entry = { value, filename };
      if (this.data.has(name)) {
        const existing = this.data.get(name);
        if (Array.isArray(existing)) {
          existing.push(entry);
        } else {
          this.data.set(name, [existing, entry]);
        }
      } else {
        this.data.set(name, entry);
      }
    }
    
    delete(name) {
      this.data.delete(name);
    }
    
    get(name) {
      const entry = this.data.get(name);
      if (Array.isArray(entry)) {
        return entry[0].value;
      }
      return entry ? entry.value : null;
    }
    
    has(name) {
      return this.data.has(name);
    }
    
    set(name, value, filename) {
      this.data.set(name, { value, filename });
    }
  };
}

// Headers polyfill
if (!global.Headers) {
  global.Headers = class Headers {
    constructor(init = {}) {
      this.headers = new Map();
      
      if (init) {
        Object.entries(init).forEach(([key, value]) => {
          this.headers.set(key.toLowerCase(), value);
        });
      }
    }
    
    get(name) {
      return this.headers.get(name.toLowerCase());
    }
    
    set(name, value) {
      this.headers.set(name.toLowerCase(), value);
    }
    
    has(name) {
      return this.headers.has(name.toLowerCase());
    }
    
    delete(name) {
      this.headers.delete(name.toLowerCase());
    }
    
    entries() {
      return this.headers.entries();
    }
    
    keys() {
      return this.headers.keys();
    }
    
    values() {
      return this.headers.values();
    }
  };
}

// AbortController polyfill
if (!global.AbortController) {
  global.AbortController = class AbortController {
    constructor() {
      this.signal = {
        aborted: false,
        addEventListener: () => {},
        removeEventListener: () => {},
        dispatchEvent: () => true
      };
    }
    
    abort() {
      this.signal.aborted = true;
    }
  };
}

// Storage polyfill for localStorage/sessionStorage
const createStorageMock = () => {
  const storage = new Map();
  
  return {
    getItem: (key) => storage.get(key) || null,
    setItem: (key, value) => storage.set(key, String(value)),
    removeItem: (key) => storage.delete(key),
    clear: () => storage.clear(),
    key: (index) => Array.from(storage.keys())[index] || null,
    get length() {
      return storage.size;
    }
  };
};

if (!global.localStorage) {
  global.localStorage = createStorageMock();
}

if (!global.sessionStorage) {
  global.sessionStorage = createStorageMock();
}