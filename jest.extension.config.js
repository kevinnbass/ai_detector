/**
 * Jest Configuration for Chrome Extension Components
 * Specialized configuration for testing extension-specific functionality
 */

const baseConfig = require('./jest.config.js');

module.exports = {
  ...baseConfig,
  
  // Display name
  displayName: 'Chrome Extension',
  
  // Test environment with Chrome extension mocks
  testEnvironment: 'jsdom',
  
  // Setup files specific to extension testing
  setupFilesAfterEnv: [
    '<rootDir>/tests/js/setup.js',
    '<rootDir>/tests/js/extension-setup.js'
  ],
  
  // Focus on extension files only
  testMatch: [
    '<rootDir>/tests/js/extension/**/*.test.js',
    '<rootDir>/extension/src/**/*.test.js'
  ],
  
  // Coverage specific to extension
  collectCoverageFrom: [
    'extension/src/**/*.js',
    '!extension/src/**/*.test.js',
    '!extension/src/vendor/**',
    '!extension/src/manifest.json'
  ],
  
  // Extension-specific coverage thresholds
  coverageThreshold: {
    global: {
      branches: 75,
      functions: 80,
      lines: 80,
      statements: 80
    },
    './extension/src/background/': {
      branches: 85,
      functions: 90,
      lines: 90,
      statements: 90
    },
    './extension/src/content/': {
      branches: 85,
      functions: 90,
      lines: 90,
      statements: 90
    },
    './extension/src/shared/': {
      branches: 90,
      functions: 95,
      lines: 95,
      statements: 95
    }
  },
  
  // Coverage directory
  coverageDirectory: 'coverage/js/extension',
  
  // Global Chrome extension mocks
  globals: {
    ...baseConfig.globals,
    chrome: {
      runtime: {},
      storage: {},
      tabs: {},
      action: {},
      contextMenus: {},
      webNavigation: {},
      scripting: {}
    }
  },
  
  // Module name mapping for extension
  moduleNameMapping: {
    ...baseConfig.moduleNameMapping,
    '^chrome$': '<rootDir>/tests/js/mocks/chrome.js',
    '^webextension-polyfill$': '<rootDir>/tests/js/mocks/webextension-polyfill.js'
  },
  
  // Test timeout for extension tests
  testTimeout: 15000,
  
  // Reporters
  reporters: [
    'default',
    ['jest-html-reporter', {
      pageTitle: 'Chrome Extension Test Report',
      outputPath: 'test-results/js/extension/index.html',
      includeFailureMsg: true,
      includeSuiteFailure: true
    }]
  ]
};