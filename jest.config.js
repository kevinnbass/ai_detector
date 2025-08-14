/**
 * Jest Configuration for AI Detector Extension
 * Comprehensive testing setup for Chrome extension components
 */

module.exports = {
  // Test environment
  testEnvironment: 'jsdom',
  
  // Setup files
  setupFilesAfterEnv: [
    '<rootDir>/tests/js/setup.js'
  ],
  
  // Test file patterns
  testMatch: [
    '<rootDir>/tests/js/**/*.test.js',
    '<rootDir>/extension/src/**/*.test.js'
  ],
  
  // Coverage collection
  collectCoverageFrom: [
    'extension/src/**/*.js',
    '!extension/src/**/*.test.js',
    '!extension/src/vendor/**',
    '!extension/src/lib/**',
    '!**/node_modules/**'
  ],
  
  // Coverage thresholds
  coverageThreshold: {
    global: {
      branches: 80,
      functions: 85,
      lines: 85,
      statements: 85
    },
    './extension/src/core/': {
      branches: 90,
      functions: 95,
      lines: 95,
      statements: 95
    }
  },
  
  // Coverage reporting
  coverageDirectory: 'coverage/js',
  coverageReporters: [
    'text',
    'text-summary',
    'lcov',
    'html',
    'json'
  ],
  
  // Module name mapping
  moduleNameMapping: {
    '^@/(.*)$': '<rootDir>/extension/src/$1',
    '^@tests/(.*)$': '<rootDir>/tests/js/$1',
    '^@shared/(.*)$': '<rootDir>/extension/src/shared/$1',
    '^@background/(.*)$': '<rootDir>/extension/src/background/$1',
    '^@content/(.*)$': '<rootDir>/extension/src/content/$1',
    '^@popup/(.*)$': '<rootDir>/extension/src/popup/$1'
  },
  
  // Transform configuration
  transform: {
    '^.+\\.js$': 'babel-jest'
  },
  
  // Module file extensions
  moduleFileExtensions: [
    'js',
    'json'
  ],
  
  // Test timeout
  testTimeout: 10000,
  
  // Verbose output
  verbose: true,
  
  // Clear mocks between tests
  clearMocks: true,
  
  // Restore mocks after each test
  restoreMocks: true,
  
  // Error on deprecated features
  errorOnDeprecated: true,
  
  // Notify mode (for watch mode)
  notify: true,
  notifyMode: 'failure-change',
  
  // Test results processor
  testResultsProcessor: '<rootDir>/tests/js/processors/results-processor.js',
  
  // Reporters
  reporters: [
    'default',
    ['jest-html-reporter', {
      pageTitle: 'AI Detector Extension Test Report',
      outputPath: 'test-results/js/index.html',
      includeFailureMsg: true,
      includeSuiteFailure: true
    }],
    ['jest-junit', {
      outputDirectory: 'test-results/js',
      outputName: 'junit.xml'
    }]
  ],
  
  // Global variables
  globals: {
    'chrome': {},
    'browser': {},
    '__DEV__': true,
    '__TEST__': true
  },
  
  // Module directories
  moduleDirectories: [
    'node_modules',
    '<rootDir>/extension/src',
    '<rootDir>/tests/js'
  ],
  
  // Setup files before framework
  setupFiles: [
    '<rootDir>/tests/js/polyfills.js'
  ]
};