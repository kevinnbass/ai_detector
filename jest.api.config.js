/**
 * Jest Configuration for API Testing
 * Specialized configuration for testing API endpoints and integration
 */

const baseConfig = require('./jest.config.js');

module.exports = {
  ...baseConfig,
  
  // Display name
  displayName: 'API Tests',
  
  // Node environment for API testing
  testEnvironment: 'node',
  
  // Setup files specific to API testing
  setupFilesAfterEnv: [
    '<rootDir>/tests/js/api-setup.js'
  ],
  
  // Focus on API test files
  testMatch: [
    '<rootDir>/tests/js/api/**/*.test.js',
    '<rootDir>/tests/js/integration/**/*.test.js'
  ],
  
  // No coverage collection for API tests (Python handles backend coverage)
  collectCoverage: false,
  
  // Coverage directory
  coverageDirectory: 'coverage/js/api',
  
  // Global variables for API testing
  globals: {
    ...baseConfig.globals,
    __API_BASE_URL__: 'http://localhost:8000',
    __TEST_TIMEOUT__: 30000
  },
  
  // Module name mapping for API tests
  moduleNameMapping: {
    '^@api/(.*)$': '<rootDir>/tests/js/api/$1',
    '^@fixtures/(.*)$': '<rootDir>/tests/js/fixtures/$1',
    '^@utils/(.*)$': '<rootDir>/tests/js/utils/$1'
  },
  
  // Longer timeout for API tests
  testTimeout: 30000,
  
  // Sequential execution for API tests to avoid conflicts
  maxWorkers: 1,
  
  // Reporters
  reporters: [
    'default',
    ['jest-html-reporter', {
      pageTitle: 'API Integration Test Report',
      outputPath: 'test-results/js/api/index.html',
      includeFailureMsg: true,
      includeSuiteFailure: true
    }]
  ],
  
  // Transform configuration for Node environment
  transform: {
    '^.+\\.js$': 'babel-jest'
  }
};