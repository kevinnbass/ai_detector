/**
 * Global Teardown for E2E Tests
 * Cleans up test environment after running E2E tests
 */

async function globalTeardown() {
  console.log('Tearing down E2E test environment...');

  // Stop API server
  if (process.env.API_SERVER_PID) {
    try {
      process.kill(process.env.API_SERVER_PID, 'SIGTERM');
      console.log('API server stopped');
    } catch (error) {
      console.error('Error stopping API server:', error);
    }
  }

  // Stop mock server
  if (process.env.MOCK_SERVER_PID) {
    try {
      process.kill(process.env.MOCK_SERVER_PID, 'SIGTERM');
      console.log('Mock server stopped');
    } catch (error) {
      console.error('Error stopping mock server:', error);
    }
  }

  console.log('E2E test environment teardown complete');
}

export default globalTeardown;