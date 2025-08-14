/**
 * Global Setup for E2E Tests
 * Sets up test environment before running E2E tests
 */

import { chromium } from '@playwright/test';
import { spawn } from 'child_process';
import path from 'path';
import fs from 'fs';

let apiServer;
let mockServer;

async function globalSetup() {
  console.log('Setting up E2E test environment...');

  // 1. Start API server
  console.log('Starting API server...');
  apiServer = spawn('python', ['src/api/server.py'], {
    cwd: path.join(__dirname, '../..'),
    stdio: 'pipe',
    env: { ...process.env, TESTING: 'true', PORT: '8000' }
  });

  // Wait for API server to start
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('API server failed to start within 30 seconds'));
    }, 30000);

    apiServer.stdout.on('data', (data) => {
      const output = data.toString();
      console.log('API Server:', output);
      if (output.includes('Server running') || output.includes('Uvicorn running')) {
        clearTimeout(timeout);
        resolve();
      }
    });

    apiServer.stderr.on('data', (data) => {
      console.error('API Server Error:', data.toString());
    });

    apiServer.on('error', (error) => {
      clearTimeout(timeout);
      reject(error);
    });
  });

  // 2. Start mock Twitter server for testing
  console.log('Starting mock server...');
  mockServer = spawn('node', ['tests/e2e/mock-server.js'], {
    cwd: path.join(__dirname, '../..'),
    stdio: 'pipe',
    env: { ...process.env, PORT: '8080' }
  });

  // Wait for mock server to start
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      reject(new Error('Mock server failed to start within 10 seconds'));
    }, 10000);

    mockServer.stdout.on('data', (data) => {
      const output = data.toString();
      console.log('Mock Server:', output);
      if (output.includes('listening on port 8080')) {
        clearTimeout(timeout);
        resolve();
      }
    });

    mockServer.stderr.on('data', (data) => {
      console.error('Mock Server Error:', data.toString());
    });

    mockServer.on('error', (error) => {
      clearTimeout(timeout);
      reject(error);
    });
  });

  // 3. Set up test data directory
  const testDataDir = path.join(__dirname, '../test-results');
  if (!fs.existsSync(testDataDir)) {
    fs.mkdirSync(testDataDir, { recursive: true });
  }

  // 4. Verify API server health
  console.log('Verifying API server health...');
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  try {
    await page.goto('http://localhost:8000/health', { timeout: 10000 });
    const response = await page.textContent('body');
    if (!response.includes('healthy')) {
      throw new Error('API server health check failed');
    }
    console.log('API server health check passed');
  } catch (error) {
    console.error('API server health check failed:', error);
    throw error;
  } finally {
    await browser.close();
  }

  // 5. Store server processes for cleanup
  process.env.API_SERVER_PID = apiServer.pid;
  process.env.MOCK_SERVER_PID = mockServer.pid;

  console.log('E2E test environment setup complete');
}

export default globalSetup;