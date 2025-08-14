/**
 * End-to-End User Journey Tests
 * Complete user workflows from browser interaction to system response
 */

import { test, expect } from '@playwright/test';
import path from 'path';

// Extension path for Chrome
const extensionPath = path.join(__dirname, '../../extension');

test.describe('AI Detector User Journeys', () => {
  
  test.describe('Extension Installation and Setup', () => {
    test('should install extension and complete initial setup', async ({ context, page }) => {
      // Note: Extension loading is configured in playwright.config.js
      
      // Navigate to extension popup
      await page.goto('chrome-extension://test-extension-id/popup.html');
      
      // Verify extension popup loads
      await expect(page.locator('h1')).toContainText('AI Text Detector');
      
      // Test initial setup flow
      await page.locator('[data-testid="setup-button"]').click();
      
      // Configure initial settings
      await page.locator('[data-testid="auto-detect-toggle"]').check();
      await page.locator('[data-testid="confidence-threshold"]').fill('0.7');
      await page.locator('[data-testid="save-settings"]').click();
      
      // Verify settings saved
      await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
      await expect(page.locator('[data-testid="success-message"]')).toContainText('Settings saved');
    });

    test('should validate extension permissions', async ({ context, page }) => {
      // Test that extension has required permissions
      const permissions = await page.evaluate(() => {
        return new Promise((resolve) => {
          chrome.permissions.getAll((permissions) => {
            resolve(permissions);
          });
        });
      });

      expect(permissions.permissions).toContain('storage');
      expect(permissions.permissions).toContain('activeTab');
      expect(permissions.origins).toContain('https://twitter.com/*');
      expect(permissions.origins).toContain('https://x.com/*');
    });
  });

  test.describe('Twitter/X Integration', () => {
    test('should detect AI text on Twitter timeline', async ({ page }) => {
      // Start API server for testing
      await page.goto('http://localhost:8000/health');
      await expect(page.locator('body')).toContainText('healthy');
      
      // Navigate to Twitter (using mock Twitter page for testing)
      await page.goto('http://localhost:8080/mock-twitter.html');
      
      // Wait for extension to inject content script
      await page.waitForTimeout(1000);
      
      // Find tweets with AI-like content
      const aiTweet = page.locator('[data-testid="tweet"]').filter({
        hasText: 'It is important to note that this analysis requires careful consideration'
      });
      
      // Wait for AI detection to complete
      await expect(aiTweet.locator('[data-ai-prediction="ai"]')).toBeVisible({ timeout: 5000 });
      
      // Verify AI indicator is shown
      await expect(aiTweet.locator('.ai-indicator')).toBeVisible();
      await expect(aiTweet.locator('.ai-confidence')).toContainText('85%');
      
      // Test clicking on AI indicator for details
      await aiTweet.locator('.ai-indicator').click();
      await expect(page.locator('.ai-details-popup')).toBeVisible();
      await expect(page.locator('.ai-details-popup')).toContainText('AI-generated content detected');
    });

    test('should handle human text correctly', async ({ page }) => {
      await page.goto('http://localhost:8080/mock-twitter.html');
      await page.waitForTimeout(1000);
      
      // Find tweet with human-like content
      const humanTweet = page.locator('[data-testid="tweet"]').filter({
        hasText: 'omg just had the best pizza ever! 🍕 totally recommend'
      });
      
      // Wait for detection to complete
      await page.waitForTimeout(2000);
      
      // Should not have AI indicator
      await expect(humanTweet.locator('.ai-indicator')).not.toBeVisible();
      
      // Or should show human indicator if configured
      await expect(humanTweet.locator('[data-ai-prediction="human"]')).toBeVisible();
    });

    test('should batch process multiple tweets efficiently', async ({ page }) => {
      await page.goto('http://localhost:8080/mock-twitter-timeline.html');
      await page.waitForTimeout(1000);
      
      // Wait for batch processing to complete
      await page.waitForTimeout(3000);
      
      // Count processed tweets
      const processedTweets = page.locator('[data-ai-prediction]');
      await expect(processedTweets).toHaveCountGreaterThan(5);
      
      // Verify mixed results (some AI, some human)
      const aiTweets = page.locator('[data-ai-prediction="ai"]');
      const humanTweets = page.locator('[data-ai-prediction="human"]');
      
      await expect(aiTweets).toHaveCountGreaterThan(0);
      await expect(humanTweets).toHaveCountGreaterThan(0);
    });

    test('should update detection as user scrolls', async ({ page }) => {
      await page.goto('http://localhost:8080/mock-twitter-infinite.html');
      
      // Initial tweets processed
      await page.waitForTimeout(2000);
      const initialCount = await page.locator('[data-ai-prediction]').count();
      
      // Scroll to load more tweets
      await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
      await page.waitForTimeout(2000);
      
      // More tweets should be processed
      const afterScrollCount = await page.locator('[data-ai-prediction]').count();
      expect(afterScrollCount).toBeGreaterThan(initialCount);
    });
  });

  test.describe('Settings and Configuration', () => {
    test('should modify detection threshold and see changes', async ({ page }) => {
      // Open extension popup
      await page.goto('chrome-extension://test-extension-id/popup.html');
      
      // Change threshold to high value
      await page.locator('[data-testid="confidence-threshold"]').fill('0.9');
      await page.locator('[data-testid="save-settings"]').click();
      
      // Navigate to test page
      await page.goto('http://localhost:8080/mock-twitter.html');
      await page.waitForTimeout(2000);
      
      // With high threshold, fewer texts should be marked as AI
      const aiTweets = page.locator('[data-ai-prediction="ai"]');
      const aiCount = await aiTweets.count();
      
      // Lower threshold
      await page.goto('chrome-extension://test-extension-id/popup.html');
      await page.locator('[data-testid="confidence-threshold"]').fill('0.5');
      await page.locator('[data-testid="save-settings"]').click();
      
      // Refresh and check again
      await page.goto('http://localhost:8080/mock-twitter.html');
      await page.waitForTimeout(2000);
      
      const aiTweetsLowThreshold = page.locator('[data-ai-prediction="ai"]');
      const aiCountLowThreshold = await aiTweetsLowThreshold.count();
      
      // Should detect more AI content with lower threshold
      expect(aiCountLowThreshold).toBeGreaterThanOrEqual(aiCount);
    });

    test('should toggle auto-detection on/off', async ({ page }) => {
      // Turn off auto-detection
      await page.goto('chrome-extension://test-extension-id/popup.html');
      await page.locator('[data-testid="auto-detect-toggle"]').uncheck();
      await page.locator('[data-testid="save-settings"]').click();
      
      // Navigate to Twitter
      await page.goto('http://localhost:8080/mock-twitter.html');
      await page.waitForTimeout(2000);
      
      // No automatic detection should occur
      await expect(page.locator('[data-ai-prediction]')).toHaveCount(0);
      
      // Manual detection should still work
      const tweet = page.locator('[data-testid="tweet"]').first();
      await tweet.hover();
      await page.locator('[data-testid="manual-detect-button"]').click();
      
      // Should show detection result
      await expect(tweet.locator('[data-ai-prediction]')).toBeVisible();
    });

    test('should configure custom API endpoint', async ({ page }) => {
      await page.goto('chrome-extension://test-extension-id/popup.html');
      
      // Switch to advanced settings
      await page.locator('[data-testid="advanced-settings-tab"]').click();
      
      // Change API endpoint
      await page.locator('[data-testid="api-endpoint"]').fill('http://localhost:8001');
      await page.locator('[data-testid="test-connection"]').click();
      
      // Should show connection error (since endpoint doesn't exist)
      await expect(page.locator('[data-testid="connection-status"]')).toContainText('Failed');
      
      // Revert to working endpoint
      await page.locator('[data-testid="api-endpoint"]').fill('http://localhost:8000');
      await page.locator('[data-testid="test-connection"]').click();
      
      // Should show success
      await expect(page.locator('[data-testid="connection-status"]')).toContainText('Connected');
    });
  });

  test.describe('Data Collection and Training', () => {
    test('should collect samples for training', async ({ page }) => {
      await page.goto('http://localhost:8080/mock-twitter.html');
      await page.waitForTimeout(1000);
      
      // Mark a tweet as definitely human
      const humanTweet = page.locator('[data-testid="tweet"]').first();
      await humanTweet.hover();
      await page.locator('[data-testid="mark-human"]').click();
      
      // Confirm the action
      await page.locator('[data-testid="confirm-human"]').click();
      
      // Should show success feedback
      await expect(page.locator('[data-testid="sample-saved"]')).toBeVisible();
      
      // Mark another tweet as definitely AI
      const aiTweet = page.locator('[data-testid="tweet"]').nth(1);
      await aiTweet.hover();
      await page.locator('[data-testid="mark-ai"]').click();
      await page.locator('[data-testid="confirm-ai"]').click();
      
      // Check collected samples in dashboard
      await page.goto('chrome-extension://test-extension-id/dashboard.html');
      await expect(page.locator('[data-testid="collected-samples"]')).toContainText('2 samples');
    });

    test('should export collected training data', async ({ page }) => {
      await page.goto('chrome-extension://test-extension-id/dashboard.html');
      
      // Should have some collected samples from previous test
      await expect(page.locator('[data-testid="collected-samples"]')).toContainText('samples');
      
      // Export data
      const downloadPromise = page.waitForEvent('download');
      await page.locator('[data-testid="export-data"]').click();
      const download = await downloadPromise;
      
      // Verify download
      expect(download.suggestedFilename()).toMatch(/training-data-\d+\.json/);
      
      // Save and verify content
      const filePath = './test-results/downloaded-data.json';
      await download.saveAs(filePath);
      
      // Read and validate content
      const fs = require('fs');
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      expect(data.samples).toBeDefined();
      expect(Array.isArray(data.samples)).toBe(true);
    });
  });

  test.describe('Performance and Error Handling', () => {
    test('should handle API server downtime gracefully', async ({ page }) => {
      // Stop the API server (simulated by using wrong URL)
      await page.goto('chrome-extension://test-extension-id/popup.html');
      await page.locator('[data-testid="advanced-settings-tab"]').click();
      await page.locator('[data-testid="api-endpoint"]').fill('http://localhost:9999');
      await page.locator('[data-testid="save-settings"]').click();
      
      // Navigate to Twitter
      await page.goto('http://localhost:8080/mock-twitter.html');
      await page.waitForTimeout(2000);
      
      // Should show offline indicator
      await expect(page.locator('[data-testid="offline-indicator"]')).toBeVisible();
      
      // Should fall back to pattern-based detection
      const tweet = page.locator('[data-testid="tweet"]').filter({
        hasText: 'important to note'
      });
      
      await expect(tweet.locator('[data-fallback-detection="true"]')).toBeVisible();
    });

    test('should handle slow API responses', async ({ page }) => {
      // Configure slow API response simulation
      await page.route('**/api/v1/detect', async route => {
        // Delay response by 3 seconds
        await new Promise(resolve => setTimeout(resolve, 3000));
        await route.fulfill({
          status: 200,
          body: JSON.stringify({
            prediction: 'ai',
            confidence: 0.8,
            ai_probability: 0.85
          })
        });
      });
      
      await page.goto('http://localhost:8080/mock-twitter.html');
      
      // Should show loading indicators
      await expect(page.locator('[data-testid="detection-loading"]')).toBeVisible();
      
      // Should eventually show results
      await expect(page.locator('[data-ai-prediction]')).toBeVisible({ timeout: 5000 });
      
      // Loading indicator should disappear
      await expect(page.locator('[data-testid="detection-loading"]')).not.toBeVisible();
    });

    test('should handle large amounts of text efficiently', async ({ page }) => {
      await page.goto('http://localhost:8080/mock-twitter-large-text.html');
      
      // Page with very long tweets should still be processed
      await page.waitForTimeout(3000);
      
      const longTweets = page.locator('[data-testid="long-tweet"]');
      await expect(longTweets.first().locator('[data-ai-prediction]')).toBeVisible();
      
      // Should not cause browser to freeze
      await page.evaluate(() => {
        // Test browser responsiveness
        return new Promise(resolve => {
          setTimeout(resolve, 100);
        });
      });
    });
  });

  test.describe('Accessibility and Usability', () => {
    test('should be accessible with keyboard navigation', async ({ page }) => {
      await page.goto('chrome-extension://test-extension-id/popup.html');
      
      // Tab through all interactive elements
      await page.keyboard.press('Tab');
      await expect(page.locator('[data-testid="auto-detect-toggle"]:focus')).toBeVisible();
      
      await page.keyboard.press('Tab');
      await expect(page.locator('[data-testid="confidence-threshold"]:focus')).toBeVisible();
      
      await page.keyboard.press('Tab');
      await expect(page.locator('[data-testid="save-settings"]:focus')).toBeVisible();
      
      // Should be able to interact with keyboard
      await page.keyboard.press('Enter');
      await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
    });

    test('should have proper ARIA labels and screen reader support', async ({ page }) => {
      await page.goto('chrome-extension://test-extension-id/popup.html');
      
      // Check ARIA labels
      await expect(page.locator('[data-testid="auto-detect-toggle"]')).toHaveAttribute('aria-label');
      await expect(page.locator('[data-testid="confidence-threshold"]')).toHaveAttribute('aria-label');
      
      // Check heading structure
      await expect(page.locator('h1')).toBeVisible();
      await expect(page.locator('h2')).toHaveCount(2); // Should have proper heading hierarchy
    });

    test('should work with high contrast mode', async ({ page }) => {
      // Simulate high contrast mode
      await page.emulateMedia({ colorScheme: 'dark' });
      await page.addStyleTag({
        content: `
          @media (prefers-contrast: high) {
            * { border: 1px solid !important; }
          }
        `
      });
      
      await page.goto('chrome-extension://test-extension-id/popup.html');
      
      // UI should still be usable
      await expect(page.locator('[data-testid="save-settings"]')).toBeVisible();
      
      // Text should be readable
      const button = page.locator('[data-testid="save-settings"]');
      const buttonStyles = await button.evaluate(el => getComputedStyle(el));
      
      // Should have sufficient contrast (this is a simplified check)
      expect(buttonStyles.backgroundColor).not.toBe(buttonStyles.color);
    });
  });

  test.describe('Browser Compatibility', () => {
    test('should work in different Chrome versions', async ({ page }) => {
      // Test basic functionality
      await page.goto('chrome-extension://test-extension-id/popup.html');
      await expect(page.locator('h1')).toContainText('AI Text Detector');
      
      // Test modern API usage gracefully degrades
      const hasModernAPIs = await page.evaluate(() => {
        return typeof fetch !== 'undefined' && 
               typeof Promise !== 'undefined' &&
               typeof chrome.storage !== 'undefined';
      });
      
      expect(hasModernAPIs).toBe(true);
    });

    test('should handle different screen sizes', async ({ page }) => {
      // Test mobile viewport
      await page.setViewportSize({ width: 375, height: 667 });
      await page.goto('chrome-extension://test-extension-id/popup.html');
      
      // UI should adapt to small screen
      await expect(page.locator('[data-testid="main-content"]')).toBeVisible();
      
      // Test desktop viewport
      await page.setViewportSize({ width: 1920, height: 1080 });
      await page.goto('chrome-extension://test-extension-id/popup.html');
      
      // UI should work on large screen
      await expect(page.locator('[data-testid="main-content"]')).toBeVisible();
    });
  });

  test.describe('Security and Privacy', () => {
    test('should not send sensitive data to API', async ({ page }) => {
      let apiRequestData = null;
      
      // Intercept API requests
      await page.route('**/api/v1/detect', async route => {
        const request = route.request();
        apiRequestData = JSON.parse(request.postData());
        
        await route.fulfill({
          status: 200,
          body: JSON.stringify({ prediction: 'human', confidence: 0.8 })
        });
      });
      
      await page.goto('http://localhost:8080/mock-twitter-sensitive.html');
      await page.waitForTimeout(2000);
      
      // Verify no sensitive information in API request
      expect(apiRequestData).toBeDefined();
      expect(apiRequestData.text).toBeDefined();
      expect(apiRequestData.text).not.toContain('@username');
      expect(apiRequestData.text).not.toContain('password');
      expect(apiRequestData.text).not.toContain('email@');
    });

    test('should respect user privacy settings', async ({ page }) => {
      await page.goto('chrome-extension://test-extension-id/popup.html');
      
      // Enable privacy mode
      await page.locator('[data-testid="privacy-mode-toggle"]').check();
      await page.locator('[data-testid="save-settings"]').click();
      
      // Navigate to Twitter
      await page.goto('http://localhost:8080/mock-twitter.html');
      
      // Should not process private/sensitive content
      const privateTweet = page.locator('[data-testid="private-tweet"]');
      await expect(privateTweet.locator('[data-ai-prediction]')).not.toBeVisible();
    });
  });
});