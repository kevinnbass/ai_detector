// Import Gemini analyzer
importScripts('gemini-analyzer.js');

// Background service worker state
let isEnabled = true;
let analysisMode = 'quick';
let analysisQueue = [];
let processingQueue = false;

chrome.runtime.onInstalled.addListener(async (details) => {
  if (details.reason === 'install') {
    console.log('🚀 GPT-4o Detector installed with LLM support');
    
    const defaultSettings = {
      enabled: true,
      threshold: 0.7,
      showOverlay: true,
      showConfidence: true,
      highlightColor: '#ff6b6b',
      quickMode: false,
      llmEnabled: false,
      analysisMode: 'quick',
      autoAnalyze: true,
      minConfidence: 0.7
    };
    
    await chrome.storage.sync.set({ 
      settings: defaultSettings,
      privacyConsent: true
    });
    
    chrome.tabs.create({
      url: chrome.runtime.getURL('setup.html')
    });
  } else if (details.reason === 'update') {
    console.log('📦 GPT-4o Detector updated');
    
    const stored = await chrome.storage.sync.get(['settings']);
    if (!stored.settings) {
      const defaultSettings = {
        enabled: true,
        threshold: 0.7,
        showOverlay: true,
        showConfidence: true,
        highlightColor: '#ff6b6b',
        quickMode: false,
        llmEnabled: false,
        analysisMode: 'quick',
        autoAnalyze: true,
        minConfidence: 0.7
      };
      
      await chrome.storage.sync.set({ settings: defaultSettings });
    }
  }
});

chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && 
      (tab.url.includes('x.com') || tab.url.includes('twitter.com'))) {
    
    chrome.storage.sync.get(['settings']).then(({ settings }) => {
      if (settings && settings.enabled) {
        setTimeout(() => {
          chrome.tabs.sendMessage(tabId, { 
            action: 'updateSettings', 
            settings: settings 
          }).catch(() => {
            // Tab might not have content script loaded yet
          });
        }, 1000);
      }
    });
  }
});

chrome.action.onClicked.addListener((tab) => {
  if (tab.url.includes('x.com') || tab.url.includes('twitter.com')) {
    chrome.action.openPopup();
  } else {
    chrome.tabs.create({
      url: 'https://x.com'
    });
  }
});

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  // Handle LLM analysis requests
  if (request.action === 'analyzeTweet') {
    handleAnalyzeTweet(request, sendResponse);
    return true; // Keep message channel open for async response
  }
  
  if (request.action === 'setApiKey') {
    handleSetApiKey(request, sendResponse);
    return true;
  }
  
  if (request.action === 'getSettings') {
    handleGetSettings(sendResponse);
    return true;
  }
  
  if (request.action === 'updateSettings') {
    handleUpdateSettings(request, sendResponse);
    return true;
  }
  
  if (request.action === 'getStats') {
    handleGetStats(sendResponse);
    return true;
  }
  
  if (request.action === 'clearCache') {
    handleClearCache(sendResponse);
    return true;
  }
  
  if (request.action === 'getTabInfo') {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        sendResponse({
          url: tabs[0].url,
          title: tabs[0].title,
          id: tabs[0].id
        });
      }
    });
    return true;
  }
  
  if (request.action === 'updateBadge') {
    const text = request.count > 0 ? request.count.toString() : '';
    chrome.action.setBadgeText({
      text: text,
      tabId: sender.tab.id
    });
    
    chrome.action.setBadgeBackgroundColor({
      color: '#ff6b6b'
    });
    
    sendResponse({ success: true });
  }
  
  if (request.action === 'logDetection') {
    console.log('GPT-4o detection event:', {
      timestamp: new Date().toISOString(),
      url: sender.url,
      confidence: request.confidence,
      patterns: request.patterns
    });
    
    sendResponse({ success: true });
  }
});

chrome.storage.onChanged.addListener((changes, namespace) => {
  if (namespace === 'sync' && changes.settings) {
    console.log('Settings updated:', changes.settings.newValue);
    
    chrome.tabs.query({}, (tabs) => {
      tabs.forEach(tab => {
        if (tab.url && (tab.url.includes('x.com') || tab.url.includes('twitter.com'))) {
          chrome.tabs.sendMessage(tab.id, {
            action: 'updateSettings',
            settings: changes.settings.newValue
          }).catch(() => {
            // Content script not loaded or tab not accessible
          });
        }
      });
    });
  }
});

setInterval(async () => {
  try {
    const stored = await chrome.storage.local.get(['detectionStats']);
    if (stored.detectionStats) {
      const stats = stored.detectionStats;
      const now = Date.now();
      
      const cutoff = now - (7 * 24 * 60 * 60 * 1000);
      stats.detections = stats.detections.filter(d => d.timestamp > cutoff);
      
      await chrome.storage.local.set({ detectionStats: stats });
    }
  } catch (error) {
    console.error('Error cleaning up stats:', error);
  }
}, 24 * 60 * 60 * 1000);

chrome.contextMenus.removeAll(() => {
  chrome.contextMenus.create({
    id: 'analyze-text',
    title: 'Analyze with GPT-4o Detector',
    contexts: ['selection']
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === 'analyze-text' && info.selectionText) {
    chrome.tabs.sendMessage(tab.id, {
      action: 'analyzeText',
      text: info.selectionText
    });
  }
});

chrome.runtime.onSuspend.addListener(() => {
  console.log('GPT-4o Detector background script suspending');
});

chrome.runtime.onStartup.addListener(() => {
  console.log('🚀 GPT-4o Detector starting up with LLM support');
});

// ============================================================================
// LLM ANALYSIS HANDLERS
// ============================================================================

/**
 * Handle tweet analysis request
 */
async function handleAnalyzeTweet(request, sendResponse) {
  const { tweetText, tweetId, options = {} } = request;
  
  if (!isEnabled) {
    sendResponse({
      success: false,
      error: 'Detection disabled'
    });
    return;
  }

  if (!tweetText) {
    sendResponse({
      success: false,
      error: 'No tweet text provided'
    });
    return;
  }

  try {
    // Add to queue for processing
    const analysisRequest = {
      tweetText,
      tweetId,
      options: {
        ...options,
        quick: options.quick !== false && analysisMode === 'quick'
      },
      timestamp: Date.now(),
      sendResponse
    };

    analysisQueue.push(analysisRequest);
    
    // Process queue if not already processing
    if (!processingQueue) {
      processAnalysisQueue();
    }

  } catch (error) {
    console.error('❌ Tweet analysis error:', error);
    sendResponse({
      success: false,
      error: error.message
    });
  }
}

/**
 * Process analysis queue with rate limiting
 */
async function processAnalysisQueue() {
  processingQueue = true;
  
  while (analysisQueue.length > 0) {
    const request = analysisQueue.shift();
    
    try {
      console.log(`🧠 Analyzing tweet: "${request.tweetText.substring(0, 50)}..."`);
      
      // Perform analysis
      const result = await geminiAnalyzer.analyzeTweet(request.tweetText, request.options);
      
      // Add metadata
      result.tweet_id = request.tweetId;
      result.analysis_mode = request.options.quick ? 'quick' : 'comprehensive';
      result.queue_time = Date.now() - request.timestamp;
      
      // Send response
      request.sendResponse({
        success: true,
        result: result
      });
      
      // Update statistics
      await updateAnalysisStats(result);
      
    } catch (error) {
      console.error('❌ Analysis failed:', error);
      request.sendResponse({
        success: false,
        error: error.message,
        fallback: await getFallbackAnalysis(request.tweetText)
      });
    }
  }
  
  processingQueue = false;
}

/**
 * Fallback analysis when LLM fails
 */
async function getFallbackAnalysis(tweetText) {
  let score = 0;
  const indicators = [];
  
  // Check for common GPT-4o patterns
  if (tweetText.includes('important to note')) {
    score += 0.3;
    indicators.push('meta_commentary');
  }
  
  if (tweetText.includes('on one hand') || tweetText.includes('on the other hand')) {
    score += 0.4;
    indicators.push('contrast_rhetoric');
  }
  
  if (/\b(?:perhaps|might|could|seems|appears)\b/i.test(tweetText)) {
    score += 0.2;
    indicators.push('hedging_language');
  }
  
  // Formal language in casual context
  if (tweetText.length > 100 && !/\b(?:lol|wtf|tbh|omg)\b/i.test(tweetText)) {
    score += 0.2;
    indicators.push('formal_register');
  }
  
  return {
    ai_probability: Math.min(score, 1.0),
    prediction: score > 0.5 ? 'ai' : 'human',
    confidence: { value: 0.6, level: 'medium' },
    key_indicators: indicators,
    reasoning: 'Fallback pattern-based analysis',
    fallback_analysis: true
  };
}

/**
 * Handle API key setting
 */
async function handleSetApiKey(request, sendResponse) {
  const { apiKey } = request;
  
  if (!apiKey) {
    sendResponse({
      success: false,
      error: 'No API key provided'
    });
    return;
  }
  
  try {
    await geminiAnalyzer.setApiKey(apiKey);
    
    // Test the API key
    const testResult = await geminiAnalyzer.analyzeTweet(
      'This is a test tweet to verify the API key works correctly.',
      { quick: true }
    );
    
    sendResponse({
      success: true,
      message: 'API key set and verified successfully',
      testResult
    });
    
  } catch (error) {
    sendResponse({
      success: false,
      error: `API key verification failed: ${error.message}`
    });
  }
}

/**
 * Handle settings retrieval
 */
async function handleGetSettings(sendResponse) {
  const settings = await chrome.storage.sync.get([
    'settings'
  ]);
  
  const hasApiKey = await chrome.storage.local.get(['geminiApiKey']);
  settings.hasApiKey = !!hasApiKey.geminiApiKey;
  
  if (settings.settings) {
    isEnabled = settings.settings.enabled !== false;
    analysisMode = settings.settings.analysisMode || 'quick';
  }
  
  sendResponse({ settings: settings.settings || {}, hasApiKey: settings.hasApiKey });
}

/**
 * Handle settings update
 */
async function handleUpdateSettings(request, sendResponse) {
  const { settings } = request;
  
  await chrome.storage.sync.set({ settings });
  
  // Update local state
  isEnabled = settings.enabled !== false;
  analysisMode = settings.analysisMode || 'quick';
  
  sendResponse({ success: true });
}

/**
 * Handle statistics request
 */
async function handleGetStats(sendResponse) {
  const stats = await chrome.storage.local.get(['analysisStats']);
  const cacheStats = geminiAnalyzer.getCacheStats();
  
  sendResponse({
    stats: stats.analysisStats || {
      totalAnalyses: 0,
      aiDetected: 0,
      humanDetected: 0,
      averageConfidence: 0,
      averageProcessingTime: 0
    },
    cacheStats
  });
}

/**
 * Handle cache clearing
 */
async function handleClearCache(sendResponse) {
  geminiAnalyzer.clearCache();
  sendResponse({ success: true });
}

/**
 * Update analysis statistics
 */
async function updateAnalysisStats(result) {
  const { analysisStats = {} } = await chrome.storage.local.get(['analysisStats']);
  
  const stats = {
    totalAnalyses: (analysisStats.totalAnalyses || 0) + 1,
    aiDetected: analysisStats.aiDetected || 0,
    humanDetected: analysisStats.humanDetected || 0,
    totalProcessingTime: (analysisStats.totalProcessingTime || 0) + (result.processing_time || 0),
    totalConfidence: (analysisStats.totalConfidence || 0) + (result.confidence?.value || result.overall_confidence?.value || 0.5)
  };
  
  if (result.prediction === 'ai') {
    stats.aiDetected++;
  } else {
    stats.humanDetected++;
  }
  
  stats.averageProcessingTime = stats.totalProcessingTime / stats.totalAnalyses;
  stats.averageConfidence = stats.totalConfidence / stats.totalAnalyses;
  
  await chrome.storage.local.set({ analysisStats: stats });
}