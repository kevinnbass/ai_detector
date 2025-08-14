(function() {
  'use strict';

  const detector = new GPT4oDetectorEngine();
  // Enhanced detector with LLM integration is available as window.enhancedDetector
  let settings = {
    enabled: true,
    threshold: 0.7,
    showOverlay: true,
    showConfidence: true,
    highlightColor: '#ff6b6b',
    quickMode: false
  };

  const PROCESSED_ATTR = 'data-gpt4o-processed';
  const RESULT_ATTR = 'data-gpt4o-result';
  const processedPosts = new WeakSet();
  let observerInstance = null;
  let processingQueue = [];
  let isProcessing = false;

  async function loadSettings() {
    try {
      const stored = await chrome.storage.sync.get(['settings']);
      if (stored.settings) {
        settings = { ...settings, ...stored.settings };
        detector.updateThreshold(settings.threshold);
      }
    } catch (error) {
      console.log('Using default settings');
    }
  }

  function getPostSelectors() {
    return [
      '[data-testid="tweetText"]',
      '[data-testid="tweet"] span',
      'article div[lang] span',
      'div[data-testid="cellInnerDiv"] span[dir="auto"]',
      '[role="article"] span'
    ];
  }

  function findPostElements() {
    const selectors = getPostSelectors();
    const elements = new Set();
    
    for (const selector of selectors) {
      const found = document.querySelectorAll(selector);
      found.forEach(el => {
        if (el.textContent && el.textContent.length > 20) {
          const article = el.closest('article') || el.closest('[data-testid="cellInnerDiv"]');
          if (article && !article.hasAttribute(PROCESSED_ATTR)) {
            elements.add({ element: el, container: article });
          }
        }
      });
    }
    
    return Array.from(elements);
  }

  function extractPostText(element) {
    let text = '';
    
    const walker = document.createTreeWalker(
      element,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode: function(node) {
          const parent = node.parentElement;
          if (parent && (parent.tagName === 'SCRIPT' || parent.tagName === 'STYLE')) {
            return NodeFilter.FILTER_REJECT;
          }
          return NodeFilter.FILTER_ACCEPT;
        }
      }
    );

    let node;
    while (node = walker.nextNode()) {
      text += node.textContent + ' ';
    }

    return text.trim();
  }

  function createWarningOverlay(result, element) {
    const existingOverlay = element.querySelector('.gpt4o-warning-overlay');
    if (existingOverlay) {
      existingOverlay.remove();
    }

    const overlay = document.createElement('div');
    overlay.className = 'gpt4o-warning-overlay';
    
    const badge = document.createElement('div');
    badge.className = 'gpt4o-badge';
    badge.innerHTML = `
      <span class="gpt4o-icon">🤖</span>
      <span class="gpt4o-label">GPT-4o Detected</span>
      ${settings.showConfidence ? `<span class="gpt4o-confidence">${Math.round(result.confidence * 100)}%</span>` : ''}
    `;
    
    overlay.appendChild(badge);

    if (result.matchedPatterns && result.matchedPatterns.length > 0) {
      const details = document.createElement('div');
      details.className = 'gpt4o-details';
      details.style.display = 'none';
      
      const patternList = result.matchedPatterns.slice(0, 3).map(p => 
        `<li>${p.description} (${p.matches} match${p.matches > 1 ? 'es' : ''})</li>`
      ).join('');
      
      details.innerHTML = `
        <div class="gpt4o-details-header">Detected Patterns:</div>
        <ul class="gpt4o-pattern-list">${patternList}</ul>
      `;
      
      overlay.appendChild(details);
      
      badge.addEventListener('click', (e) => {
        e.stopPropagation();
        details.style.display = details.style.display === 'none' ? 'block' : 'none';
      });
    }

    return overlay;
  }

  function applyVisualIndicator(container, result) {
    if (result.isGPT4o) {
      container.style.borderLeft = `3px solid ${settings.highlightColor}`;
      container.style.paddingLeft = '8px';
      container.style.transition = 'all 0.3s ease';
      
      if (settings.showOverlay) {
        const overlay = createWarningOverlay(result, container);
        container.style.position = 'relative';
        container.insertBefore(overlay, container.firstChild);
      }
      
      container.setAttribute(RESULT_ATTR, 'gpt4o');
      container.setAttribute('data-gpt4o-confidence', result.confidence);
    } else {
      container.setAttribute(RESULT_ATTR, 'human');
    }
  }

  async function processPost(postData) {
    const { element, container } = postData;
    
    if (processedPosts.has(container)) {
      return;
    }
    
    const text = extractPostText(element);
    
    if (text.length < 10) {
      return;
    }

    const startTime = performance.now();
    
    // Use enhanced detector if available (with LLM integration)
    let result;
    if (window.enhancedDetector && settings.llmEnabled) {
      try {
        result = await window.enhancedDetector.analyzePost(element, text);
        // Convert enhanced result format to traditional format for compatibility
        result = {
          isGPT4o: result.prediction === 'gpt4o' || result.prediction === 'ai',
          confidence: result.confidence,
          probability: result.probability,
          indicators: result.indicators,
          matchedPatterns: result.indicators?.map(ind => ({ 
            description: ind, 
            matches: 1 
          })) || [],
          reasoning: result.reasoning,
          analysis_method: result.analysis_method || 'traditional',
          llm_enhanced: result.llm_enhanced,
          processing_time: result.processing_time
        };
      } catch (error) {
        console.error('❌ Enhanced analysis failed, falling back to traditional:', error);
        result = settings.quickMode ? 
          detector.quickDetect(text) : 
          detector.detect(text);
      }
    } else {
      // Use traditional detector only
      result = settings.quickMode ? 
        detector.quickDetect(text) : 
        detector.detect(text);
    }
    
    const processingTime = performance.now() - startTime;

    if (processingTime > 2000) { // Increased threshold for LLM analysis
      console.warn(`Slow detection: ${processingTime}ms for text length ${text.length}`);
    }

    applyVisualIndicator(container, result);
    container.setAttribute(PROCESSED_ATTR, 'true');
    processedPosts.add(container);

    if (result.isGPT4o) {
      const analysisMethod = result.llm_enhanced ? 
        (result.analysis_method || 'LLM-enhanced') : 'Traditional';
      
      console.log(`🎯 GPT-4o detected (${analysisMethod}):`, {
        confidence: result.confidence,
        probability: result.probability,
        patterns: result.matchedPatterns,
        reasoning: result.reasoning,
        text: text.substring(0, 100) + '...',
        processing_time: processingTime.toFixed(1) + 'ms'
      });

      // Update badge count
      chrome.runtime.sendMessage({ 
        action: 'updateBadge', 
        count: document.querySelectorAll('[data-gpt4o-result="gpt4o"]').length 
      }).catch(() => {});
    }
  }

  async function processQueue() {
    if (isProcessing || processingQueue.length === 0) {
      return;
    }

    isProcessing = true;
    const batch = processingQueue.splice(0, 10);

    for (const postData of batch) {
      await processPost(postData);
      await new Promise(resolve => setTimeout(resolve, 10));
    }

    isProcessing = false;
    
    if (processingQueue.length > 0) {
      requestAnimationFrame(processQueue);
    }
  }

  function scanAndProcess() {
    if (!settings.enabled) {
      return;
    }

    const posts = findPostElements();
    
    posts.forEach(postData => {
      if (!processedPosts.has(postData.container)) {
        processingQueue.push(postData);
      }
    });

    processQueue();
  }

  function setupObserver() {
    if (observerInstance) {
      observerInstance.disconnect();
    }

    const targetNode = document.body;
    const config = {
      childList: true,
      subtree: true,
      characterData: false,
      attributes: false
    };

    let debounceTimer;
    const debouncedScan = () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(scanAndProcess, 300);
    };

    observerInstance = new MutationObserver((mutations) => {
      const hasNewContent = mutations.some(mutation => {
        return mutation.addedNodes.length > 0 &&
               Array.from(mutation.addedNodes).some(node => 
                 node.nodeType === 1 && 
                 (node.matches && (node.matches('article') || node.querySelector('article')))
               );
      });

      if (hasNewContent) {
        debouncedScan();
      }
    });

    observerInstance.observe(targetNode, config);
  }

  function handleScroll() {
    let scrollTimer;
    window.addEventListener('scroll', () => {
      clearTimeout(scrollTimer);
      scrollTimer = setTimeout(scanAndProcess, 500);
    }, { passive: true });
  }

  function cleanup() {
    if (observerInstance) {
      observerInstance.disconnect();
      observerInstance = null;
    }
    
    processingQueue = [];
    isProcessing = false;
    
    document.querySelectorAll('.gpt4o-warning-overlay').forEach(el => el.remove());
    document.querySelectorAll(`[${PROCESSED_ATTR}]`).forEach(el => {
      el.removeAttribute(PROCESSED_ATTR);
      el.removeAttribute(RESULT_ATTR);
      el.style.borderLeft = '';
      el.style.paddingLeft = '';
    });
  }

  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'updateSettings') {
      settings = { ...settings, ...request.settings };
      detector.updateThreshold(settings.threshold);
      
      // Update enhanced detector settings if available
      if (window.enhancedDetector) {
        window.enhancedDetector.updateLLMSettings(settings);
      }
      
      if (settings.enabled) {
        scanAndProcess();
      } else {
        cleanup();
      }
      
      sendResponse({ success: true });
    } else if (request.action === 'getStats') {
      const stats = detector.getStats();
      stats.processedCount = document.querySelectorAll(`[${PROCESSED_ATTR}]`).length;
      stats.detectedCount = document.querySelectorAll(`[${RESULT_ATTR}="gpt4o"]`).length;
      
      // Add enhanced detector stats if available
      if (window.enhancedDetector) {
        const enhancedStats = window.enhancedDetector.getStats();
        stats.enhanced = enhancedStats;
      }
      
      sendResponse(stats);
    } else if (request.action === 'rescan') {
      cleanup();
      scanAndProcess();
      sendResponse({ success: true });
    } else if (request.action === 'clearCache') {
      if (window.enhancedDetector) {
        window.enhancedDetector.clearCaches();
      }
      sendResponse({ success: true });
    } else if (request.action === 'ping') {
      sendResponse({ success: true, version: '2.0.0-llm' });
    }
    
    return true;
  });

  async function initialize() {
    console.log('GPT-4o Detector initializing...');
    
    await loadSettings();
    await detector.loadExternalRules();
    
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => {
        scanAndProcess();
        setupObserver();
        handleScroll();
      });
    } else {
      scanAndProcess();
      setupObserver();
      handleScroll();
    }

    console.log('GPT-4o Detector initialized');
  }

  initialize();

  window.addEventListener('beforeunload', cleanup);
})();