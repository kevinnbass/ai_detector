document.addEventListener('DOMContentLoaded', async function() {
  const elements = {
    enableToggle: document.getElementById('enableToggle'),
    overlayToggle: document.getElementById('overlayToggle'),
    confidenceToggle: document.getElementById('confidenceToggle'),
    quickModeToggle: document.getElementById('quickModeToggle'),
    llmToggle: document.getElementById('llmToggle'),
    analysisModeSelect: document.getElementById('analysisModeSelect'),
    analysisMode: document.getElementById('analysisMode'),
    thresholdSlider: document.getElementById('thresholdSlider'),
    thresholdValue: document.getElementById('thresholdValue'),
    colorPicker: document.getElementById('colorPicker'),
    rescanBtn: document.getElementById('rescanBtn'),
    clearCacheBtn: document.getElementById('clearCacheBtn'),
    setupLLMBtn: document.getElementById('setupLLMBtn'),
    privacyConsent: document.getElementById('privacyConsent'),
    statusDot: document.getElementById('statusDot'),
    statusText: document.getElementById('statusText'),
    postsAnalyzed: document.getElementById('postsAnalyzed'),
    gptDetected: document.getElementById('gptDetected'),
    llmEnhanced: document.getElementById('llmEnhanced'),
    llmStats: document.getElementById('llmStats'),
    reportIssue: document.getElementById('reportIssue'),
    about: document.getElementById('about')
  };

  let settings = {
    enabled: true,
    threshold: 0.7,
    showOverlay: true,
    showConfidence: true,
    highlightColor: '#ff6b6b',
    quickMode: false,
    llmEnabled: false,
    analysisMode: 'quick'
  };
  
  let hasApiKey = false;

  async function loadSettings() {
    try {
      const stored = await chrome.storage.sync.get(['settings', 'privacyConsent']);
      if (stored.settings) {
        settings = { ...settings, ...stored.settings };
      }
      if (stored.privacyConsent !== undefined) {
        elements.privacyConsent.checked = stored.privacyConsent;
      }
      
      // Check for API key
      try {
        const response = await chrome.runtime.sendMessage({ action: 'getSettings' });
        hasApiKey = response && response.hasApiKey;
      } catch (error) {
        hasApiKey = false;
      }
      
      updateUI();
    } catch (error) {
      console.error('Error loading settings:', error);
    }
  }

  async function saveSettings() {
    try {
      await chrome.storage.sync.set({ 
        settings: settings,
        privacyConsent: elements.privacyConsent.checked
      });
      
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: 'updateSettings',
          settings: settings
        });
      }
    } catch (error) {
      console.error('Error saving settings:', error);
    }
  }

  function updateUI() {
    elements.enableToggle.checked = settings.enabled;
    elements.overlayToggle.checked = settings.showOverlay;
    elements.confidenceToggle.checked = settings.showConfidence;
    elements.quickModeToggle.checked = settings.quickMode;
    elements.llmToggle.checked = settings.llmEnabled && hasApiKey;
    elements.analysisModeSelect.value = settings.analysisMode || 'quick';
    elements.thresholdSlider.value = settings.threshold * 100;
    elements.thresholdValue.textContent = `${Math.round(settings.threshold * 100)}%`;
    elements.colorPicker.value = settings.highlightColor;
    
    // Show/hide LLM-specific UI elements
    if (hasApiKey) {
      elements.setupLLMBtn.style.display = 'none';
      elements.analysisMode.style.display = settings.llmEnabled ? 'block' : 'none';
    } else {
      elements.setupLLMBtn.style.display = 'block';
      elements.llmToggle.checked = false;
      elements.llmToggle.disabled = true;
      elements.analysisMode.style.display = 'none';
    }
    
    updateStatus();
  }

  function updateStatus() {
    if (settings.enabled) {
      elements.statusDot.classList.remove('inactive');
      elements.statusText.textContent = 'Active';
    } else {
      elements.statusDot.classList.add('inactive');
      elements.statusText.textContent = 'Inactive';
    }
  }

  async function updateStats() {
    try {
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'getStats' }, (response) => {
          if (response) {
            elements.postsAnalyzed.textContent = response.processedCount || 0;
            elements.gptDetected.textContent = response.detectedCount || 0;
            
            // Show LLM stats if available
            if (response.enhanced) {
              elements.llmEnhanced.textContent = response.enhanced.llm_enhanced || 0;
              elements.llmStats.style.display = 'block';
            } else {
              elements.llmStats.style.display = 'none';
            }
          }
        });
      }

      // Also get background service worker stats
      if (hasApiKey && settings.llmEnabled) {
        try {
          const backgroundStats = await chrome.runtime.sendMessage({ action: 'getStats' });
          if (backgroundStats && backgroundStats.stats) {
            // Could show additional LLM stats from background service worker
            console.log('Background LLM stats:', backgroundStats);
          }
        } catch (error) {
          // Background stats not available
        }
      }
    } catch (error) {
      console.error('Error updating stats:', error);
    }
  }

  elements.enableToggle.addEventListener('change', (e) => {
    settings.enabled = e.target.checked;
    updateStatus();
    saveSettings();
  });

  elements.overlayToggle.addEventListener('change', (e) => {
    settings.showOverlay = e.target.checked;
    saveSettings();
  });

  elements.confidenceToggle.addEventListener('change', (e) => {
    settings.showConfidence = e.target.checked;
    saveSettings();
  });

  elements.quickModeToggle.addEventListener('change', (e) => {
    settings.quickMode = e.target.checked;
    saveSettings();
  });

  elements.llmToggle.addEventListener('change', (e) => {
    settings.llmEnabled = e.target.checked && hasApiKey;
    elements.analysisMode.style.display = settings.llmEnabled ? 'block' : 'none';
    saveSettings();
  });

  elements.analysisModeSelect.addEventListener('change', (e) => {
    settings.analysisMode = e.target.value;
    saveSettings();
  });

  elements.setupLLMBtn.addEventListener('click', () => {
    chrome.tabs.create({ url: chrome.runtime.getURL('setup.html') });
  });

  elements.thresholdSlider.addEventListener('input', (e) => {
    const value = e.target.value;
    settings.threshold = value / 100;
    elements.thresholdValue.textContent = `${value}%`;
  });

  elements.thresholdSlider.addEventListener('change', () => {
    saveSettings();
  });

  elements.colorPicker.addEventListener('change', (e) => {
    settings.highlightColor = e.target.value;
    saveSettings();
  });

  elements.rescanBtn.addEventListener('click', async () => {
    elements.rescanBtn.disabled = true;
    elements.rescanBtn.textContent = 'Rescanning...';
    
    try {
      const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'rescan' }, () => {
          elements.rescanBtn.textContent = 'Rescan Page';
          elements.rescanBtn.disabled = false;
          setTimeout(updateStats, 500);
        });
      }
    } catch (error) {
      console.error('Error rescanning:', error);
      elements.rescanBtn.textContent = 'Rescan Page';
      elements.rescanBtn.disabled = false;
    }
  });

  elements.clearCacheBtn.addEventListener('click', () => {
    if (confirm('Clear the detection cache? This will reset all stored detection results.')) {
      chrome.storage.local.clear(() => {
        elements.clearCacheBtn.textContent = 'Cache Cleared!';
        setTimeout(() => {
          elements.clearCacheBtn.textContent = 'Clear Cache';
        }, 2000);
      });
    }
  });

  elements.privacyConsent.addEventListener('change', () => {
    saveSettings();
  });

  elements.reportIssue.addEventListener('click', (e) => {
    e.preventDefault();
    chrome.tabs.create({ 
      url: 'https://github.com/yourusername/gpt4o-detector/issues' 
    });
  });

  elements.about.addEventListener('click', (e) => {
    e.preventDefault();
    alert('GPT-4o Detector v1.0.0\n\nDetects GPT-4o generated text on X (Twitter) using pattern analysis.\n\nDeveloped with privacy in mind - all detection happens locally in your browser.');
  });

  await loadSettings();
  await updateStats();
  
  setInterval(updateStats, 5000);
});