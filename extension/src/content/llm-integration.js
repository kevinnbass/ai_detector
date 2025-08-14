/**
 * LLM INTEGRATION FOR CONTENT SCRIPT
 * =================================
 * 
 * Adds LLM-powered analysis to the existing content script functionality.
 * Integrates with the background service worker for Gemini API calls.
 */

class LLMTweetAnalyzer {
    constructor() {
        this.isEnabled = false;
        this.analysisMode = 'quick';
        this.pendingAnalyses = new Map();
        this.analysisCache = new Map();
        this.maxCacheSize = 500;
        this.analysisQueue = [];
        this.processingQueue = false;
        
        this.initializeSettings();
    }

    /**
     * Initialize LLM settings
     */
    async initializeSettings() {
        try {
            const response = await chrome.runtime.sendMessage({ action: 'getSettings' });
            if (response && response.settings) {
                this.isEnabled = response.settings.llmEnabled && response.hasApiKey;
                this.analysisMode = response.settings.analysisMode || 'quick';
                console.log(`🧠 LLM Analysis: ${this.isEnabled ? 'Enabled' : 'Disabled'}`);
            }
        } catch (error) {
            console.error('❌ Failed to initialize LLM settings:', error);
        }
    }

    /**
     * Update settings from popup or background
     */
    updateSettings(newSettings) {
        this.isEnabled = newSettings.llmEnabled;
        this.analysisMode = newSettings.analysisMode || 'quick';
        console.log(`🔧 LLM settings updated: ${this.isEnabled ? 'Enabled' : 'Disabled'}`);
    }

    /**
     * Analyze tweet with LLM
     */
    async analyzeTweet(tweetText, tweetId, options = {}) {
        if (!this.isEnabled) {
            return null;
        }

        if (!tweetText || tweetText.length < 10) {
            return null;
        }

        // Check cache first
        const cacheKey = this.getCacheKey(tweetText);
        if (this.analysisCache.has(cacheKey)) {
            return this.analysisCache.get(cacheKey);
        }

        // Check if already pending
        if (this.pendingAnalyses.has(tweetId)) {
            return await this.pendingAnalyses.get(tweetId);
        }

        // Create analysis promise
        const analysisPromise = this.performAnalysis(tweetText, tweetId, options);
        this.pendingAnalyses.set(tweetId, analysisPromise);

        try {
            const result = await analysisPromise;
            
            // Cache result
            if (result && result.success) {
                this.cacheResult(cacheKey, result);
            }
            
            return result;
        } finally {
            this.pendingAnalyses.delete(tweetId);
        }
    }

    /**
     * Perform actual LLM analysis via background script
     */
    async performAnalysis(tweetText, tweetId, options) {
        try {
            const response = await chrome.runtime.sendMessage({
                action: 'analyzeTweet',
                tweetText: tweetText,
                tweetId: tweetId,
                options: {
                    quick: this.analysisMode === 'quick',
                    ...options
                }
            });

            if (response && response.success) {
                console.log(`✅ LLM analysis complete for tweet ${tweetId}: ${response.result.prediction} (${(response.result.ai_probability * 100).toFixed(1)}%)`);
                return response;
            } else if (response && response.fallback) {
                console.log(`⚠️ LLM failed, using fallback for tweet ${tweetId}`);
                return {
                    success: true,
                    result: response.fallback,
                    fallback: true
                };
            } else {
                console.error('❌ LLM analysis failed:', response?.error || 'Unknown error');
                return null;
            }
        } catch (error) {
            console.error('❌ LLM analysis error:', error);
            return null;
        }
    }

    /**
     * Batch analyze multiple tweets
     */
    async batchAnalyze(tweets) {
        if (!this.isEnabled) {
            return [];
        }

        const results = [];
        for (const tweet of tweets) {
            try {
                const result = await this.analyzeTweet(tweet.text, tweet.id);
                results.push({
                    tweetId: tweet.id,
                    result: result
                });
                
                // Rate limiting - wait between analyses
                await this.delay(1000);
            } catch (error) {
                console.error(`❌ Batch analysis failed for tweet ${tweet.id}:`, error);
                results.push({
                    tweetId: tweet.id,
                    result: null,
                    error: error.message
                });
            }
        }

        return results;
    }

    /**
     * Generate cache key for a tweet
     */
    getCacheKey(tweetText) {
        // Simple hash function
        let hash = 0;
        for (let i = 0; i < tweetText.length; i++) {
            const char = tweetText.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return hash.toString();
    }

    /**
     * Cache analysis result
     */
    cacheResult(key, result) {
        // Implement LRU cache
        if (this.analysisCache.size >= this.maxCacheSize) {
            const firstKey = this.analysisCache.keys().next().value;
            this.analysisCache.delete(firstKey);
        }
        
        this.analysisCache.set(key, {
            ...result,
            cached_at: Date.now()
        });
    }

    /**
     * Clear analysis cache
     */
    clearCache() {
        this.analysisCache.clear();
        chrome.runtime.sendMessage({ action: 'clearCache' });
        console.log('🧹 LLM analysis cache cleared');
    }

    /**
     * Get cache statistics
     */
    getCacheStats() {
        return {
            size: this.analysisCache.size,
            maxSize: this.maxCacheSize
        };
    }

    /**
     * Utility delay function
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Check if LLM analysis is available
     */
    isAvailable() {
        return this.isEnabled;
    }

    /**
     * Get analysis mode
     */
    getAnalysisMode() {
        return this.analysisMode;
    }
}

/**
 * Enhanced tweet detection with LLM integration
 */
class EnhancedTweetDetector {
    constructor() {
        this.traditionalDetector = detector; // Use existing detector
        this.llmAnalyzer = new LLMTweetAnalyzer();
        this.combinedResults = new Map();
    }

    /**
     * Analyze tweet with both traditional and LLM methods
     */
    async analyzePost(element, tweetText) {
        const tweetId = this.generateTweetId(element);
        
        // Run traditional detection first (fast)
        const traditionalResult = this.traditionalDetector.analyzeText(tweetText);
        
        // If LLM is available and traditional result is uncertain, use LLM
        let llmResult = null;
        let finalResult = traditionalResult;

        if (this.llmAnalyzer.isAvailable()) {
            // Use LLM for uncertain cases or high-confidence traditional results
            const shouldUseLLM = 
                traditionalResult.confidence < 0.8 || // Uncertain traditional result
                traditionalResult.confidence > 0.9;   // High confidence to verify

            if (shouldUseLLM) {
                try {
                    const llmResponse = await this.llmAnalyzer.analyzeTweet(tweetText, tweetId);
                    
                    if (llmResponse && llmResponse.success) {
                        llmResult = llmResponse.result;
                        
                        // Combine traditional and LLM results
                        finalResult = this.combineResults(traditionalResult, llmResult);
                        finalResult.llm_enhanced = true;
                        finalResult.llm_mode = llmResponse.fallback ? 'fallback' : 'full';
                    }
                } catch (error) {
                    console.error('❌ LLM analysis failed, using traditional only:', error);
                }
            }
        }

        // Store combined result
        this.combinedResults.set(tweetId, {
            traditional: traditionalResult,
            llm: llmResult,
            final: finalResult,
            timestamp: Date.now()
        });

        return finalResult;
    }

    /**
     * Combine traditional ML and LLM results using ensemble method
     */
    combineResults(traditional, llm) {
        // Weighted ensemble combination
        const traditionWeight = 0.4;
        const llmWeight = 0.6;

        // Combine probabilities
        const combinedProbability = 
            (traditional.probability * traditionWeight) + 
            (llm.ai_probability * llmWeight);

        // Use higher confidence for final confidence
        const combinedConfidence = Math.max(
            traditional.confidence,
            llm.confidence?.value || llm.overall_confidence?.value || 0.5
        );

        // Combine indicators
        const combinedIndicators = [
            ...traditional.indicators,
            ...(llm.key_indicators || [])
        ];

        // Determine final prediction
        const prediction = combinedProbability > 0.5 ? 'gpt4o' : 'human';

        return {
            prediction: prediction,
            probability: combinedProbability,
            confidence: combinedConfidence,
            indicators: combinedIndicators,
            reasoning: this.generateCombinedReasoning(traditional, llm),
            traditional_score: traditional.probability,
            llm_score: llm.ai_probability,
            llm_prediction: llm.prediction,
            analysis_method: 'ensemble',
            processing_time: llm.processing_time || 0
        };
    }

    /**
     * Generate combined reasoning explanation
     */
    generateCombinedReasoning(traditional, llm) {
        const agreement = Math.abs(traditional.probability - llm.ai_probability) < 0.3;
        
        if (agreement) {
            return `Traditional ML and LLM analysis agree. ${llm.reasoning || 'Consistent patterns detected across both methods.'}`;
        } else {
            return `Traditional ML (${(traditional.probability * 100).toFixed(1)}%) and LLM (${(llm.ai_probability * 100).toFixed(1)}%) show different confidence levels. Combined analysis provides balanced assessment. ${llm.reasoning || ''}`;
        }
    }

    /**
     * Generate unique tweet ID from element
     */
    generateTweetId(element) {
        // Try to find tweet ID from DOM
        let tweetElement = element;
        while (tweetElement && !tweetElement.getAttribute('data-tweet-id')) {
            tweetElement = tweetElement.closest('[data-testid="tweet"], article, [data-tweet-id]');
            if (!tweetElement) break;
        }

        if (tweetElement) {
            const tweetId = tweetElement.getAttribute('data-tweet-id') ||
                           tweetElement.querySelector('[href*="/status/"]')?.href?.match(/\/status\/(\d+)/)?.[1];
            if (tweetId) return tweetId;
        }

        // Fallback: generate ID from text content and position
        const textHash = this.hashString(element.textContent);
        const position = Array.from(document.querySelectorAll('[data-testid="tweetText"], span')).indexOf(element);
        return `generated_${textHash}_${position}`;
    }

    /**
     * Simple string hash function
     */
    hashString(str) {
        let hash = 0;
        for (let i = 0; i < str.length; i++) {
            const char = str.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return Math.abs(hash).toString(36);
    }

    /**
     * Get analysis history for a tweet
     */
    getAnalysisHistory(tweetId) {
        return this.combinedResults.get(tweetId);
    }

    /**
     * Update LLM settings
     */
    updateLLMSettings(settings) {
        this.llmAnalyzer.updateSettings(settings);
    }

    /**
     * Clear all caches
     */
    clearCaches() {
        this.combinedResults.clear();
        this.llmAnalyzer.clearCache();
    }

    /**
     * Get statistics
     */
    getStats() {
        const totalAnalyses = this.combinedResults.size;
        const llmEnhanced = Array.from(this.combinedResults.values())
            .filter(r => r.final.llm_enhanced).length;
        
        return {
            total_analyses: totalAnalyses,
            llm_enhanced: llmEnhanced,
            traditional_only: totalAnalyses - llmEnhanced,
            cache_stats: this.llmAnalyzer.getCacheStats()
        };
    }
}

// Create global enhanced detector
const enhancedDetector = new EnhancedTweetDetector();

// Export for use in content script
if (typeof window !== 'undefined') {
    window.enhancedDetector = enhancedDetector;
    window.llmAnalyzer = enhancedDetector.llmAnalyzer;
}