class GPT4oDetectorEngine {
  constructor() {
    this.rules = [];
    this.threshold = 0.7;
    this.initialized = false;
    this.cache = new Map();
    this.maxCacheSize = 1000;
    this.initializeRules();
  }

  initializeRules() {
    // Updated rules based on data-driven pattern analysis
    this.rules = [
      {
        id: 'HEDGE_01',
        pattern: 'excessive_hedging',
        regex: /\b(it's\s+important\s+to\s+note|it's\s+worth\s+considering|merit\s+thorough\s+examination|perhaps|maybe|possibly|might|could|seems|appears|likely|probably|generally|typically|often|sometimes)\b/gi,
        threshold: 2,
        weight: 0.18, // Increased weight based on 87% reliability
        description: 'Excessive hedging and qualifying language'
      },
      {
        id: 'BALANCE_01', 
        pattern: 'balanced_presentation',
        regex: /(on\s+one\s+hand.*on\s+the\s+other\s+hand|both\s+advantages\s+and\s+disadvantages|advantages.*disadvantages|pros.*cons|benefits.*drawbacks|positive.*negative|strengths.*weaknesses|opportunities.*challenges)/gi,
        threshold: 1,
        weight: 0.30, // Increased weight based on 85% reliability
        description: 'Systematic balanced presentation'
      },
      {
        id: 'FORMAL_01',
        pattern: 'formal_transitions',
        regex: /\b(furthermore|moreover|consequently|therefore|thus|hence|accordingly|nevertheless|nonetheless|however|in\s+addition)\b/gi,
        threshold: 1,
        weight: 0.25, // Increased weight based on 82% reliability
        description: 'Formal transitional phrases in casual context'
      },
      {
        id: 'META_01',
        pattern: 'meta_commentary',
        regex: /(when\s+examining|in\s+considering|we\s+must\s+acknowledge|it's\s+crucial\s+to\s+understand|merit\s+careful\s+consideration)/gi,
        threshold: 1,
        weight: 0.23, // Based on 79% reliability
        description: 'Meta-commentary about analysis process'
      },
      {
        id: 'QUAL_01',
        pattern: 'excessive_qualifiers',
        regex: /(it's\s+(important|worth|crucial|essential)\s+(to\s+)?(note|noting|mention|consider)|keep\s+in\s+mind|bear\s+in\s+mind|remember\s+that|consider\s+that)/gi,
        threshold: 1,
        weight: 0.20,
        description: 'Excessive qualifier phrases'
      },
      {
        id: 'CONTRAST_01',
        pattern: 'contrast_rhetoric',
        regex: /(not\s+\w+,?\s+but\s+\w+|while\s+.*,\s+|although\s+.*,\s+|however,?\s+)/gi,
        threshold: 1,
        weight: 0.22,
        description: 'Contrast constructions (not X, but Y)'
      },
      {
        id: 'LIST_01',
        pattern: 'structured_lists',
        regex: /(firstly|secondly|thirdly|first,|second,|third,|\d\.|•|-\s+)/gi,
        threshold: 2,
        weight: 0.15,
        description: 'Structured list formatting'
      },
      {
        id: 'EXPLAIN_01',
        pattern: 'explanatory_style',
        regex: /(essentially|basically|in\s+other\s+words|simply\s+put|to\s+put\s+it\s+simply|in\s+essence)/gi,
        threshold: 1,
        weight: 0.15,
        description: 'Explanatory language'
      },
      {
        id: 'CAVEAT_01',
        pattern: 'caveats',
        regex: /(that\s+said|having\s+said\s+that|with\s+that\s+in\s+mind|that\s+being\s+said|to\s+be\s+fair)/gi,
        threshold: 1,
        weight: 0.18,
        description: 'Caveats and disclaimers'
      },
      // New rules based on human indicators (negative scoring)
      {
        id: 'HUMAN_01',
        pattern: 'casual_authenticity',
        regex: /\b(tbh|lol|rn|lmao|damn|honestly|insane|wtf|omg|fml|ngl)\b/gi,
        threshold: 1,
        weight: -0.25, // Negative weight - reduces AI probability
        description: 'Casual authentic language (human indicator)'
      },
      {
        id: 'HUMAN_02',
        pattern: 'natural_errors',
        regex: /\b(i\s+[a-z]|[a-z]+\s+but\s+[a-z]|way\s+too\s+[a-z]+|kinda\s+|sorta\s+|gonna\s+|wanna\s+)\b/gi,
        threshold: 1,
        weight: -0.22, // Negative weight
        description: 'Natural errors and informal grammar (human indicator)'
      },
      {
        id: 'HUMAN_03',
        pattern: 'emotional_spontaneity',
        regex: /(i\s+literally\s+cannot\s+even|honestly\s+it's\s+insane|coding\s+is\s+pain|i\s+can\s+barely|omfg|this\s+is\s+so\s+[a-z]+)/gi,
        threshold: 1,
        weight: -0.20, // Negative weight
        description: 'Genuine emotional reactions (human indicator)'
      }
    ];
    this.initialized = true;
  }

  async loadExternalRules() {
    try {
      const response = await fetch(chrome.runtime.getURL('content/detection-rules.json'));
      if (response.ok) {
        const data = await response.json();
        if (data.rules && data.rules.length > 0) {
          this.rules = data.rules.map(rule => ({
            ...rule,
            regex: new RegExp(rule.regex, 'gi')
          }));
          this.threshold = data.threshold || 0.7;
        }
      }
    } catch (error) {
      console.log('Using default rules');
    }
  }

  detect(text) {
    if (!text || text.length < 20) {
      return {
        isGPT4o: false,
        confidence: 0,
        matchedPatterns: [],
        explanation: 'Text too short for analysis'
      };
    }

    const cacheKey = this.hashText(text);
    if (this.cache.has(cacheKey)) {
      return this.cache.get(cacheKey);
    }

    const result = this.performDetection(text);
    
    if (this.cache.size >= this.maxCacheSize) {
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(cacheKey, result);
    
    return result;
  }

  performDetection(text) {
    const textLower = text.toLowerCase();
    const matchedPatterns = [];
    const patternScores = {};
    let totalScore = 0;

    for (const rule of this.rules) {
      const matches = (text.match(rule.regex) || []).length;
      
      if (matches > 0) {
        const normalizedMatches = Math.min(matches / rule.threshold, 2.0);
        const scoreContribution = normalizedMatches * rule.weight;
        totalScore += scoreContribution;
        patternScores[rule.pattern] = scoreContribution;
        
        if (matches >= rule.threshold) {
          matchedPatterns.push({
            pattern: rule.pattern,
            description: rule.description,
            matches: matches
          });
        }
      }
    }

    const statisticalScore = this.getStatisticalScore(text);
    totalScore = (totalScore * 0.7) + (statisticalScore * 0.3);

    const confidence = Math.min(totalScore, 1.0);
    const isGPT4o = confidence >= this.threshold;

    return {
      isGPT4o,
      confidence,
      matchedPatterns,
      patternScores,
      explanation: this.generateExplanation(isGPT4o, confidence, matchedPatterns)
    };
  }

  getStatisticalScore(text) {
    const sentences = text.split(/[.!?]+/).filter(s => s.trim());
    const words = text.split(/\s+/);
    
    let score = 0;

    if (sentences.length > 0) {
      const avgSentenceLength = words.length / sentences.length;
      if (avgSentenceLength > 15 && avgSentenceLength < 25) {
        score += 0.3;
      }
    }

    const uniqueWords = new Set(words.map(w => w.toLowerCase()));
    const lexicalDiversity = uniqueWords.size / words.length;
    if (lexicalDiversity < 0.6) {
      score += 0.2;
    }

    const sentenceLengths = sentences.map(s => s.split(/\s+/).length);
    const variance = this.calculateVariance(sentenceLengths);
    if (variance < 5) {
      score += 0.3;
    }

    const punctuationCount = (text.match(/[.,;:!?()[\]{}"\'-]/g) || []).length;
    const punctuationRatio = punctuationCount / text.length;
    if (punctuationRatio > 0.08) {
      score += 0.2;
    }

    return Math.min(score, 1.0);
  }

  calculateVariance(numbers) {
    if (numbers.length === 0) return 0;
    const mean = numbers.reduce((a, b) => a + b, 0) / numbers.length;
    const squaredDiffs = numbers.map(n => Math.pow(n - mean, 2));
    return squaredDiffs.reduce((a, b) => a + b, 0) / numbers.length;
  }

  generateExplanation(isGPT4o, confidence, patterns) {
    if (isGPT4o) {
      const confidencePercent = Math.round(confidence * 100);
      let prefix;
      
      if (confidence > 0.9) {
        prefix = 'Very likely GPT-4o';
      } else if (confidence > 0.8) {
        prefix = 'Likely GPT-4o';
      } else {
        prefix = 'Possibly GPT-4o';
      }

      if (patterns.length > 0) {
        const topPatterns = patterns.slice(0, 3).map(p => p.description).join(', ');
        return `${prefix} (${confidencePercent}% confidence). Detected: ${topPatterns}`;
      } else {
        return `${prefix} (${confidencePercent}% confidence)`;
      }
    } else {
      const humanConfidence = Math.round((1 - confidence) * 100);
      return `Likely human-written (${humanConfidence}% confidence)`;
    }
  }

  hashText(text) {
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      const char = text.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash.toString(36);
  }

  quickDetect(text) {
    const quickPatterns = [
      { regex: /not\s+\w+,?\s+but\s+\w+/i, weight: 0.3 },
      { regex: /it's\s+(important|worth)\s+to\s+note/i, weight: 0.25 },
      { regex: /(firstly|secondly|thirdly)/i, weight: 0.2 },
      { regex: /\b(perhaps|maybe|possibly|might)\b/i, weight: 0.15 },
      { regex: /(advantages.*disadvantages|pros.*cons)/i, weight: 0.3 }
    ];

    let score = 0;
    for (const pattern of quickPatterns) {
      if (pattern.regex.test(text)) {
        score += pattern.weight;
      }
    }

    const confidence = Math.min(score, 1.0);
    return {
      isGPT4o: confidence >= 0.5,
      confidence
    };
  }

  batchDetect(texts) {
    return texts.map(text => this.detect(text));
  }

  clearCache() {
    this.cache.clear();
  }

  updateThreshold(newThreshold) {
    this.threshold = Math.max(0, Math.min(1, newThreshold));
  }

  getStats() {
    return {
      rulesCount: this.rules.length,
      threshold: this.threshold,
      cacheSize: this.cache.size,
      initialized: this.initialized
    };
  }
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = GPT4oDetectorEngine;
}