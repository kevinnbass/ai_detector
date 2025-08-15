# Complexity Refactoring Examples

def statistical_detection(self, text: str) -> DetectionResult:
    """Refactored statistical detection with reduced complexity"""
    features = self.extract_statistical_features(text)
    
    # Use score rules pattern instead of multiple if statements
    score_rules = [
        self._check_sentence_length_rule(features),
        self._check_lexical_diversity_rule(features),
        self._check_punctuation_rule(features),
        self._check_variance_rule(features),
        self._check_structure_rule(features)
    ]
    
    gpt4o_score = sum(rule['score'] for rule in score_rules)
    matched_indicators = [rule['indicator'] for rule in score_rules if rule['score'] > 0]
    
    confidence = min(gpt4o_score, 1.0)
    is_gpt4o = confidence >= self.threshold
    
    return DetectionResult(
        is_gpt4o=is_gpt4o,
        confidence=confidence,
        matched_patterns=matched_indicators,
        pattern_scores={'statistical': confidence},
        explanation=f"Statistical analysis confidence: {confidence:.2%}"
    )

def _check_sentence_length_rule(self, features: Dict[str, float]) -> Dict[str, Any]:
    """Check sentence length rule"""
    avg_length = features['avg_sentence_length']
    if 15 < avg_length < 25:
        return {'score': 0.2, 'indicator': "Consistent medium sentence length"}
    return {'score': 0, 'indicator': None}

def _check_lexical_diversity_rule(self, features: Dict[str, float]) -> Dict[str, Any]:
    """Check lexical diversity rule"""
    if features['lexical_diversity'] < 0.6:
        return {'score': 0.15, 'indicator': "Low lexical diversity"}
    return {'score': 0, 'indicator': None}

def _check_punctuation_rule(self, features: Dict[str, float]) -> Dict[str, Any]:
    """Check punctuation rule"""
    if features['punctuation_ratio'] > 0.08:
        return {'score': 0.15, 'indicator': "High punctuation usage"}
    return {'score': 0, 'indicator': None}

def _check_variance_rule(self, features: Dict[str, float]) -> Dict[str, Any]:
    """Check sentence variance rule"""
    if features['sentence_length_variance'] < 5:
        return {'score': 0.25, 'indicator': "Low sentence length variance"}
    return {'score': 0, 'indicator': None}

def _check_structure_rule(self, features: Dict[str, float]) -> Dict[str, Any]:
    """Check paragraph structure rule"""
    if features['paragraph_structure_score'] > 0.7:
        return {'score': 0.25, 'indicator': "Structured paragraph format"}
    return {'score': 0, 'indicator': None}

def extract_statistical_features(self, text: str) -> Dict[str, float]:
    """Refactored feature extraction with reduced complexity"""
    sentences = self._extract_sentences(text)
    words = text.split()
    
    features = {}
    features.update(self._calculate_sentence_features(sentences))
    features.update(self._calculate_word_features(words, text))
    features.update(self._calculate_structure_features(text))
    
    return features

def _extract_sentences(self, text: str) -> List[str]:
    """Extract sentences from text"""
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

def _calculate_sentence_features(self, sentences: List[str]) -> Dict[str, float]:
    """Calculate sentence-related features"""
    if not sentences:
        return {
            'avg_sentence_length': 0,
            'sentence_length_variance': 0
        }
    
    sentence_lengths = [len(s.split()) for s in sentences]
    return {
        'avg_sentence_length': np.mean(sentence_lengths),
        'sentence_length_variance': np.var(sentence_lengths)
    }

def _calculate_word_features(self, words: List[str], text: str) -> Dict[str, float]:
    """Calculate word-related features"""
    if not words:
        return {
            'total_words': 0,
            'unique_words': 0,
            'lexical_diversity': 0,
            'punctuation_ratio': 0
        }
    
    punctuation_count = sum(1 for char in text if char in '.,;:!?()[]{}"'-')
    
    return {
        'total_words': len(words),
        'unique_words': len(set(words)),
        'lexical_diversity': len(set(words)) / len(words),
        'punctuation_ratio': punctuation_count / len(text)
    }

def _calculate_structure_features(self, text: str) -> Dict[str, float]:
    """Calculate structural features"""
    paragraphs = text.split('\n\n')
    structure_score = 1.0 if len(paragraphs) > 1 else 0.3
    
    return {
        'paragraph_structure_score': structure_score
    }