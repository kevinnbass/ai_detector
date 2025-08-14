import json
import re
from typing import Dict, List, Any, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum

class DetectionMethod(Enum):
    PATTERN_BASED = "pattern_based"
    STATISTICAL = "statistical"
    HYBRID = "hybrid"

@dataclass
class DetectionResult:
    is_gpt4o: bool
    confidence: float
    matched_patterns: List[str]
    pattern_scores: Dict[str, float]
    explanation: str

class GPT4oDetector:
    def __init__(self, patterns_file: str = None):
        self.patterns = []
        self.rules = []
        self.threshold = 0.7
        
        if patterns_file:
            self.load_patterns(patterns_file)
        else:
            self.initialize_default_patterns()
    
    def initialize_default_patterns(self):
        self.rules = [
            {
                'rule_id': 'HEDGE_01',
                'pattern': 'excessive_hedging',
                'regex': r'\b(perhaps|maybe|possibly|might|could|seems|appears|likely|probably|generally|typically|often|sometimes)\b',
                'threshold': 3,
                'weight': 0.15,
                'description': 'Excessive use of hedging language'
            },
            {
                'rule_id': 'CONTRAST_01',
                'pattern': 'contrast_rhetoric',
                'regex': r'(not\s+\w+,?\s+but\s+\w+|while\s+.*,\s+|although\s+.*,\s+|however,?\s+|on\s+the\s+other\s+hand)',
                'threshold': 1,
                'weight': 0.25,
                'description': 'Frequent contrast constructions'
            },
            {
                'rule_id': 'FORMAL_01',
                'pattern': 'formal_in_casual',
                'regex': r'\b(furthermore|moreover|consequently|therefore|thus|hence|accordingly|nevertheless|nonetheless)\b',
                'threshold': 1,
                'weight': 0.20,
                'description': 'Formal language in casual context'
            },
            {
                'rule_id': 'LIST_01',
                'pattern': 'structured_lists',
                'regex': r'(firstly|secondly|thirdly|first,|second,|third,|\d\.|•|-\s+)',
                'threshold': 2,
                'weight': 0.15,
                'description': 'Structured list formatting'
            },
            {
                'rule_id': 'QUAL_01',
                'pattern': 'excessive_qualifiers',
                'regex': r"(it's\s+(important|worth|crucial|essential)\s+(to\s+)?(note|noting|mention|consider)|keep\s+in\s+mind|bear\s+in\s+mind|remember\s+that|consider\s+that)",
                'threshold': 1,
                'weight': 0.20,
                'description': 'Overuse of qualifier phrases'
            },
            {
                'rule_id': 'BALANCE_01',
                'pattern': 'balanced_presentation',
                'regex': r'(advantages.*disadvantages|pros.*cons|benefits.*drawbacks|positive.*negative|strengths.*weaknesses)',
                'threshold': 1,
                'weight': 0.25,
                'description': 'Balanced pros/cons presentation'
            },
            {
                'rule_id': 'EXPLAIN_01',
                'pattern': 'explanatory_style',
                'regex': r'(essentially|basically|in\s+other\s+words|simply\s+put|to\s+put\s+it\s+simply|in\s+essence)',
                'threshold': 1,
                'weight': 0.15,
                'description': 'Explanatory language patterns'
            },
            {
                'rule_id': 'CAVEAT_01',
                'pattern': 'caveats',
                'regex': r'(that\s+said|having\s+said\s+that|with\s+that\s+in\s+mind|that\s+being\s+said|to\s+be\s+fair)',
                'threshold': 1,
                'weight': 0.18,
                'description': 'Frequent caveats and disclaimers'
            }
        ]
    
    def load_patterns(self, patterns_file: str):
        try:
            with open(patterns_file, 'r') as f:
                data = json.load(f)
                self.rules = data.get('detection_rules', [])
                self.threshold = data.get('confidence_threshold', 0.7)
        except Exception as e:
            print(f"Error loading patterns: {e}")
            self.initialize_default_patterns()
    
    def detect(self, text: str, method: DetectionMethod = DetectionMethod.HYBRID) -> DetectionResult:
        if method == DetectionMethod.PATTERN_BASED:
            return self.pattern_based_detection(text)
        elif method == DetectionMethod.STATISTICAL:
            return self.statistical_detection(text)
        else:
            return self.hybrid_detection(text)
    
    def pattern_based_detection(self, text: str) -> DetectionResult:
        text_lower = text.lower()
        matched_patterns = []
        pattern_scores = {}
        total_score = 0
        
        for rule in self.rules:
            matches = len(re.findall(rule['regex'], text_lower))
            if matches > 0:
                normalized_matches = min(matches / rule['threshold'], 2.0)
                score_contribution = normalized_matches * rule['weight']
                total_score += score_contribution
                pattern_scores[rule['pattern']] = score_contribution
                
                if matches >= rule['threshold']:
                    matched_patterns.append(rule['description'])
        
        confidence = min(total_score, 1.0)
        is_gpt4o = confidence >= self.threshold
        
        explanation = self.generate_explanation(is_gpt4o, confidence, matched_patterns)
        
        return DetectionResult(
            is_gpt4o=is_gpt4o,
            confidence=confidence,
            matched_patterns=matched_patterns,
            pattern_scores=pattern_scores,
            explanation=explanation
        )
    
    def statistical_detection(self, text: str) -> DetectionResult:
        features = self.extract_statistical_features(text)
        
        gpt4o_score = 0
        matched_indicators = []
        
        if features['avg_sentence_length'] > 15 and features['avg_sentence_length'] < 25:
            gpt4o_score += 0.2
            matched_indicators.append("Consistent medium sentence length")
        
        if features['lexical_diversity'] < 0.6:
            gpt4o_score += 0.15
            matched_indicators.append("Low lexical diversity")
        
        if features['punctuation_ratio'] > 0.08:
            gpt4o_score += 0.15
            matched_indicators.append("High punctuation usage")
        
        if features['sentence_length_variance'] < 5:
            gpt4o_score += 0.25
            matched_indicators.append("Low sentence length variance")
        
        if features['paragraph_structure_score'] > 0.7:
            gpt4o_score += 0.25
            matched_indicators.append("Structured paragraph format")
        
        confidence = min(gpt4o_score, 1.0)
        is_gpt4o = confidence >= self.threshold
        
        return DetectionResult(
            is_gpt4o=is_gpt4o,
            confidence=confidence,
            matched_patterns=matched_indicators,
            pattern_scores={'statistical': confidence},
            explanation=f"Statistical analysis confidence: {confidence:.2%}"
        )
    
    def hybrid_detection(self, text: str) -> DetectionResult:
        pattern_result = self.pattern_based_detection(text)
        statistical_result = self.statistical_detection(text)
        
        combined_confidence = (pattern_result.confidence * 0.7 + statistical_result.confidence * 0.3)
        
        all_patterns = pattern_result.matched_patterns + statistical_result.matched_patterns
        all_scores = {**pattern_result.pattern_scores, **statistical_result.pattern_scores}
        
        is_gpt4o = combined_confidence >= self.threshold
        
        explanation = f"Hybrid detection: Pattern confidence {pattern_result.confidence:.2%}, Statistical confidence {statistical_result.confidence:.2%}"
        
        return DetectionResult(
            is_gpt4o=is_gpt4o,
            confidence=combined_confidence,
            matched_patterns=all_patterns,
            pattern_scores=all_scores,
            explanation=explanation
        )
    
    def extract_statistical_features(self, text: str) -> Dict[str, float]:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = text.split()
        
        features = {}
        
        if sentences:
            sentence_lengths = [len(s.split()) for s in sentences]
            features['avg_sentence_length'] = np.mean(sentence_lengths)
            features['sentence_length_variance'] = np.var(sentence_lengths)
        else:
            features['avg_sentence_length'] = 0
            features['sentence_length_variance'] = 0
        
        if words:
            features['total_words'] = len(words)
            features['unique_words'] = len(set(words))
            features['lexical_diversity'] = features['unique_words'] / features['total_words']
            
            punctuation_count = sum(1 for char in text if char in '.,;:!?()[]{}"\'-')
            features['punctuation_ratio'] = punctuation_count / len(text)
        else:
            features['total_words'] = 0
            features['unique_words'] = 0
            features['lexical_diversity'] = 0
            features['punctuation_ratio'] = 0
        
        paragraphs = text.split('\n\n')
        if len(paragraphs) > 1:
            features['paragraph_structure_score'] = 1.0
        else:
            features['paragraph_structure_score'] = 0.3
        
        return features
    
    def generate_explanation(self, is_gpt4o: bool, confidence: float, patterns: List[str]) -> str:
        if is_gpt4o:
            if confidence > 0.9:
                prefix = "Very likely GPT-4o generated"
            elif confidence > 0.8:
                prefix = "Likely GPT-4o generated"
            else:
                prefix = "Possibly GPT-4o generated"
            
            if patterns:
                pattern_str = ", ".join(patterns[:3])
                return f"{prefix} ({confidence:.1%} confidence). Detected: {pattern_str}"
            else:
                return f"{prefix} ({confidence:.1%} confidence)"
        else:
            return f"Likely human-written ({(1-confidence):.1%} confidence)"
    
    def batch_detect(self, texts: List[str]) -> List[DetectionResult]:
        results = []
        for text in texts:
            results.append(self.detect(text))
        return results
    
    def get_detection_stats(self, results: List[DetectionResult]) -> Dict[str, Any]:
        total = len(results)
        gpt4o_count = sum(1 for r in results if r.is_gpt4o)
        avg_confidence = np.mean([r.confidence for r in results])
        
        pattern_frequency = {}
        for result in results:
            for pattern in result.matched_patterns:
                pattern_frequency[pattern] = pattern_frequency.get(pattern, 0) + 1
        
        return {
            'total_analyzed': total,
            'gpt4o_detected': gpt4o_count,
            'human_detected': total - gpt4o_count,
            'average_confidence': avg_confidence,
            'most_common_patterns': sorted(pattern_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
        }

class FastDetector:
    def __init__(self):
        self.quick_patterns = [
            (r'not\s+\w+,?\s+but\s+\w+', 0.3),
            (r"it's\s+(important|worth)\s+to\s+note", 0.25),
            (r'(firstly|secondly|thirdly)', 0.2),
            (r'\b(perhaps|maybe|possibly|might)\b', 0.15),
            (r'(advantages.*disadvantages|pros.*cons)', 0.3),
        ]
    
    def quick_detect(self, text: str) -> Tuple[bool, float]:
        text_lower = text.lower()
        score = 0
        
        for pattern, weight in self.quick_patterns:
            if re.search(pattern, text_lower):
                score += weight
        
        confidence = min(score, 1.0)
        return confidence >= 0.5, confidence

def export_for_extension(detector: GPT4oDetector, output_file: str = 'detection_rules.json'):
    export_data = {
        'rules': [],
        'threshold': detector.threshold,
        'version': '1.0.0'
    }
    
    for rule in detector.rules:
        export_data['rules'].append({
            'id': rule['rule_id'],
            'pattern': rule['pattern'],
            'regex': rule['regex'],
            'weight': rule['weight'],
            'threshold': rule['threshold'],
            'description': rule['description']
        })
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"Detection rules exported to {output_file}")

if __name__ == "__main__":
    detector = GPT4oDetector()
    
    test_texts = [
        "While AI is advancing rapidly, it's important to note that there are both advantages and disadvantages. On one hand, productivity increases. On the other hand, job displacement is a concern.",
        "AI is crazy fast now. honestly can't keep up with all the updates. feels like skynet incoming lol",
        "The implications of quantum computing are profound. Firstly, cryptography will need reimagining. Secondly, drug discovery could accelerate. However, practical implementation remains challenging.",
        "quantum computers gonna break everything we know about encryption. wild times ahead",
    ]
    
    print("GPT-4o Detection System Test")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nText {i}: {text[:80]}...")
        result = detector.detect(text)
        print(f"Result: {'GPT-4o' if result.is_gpt4o else 'Human'} ({result.confidence:.1%} confidence)")
        if result.matched_patterns:
            print(f"Patterns: {', '.join(result.matched_patterns[:3])}")
    
    export_for_extension(detector, '../data/detection_rules.json')
    
    fast = FastDetector()
    print("\n" + "=" * 50)
    print("Fast Detection Test:")
    for i, text in enumerate(test_texts, 1):
        is_gpt, conf = fast.quick_detect(text)
        print(f"Text {i}: {'GPT-4o' if is_gpt else 'Human'} ({conf:.1%})")