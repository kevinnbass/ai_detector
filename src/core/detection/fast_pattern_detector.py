"""
Fast pattern-based AI text detection optimized for sub-100ms performance.

Uses pre-compiled patterns, efficient string matching, and optimized algorithms
to achieve maximum detection speed while maintaining reasonable accuracy.
"""

import re
import time
from typing import Dict, Any, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from src.core.monitoring import get_logger, get_metrics_collector


class PatternType(Enum):
    """Types of AI text patterns."""
    TRANSITION_WORDS = "transition_words"
    FORMAL_LANGUAGE = "formal_language"
    HEDGING_LANGUAGE = "hedging_language"
    ACADEMIC_PHRASES = "academic_phrases"
    REPETITIVE_STRUCTURES = "repetitive_structures"


@dataclass
class PatternMatch:
    """Represents a pattern match in text."""
    pattern_type: PatternType
    match_text: str
    position: int
    confidence: float
    weight: float


class FastPatternDetector:
    """Ultra-fast pattern-based AI text detection."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
        
        # Pre-compile all regex patterns for maximum speed
        self._compile_patterns()
        
        # Pre-built word sets for O(1) lookups
        self._build_word_sets()
        
        # Pattern weights for scoring
        self.pattern_weights = {
            PatternType.TRANSITION_WORDS: 0.2,
            PatternType.FORMAL_LANGUAGE: 0.25,
            PatternType.HEDGING_LANGUAGE: 0.15,
            PatternType.ACADEMIC_PHRASES: 0.3,
            PatternType.REPETITIVE_STRUCTURES: 0.1
        }
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for maximum performance."""
        # Transition word patterns
        self.transition_regex = re.compile(
            r'\b(?:furthermore|however|therefore|moreover|additionally|'
            r'consequently|nevertheless|meanwhile|subsequently|thus|hence)\b',
            re.IGNORECASE
        )
        
        # Academic phrase patterns
        self.academic_regex = re.compile(
            r'(?:comprehensive analysis|multifaceted nature|careful consideration|'
            r'it should be noted|it is important to|one must consider|'
            r'in conclusion|to summarize|in summary)\b',
            re.IGNORECASE
        )
        
        # Hedging language patterns
        self.hedging_regex = re.compile(
            r'\b(?:might|could|may|perhaps|possibly|likely|seems|appears|'
            r'suggests|indicates|tend to|appear to)\b',
            re.IGNORECASE
        )
        
        # Formal language patterns
        self.formal_regex = re.compile(
            r'\b(?:utilize|implement|demonstrate|facilitate|establish|'
            r'paradigm|methodology|comprehensive|substantial|significant)\b',
            re.IGNORECASE
        )
        
        # Repetitive structure patterns
        self.repetitive_regex = re.compile(
            r'(\b\w+\b)(?:\s+\w+){0,3}\s+\1\b',  # Word repetition within 4 words
            re.IGNORECASE
        )
    
    def _build_word_sets(self):
        """Build word sets for fast O(1) lookups."""
        self.ai_indicators = {
            # Formal/academic words
            "furthermore", "however", "therefore", "moreover", "additionally",
            "consequently", "nevertheless", "meanwhile", "subsequently",
            "comprehensive", "multifaceted", "paradigm", "methodology",
            "facilitate", "demonstrate", "substantial", "significant",
            
            # Hedging words
            "might", "could", "perhaps", "possibly", "likely", "seems",
            "appears", "suggests", "indicates", "tend", "appear",
            
            # Academic phrases (as single tokens after preprocessing)
            "analysis", "consideration", "conclusion", "summary", "examine"
        }
        
        self.human_indicators = {
            # Casual language
            "lol", "haha", "omg", "wtf", "tbh", "imo", "idk", "ngl",
            "gonna", "wanna", "gotta", "dunno", "kinda", "sorta",
            
            # Emotional expressions
            "love", "hate", "awesome", "terrible", "amazing", "awful",
            "excited", "angry", "happy", "sad", "frustrated",
            
            # Contractions (stemmed)
            "cant", "dont", "wont", "isnt", "arent", "wasnt", "werent"
        }
        
        # Emoji patterns (common ones)
        self.emoji_pattern = re.compile(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
            r'\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001f900-\U0001f9ff'
            r'\U0001f600-\U0001f64f]'
        )
    
    def detect(self, text: str) -> Dict[str, Any]:
        """Perform fast pattern-based detection."""
        start_time = time.time()
        
        # Quick length check
        if len(text) < 10:
            return self._create_result(False, 0.1, "too_short", start_time, [])
        
        # Normalize text for processing
        normalized_text = self._fast_normalize(text)
        words = normalized_text.split()
        
        if len(words) < 3:
            return self._create_result(False, 0.2, "too_few_words", start_time, [])
        
        # Fast emoji check
        if self.emoji_pattern.search(text):
            return self._create_result(False, 0.85, "emoji_present", start_time, [])
        
        # Pattern matching
        matches = []
        
        # Check each pattern type
        matches.extend(self._check_transition_words(normalized_text))
        matches.extend(self._check_formal_language(normalized_text))
        matches.extend(self._check_hedging_language(normalized_text))
        matches.extend(self._check_academic_phrases(normalized_text))
        matches.extend(self._check_repetitive_structures(normalized_text))
        
        # Fast word-level analysis
        ai_score, human_score = self._fast_word_analysis(words)
        
        # Calculate final score
        pattern_score = self._calculate_pattern_score(matches, len(words))
        final_score = (pattern_score + ai_score - human_score) / 2
        
        # Determine result
        is_ai = final_score > 0.5
        confidence = min(max(abs(final_score - 0.5) * 2, 0.1), 0.95)
        
        return self._create_result(is_ai, confidence, "pattern_analysis", start_time, matches)
    
    def _fast_normalize(self, text: str) -> str:
        """Fast text normalization for pattern matching."""
        # Convert to lowercase and remove extra whitespace
        text = re.sub(r'\s+', ' ', text.lower().strip())
        
        # Remove punctuation for word analysis (keep for pattern matching)
        return text
    
    def _check_transition_words(self, text: str) -> List[PatternMatch]:
        """Check for transition word patterns."""
        matches = []
        for match in self.transition_regex.finditer(text):
            matches.append(PatternMatch(
                pattern_type=PatternType.TRANSITION_WORDS,
                match_text=match.group(),
                position=match.start(),
                confidence=0.7,
                weight=self.pattern_weights[PatternType.TRANSITION_WORDS]
            ))
        return matches
    
    def _check_formal_language(self, text: str) -> List[PatternMatch]:
        """Check for formal language patterns."""
        matches = []
        for match in self.formal_regex.finditer(text):
            matches.append(PatternMatch(
                pattern_type=PatternType.FORMAL_LANGUAGE,
                match_text=match.group(),
                position=match.start(),
                confidence=0.8,
                weight=self.pattern_weights[PatternType.FORMAL_LANGUAGE]
            ))
        return matches
    
    def _check_hedging_language(self, text: str) -> List[PatternMatch]:
        """Check for hedging language patterns."""
        matches = []
        for match in self.hedging_regex.finditer(text):
            matches.append(PatternMatch(
                pattern_type=PatternType.HEDGING_LANGUAGE,
                match_text=match.group(),
                position=match.start(),
                confidence=0.6,
                weight=self.pattern_weights[PatternType.HEDGING_LANGUAGE]
            ))
        return matches
    
    def _check_academic_phrases(self, text: str) -> List[PatternMatch]:
        """Check for academic phrase patterns."""
        matches = []
        for match in self.academic_regex.finditer(text):
            matches.append(PatternMatch(
                pattern_type=PatternType.ACADEMIC_PHRASES,
                match_text=match.group(),
                position=match.start(),
                confidence=0.9,
                weight=self.pattern_weights[PatternType.ACADEMIC_PHRASES]
            ))
        return matches
    
    def _check_repetitive_structures(self, text: str) -> List[PatternMatch]:
        """Check for repetitive structure patterns."""
        matches = []
        for match in self.repetitive_regex.finditer(text):
            matches.append(PatternMatch(
                pattern_type=PatternType.REPETITIVE_STRUCTURES,
                match_text=match.group(),
                position=match.start(),
                confidence=0.5,
                weight=self.pattern_weights[PatternType.REPETITIVE_STRUCTURES]
            ))
        return matches
    
    def _fast_word_analysis(self, words: List[str]) -> Tuple[float, float]:
        """Fast word-level analysis using pre-built sets."""
        if not words:
            return 0.0, 0.0
        
        ai_count = 0
        human_count = 0
        
        # Count AI and human indicators
        for word in words:
            # Remove punctuation for word matching
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            if clean_word in self.ai_indicators:
                ai_count += 1
            elif clean_word in self.human_indicators:
                human_count += 1
        
        # Normalize by word count
        ai_score = min(ai_count / len(words) * 5, 1.0)  # Scale factor 5
        human_score = min(human_count / len(words) * 5, 1.0)
        
        return ai_score, human_score
    
    def _calculate_pattern_score(self, matches: List[PatternMatch], word_count: int) -> float:
        """Calculate overall pattern score."""
        if not matches:
            return 0.0
        
        # Weight matches by type and frequency
        type_scores = {}
        
        for match in matches:
            pattern_type = match.pattern_type
            if pattern_type not in type_scores:
                type_scores[pattern_type] = []
            
            type_scores[pattern_type].append(match.confidence * match.weight)
        
        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0
        
        for pattern_type, scores in type_scores.items():
            # Use average score for each pattern type to avoid over-counting
            avg_score = sum(scores) / len(scores)
            type_weight = self.pattern_weights[pattern_type]
            
            total_score += avg_score * type_weight
            total_weight += type_weight
        
        # Normalize by text length
        length_factor = min(word_count / 50, 1.0)  # Normalize around 50 words
        
        return (total_score / max(total_weight, 0.1)) * length_factor
    
    def _create_result(self, is_ai: bool, confidence: float, method: str, 
                      start_time: float, matches: List[PatternMatch]) -> Dict[str, Any]:
        """Create standardized result dictionary."""
        processing_time = (time.time() - start_time) * 1000
        
        # Record metrics
        self.metrics.observe_histogram("fast_pattern_detection_ms", processing_time)
        self.metrics.increment_counter("fast_pattern_detections_total")
        
        return {
            "is_ai_generated": is_ai,
            "confidence_score": confidence,
            "processing_time_ms": processing_time,
            "method_used": f"fast_pattern_{method}",
            "detection_details": {
                "pattern_matches": len(matches),
                "match_types": list(set(match.pattern_type.value for match in matches)),
                "pattern_score": self._calculate_pattern_score(matches, 1) if matches else 0.0
            },
            "matches": [
                {
                    "type": match.pattern_type.value,
                    "text": match.match_text,
                    "position": match.position,
                    "confidence": match.confidence
                }
                for match in matches[:10]  # Limit to first 10 matches
            ]
        }
    
    def benchmark(self, test_texts: List[str]) -> Dict[str, Any]:
        """Benchmark the fast pattern detector."""
        start_time = time.time()
        results = []
        
        for text in test_texts:
            result = self.detect(text)
            results.append(result)
        
        total_time = (time.time() - start_time) * 1000
        avg_time = total_time / len(test_texts)
        
        return {
            "total_texts": len(test_texts),
            "total_time_ms": total_time,
            "average_time_ms": avg_time,
            "texts_per_second": len(test_texts) / (total_time / 1000),
            "sub_100ms_count": sum(1 for r in results if r["processing_time_ms"] < 100),
            "sub_50ms_count": sum(1 for r in results if r["processing_time_ms"] < 50),
            "sub_10ms_count": sum(1 for r in results if r["processing_time_ms"] < 10),
            "performance_grade": "A" if avg_time < 10 else "B" if avg_time < 50 else "C" if avg_time < 100 else "D"
        }


class HeuristicDetector:
    """Ultra-fast heuristic-based detection for immediate results."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.metrics = get_metrics_collector()
    
    def detect(self, text: str) -> Dict[str, Any]:
        """Perform ultra-fast heuristic detection."""
        start_time = time.time()
        
        # Length heuristics
        if len(text) < 10:
            return self._result(False, 0.9, "too_short", start_time)
        
        if len(text) > 2000:
            return self._result(True, 0.7, "very_long", start_time)
        
        text_lower = text.lower()
        
        # Emoji check (strong human indicator)
        if any(char in text for char in "😂😊😢😍😱💀🔥"):
            return self._result(False, 0.9, "emoji_present", start_time)
        
        # Internet slang (strong human indicator)
        slang_words = ["lol", "lmao", "omg", "wtf", "tbh", "imo", "ngl"]
        if any(word in text_lower for word in slang_words):
            return self._result(False, 0.85, "slang_detected", start_time)
        
        # Strong AI indicators
        ai_phrases = [
            "comprehensive analysis", "multifaceted nature", "paradigm",
            "it should be noted", "careful consideration"
        ]
        if any(phrase in text_lower for phrase in ai_phrases):
            return self._result(True, 0.8, "ai_phrases", start_time)
        
        # Word length heuristic
        words = text.split()
        if words:
            avg_word_length = sum(len(word) for word in words) / len(words)
            if avg_word_length > 7:
                return self._result(True, 0.6, "long_words", start_time)
            if avg_word_length < 4:
                return self._result(False, 0.6, "short_words", start_time)
        
        # Default uncertain result
        return self._result(None, 0.5, "uncertain", start_time)
    
    def _result(self, is_ai: Optional[bool], confidence: float, reason: str, start_time: float) -> Dict[str, Any]:
        """Create heuristic result."""
        processing_time = (time.time() - start_time) * 1000
        
        self.metrics.observe_histogram("heuristic_detection_ms", processing_time)
        self.metrics.increment_counter("heuristic_detections_total")
        
        return {
            "is_ai_generated": is_ai,
            "confidence_score": confidence,
            "processing_time_ms": processing_time,
            "method_used": f"heuristic_{reason}",
            "detection_details": {
                "heuristic_reason": reason,
                "certainty": "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
            }
        }