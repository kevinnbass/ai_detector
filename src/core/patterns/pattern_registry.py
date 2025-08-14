"""
Pattern Registry - Single Source of Truth for All Detection Patterns
Consolidates patterns from detector.py, detector-engine.js, detection-rules.json, gpt4o_miner.py
"""

import json
import re
from typing import Dict, List, Any, Optional, Pattern
from dataclasses import dataclass, asdict
from enum import Enum
import os


class PatternType(Enum):
    """Pattern types"""
    AI_INDICATOR = "ai_indicator"
    HUMAN_INDICATOR = "human_indicator"
    NEUTRAL = "neutral"


class PatternCategory(Enum):
    """Pattern categories"""
    HEDGING = "hedging"
    BALANCE = "balance"
    FORMAL = "formal"
    META = "meta"
    QUALIFIER = "qualifier"
    CONTRAST = "contrast"
    LIST = "list"
    EXPLANATORY = "explanatory"
    CAVEAT = "caveat"
    CASUAL = "casual"
    EMOTIONAL = "emotional"
    ERROR = "error"


@dataclass
class DetectionPattern:
    """Single detection pattern"""
    id: str
    name: str
    category: PatternCategory
    type: PatternType
    regex: str
    weight: float
    threshold: int
    reliability: float
    description: str
    examples: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['category'] = self.category.value
        data['type'] = self.type.value
        return data
    
    def to_js_format(self) -> Dict:
        """Convert to JavaScript format"""
        return {
            'id': self.id,
            'pattern': self.name,
            'regex': self.regex,
            'weight': self.weight,
            'threshold': self.threshold,
            'description': self.description
        }
    
    def compile_regex(self) -> Pattern:
        """Compile regex pattern"""
        return re.compile(self.regex, re.IGNORECASE)


class PatternRegistry:
    """
    Central pattern registry - single source of truth for all detection patterns
    """
    
    # Version for pattern compatibility
    VERSION = "2.1.0"
    
    def __init__(self):
        self.patterns: Dict[str, DetectionPattern] = {}
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Initialize all patterns"""
        
        # AI Indicator Patterns (based on data analysis)
        self.add_pattern(DetectionPattern(
            id="AI_HEDGE_01",
            name="excessive_hedging",
            category=PatternCategory.HEDGING,
            type=PatternType.AI_INDICATOR,
            regex=r"\b(it's\s+important\s+to\s+note|it's\s+worth\s+considering|merit\s+thorough\s+examination|perhaps|maybe|possibly|might|could|seems|appears|likely|probably|generally|typically|often|sometimes)\b",
            weight=0.18,
            threshold=2,
            reliability=0.87,
            description="Excessive hedging and qualifying language",
            examples=["it's important to note", "it's worth considering", "merit thorough examination"]
        ))
        
        self.add_pattern(DetectionPattern(
            id="AI_BALANCE_01",
            name="balanced_presentation",
            category=PatternCategory.BALANCE,
            type=PatternType.AI_INDICATOR,
            regex=r"(on\s+one\s+hand.*on\s+the\s+other\s+hand|both\s+advantages\s+and\s+disadvantages|advantages.*disadvantages|pros.*cons|benefits.*drawbacks|positive.*negative|strengths.*weaknesses|opportunities.*challenges)",
            weight=0.30,
            threshold=1,
            reliability=0.85,
            description="Systematic balanced presentation",
            examples=["On one hand... On the other hand", "both advantages and disadvantages"]
        ))
        
        self.add_pattern(DetectionPattern(
            id="AI_FORMAL_01",
            name="formal_transitions",
            category=PatternCategory.FORMAL,
            type=PatternType.AI_INDICATOR,
            regex=r"\b(furthermore|moreover|consequently|therefore|thus|hence|accordingly|nevertheless|nonetheless|however|in\s+addition)\b",
            weight=0.25,
            threshold=1,
            reliability=0.82,
            description="Formal transitional phrases in casual context",
            examples=["Furthermore", "Moreover", "Consequently"]
        ))
        
        self.add_pattern(DetectionPattern(
            id="AI_META_01",
            name="meta_commentary",
            category=PatternCategory.META,
            type=PatternType.AI_INDICATOR,
            regex=r"(when\s+examining|in\s+considering|we\s+must\s+acknowledge|it's\s+crucial\s+to\s+understand|merit\s+careful\s+consideration)",
            weight=0.23,
            threshold=1,
            reliability=0.79,
            description="Meta-commentary about analysis process",
            examples=["when examining", "we must acknowledge", "merit careful consideration"]
        ))
        
        self.add_pattern(DetectionPattern(
            id="AI_QUAL_01",
            name="excessive_qualifiers",
            category=PatternCategory.QUALIFIER,
            type=PatternType.AI_INDICATOR,
            regex=r"(it's\s+(important|worth|crucial|essential)\s+(to\s+)?(note|noting|mention|consider)|keep\s+in\s+mind|bear\s+in\s+mind|remember\s+that|consider\s+that)",
            weight=0.20,
            threshold=1,
            reliability=0.75,
            description="Excessive qualifier phrases",
            examples=["it's important to note", "keep in mind", "consider that"]
        ))
        
        self.add_pattern(DetectionPattern(
            id="AI_CONTRAST_01",
            name="contrast_rhetoric",
            category=PatternCategory.CONTRAST,
            type=PatternType.AI_INDICATOR,
            regex=r"(not\s+\w+,?\s+but\s+\w+|while\s+.*,\s+|although\s+.*,\s+|however,?\s+)",
            weight=0.22,
            threshold=1,
            reliability=0.73,
            description="Contrast constructions (not X, but Y)",
            examples=["not only... but also", "while... however"]
        ))
        
        self.add_pattern(DetectionPattern(
            id="AI_LIST_01",
            name="structured_lists",
            category=PatternCategory.LIST,
            type=PatternType.AI_INDICATOR,
            regex=r"(firstly|secondly|thirdly|first,|second,|third,|\d\.|•|-\s+)",
            weight=0.15,
            threshold=2,
            reliability=0.68,
            description="Structured list formatting",
            examples=["Firstly", "1.", "• "]
        ))
        
        # Human Indicator Patterns (negative weights)
        self.add_pattern(DetectionPattern(
            id="HUMAN_CASUAL_01",
            name="casual_authenticity",
            category=PatternCategory.CASUAL,
            type=PatternType.HUMAN_INDICATOR,
            regex=r"\b(tbh|lol|rn|lmao|damn|honestly|insane|wtf|omg|fml|ngl|bruh|yolo|smh|imo|imho|afaik|iirc|fwiw|tl;?dr)\b",
            weight=-0.25,
            threshold=1,
            reliability=0.88,
            description="Casual authentic language (human indicator)",
            examples=["tbh", "lol", "rn", "lmao", "damn"]
        ))
        
        self.add_pattern(DetectionPattern(
            id="HUMAN_ERROR_01",
            name="natural_errors",
            category=PatternCategory.ERROR,
            type=PatternType.HUMAN_INDICATOR,
            regex=r"\b(i\s+[a-z]|[a-z]+\s+but\s+[a-z]|way\s+too\s+[a-z]+|kinda\s+|sorta\s+|gonna\s+|wanna\s+|gotta\s+|coulda\s+|shoulda\s+|woulda\s+|dunno\s+|gimme\s+|lemme\s+|ain't\s+|y'all\s+)",
            weight=-0.22,
            threshold=1,
            reliability=0.86,
            description="Natural errors and informal grammar (human indicator)",
            examples=["i can", "kinda", "gonna", "wanna"]
        ))
        
        self.add_pattern(DetectionPattern(
            id="HUMAN_EMOTION_01",
            name="emotional_spontaneity",
            category=PatternCategory.EMOTIONAL,
            type=PatternType.HUMAN_INDICATOR,
            regex=r"(i\s+literally\s+cannot\s+even|honestly\s+it's\s+insane|coding\s+is\s+pain|i\s+can\s+barely|omfg|this\s+is\s+so\s+[a-z]+|holy\s+shit|jesus\s+christ|for\s+fuck'?s?\s+sake|what\s+the\s+actual)",
            weight=-0.20,
            threshold=1,
            reliability=0.84,
            description="Genuine emotional reactions (human indicator)",
            examples=["I literally cannot even", "honestly it's insane", "holy shit"]
        ))
    
    def add_pattern(self, pattern: DetectionPattern) -> None:
        """Add a pattern to the registry"""
        self.patterns[pattern.id] = pattern
    
    def get_pattern(self, pattern_id: str) -> Optional[DetectionPattern]:
        """Get a pattern by ID"""
        return self.patterns.get(pattern_id)
    
    def get_patterns_by_type(self, pattern_type: PatternType) -> List[DetectionPattern]:
        """Get all patterns of a specific type"""
        return [p for p in self.patterns.values() if p.type == pattern_type]
    
    def get_patterns_by_category(self, category: PatternCategory) -> List[DetectionPattern]:
        """Get all patterns in a category"""
        return [p for p in self.patterns.values() if p.category == category]
    
    def export_to_json(self, filepath: Optional[str] = None) -> str:
        """Export patterns to JSON format"""
        data = {
            'version': self.VERSION,
            'patterns': [p.to_dict() for p in self.patterns.values()]
        }
        
        json_str = json.dumps(data, indent=2)
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        
        return json_str
    
    def export_to_javascript(self, filepath: Optional[str] = None) -> str:
        """Export patterns to JavaScript format"""
        js_patterns = [p.to_js_format() for p in self.patterns.values()]
        
        js_content = f"""// Auto-generated pattern definitions v{self.VERSION}
// DO NOT EDIT - Generated from pattern_registry.py

const detectionPatterns = {json.dumps(js_patterns, indent=2)};

export default detectionPatterns;
"""
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(js_content)
        
        return js_content
    
    def export_to_python(self, filepath: Optional[str] = None) -> str:
        """Export patterns to Python format"""
        py_content = f'''"""
Auto-generated pattern definitions v{self.VERSION}
DO NOT EDIT - Generated from pattern_registry.py
"""

DETECTION_PATTERNS = {{
'''
        
        for pattern_id, pattern in self.patterns.items():
            py_content += f'''    "{pattern_id}": {{
        "name": "{pattern.name}",
        "category": "{pattern.category.value}",
        "type": "{pattern.type.value}",
        "regex": r"{pattern.regex}",
        "weight": {pattern.weight},
        "threshold": {pattern.threshold},
        "reliability": {pattern.reliability},
        "description": "{pattern.description}",
        "examples": {pattern.examples}
    }},
'''
        
        py_content += "}\n"
        
        if filepath:
            with open(filepath, 'w') as f:
                f.write(py_content)
        
        return py_content
    
    def validate_patterns(self) -> Dict[str, List[str]]:
        """Validate all patterns for issues"""
        issues = {
            'errors': [],
            'warnings': []
        }
        
        # Check for duplicate IDs
        ids = [p.id for p in self.patterns.values()]
        if len(ids) != len(set(ids)):
            issues['errors'].append("Duplicate pattern IDs found")
        
        # Check for invalid regex
        for pattern in self.patterns.values():
            try:
                re.compile(pattern.regex)
            except re.error as e:
                issues['errors'].append(f"Invalid regex in {pattern.id}: {e}")
        
        # Check for weight issues
        for pattern in self.patterns.values():
            if abs(pattern.weight) > 1.0:
                issues['warnings'].append(f"Weight out of range in {pattern.id}: {pattern.weight}")
            
            if pattern.reliability < 0 or pattern.reliability > 1:
                issues['warnings'].append(f"Reliability out of range in {pattern.id}: {pattern.reliability}")
        
        # Check for missing examples
        for pattern in self.patterns.values():
            if not pattern.examples:
                issues['warnings'].append(f"No examples provided for {pattern.id}")
        
        return issues
    
    def apply_patterns(self, text: str) -> Dict[str, Any]:
        """Apply all patterns to text and return analysis"""
        results = {
            'ai_score': 0.0,
            'human_score': 0.0,
            'matches': [],
            'ai_patterns_found': [],
            'human_patterns_found': []
        }
        
        for pattern in self.patterns.values():
            regex = pattern.compile_regex()
            matches = regex.findall(text)
            
            if len(matches) >= pattern.threshold:
                match_info = {
                    'pattern_id': pattern.id,
                    'pattern_name': pattern.name,
                    'category': pattern.category.value,
                    'matches': matches[:5],  # Limit to first 5 matches
                    'count': len(matches),
                    'weight': pattern.weight,
                    'reliability': pattern.reliability
                }
                
                results['matches'].append(match_info)
                
                # Calculate scores
                score_contribution = pattern.weight * pattern.reliability * min(len(matches) / pattern.threshold, 2.0)
                
                if pattern.type == PatternType.AI_INDICATOR:
                    results['ai_score'] += score_contribution
                    results['ai_patterns_found'].append(pattern.name)
                elif pattern.type == PatternType.HUMAN_INDICATOR:
                    results['human_score'] += abs(score_contribution)
                    results['human_patterns_found'].append(pattern.name)
        
        # Normalize scores
        total_score = results['ai_score'] + results['human_score']
        if total_score > 0:
            results['ai_probability'] = results['ai_score'] / total_score
        else:
            results['ai_probability'] = 0.5
        
        # Determine prediction
        if results['ai_probability'] > 0.7:
            results['prediction'] = 'ai'
            results['confidence'] = 'high'
        elif results['ai_probability'] > 0.5:
            results['prediction'] = 'ai'
            results['confidence'] = 'medium'
        elif results['ai_probability'] < 0.3:
            results['prediction'] = 'human'
            results['confidence'] = 'high'
        elif results['ai_probability'] < 0.5:
            results['prediction'] = 'human'
            results['confidence'] = 'medium'
        else:
            results['prediction'] = 'uncertain'
            results['confidence'] = 'low'
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            'version': self.VERSION,
            'total_patterns': len(self.patterns),
            'ai_indicators': len(self.get_patterns_by_type(PatternType.AI_INDICATOR)),
            'human_indicators': len(self.get_patterns_by_type(PatternType.HUMAN_INDICATOR)),
            'categories': {cat.value: len(self.get_patterns_by_category(cat)) 
                         for cat in PatternCategory},
            'average_weight': sum(abs(p.weight) for p in self.patterns.values()) / len(self.patterns),
            'average_reliability': sum(p.reliability for p in self.patterns.values()) / len(self.patterns)
        }


# Singleton instance
_registry_instance = None

def get_registry() -> PatternRegistry:
    """Get the singleton pattern registry instance"""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = PatternRegistry()
    return _registry_instance


# Export patterns on module load
def export_all_formats(base_path: str = "."):
    """Export patterns to all formats"""
    registry = get_registry()
    
    # Validate first
    issues = registry.validate_patterns()
    if issues['errors']:
        raise ValueError(f"Pattern validation failed: {issues['errors']}")
    
    # Export to different formats
    registry.export_to_json(os.path.join(base_path, "patterns.json"))
    registry.export_to_javascript(os.path.join(base_path, "patterns.js"))
    registry.export_to_python(os.path.join(base_path, "patterns_generated.py"))
    
    print(f"Exported {len(registry.patterns)} patterns to all formats")


if __name__ == "__main__":
    # Demo usage
    registry = get_registry()
    
    # Validate patterns
    issues = registry.validate_patterns()
    print(f"Validation issues: {issues}")
    
    # Get statistics
    stats = registry.get_statistics()
    print(f"Registry statistics: {stats}")
    
    # Test pattern matching
    test_text = "It's important to note that while AI has advantages, on the other hand, there are disadvantages."
    results = registry.apply_patterns(test_text)
    print(f"Analysis results: AI probability = {results['ai_probability']:.2f}, Prediction = {results['prediction']}")
    
    # Export patterns
    export_all_formats()