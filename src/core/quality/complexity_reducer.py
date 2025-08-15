"""
Complexity Reducer - Analyzes and reduces cyclomatic complexity
Identifies functions with high complexity and provides refactoring patterns
"""

import ast
import os
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class ComplexityMetric:
    """Cyclomatic complexity metric for a function"""
    function_name: str
    file_path: str
    line_number: int
    complexity: int
    suggestions: List[str]
    refactored_code: Optional[str] = None


class CyclomaticComplexityAnalyzer(ast.NodeVisitor):
    """AST visitor to calculate cyclomatic complexity"""
    
    def __init__(self):
        self.complexity = 1  # Base complexity
        self.current_function = None
        self.functions = {}
        
    def visit_FunctionDef(self, node):
        """Visit function definition"""
        old_complexity = self.complexity
        old_function = self.current_function
        
        self.complexity = 1  # Reset for new function
        self.current_function = node.name
        
        # Visit function body
        self.generic_visit(node)
        
        # Store complexity
        self.functions[node.name] = {
            'complexity': self.complexity,
            'line_number': node.lineno,
            'node': node
        }
        
        # Restore previous state
        self.complexity = old_complexity
        self.current_function = old_function
    
    def _increment_complexity(self, node):
        """Increment complexity for decision points"""
        if self.current_function:
            self.complexity += 1
    
    def visit_If(self, node):
        """Visit if statement"""
        self._increment_complexity(node)
        self.generic_visit(node)
    
    def visit_While(self, node):
        """Visit while loop"""
        self._increment_complexity(node)
        self.generic_visit(node)
    
    def visit_For(self, node):
        """Visit for loop"""
        self._increment_complexity(node)
        self.generic_visit(node)
    
    def visit_ExceptHandler(self, node):
        """Visit exception handler"""
        self._increment_complexity(node)
        self.generic_visit(node)
    
    def visit_With(self, node):
        """Visit with statement"""
        self._increment_complexity(node)
        self.generic_visit(node)
    
    def visit_BoolOp(self, node):
        """Visit boolean operation (and/or)"""
        # Each additional condition adds complexity
        self.complexity += len(node.values) - 1
        self.generic_visit(node)


class ComplexityReducer:
    """Main complexity reduction engine"""
    
    def __init__(self, max_complexity: int = 10):
        self.max_complexity = max_complexity
        self.high_complexity_functions = []
        
    def analyze_file(self, file_path: str) -> List[ComplexityMetric]:
        """Analyze complexity of all functions in a file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            analyzer = CyclomaticComplexityAnalyzer()
            analyzer.visit(tree)
            
            metrics = []
            for func_name, data in analyzer.functions.items():
                if data['complexity'] > self.max_complexity:
                    suggestions = self._generate_suggestions(data['node'])
                    metric = ComplexityMetric(
                        function_name=func_name,
                        file_path=file_path,
                        line_number=data['line_number'],
                        complexity=data['complexity'],
                        suggestions=suggestions
                    )
                    metrics.append(metric)
            
            return metrics
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return []
    
    def _generate_suggestions(self, node: ast.FunctionDef) -> List[str]:
        """Generate refactoring suggestions for high complexity function"""
        suggestions = []
        
        # Analyze function structure
        if_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.If))
        for_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.For))
        while_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.While))
        
        if if_count > 5:
            suggestions.append("Extract conditional logic into separate methods")
            suggestions.append("Consider using strategy pattern for complex conditionals")
        
        if for_count > 2:
            suggestions.append("Extract loop logic into helper functions")
        
        if len(node.body) > 20:
            suggestions.append("Break function into smaller functions")
            suggestions.append("Apply single responsibility principle")
        
        # Check for nested structures
        nested_depth = self._calculate_nesting_depth(node)
        if nested_depth > 3:
            suggestions.append("Reduce nesting depth using early returns")
            suggestions.append("Extract nested logic into separate methods")
        
        # Check for repeated patterns
        if self._has_repeated_patterns(node):
            suggestions.append("Extract common patterns into utility functions")
        
        return suggestions
    
    def _calculate_nesting_depth(self, node: ast.FunctionDef) -> int:
        """Calculate maximum nesting depth"""
        max_depth = 0
        
        def calc_depth(n, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            for child in ast.iter_child_nodes(n):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.With)):
                    calc_depth(child, current_depth + 1)
                else:
                    calc_depth(child, current_depth)
        
        calc_depth(node)
        return max_depth
    
    def _has_repeated_patterns(self, node: ast.FunctionDef) -> bool:
        """Check for repeated code patterns"""
        # Simple heuristic: look for similar variable assignments
        assignments = []
        for n in ast.walk(node):
            if isinstance(n, ast.Assign):
                assignments.append(ast.unparse(n) if hasattr(ast, 'unparse') else str(n))
        
        # If more than 30% of assignments are similar, suggest extraction
        unique_patterns = set(assignments)
        return len(assignments) > 5 and len(unique_patterns) / len(assignments) < 0.7
    
    def analyze_codebase(self, base_path: str) -> List[ComplexityMetric]:
        """Analyze entire codebase for high complexity functions"""
        all_metrics = []
        
        for root, dirs, files in os.walk(base_path):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    metrics = self.analyze_file(file_path)
                    all_metrics.extend(metrics)
        
        return sorted(all_metrics, key=lambda x: x.complexity, reverse=True)
    
    def refactor_statistical_detection(self) -> str:
        """Refactor the statistical_detection method to reduce complexity"""
        return '''def statistical_detection(self, text: str) -> DetectionResult:
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
    return {'score': 0, 'indicator': None}'''
    
    def refactor_extract_features(self) -> str:
        """Refactor extract_statistical_features to reduce complexity"""
        return '''def extract_statistical_features(self, text: str) -> Dict[str, float]:
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
    
    punctuation_count = sum(1 for char in text if char in '.,;:!?()[]{}"\'-')
    
    return {
        'total_words': len(words),
        'unique_words': len(set(words)),
        'lexical_diversity': len(set(words)) / len(words),
        'punctuation_ratio': punctuation_count / len(text)
    }

def _calculate_structure_features(self, text: str) -> Dict[str, float]:
    """Calculate structural features"""
    paragraphs = text.split('\\n\\n')
    structure_score = 1.0 if len(paragraphs) > 1 else 0.3
    
    return {
        'paragraph_structure_score': structure_score
    }'''
    
    def generate_report(self, metrics: List[ComplexityMetric]) -> str:
        """Generate complexity analysis report"""
        if not metrics:
            return "All functions have complexity <= 10. No refactoring needed."
        
        report = f"Cyclomatic Complexity Analysis Report\n"
        report += f"=" * 50 + "\n\n"
        report += f"Found {len(metrics)} functions with complexity > {self.max_complexity}\n\n"
        
        for i, metric in enumerate(metrics, 1):
            report += f"{i}. {metric.function_name} (complexity: {metric.complexity})\n"
            report += f"   File: {metric.file_path}:{metric.line_number}\n"
            report += f"   Suggestions:\n"
            for suggestion in metric.suggestions:
                report += f"   - {suggestion}\n"
            report += "\n"
        
        report += "Refactoring Priority:\n"
        report += "1. Functions with complexity > 15 (critical)\n"
        report += "2. Functions with complexity 11-15 (high)\n"
        report += "3. Functions with complexity 10-11 (medium)\n\n"
        
        report += "Recommended Patterns:\n"
        report += "- Strategy Pattern for complex conditionals\n"
        report += "- Extract Method for long functions\n"
        report += "- Guard Clauses for early returns\n"
        report += "- Chain of Responsibility for validation\n"
        
        return report


def analyze_and_reduce_complexity():
    """Main function to analyze and reduce complexity"""
    reducer = ComplexityReducer(max_complexity=10)
    
    # Analyze current codebase
    print("Analyzing codebase for high complexity functions...")
    base_path = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    metrics = reducer.analyze_codebase(base_path)
    
    # Generate report
    report = reducer.generate_report(metrics)
    print(report)
    
    # Save report
    report_path = os.path.join(base_path, "docs", "complexity_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")
    
    # Generate refactored examples
    if metrics:
        print("\nGenerating refactored examples...")
        examples_path = os.path.join(base_path, "docs", "complexity_refactoring_examples.py")
        
        with open(examples_path, 'w') as f:
            f.write("# Complexity Refactoring Examples\n\n")
            f.write(reducer.refactor_statistical_detection())
            f.write("\n\n")
            f.write(reducer.refactor_extract_features())
        
        print(f"Refactoring examples saved to: {examples_path}")
    
    return len(metrics) == 0  # True if all functions have low complexity


if __name__ == "__main__":
    success = analyze_and_reduce_complexity()
    if success:
        print("All functions have acceptable complexity!")
    else:
        print("Some functions need refactoring to reduce complexity.")