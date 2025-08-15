"""
Cohesion Analyzer - Measures and improves cohesion within modules
Analyzes method relationships and data dependencies to calculate cohesion metrics
"""

import ast
import os
import re
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import math


@dataclass
class CohesionMetric:
    """Cohesion metric for a module"""
    module_name: str
    file_path: str
    classes: List[str]
    functions: List[str]
    cohesion_score: float
    method_interactions: int
    shared_data: int
    suggestions: List[str]


class CohesionAnalyzer:
    """Analyzes cohesion within modules"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        
    def analyze_module_cohesion(self) -> List[CohesionMetric]:
        """Analyze cohesion for all modules"""
        metrics = []
        
        for root, dirs, files in os.walk(self.base_path):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    file_path = Path(root) / file
                    module_name = self._get_module_name(file_path)
                    
                    try:
                        cohesion_metric = self._analyze_file_cohesion(file_path, module_name)
                        if cohesion_metric:
                            metrics.append(cohesion_metric)
                    except Exception as e:
                        print(f"Error analyzing {file_path}: {e}")
        
        return sorted(metrics, key=lambda x: x.cohesion_score)
    
    def _get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name"""
        relative_path = file_path.relative_to(self.base_path)
        parts = list(relative_path.parts)
        
        # Remove .py extension
        if parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]
        
        # Remove __init__ files
        if parts[-1] == '__init__':
            parts = parts[:-1]
        
        return '.'.join(parts)
    
    def _analyze_file_cohesion(self, file_path: Path, module_name: str) -> Optional[CohesionMetric]:
        """Analyze cohesion for a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Extract module elements
            classes = []
            functions = []
            class_methods = defaultdict(list)
            function_calls = defaultdict(set)
            shared_variables = set()
            
            # First pass: collect classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                    
                    # Extract methods from class
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            class_methods[node.name].append(item.name)
                
                elif isinstance(node, ast.FunctionDef) and not self._is_nested_function(node, tree):
                    functions.append(node.name)
            
            # Second pass: analyze interactions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Analyze function calls within this function
                    calls = self._extract_function_calls(node)
                    function_calls[node.name].update(calls)
                    
                    # Analyze shared variables
                    variables = self._extract_variables(node)
                    shared_variables.update(variables)
            
            # Calculate cohesion metrics
            cohesion_score = self._calculate_cohesion(
                classes, functions, class_methods, function_calls, shared_variables
            )
            
            method_interactions = sum(len(calls) for calls in function_calls.values())
            shared_data = len(shared_variables)
            
            suggestions = self._generate_cohesion_suggestions(
                cohesion_score, classes, functions, method_interactions, shared_data
            )
            
            return CohesionMetric(
                module_name=module_name,
                file_path=str(file_path),
                classes=classes,
                functions=functions,
                cohesion_score=cohesion_score,
                method_interactions=method_interactions,
                shared_data=shared_data,
                suggestions=suggestions
            )
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    def _is_nested_function(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if function is nested inside another function or class"""
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node != func_node:
                for child in ast.walk(node):
                    if child == func_node:
                        return True
        return False
    
    def _extract_function_calls(self, func_node: ast.FunctionDef) -> Set[str]:
        """Extract function calls from a function"""
        calls = set()
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        calls.add(f"{node.func.value.id}.{node.func.attr}")
        
        return calls
    
    def _extract_variables(self, func_node: ast.FunctionDef) -> Set[str]:
        """Extract variables used in a function"""
        variables = set()
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Name):
                variables.add(node.id)
        
        return variables
    
    def _calculate_cohesion(self, classes: List[str], functions: List[str], 
                          class_methods: Dict[str, List[str]], 
                          function_calls: Dict[str, Set[str]], 
                          shared_variables: Set[str]) -> float:
        """Calculate cohesion score using LCOM (Lack of Cohesion of Methods) and other metrics"""
        
        if not classes and not functions:
            return 1.0  # Empty module is perfectly cohesive
        
        total_cohesion = 0.0
        cohesion_factors = 0
        
        # Class cohesion (LCOM metric)
        for class_name in classes:
            methods = class_methods.get(class_name, [])
            if len(methods) > 1:
                class_cohesion = self._calculate_class_cohesion(methods, function_calls)
                total_cohesion += class_cohesion
                cohesion_factors += 1
        
        # Function cohesion (based on shared data and calls)
        if functions:
            function_cohesion = self._calculate_function_cohesion(functions, function_calls, shared_variables)
            total_cohesion += function_cohesion
            cohesion_factors += 1
        
        # Module organization cohesion
        organization_cohesion = self._calculate_organization_cohesion(classes, functions)
        total_cohesion += organization_cohesion
        cohesion_factors += 1
        
        return total_cohesion / cohesion_factors if cohesion_factors > 0 else 0.0
    
    def _calculate_class_cohesion(self, methods: List[str], function_calls: Dict[str, Set[str]]) -> float:
        """Calculate cohesion within a class using LCOM"""
        if len(methods) <= 1:
            return 1.0
        
        # Count method pairs that share variables or call each other
        shared_pairs = 0
        total_pairs = 0
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                total_pairs += 1
                
                # Check if methods call each other or share variables
                calls1 = function_calls.get(method1, set())
                calls2 = function_calls.get(method2, set())
                
                if method2 in calls1 or method1 in calls2:
                    shared_pairs += 1
                elif calls1.intersection(calls2):  # Share common function calls
                    shared_pairs += 1
        
        if total_pairs == 0:
            return 1.0
        
        # Cohesion = shared pairs / total pairs
        return shared_pairs / total_pairs
    
    def _calculate_function_cohesion(self, functions: List[str], 
                                   function_calls: Dict[str, Set[str]], 
                                   shared_variables: Set[str]) -> float:
        """Calculate cohesion among module functions"""
        if len(functions) <= 1:
            return 1.0
        
        # Functions that call other functions in the module
        internal_calls = 0
        total_functions = len(functions)
        
        for func in functions:
            calls = function_calls.get(func, set())
            for call in calls:
                if call in functions:
                    internal_calls += 1
        
        # Normalize by possible internal calls
        max_possible_calls = total_functions * (total_functions - 1)
        call_cohesion = internal_calls / max_possible_calls if max_possible_calls > 0 else 0
        
        # Data cohesion based on shared variables
        data_cohesion = min(len(shared_variables) / (total_functions * 2), 1.0)
        
        return (call_cohesion + data_cohesion) / 2
    
    def _calculate_organization_cohesion(self, classes: List[str], functions: List[str]) -> float:
        """Calculate organizational cohesion (single responsibility)"""
        total_elements = len(classes) + len(functions)
        
        if total_elements == 0:
            return 1.0
        
        # Prefer modules with focused responsibility
        if total_elements <= 3:
            return 1.0  # Small, focused module
        elif total_elements <= 7:
            return 0.8  # Medium module
        elif total_elements <= 15:
            return 0.6  # Large module
        else:
            return 0.4  # Very large module - likely low cohesion
    
    def _generate_cohesion_suggestions(self, cohesion_score: float, classes: List[str], 
                                     functions: List[str], method_interactions: int, 
                                     shared_data: int) -> List[str]:
        """Generate suggestions for improving cohesion"""
        suggestions = []
        
        total_elements = len(classes) + len(functions)
        
        if cohesion_score < 0.3:
            suggestions.append("CRITICAL: Very low cohesion - major refactoring needed")
            suggestions.append("Split module into multiple focused modules")
            suggestions.append("Group related functions into classes")
        
        elif cohesion_score < 0.5:
            suggestions.append("Low cohesion - consider refactoring")
            suggestions.append("Extract related methods into separate classes")
            suggestions.append("Remove unrelated functionality")
        
        elif cohesion_score < 0.7:
            suggestions.append("Moderate cohesion - some improvements possible")
            suggestions.append("Review method responsibilities")
            suggestions.append("Consider extracting utility functions")
        
        # Specific suggestions based on structure
        if total_elements > 20:
            suggestions.append("Module too large - split into smaller modules")
        
        if len(classes) > 5 and len(functions) > 10:
            suggestions.append("Mixed classes and functions - consider separating")
        
        if method_interactions == 0 and total_elements > 3:
            suggestions.append("No method interactions - check if module is cohesive")
        
        if shared_data > total_elements * 3:
            suggestions.append("Too many shared variables - encapsulate in classes")
        
        if not suggestions:
            suggestions.append("Good cohesion - maintain current structure")
        
        return suggestions
    
    def calculate_overall_cohesion(self, metrics: List[CohesionMetric]) -> Dict[str, Any]:
        """Calculate overall cohesion statistics"""
        if not metrics:
            return {}
        
        total_modules = len(metrics)
        cohesion_scores = [m.cohesion_score for m in metrics]
        
        avg_cohesion = sum(cohesion_scores) / total_modules
        min_cohesion = min(cohesion_scores)
        max_cohesion = max(cohesion_scores)
        
        # Cohesion distribution
        high_cohesion = len([m for m in metrics if m.cohesion_score >= 0.7])
        medium_cohesion = len([m for m in metrics if 0.4 <= m.cohesion_score < 0.7])
        low_cohesion = len([m for m in metrics if m.cohesion_score < 0.4])
        
        return {
            'total_modules': total_modules,
            'avg_cohesion': avg_cohesion,
            'min_cohesion': min_cohesion,
            'max_cohesion': max_cohesion,
            'high_cohesion_modules': high_cohesion,
            'medium_cohesion_modules': medium_cohesion,
            'low_cohesion_modules': low_cohesion,
            'target_met': avg_cohesion >= 0.7
        }
    
    def generate_cohesion_report(self, metrics: List[CohesionMetric]) -> str:
        """Generate cohesion analysis report"""
        overall = self.calculate_overall_cohesion(metrics)
        
        if not metrics:
            return "No modules found for cohesion analysis."
        
        report = "Module Cohesion Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        # Overall statistics
        report += f"Overall Cohesion Statistics:\n"
        report += f"  Total modules: {overall['total_modules']}\n"
        report += f"  Average cohesion: {overall['avg_cohesion']:.3f}\n"
        report += f"  Minimum cohesion: {overall['min_cohesion']:.3f}\n"
        report += f"  Maximum cohesion: {overall['max_cohesion']:.3f}\n\n"
        
        # Cohesion distribution
        report += f"Cohesion Distribution:\n"
        report += f"  High cohesion (>=0.7): {overall['high_cohesion_modules']} modules\n"
        report += f"  Medium cohesion (0.4-0.7): {overall['medium_cohesion_modules']} modules\n"
        report += f"  Low cohesion (<0.4): {overall['low_cohesion_modules']} modules\n\n"
        
        # Assessment
        if overall['target_met']:
            report += "Assessment: SUCCESS - High cohesion target achieved!\n\n"
        elif overall['avg_cohesion'] >= 0.6:
            report += "Assessment: GOOD - Approaching target cohesion\n\n"
        elif overall['avg_cohesion'] >= 0.5:
            report += "Assessment: MODERATE - Some cohesion improvements needed\n\n"
        else:
            report += "Assessment: NEEDS IMPROVEMENT - Low cohesion detected\n\n"
        
        # Low cohesion modules
        low_cohesion = [m for m in metrics if m.cohesion_score < 0.5]
        if low_cohesion:
            report += f"Low Cohesion Modules ({len(low_cohesion)}):\n"
            for i, metric in enumerate(low_cohesion[:10], 1):  # Top 10
                report += f"{i}. {metric.module_name} (cohesion: {metric.cohesion_score:.3f})\n"
                report += f"   Classes: {len(metric.classes)}, Functions: {len(metric.functions)}\n"
                report += f"   Suggestions:\n"
                for suggestion in metric.suggestions[:3]:
                    report += f"   - {suggestion}\n"
                report += "\n"
        
        # High cohesion modules (examples)
        high_cohesion = [m for m in metrics if m.cohesion_score >= 0.8]
        if high_cohesion:
            report += f"High Cohesion Examples ({len(high_cohesion)}):\n"
            for i, metric in enumerate(high_cohesion[:5], 1):  # Top 5
                report += f"{i}. {metric.module_name} (cohesion: {metric.cohesion_score:.3f})\n"
        
        if high_cohesion:
            report += "\n"
        
        report += "Recommended Actions:\n"
        if not overall['target_met']:
            if overall['avg_cohesion'] < 0.5:
                report += "1. CRITICAL: Major refactoring needed for low cohesion modules\n"
                report += "2. Split large modules into focused, single-purpose modules\n"
                report += "3. Group related functions into classes\n"
                report += "4. Remove unrelated functionality from modules\n"
            else:
                report += "1. Focus on modules with cohesion < 0.6\n"
                report += "2. Review method responsibilities and interactions\n"
                report += "3. Extract utility functions to appropriate modules\n"
                report += "4. Consider combining small related modules\n"
        else:
            report += "1. Maintain current high cohesion\n"
            report += "2. Monitor new additions for cohesion impact\n"
            report += "3. Continue applying single responsibility principle\n"
        
        return report


def analyze_module_cohesion():
    """Main function to analyze module cohesion"""
    print("Analyzing module cohesion...")
    
    base_path = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    analyzer = CohesionAnalyzer(base_path)
    
    # Analyze cohesion
    metrics = analyzer.analyze_module_cohesion()
    
    # Generate report
    report = analyzer.generate_cohesion_report(metrics)
    print(report)
    
    # Save report
    report_path = os.path.join(base_path, "docs", "cohesion_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")
    
    # Check if target met
    overall = analyzer.calculate_overall_cohesion(metrics)
    target_met = overall.get('target_met', False)
    
    if target_met:
        print("SUCCESS: High cohesion target (>0.7) achieved!")
    else:
        avg_cohesion = overall.get('avg_cohesion', 0)
        print(f"Target not met: Average cohesion {avg_cohesion:.3f} < 0.7")
    
    return target_met


if __name__ == "__main__":
    success = analyze_module_cohesion()
    if success:
        print("High cohesion target achieved!")
    else:
        print("Further cohesion improvements needed.")