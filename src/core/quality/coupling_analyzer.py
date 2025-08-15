"""
Coupling Analyzer - Measures and reduces coupling between modules
Analyzes import dependencies and provides refactoring suggestions for loose coupling
"""

import ast
import os
import re
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import networkx as nx
import json


@dataclass
class CouplingMetric:
    """Coupling metric for a module"""
    module_name: str
    file_path: str
    imports: List[str]
    imported_by: List[str]
    coupling_in: int  # Afferent coupling (modules depending on this)
    coupling_out: int  # Efferent coupling (modules this depends on)
    instability: float  # Ce / (Ca + Ce)
    suggestions: List[str]


class CouplingAnalyzer:
    """Analyzes coupling between modules in the codebase"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.modules = {}
        self.import_graph = nx.DiGraph()
        self.module_imports = {}
        
    def analyze_module_coupling(self) -> List[CouplingMetric]:
        """Analyze coupling for all modules"""
        self._scan_modules()
        self._build_import_graph()
        return self._calculate_coupling_metrics()
    
    def _scan_modules(self):
        """Scan all Python modules in the codebase"""
        for root, dirs, files in os.walk(self.base_path):
            # Skip certain directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules']]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('test_'):
                    file_path = Path(root) / file
                    module_name = self._get_module_name(file_path)
                    
                    imports = self._extract_imports(file_path)
                    self.modules[module_name] = {
                        'file_path': str(file_path),
                        'imports': imports
                    }
                    self.module_imports[module_name] = imports
    
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
    
    def _extract_imports(self, file_path: Path) -> List[str]:
        """Extract imports from a Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            # Filter to only include project modules
            project_imports = []
            for imp in imports:
                if self._is_project_module(imp):
                    project_imports.append(imp)
            
            return project_imports
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return []
    
    def _is_project_module(self, module_name: str) -> bool:
        """Check if module is part of the project"""
        # Consider modules starting with 'src' or local relative imports
        return (
            module_name.startswith('src.') or
            module_name.startswith('core.') or
            module_name.startswith('data.') or
            module_name.startswith('training.') or
            module_name.startswith('integrations.') or
            module_name.startswith('api.') or
            module_name.startswith('utils.') or
            '.' not in module_name  # Local modules without dots
        )
    
    def _build_import_graph(self):
        """Build directed graph of module dependencies"""
        for module, data in self.modules.items():
            self.import_graph.add_node(module)
            
            for imported_module in data['imports']:
                # Find closest matching module in our codebase
                matching_module = self._find_matching_module(imported_module)
                if matching_module and matching_module != module:
                    self.import_graph.add_edge(module, matching_module)
    
    def _find_matching_module(self, import_name: str) -> str:
        """Find matching module in our codebase"""
        # Direct match
        if import_name in self.modules:
            return import_name
        
        # Partial match - find the best match
        candidates = []
        for module in self.modules:
            if module.endswith(import_name) or import_name.endswith(module):
                candidates.append(module)
        
        # Return the most specific match
        if candidates:
            return min(candidates, key=len)
        
        return None
    
    def _calculate_coupling_metrics(self) -> List[CouplingMetric]:
        """Calculate coupling metrics for all modules"""
        metrics = []
        
        for module in self.modules:
            # Efferent coupling (outgoing dependencies)
            coupling_out = len(list(self.import_graph.successors(module)))
            
            # Afferent coupling (incoming dependencies)
            coupling_in = len(list(self.import_graph.predecessors(module)))
            
            # Instability metric: Ce / (Ca + Ce)
            total_coupling = coupling_in + coupling_out
            instability = coupling_out / total_coupling if total_coupling > 0 else 0
            
            imports = self.modules[module]['imports']
            imported_by = list(self.import_graph.predecessors(module))
            
            suggestions = self._generate_coupling_suggestions(
                module, coupling_in, coupling_out, instability
            )
            
            metric = CouplingMetric(
                module_name=module,
                file_path=self.modules[module]['file_path'],
                imports=imports,
                imported_by=imported_by,
                coupling_in=coupling_in,
                coupling_out=coupling_out,
                instability=instability,
                suggestions=suggestions
            )
            
            metrics.append(metric)
        
        return sorted(metrics, key=lambda x: x.instability, reverse=True)
    
    def _generate_coupling_suggestions(self, module: str, ca: int, ce: int, instability: float) -> List[str]:
        """Generate suggestions for reducing coupling"""
        suggestions = []
        
        if instability > 0.7:
            suggestions.append("High instability - consider stabilizing with interfaces")
            suggestions.append("Extract stable abstractions from concrete implementations")
        
        if ce > 5:
            suggestions.append("High efferent coupling - break into smaller modules")
            suggestions.append("Apply Dependency Inversion Principle")
            suggestions.append("Use Facade pattern to reduce dependencies")
        
        if ca > 8:
            suggestions.append("High afferent coupling - module is too central")
            suggestions.append("Consider splitting into multiple focused modules")
            suggestions.append("Extract common functionality to shared utilities")
        
        if ce > 3 and ca > 3:
            suggestions.append("High bidirectional coupling - introduce mediator pattern")
            suggestions.append("Use event-driven architecture to decouple components")
        
        if instability < 0.2 and ca > 2:
            suggestions.append("Very stable module - ensure it's truly stable")
            suggestions.append("Consider making this an abstract interface")
        
        return suggestions
    
    def calculate_overall_coupling(self, metrics: List[CouplingMetric]) -> Dict[str, float]:
        """Calculate overall coupling statistics"""
        if not metrics:
            return {}
        
        total_modules = len(metrics)
        total_dependencies = sum(m.coupling_out for m in metrics)
        
        # Average coupling metrics
        avg_coupling_in = sum(m.coupling_in for m in metrics) / total_modules
        avg_coupling_out = sum(m.coupling_out for m in metrics) / total_modules
        avg_instability = sum(m.instability for m in metrics) / total_modules
        
        # Coupling distribution
        low_coupling = len([m for m in metrics if m.coupling_out <= 2])
        medium_coupling = len([m for m in metrics if 2 < m.coupling_out <= 5])
        high_coupling = len([m for m in metrics if m.coupling_out > 5])
        
        return {
            'total_modules': total_modules,
            'total_dependencies': total_dependencies,
            'avg_coupling_in': avg_coupling_in,
            'avg_coupling_out': avg_coupling_out,
            'avg_instability': avg_instability,
            'low_coupling_modules': low_coupling,
            'medium_coupling_modules': medium_coupling,
            'high_coupling_modules': high_coupling,
            'coupling_score': avg_coupling_out  # Lower is better
        }
    
    def generate_coupling_report(self, metrics: List[CouplingMetric]) -> str:
        """Generate coupling analysis report"""
        overall = self.calculate_overall_coupling(metrics)
        
        if not metrics:
            return "No modules found for coupling analysis."
        
        report = "Module Coupling Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        # Overall statistics
        report += f"Overall Coupling Statistics:\n"
        report += f"  Total modules: {overall['total_modules']}\n"
        report += f"  Total dependencies: {overall['total_dependencies']}\n"
        report += f"  Average coupling out: {overall['avg_coupling_out']:.2f}\n"
        report += f"  Average coupling in: {overall['avg_coupling_in']:.2f}\n"
        report += f"  Average instability: {overall['avg_instability']:.2f}\n\n"
        
        # Coupling distribution
        report += f"Coupling Distribution:\n"
        report += f"  Low coupling (0-2): {overall['low_coupling_modules']} modules\n"
        report += f"  Medium coupling (3-5): {overall['medium_coupling_modules']} modules\n"
        report += f"  High coupling (6+): {overall['high_coupling_modules']} modules\n\n"
        
        # Assessment
        coupling_score = overall['coupling_score']
        if coupling_score < 3:
            report += "Assessment: EXCELLENT - Low coupling achieved\n\n"
        elif coupling_score < 5:
            report += "Assessment: GOOD - Moderate coupling, some improvements possible\n\n"
        else:
            report += "Assessment: NEEDS IMPROVEMENT - High coupling detected\n\n"
        
        # High coupling modules
        high_coupling = [m for m in metrics if m.coupling_out > 5 or m.instability > 0.8]
        if high_coupling:
            report += f"High Coupling Modules ({len(high_coupling)}):\n"
            for i, metric in enumerate(high_coupling[:10], 1):  # Top 10
                report += f"{i}. {metric.module_name}\n"
                report += f"   Coupling: {metric.coupling_out} out, {metric.coupling_in} in\n"
                report += f"   Instability: {metric.instability:.2f}\n"
                report += f"   Suggestions:\n"
                for suggestion in metric.suggestions[:3]:
                    report += f"   - {suggestion}\n"
                report += "\n"
        
        report += "Recommended Actions:\n"
        if coupling_score > 5:
            report += "1. CRITICAL: Reduce efferent coupling in high-dependency modules\n"
            report += "2. Apply Dependency Inversion Principle\n"
            report += "3. Extract interfaces and abstract classes\n"
            report += "4. Use Facade pattern for complex subsystems\n"
        elif coupling_score > 3:
            report += "1. Consider breaking large modules into smaller ones\n"
            report += "2. Extract common utilities to reduce duplication\n"
            report += "3. Review high-instability modules for stability\n"
        else:
            report += "1. Maintain current low coupling\n"
            report += "2. Monitor new dependencies carefully\n"
            report += "3. Consider extracting reusable components\n"
        
        return report
    
    def export_dependency_graph(self, output_file: str = "dependency_graph.json"):
        """Export dependency graph for visualization"""
        graph_data = {
            'nodes': [],
            'edges': []
        }
        
        for module in self.modules:
            coupling_out = len(list(self.import_graph.successors(module)))
            coupling_in = len(list(self.import_graph.predecessors(module)))
            
            graph_data['nodes'].append({
                'id': module,
                'label': module.split('.')[-1],  # Short name
                'coupling_out': coupling_out,
                'coupling_in': coupling_in,
                'size': coupling_in + coupling_out + 1,
                'group': module.split('.')[0] if '.' in module else 'root'
            })
        
        for edge in self.import_graph.edges():
            graph_data['edges'].append({
                'source': edge[0],
                'target': edge[1]
            })
        
        with open(output_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        return graph_data


def analyze_module_coupling():
    """Main function to analyze module coupling"""
    print("Analyzing module coupling...")
    
    base_path = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    analyzer = CouplingAnalyzer(base_path)
    
    # Analyze coupling
    metrics = analyzer.analyze_module_coupling()
    
    # Generate report
    report = analyzer.generate_coupling_report(metrics)
    print(report)
    
    # Save report
    report_path = os.path.join(base_path, "docs", "coupling_report.md")
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")
    
    # Export dependency graph
    graph_file = os.path.join(base_path, "docs", "dependency_graph.json")
    analyzer.export_dependency_graph(graph_file)
    print(f"Dependency graph saved to: {graph_file}")
    
    # Check if target met
    overall = analyzer.calculate_overall_coupling(metrics)
    coupling_score = overall.get('coupling_score', 10)  # Default high if no data
    
    target_met = coupling_score < 3.0  # Target: average coupling < 3
    
    if target_met:
        print("SUCCESS: Low coupling target achieved!")
    else:
        print(f"Target not met: Average coupling {coupling_score:.2f} > 3.0")
    
    return target_met


if __name__ == "__main__":
    success = analyze_module_coupling()
    if success:
        print("Low coupling target achieved!")
    else:
        print("Further coupling reduction needed.")