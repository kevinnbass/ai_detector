Module Coupling Analysis Report
==================================================

Overall Coupling Statistics:
  Total modules: 111
  Total dependencies: 197
  Average coupling out: 1.77
  Average coupling in: 1.77
  Average instability: 0.50

Coupling Distribution:
  Low coupling (0-2): 84 modules
  Medium coupling (3-5): 21 modules
  High coupling (6+): 6 modules

Assessment: EXCELLENT - Low coupling achieved

High Coupling Modules (35):
1. validate_throughput
   Coupling: 1 out, 0 in
   Instability: 1.00
   Suggestions:
   - High instability - consider stabilizing with interfaces
   - Extract stable abstractions from concrete implementations

2. docs.api.openapi_spec
   Coupling: 1 out, 0 in
   Instability: 1.00
   Suggestions:
   - High instability - consider stabilizing with interfaces
   - Extract stable abstractions from concrete implementations

3. scripts.demo_optimization
   Coupling: 1 out, 0 in
   Instability: 1.00
   Suggestions:
   - High instability - consider stabilizing with interfaces
   - Extract stable abstractions from concrete implementations

4. scripts.run_optimization
   Coupling: 1 out, 0 in
   Instability: 1.00
   Suggestions:
   - High instability - consider stabilizing with interfaces
   - Extract stable abstractions from concrete implementations

5. src.api.monitoring_routes
   Coupling: 1 out, 0 in
   Instability: 1.00
   Suggestions:
   - High instability - consider stabilizing with interfaces
   - Extract stable abstractions from concrete implementations

6. src.api
   Coupling: 2 out, 0 in
   Instability: 1.00
   Suggestions:
   - High instability - consider stabilizing with interfaces
   - Extract stable abstractions from concrete implementations

7. src.api.performance.connection_optimizer
   Coupling: 1 out, 0 in
   Instability: 1.00
   Suggestions:
   - High instability - consider stabilizing with interfaces
   - Extract stable abstractions from concrete implementations

8. src.api.rest.optimized_app
   Coupling: 3 out, 0 in
   Instability: 1.00
   Suggestions:
   - High instability - consider stabilizing with interfaces
   - Extract stable abstractions from concrete implementations

9. src.api.websocket.routes
   Coupling: 2 out, 0 in
   Instability: 1.00
   Suggestions:
   - High instability - consider stabilizing with interfaces
   - Extract stable abstractions from concrete implementations

10. src.core.abstractions
   Coupling: 3 out, 0 in
   Instability: 1.00
   Suggestions:
   - High instability - consider stabilizing with interfaces
   - Extract stable abstractions from concrete implementations

Recommended Actions:
1. Maintain current low coupling
2. Monitor new dependencies carefully
3. Consider extracting reusable components
