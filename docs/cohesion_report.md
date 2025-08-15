Module Cohesion Analysis Report
==================================================

Overall Cohesion Statistics:
  Total modules: 111
  Average cohesion: 0.589
  Minimum cohesion: 0.031
  Maximum cohesion: 1.000

Cohesion Distribution:
  High cohesion (>=0.7): 47 modules
  Medium cohesion (0.4-0.7): 31 modules
  Low cohesion (<0.4): 33 modules

Assessment: MODERATE - Some cohesion improvements needed

Low Cohesion Modules (48):
1. src.core.interfaces.extension_interfaces (cohesion: 0.031)
   Classes: 18, Functions: 0
   Suggestions:
   - CRITICAL: Very low cohesion - major refactoring needed
   - Split module into multiple focused modules
   - Group related functions into classes

2. src.core.interfaces.api_interfaces (cohesion: 0.033)
   Classes: 18, Functions: 0
   Suggestions:
   - CRITICAL: Very low cohesion - major refactoring needed
   - Split module into multiple focused modules
   - Group related functions into classes

3. src.core.interfaces.data_interfaces (cohesion: 0.033)
   Classes: 19, Functions: 0
   Suggestions:
   - CRITICAL: Very low cohesion - major refactoring needed
   - Split module into multiple focused modules
   - Group related functions into classes

4. src.core.interfaces.ml_interfaces (cohesion: 0.033)
   Classes: 16, Functions: 0
   Suggestions:
   - CRITICAL: Very low cohesion - major refactoring needed
   - Split module into multiple focused modules
   - Group related functions into classes

5. src.core.interfaces.pipeline_interfaces (cohesion: 0.033)
   Classes: 16, Functions: 0
   Suggestions:
   - CRITICAL: Very low cohesion - major refactoring needed
   - Split module into multiple focused modules
   - Group related functions into classes

6. src.core.interfaces.base_interfaces (cohesion: 0.036)
   Classes: 17, Functions: 0
   Suggestions:
   - CRITICAL: Very low cohesion - major refactoring needed
   - Split module into multiple focused modules
   - Group related functions into classes

7. src.core.interfaces.detector_interfaces (cohesion: 0.055)
   Classes: 15, Functions: 0
   Suggestions:
   - CRITICAL: Very low cohesion - major refactoring needed
   - Split module into multiple focused modules
   - Group related functions into classes

8. src.core.interfaces.service_interfaces (cohesion: 0.067)
   Classes: 15, Functions: 0
   Suggestions:
   - CRITICAL: Very low cohesion - major refactoring needed
   - Split module into multiple focused modules
   - Group related functions into classes

9. src.api.performance.connection_optimizer (cohesion: 0.120)
   Classes: 8, Functions: 0
   Suggestions:
   - CRITICAL: Very low cohesion - major refactoring needed
   - Split module into multiple focused modules
   - Group related functions into classes

10. src.core.dependency_injection.container (cohesion: 0.144)
   Classes: 8, Functions: 3
   Suggestions:
   - CRITICAL: Very low cohesion - major refactoring needed
   - Split module into multiple focused modules
   - Group related functions into classes

High Cohesion Examples (34):
1. src.api.rest.optimized_app (cohesion: 0.800)
2. src.core.interfaces.integration_examples (cohesion: 0.800)
3. src.training.trainers.active_learner (cohesion: 0.810)
4. src.core.quality.cohesion_analyzer (cohesion: 0.813)
5. src.training.trainers.enhanced_trainer (cohesion: 0.833)

Recommended Actions:
1. Focus on modules with cohesion < 0.6
2. Review method responsibilities and interactions
3. Extract utility functions to appropriate modules
4. Consider combining small related modules
