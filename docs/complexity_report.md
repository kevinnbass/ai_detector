Cyclomatic Complexity Analysis Report
==================================================

Found 17 functions with complexity > 10

1. generate_api_documentation (complexity: 16)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\docs\api\openapi_spec.py:35
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals
   - Extract loop logic into helper functions
   - Break function into smaller functions
   - Apply single responsibility principle
   - Reduce nesting depth using early returns
   - Extract nested logic into separate methods

2. _matches_filters (complexity: 16)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\src\core\repositories\base_repository.py:112
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals
   - Reduce nesting depth using early returns
   - Extract nested logic into separate methods

3. simple_data_collector (complexity: 16)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\src\data\collectors\simple_collector.py:11
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals
   - Reduce nesting depth using early returns
   - Extract nested logic into separate methods

4. identify_gpt4o_patterns (complexity: 14)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\src\core\patterns\gpt4o_miner.py:113
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals
   - Extract loop logic into helper functions

5. synonym_replacement (complexity: 14)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\src\data\processors\data_augmenter.py:28
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals
   - Extract loop logic into helper functions
   - Reduce nesting depth using early returns
   - Extract nested logic into separate methods

6. _generate_enhanced_report (complexity: 13)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\src\training\trainers\enhanced_trainer.py:293
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals

7. _analyze_performance_results (complexity: 13)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\tests\performance\run_performance_tests.py:399
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals
   - Reduce nesting depth using early returns
   - Extract nested logic into separate methods

8. _convert_query_to_filters (complexity: 12)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\src\core\abstractions\data_access_layer.py:211
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals
   - Reduce nesting depth using early returns
   - Extract nested logic into separate methods

9. validate_training_request (complexity: 12)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\src\core\abstractions\presentation_layer.py:179
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals
   - Reduce nesting depth using early returns
   - Extract nested logic into separate methods

10. _should_retry (complexity: 12)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\src\core\api_client\retry_handler.py:139
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals

11. interactive_labeling_session (complexity: 12)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\src\data\collectors\data_collector.py:280
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals
   - Reduce nesting depth using early returns
   - Extract nested logic into separate methods

12. _create_ensemble_prediction (complexity: 12)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\src\integrations\gemini\gemini_structured_analyzer.py:1331
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals
   - Reduce nesting depth using early returns
   - Extract nested logic into separate methods

13. comprehensive_validation_report (complexity: 12)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\src\training\validators\validator.py:214
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals
   - Break function into smaller functions
   - Apply single responsibility principle

14. _write_summary_report (complexity: 12)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\tests\performance\run_performance_tests.py:557
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals
   - Extract loop logic into helper functions

15. mock_database (complexity: 12)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\tests\python\conftest.py:276
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals
   - Extract loop logic into helper functions
   - Reduce nesting depth using early returns
   - Extract nested logic into separate methods

16. interactive_collection (complexity: 11)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\src\data\collectors\tweet_data_collector.py:153
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals
   - Reduce nesting depth using early returns
   - Extract nested logic into separate methods

17. _check_performance_requirements (complexity: 11)
   File: C:\Users\Kevin\ai_detector\src\core\quality\..\..\..\tests\performance\run_performance_tests.py:466
   Suggestions:
   - Extract conditional logic into separate methods
   - Consider using strategy pattern for complex conditionals

Refactoring Priority:
1. Functions with complexity > 15 (critical)
2. Functions with complexity 11-15 (high)
3. Functions with complexity 10-11 (medium)

Recommended Patterns:
- Strategy Pattern for complex conditionals
- Extract Method for long functions
- Guard Clauses for early returns
- Chain of Responsibility for validation
